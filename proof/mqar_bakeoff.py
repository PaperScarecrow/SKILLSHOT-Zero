#!/usr/bin/env python
"""
SKILLSHOT proof — Track B: long-context mechanism bake-off (CPU-only).

One MQAR (multi-query associative recall = the clean needle-in-a-haystack proxy)
harness, four token-mixers on EQUAL footing:

  dense   : full causal softmax attention            O(L^2) compute, O(L^2) mem
  topk    : top-k sparse softmax (the e4b lever)      O(L^2) compute (full scores), sparse read
  miras   : MIRAS/TITANS linear delta-rule memory     O(L) compute, O(d_k*d_v) state  <- the sub-quadratic one
  hier    : O(log n) hierarchical band summaries      O(L log L) compute (the structured-recall probe)

Two experiments:
  --task recall : train a fresh small model per (mixer, L, binding-regime), measure recall.
                  SPARSE bindings (few needles) vs DENSE bindings (needles ~ L/2) -> exposes the Omega(L) wall.
  --task timing : untrained forward-pass wall-clock vs L for each mixer -> the actual asymptotic-cost claim.
  --task smoke  : tiny fast run to validate correctness.

CPU-only by construction (we never touch cuda). Ports:
  - MirasMemoryGate math from skillshot/skillshot/memory_gate.py (batched here).
  - dense/hier from Nyxara/core/ologn_attention4.py.
  - top-k from integration/e4b_equivalence_harness.py (topk_eager).
"""
from __future__ import annotations
import argparse, json, time, math, os
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never touch the GPU
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cpu"
torch.manual_seed(0)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- data (MQAR)
class MQAR:
    """key/value pairs, then a query; predict the queried key's value.
    SPARSE: n_pairs fixed small. DENSE: n_pairs ~ L/2 (whole context is bindings)."""
    def __init__(self, n_keys, n_vals):
        self.n_keys, self.n_vals = n_keys, n_vals
        self.PAD = 0
        self.KEY0 = 1
        self.VAL0 = 1 + n_keys
        self.QMARK = 1 + n_keys + n_vals
        self.VOCAB = self.QMARK + 1
        self.g = np.random.default_rng(0)

    def reseed(self, seed=0):  # per-config reset -> every mixer sees identical data
        self.g = np.random.default_rng(seed)

    def batch(self, L, n_pairs, B):
        n_pairs = min(n_pairs, self.n_keys, (L - 2) // 2)
        X = np.full((B, L), self.PAD, np.int64)
        Y = np.zeros(B, np.int64)
        for b in range(B):
            keys = self.g.choice(self.n_keys, size=n_pairs, replace=False)
            vals = self.g.integers(0, self.n_vals, size=n_pairs)
            seq = [self.PAD] * L
            for j in range(n_pairs):
                seq[2 * j] = self.KEY0 + keys[j]
                seq[2 * j + 1] = self.VAL0 + vals[j]
            qi = self.g.integers(0, n_pairs)
            seq[L - 2] = self.QMARK
            seq[L - 1] = self.KEY0 + keys[qi]
            X[b] = seq
            Y[b] = self.VAL0 + vals[qi]
        return torch.tensor(X), torch.tensor(Y)


# ---------------------------------------------------------------- mixers
class DenseAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.scale = d ** -0.5

    def forward(self, x):
        B, L, d = x.shape
        Q, K, V = self.q(x), self.k(x), self.v(x)
        sc = (Q @ K.transpose(1, 2)) * self.scale
        sc = sc + torch.triu(torch.full((L, L), float("-inf")), 1)
        return self.o(torch.softmax(sc, -1) @ V)


class TopKAttn(nn.Module):
    """full O(L^2) scores, keep top-k per query (the e4b topk_eager lever)."""
    def __init__(self, d, k=16):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.scale = d ** -0.5
        self.topk = k

    def forward(self, x):
        B, L, d = x.shape
        Q, K, V = self.q(x), self.k(x), self.v(x)
        sc = (Q @ K.transpose(1, 2)) * self.scale
        sc = sc + torch.triu(torch.full((L, L), float("-inf")), 1)
        k_eff = min(self.topk, L)
        kth = sc.topk(k_eff, dim=-1).values[..., -1:]      # (B,L,1) kth-largest per row
        sc = sc.masked_fill(sc < kth, float("-inf"))
        return self.o(torch.softmax(sc, -1) @ V)


class MirasMixer(nn.Module):
    """Batched MIRAS/TITANS delta-rule matrix memory (port of MirasMemoryGate).
    Sequential recurrence over T (O(L)); vectorized over batch. State M:(B,dk,dv)."""
    def __init__(self, d, dk=None, dv=None):
        super().__init__()
        self.dk = dk or d
        self.dv = dv or d
        # depthwise causal short conv: lets the linear memory bind adjacent tokens
        # (key->value). Standard in DeltaNet/Mamba/Based; required for assoc. recall.
        self.kconv = 3
        self.conv = nn.Conv1d(d, d, kernel_size=self.kconv, groups=d, bias=True)
        self.q_proj = nn.Linear(d, self.dk, bias=False)
        self.k_proj = nn.Linear(d, self.dk, bias=False)
        self.v_proj = nn.Linear(d, self.dv, bias=False)
        self.retention_head = nn.Linear(d, 1)   # alpha_t  (MIRAS retention gate, ->(0,1))
        self.surprise_head = nn.Linear(d, 1)    # beta_t   (MIRAS write strength, ->(0,1))
        self.momentum = nn.Parameter(torch.tensor(2.2))  # TITANS momentum (sigmoid->~0.9)
        self.read_proj = nn.Linear(self.dv, d, bias=False)
        self.out_gate = nn.Sequential(
            nn.Linear(d * 2, max(8, d // 4)), nn.SiLU(),
            nn.Linear(max(8, d // 4), d), nn.Sigmoid())

    def _short_conv(self, x):                                 # strictly-causal depthwise conv
        B, T, d = x.shape
        z = F.pad(x.transpose(1, 2), (self.kconv - 1, 0))     # left-pad k-1
        return self.conv(z)[..., :T].transpose(1, 2)

    def forward(self, x):
        B, T, d = x.shape
        xc = self._short_conv(x)
        q, k, v = self.q_proj(xc), self.k_proj(xc), self.v_proj(xc)
        q = F.normalize(q, dim=-1)                             # unit keys/queries -> bounded outer products
        k = F.normalize(k, dim=-1)
        alpha = torch.sigmoid(self.retention_head(x))         # (B,T,1) retention in (0,1)
        beta = torch.sigmoid(self.surprise_head(x))           # (B,T,1) write strength in (0,1)
        mom = torch.sigmoid(self.momentum)                    # scalar in (0,1)
        M = torch.zeros(B, self.dk, self.dv)
        S = torch.zeros(B, self.dk, self.dv)
        reads = []
        for t in range(T):
            q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]         # (B,dk),(B,dk),(B,dv)
            reads.append(torch.einsum("bkv,bk->bv", M, q_t))  # read BEFORE write
            Mk = torch.einsum("bkv,bk->bv", M, k_t)           # (B,dv) current recall of k_t
            surprise = torch.einsum("bk,bv->bkv", k_t, v_t - Mk)   # delta rule (self-correcting)
            S = mom * S + (1.0 - mom) * surprise              # bounded EMA momentum (TITANS)
            M = alpha[:, t].unsqueeze(-1) * M + beta[:, t].unsqueeze(-1) * S  # gated write (MIRAS)
        read = self.read_proj(torch.stack(reads, 1))          # (B,T,d)
        g = self.out_gate(torch.cat([x, read], -1))
        return g * read                                       # contribution; Block adds residual


class HierAttn(nn.Module):
    """O(log n) hierarchical band summaries (port of ologn_attention4 'hier')."""
    def __init__(self, d, S=8, w_local=4):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.scale = d ** -0.5
        self.S, self.w = S, w_local

    def forward(self, x):
        B, L, d = x.shape
        Q, K, V = self.q(x), self.k(x), self.v(x)
        idx = torch.arange(L)
        csK = torch.cat([torch.zeros(B, 1, d), K.cumsum(1)], 1)
        csV = torch.cat([torch.zeros(B, 1, d), V.cumsum(1)], 1)
        keys, vals, valid = [], [], []
        for w in range(self.w):
            pos = (idx - w).clamp(0)
            keys.append(K[:, pos]); vals.append(V[:, pos]); valid.append(idx - w >= 0)
        l = 1
        while self.w * (2 ** (l - 1)) < L:
            bw = self.w * (2 ** (l - 1)); ns = min(self.S, bw); sw = max(1, bw // ns)
            base = idx + 1 - self.w * (2 ** l)
            for j in range(ns):
                lo = base + j * sw; hi = base + (j + 1) * sw
                loc = lo.clamp(0, L); hic = hi.clamp(0, L); cnt = (hic - loc).clamp(min=1)
                keys.append((csK[:, hic] - csK[:, loc]) / cnt[None, :, None])
                vals.append((csV[:, hic] - csV[:, loc]) / cnt[None, :, None])
                valid.append(lo >= 0)
            l += 1
        Kk = torch.stack(keys, 2); Vk = torch.stack(vals, 2); vm = torch.stack(valid, 1)
        sc = (Q[:, :, None, :] * Kk).sum(-1) * self.scale
        sc = sc.masked_fill(~vm[None], float("-inf"))
        a = torch.softmax(sc, -1)
        return self.o((a[..., None] * Vk).sum(2))


def make_mixer(kind, d, topk, hier_S):
    if kind == "dense": return DenseAttn(d)
    if kind == "topk":  return TopKAttn(d, k=topk)
    if kind == "miras": return MirasMixer(d)
    if kind == "hier":  return HierAttn(d, S=hier_S)
    raise ValueError(kind)


class Block(nn.Module):
    def __init__(self, kind, d, topk, hier_S):
        super().__init__()
        self.mix = make_mixer(kind, d, topk, hier_S)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.ff(self.n2(x))


class Net(nn.Module):
    def __init__(self, kind, vocab, d, nlayers, max_len, topk, hier_S):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
        self.blocks = nn.ModuleList([Block(kind, d, topk, hier_S) for _ in range(nlayers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.emb(x) + self.pos[:x.shape[1]][None]
        for b in self.blocks:
            h = b(h)
        return self.head(h[:, -1])


def n_params(m): return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- experiments
def run_recall(args):
    regimes = {"sparse": (lambda L: args.sparse_pairs),
               "dense":  (lambda L: max(args.sparse_pairs, (L - 2) // 2))}
    regimes = {k: v for k, v in regimes.items() if k in args.regimes}
    max_pairs = max(max(r(L) for L in args.lens) for r in regimes.values())
    n_kv = max(args.sparse_pairs, max_pairs, args.min_vocab) + 2
    data = MQAR(n_kv, n_kv)
    chance = 1.0 / data.n_vals
    log(f"recall: vocab={data.VOCAB} chance={chance:.3f} d_model={args.d} layers={args.layers} "
        f"batch={args.batch} steps={args.steps} mixers={args.mixers}")
    results = {"meta": {"task": "recall", "chance": chance, "d_model": args.d,
                        "layers": args.layers, "batch": args.batch, "steps": args.steps,
                        "lens": args.lens, "topk": args.topk, "hier_S": args.hier_S,
                        "sparse_pairs": args.sparse_pairs}, "runs": []}
    for regime, pf in regimes.items():
        print(f"\n=== bindings: {regime.upper()} ===")
        header = f"  {'L':>5} {'pairs':>6} " + " ".join(f"{m:>8}" for m in args.mixers)
        print(header)
        for L in args.lens:
            n_pairs = min(pf(L), data.n_keys, (L - 2) // 2)
            row = {}
            for kind in args.mixers:
                acc = _train_eval(kind, data, L, n_pairs, args)
                row[kind] = acc
                results["runs"].append({"regime": regime, "L": L, "n_pairs": n_pairs,
                                        "mixer": kind, "recall": acc})
            print(f"  {L:>5} {n_pairs:>6} " + " ".join(f"{row[m]:>8.3f}" for m in args.mixers))
    _save(args, results)
    _verdict_recall(results, chance)
    return results


def _train_eval(kind, data, L, n_pairs, args):
    torch.manual_seed(0)            # identical init per mixer
    data.reseed(0)                  # identical data stream per mixer
    m = Net(kind, data.VOCAB, args.d, args.layers, max(args.lens) + 4,
            args.topk, args.hier_S).to(DEV)
    opt = torch.optim.Adam(m.parameters(), args.lr)
    lf = nn.CrossEntropyLoss()
    m.train()
    t0 = time.time()
    for st in range(args.steps):
        X, Y = data.batch(L, n_pairs, args.batch)
        opt.zero_grad(); loss = lf(m(X), Y); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)  # tame BPTT explosions
        opt.step()
    m.eval(); cor = tot = 0
    with torch.no_grad():
        for _ in range(args.eval_iters):
            X, Y = data.batch(L, n_pairs, args.eval_batch)
            cor += (m(X).argmax(1) == Y).sum().item(); tot += len(Y)
    dt = time.time() - t0
    log(f"  {kind:>5} L={L:<5} pairs={n_pairs:<3} acc={cor/tot:.3f} "
        f"({dt:.0f}s, {n_params(m)/1e3:.0f}k params)")
    return cor / tot


def run_timing(args):
    data = MQAR(args.sparse_pairs + 2, args.sparse_pairs + 2)
    log(f"timing: forward-pass wall-clock vs L, mixers={args.mixers}, reps={args.reps}")
    results = {"meta": {"task": "timing", "d_model": args.d, "layers": args.layers,
                        "batch": args.time_batch, "lens": args.time_lens,
                        "topk": args.topk, "reps": args.reps}, "runs": []}
    print(f"\n  {'L':>6} " + " ".join(f"{m+' ms':>12}" for m in args.mixers))
    times = {m: [] for m in args.mixers}
    for L in args.time_lens:
        cells = {}
        for kind in args.mixers:
            torch.manual_seed(0)
            m = Net(kind, data.VOCAB, args.d, args.layers, L + 4, args.topk, args.hier_S).to(DEV)
            m.eval()
            X, _ = data.batch(L, args.sparse_pairs, args.time_batch)
            with torch.no_grad():
                m(X)  # warmup
                reps = []
                for _ in range(args.reps):
                    t0 = time.perf_counter(); m(X); reps.append((time.perf_counter() - t0) * 1e3)
            ms = float(np.median(reps))
            cells[kind] = ms; times[kind].append(ms)
            results["runs"].append({"L": L, "mixer": kind, "ms": ms})
        print(f"  {L:>6} " + " ".join(f"{cells[m]:>12.1f}" for m in args.mixers))
    _save(args, results)
    _verdict_timing(results, args)
    return results


def _slope(xs, ys):
    """log-log least-squares slope: time ~ L^slope."""
    lx = np.log(np.array(xs, float)); ly = np.log(np.array(ys, float))
    return float(np.polyfit(lx, ly, 1)[0])


def _verdict_timing(results, args):
    print("\n  log-log scaling slope (time ~ L^s):")
    by = {}
    for r in results["runs"]:
        by.setdefault(r["mixer"], ([], []))
        by[r["mixer"]][0].append(r["L"]); by[r["mixer"]][1].append(r["ms"])
    for m in args.mixers:
        xs, ys = by[m]
        s = _slope(xs, ys)
        tag = "~O(L) LINEAR" if s < 1.4 else ("~O(L^2) QUADRATIC" if s > 1.7 else "~superlinear")
        print(f"    {m:>6}: slope={s:.2f}  {tag}")
    print("    -> miras should be ~1 (linear); dense/topk ~2 (quadratic). That gap IS the sub-quadratic win.")


def _verdict_recall(results, chance):
    # Omega(L) wall: does recall decay with L under DENSE bindings?
    dense = [r for r in results["runs"] if r["regime"] == "dense"]
    if not dense:
        return
    print("\n  Omega(L) check (DENSE bindings, recall vs L):")
    by = {}
    for r in dense:
        by.setdefault(r["mixer"], []).append((r["L"], r["recall"]))
    for m, pts in by.items():
        pts.sort()
        first, last = pts[0][1], pts[-1][1]
        trend = "DECAYS" if first - last > 0.15 else "holds"
        print(f"    {m:>6}: {pts[0][0]}->{pts[-1][0]} tok: {first:.2f}->{last:.2f}  ({trend})")
    print("    -> arbitrary (dense) recall is Omega(L): expect decay at fixed state. Sparse needle is the easy bar.")


def _save(args, results):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"bakeoff_{results['meta']['task']}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"saved -> {path}")


# ---------------------------------------------------------------- cli
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["smoke", "recall", "timing"], default="smoke")
    p.add_argument("--mixers", nargs="+", default=["dense", "topk", "miras", "hier"])
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lens", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--sparse_pairs", type=int, default=6)
    p.add_argument("--regimes", nargs="+", default=["sparse", "dense"])
    p.add_argument("--min_vocab", type=int, default=32)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--hier_S", type=int, default=8)
    p.add_argument("--eval_iters", type=int, default=6)
    p.add_argument("--eval_batch", type=int, default=256)
    # timing
    p.add_argument("--time_lens", nargs="+", type=int, default=[128, 256, 512, 1024, 2048])
    p.add_argument("--time_batch", type=int, default=8)
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args()

    if args.task == "smoke":
        log("SMOKE: tiny correctness check")
        args.steps = 60; args.lens = [64]; args.batch = 32
        args.eval_iters = 2; args.eval_batch = 128
        run_recall(args)
        args.time_lens = [128, 256, 512]; args.reps = 3
        run_timing(args)
    elif args.task == "recall":
        run_recall(args)
    else:
        run_timing(args)


if __name__ == "__main__":
    main()
