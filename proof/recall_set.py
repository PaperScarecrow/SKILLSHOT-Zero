#!/usr/bin/env python
"""
SKILLSHOT proof — RECALL SET (CPU): the harder recall regimes beyond single-needle.

  multi-hop   : H-hop CHAINED recall. Bindings form a chain k0->k1->...->kH (plus distractors);
                query k0, answer = kH. Tests COMPOSITIONAL recall (the honest hard bar).
  multi-needle: CAPACITY. One query, but vary the number of bindings N in context. Tests the
                state-capacity / Omega(L) wall (a bounded-state memory must fail once N exceeds it).

Four mixers on equal footing (reused, stabilized, from mqar_bakeoff): dense softmax, top-k sparse
softmax, MIRAS/TITANS linear memory (O(n)), O(log n) hierarchical. (Compute-scaling/timing test is
intentionally skipped — this is a PoC about recall quality, not wall-clock.)
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from mqar_bakeoff import Net, MQAR, DEV   # reuse the stabilized mixers + Net


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class MultiHop:
    """Shared-symbol vocab so a value can be the next key (chaining). Answer = node H hops from k0."""
    def __init__(self, n_sym):
        self.n_sym = n_sym
        self.PAD = 0
        self.SYM0 = 1
        self.QMARK = 1 + n_sym
        self.VOCAB = self.QMARK + 1
        self.g = np.random.default_rng(0)

    def reseed(self, s=0):
        self.g = np.random.default_rng(s)

    def batch(self, L, n_pairs, hops, B):
        cap = (L - 2) // 2
        n_pairs = max(hops, min(n_pairs, cap, self.n_sym - 1))
        X = np.full((B, L), self.PAD, np.int64)
        Y = np.zeros(B, np.int64)
        for b in range(B):
            perm = self.g.permutation(self.n_sym)
            chain = perm[:hops + 1]                              # k0..kH
            used = set(int(c) for c in chain)
            keys = [int(chain[i]) for i in range(hops)]          # chain keys
            vals = [int(chain[i + 1]) for i in range(hops)]      # -> next node
            ptr = hops + 1
            while len(keys) < n_pairs and ptr < self.n_sym:      # distractor pairs (distinct keys)
                k = int(perm[ptr]); ptr += 1
                if k in used:
                    continue
                used.add(k)
                keys.append(k); vals.append(int(self.g.integers(0, self.n_sym)))
            pairs = list(zip(keys, vals))
            self.g.shuffle(pairs)
            seq = [self.PAD] * L
            for j, (k, v) in enumerate(pairs):
                seq[2 * j] = self.SYM0 + k
                seq[2 * j + 1] = self.SYM0 + v
            seq[L - 2] = self.QMARK
            seq[L - 1] = self.SYM0 + int(chain[0])               # query k0
            X[b] = seq
            Y[b] = self.SYM0 + int(chain[hops])                  # answer kH
        return torch.tensor(X), torch.tensor(Y)


def train_eval(kind, vocab, batch_fn, max_len, args):
    torch.manual_seed(0)
    m = Net(kind, vocab, args.d, args.layers, max_len, args.topk, args.hier_S).to(DEV)
    opt = torch.optim.Adam(m.parameters(), args.lr)
    lf = nn.CrossEntropyLoss()
    m.train()
    t0 = time.time()
    for _ in range(args.steps):
        X, Y = batch_fn()
        opt.zero_grad(); lf(m(X), Y).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval(); cor = tot = 0
    with torch.no_grad():
        for _ in range(args.eval_iters):
            X, Y = batch_fn()
            cor += (m(X).argmax(1) == Y).sum().item(); tot += len(Y)
    return cor / tot, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mixers", nargs="+", default=["dense", "topk", "miras", "hier"])
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--hops", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--needles", nargs="+", type=int, default=[4, 16, 32])
    p.add_argument("--n_sym", type=int, default=48)
    p.add_argument("--hop_pairs", type=int, default=10)
    p.add_argument("--needle_vocab", type=int, default=64)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--hier_S", type=int, default=8)
    p.add_argument("--eval_iters", type=int, default=6)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.steps = 60; args.hops = [1, 2]; args.needles = [4, 16]

    results = {"meta": {"d": args.d, "layers": args.layers, "L": args.L, "steps": args.steps},
               "multihop": [], "multineedle": []}

    # ---- multi-hop (compositional) ----
    mh = MultiHop(args.n_sym)
    log(f"MULTI-HOP: L={args.L} pairs={args.hop_pairs} chance={1/args.n_sym:.3f} steps={args.steps}")
    print(f"  {'hops':>5} " + " ".join(f"{m:>8}" for m in args.mixers))
    for H in args.hops:
        row = {}
        for kind in args.mixers:
            mh.reseed(0)
            bf = (lambda H=H: mh.batch(args.L, args.hop_pairs, H, args.batch))
            acc, dt = train_eval(kind, mh.VOCAB, bf, args.L + 4, args)
            row[kind] = acc
            results["multihop"].append({"hops": H, "mixer": kind, "acc": acc})
            log(f"    hop={H} {kind:>6} acc={acc:.3f} ({dt:.0f}s)")
        print(f"  {H:>5} " + " ".join(f"{row[m]:>8.3f}" for m in args.mixers))

    # ---- multi-needle (capacity) ----
    md = MQAR(args.needle_vocab, args.needle_vocab)
    log(f"MULTI-NEEDLE: L={args.L} chance={1/args.needle_vocab:.3f} vary #bindings, steps={args.steps}")
    print(f"  {'pairs':>5} " + " ".join(f"{m:>8}" for m in args.mixers))
    for N in args.needles:
        row = {}
        for kind in args.mixers:
            md.reseed(0)
            bf = (lambda N=N: md.batch(args.L, N, args.batch))
            acc, dt = train_eval(kind, md.VOCAB, bf, args.L + 4, args)
            row[kind] = acc
            results["multineedle"].append({"pairs": N, "mixer": kind, "acc": acc})
            log(f"    N={N} {kind:>6} acc={acc:.3f} ({dt:.0f}s)")
        print(f"  {N:>5} " + " ".join(f"{row[m]:>8.3f}" for m in args.mixers))

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recall_set_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"saved -> {path}")

    # ---- verdicts: does each mixer degrade with hop-depth / binding-count? ----
    def trend(rows, key, mixer):
        pts = sorted((r[key], r["acc"]) for r in rows if r["mixer"] == mixer)
        return pts[0], pts[-1]
    log("  VERDICTS:")
    for mixer in args.mixers:
        (h0, a0), (h1, a1) = trend(results["multihop"], "hops", mixer)
        tag = "DEGRADES" if a0 - a1 > 0.2 else "holds"
        log(f"    multihop  {mixer:>6}: {h0}->{h1} hops  {a0:.2f}->{a1:.2f}  ({tag})")
    for mixer in args.mixers:
        (n0, a0), (n1, a1) = trend(results["multineedle"], "pairs", mixer)
        tag = "DEGRADES" if a0 - a1 > 0.2 else "holds"
        log(f"    needle    {mixer:>6}: {n0}->{n1} bindings  {a0:.2f}->{a1:.2f}  ({tag})")


if __name__ == "__main__":
    main()
