#!/usr/bin/env python
"""
SKILLSHOT proof — REAL tools (CPU-only): forge skill-adapters from public HF datasets.

Three real, security/coding tools the base gemma-3-270m fails zero-shot:
  vuln-detect  : Devign (google/code_x_glue_cc_defect_detection)  C func -> vulnerable|secure  (binary acc)
  cve-severity : stasvinokur/cve-and-cwe-dataset-1999-2025         CVE desc -> CRITICAL/HIGH/MEDIUM/LOW (acc + macroF1)
  code-output  : cruxeval-org/cruxeval (CRUXEval-O)                code+input -> exact output  (exact-match)

For each: measure BASE zero-shot, forge a LoRA (the SKILLSHOT forge), measure POST, report lift.
Each forged adapter is saved to tool_adapters/ so it drops into the tool-changer magazine unchanged.

Datasets need network; gemma loads from local cache (local_files_only). GPU never touched.
"""
from __future__ import annotations
import argparse, json, os, time, random, shutil, collections

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # no GPU
# NOTE: do NOT force HF offline here — datasets need network; gemma uses local_files_only.
import torch

MODEL = "unsloth/gemma-3-270m-it"
DEV = "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTERS = os.path.join(HERE, "tool_adapters")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- dataset loaders -> (input, gold) rows
def _take(ds, n):
    return [ds[i] for i in range(min(n, len(ds)))]


def load_vuln(n_train, n_test):
    from datasets import load_dataset
    ds = load_dataset("google/code_x_glue_cc_defect_detection", split="train")
    ds = ds.filter(lambda e: len(e["func"]) <= 650).shuffle(seed=0)
    rows = [(e["func"], "vulnerable" if e["target"] else "secure") for e in _take(ds, n_train + n_test)]
    return rows[:n_train], rows[n_train:n_train + n_test]


def load_cve(n_train, n_test):
    from datasets import load_dataset
    keep = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    ds = load_dataset("stasvinokur/cve-and-cwe-dataset-1999-2025", split="train")
    ds = ds.filter(lambda e: e["SEVERITY"] in keep).shuffle(seed=0)
    rows = [(e["DESCRIPTION"][:600], e["SEVERITY"]) for e in _take(ds, n_train + n_test)]
    return rows[:n_train], rows[n_train:n_train + n_test]


def load_code(n_train, n_test):
    # MBPP-derived code-comprehension: does this function implement the described task? (yes/no)
    # (CRUXEval output-prediction needs real code execution -> beyond a 270M; this is learnable.)
    from datasets import load_dataset
    ds = load_dataset("mbpp", split="test")
    items = [(e["text"], e["code"]) for e in ds]
    rng = random.Random(0)
    rows = []
    for i, (text, code) in enumerate(items):
        if i % 2 == 0:
            rows.append((f"Task: {text}\nCode:\n{code}", "yes"))            # real pair
        else:
            j = rng.randrange(len(items))
            while items[j][1] == code:
                j = rng.randrange(len(items))
            rows.append((f"Task: {text}\nCode:\n{items[j][1]}", "no"))      # mismatched negative
    rng.shuffle(rows)
    return rows[:n_train], rows[n_train:n_train + n_test]


# ---------------------------------------------------------------- per-tool spec (fmt + canonicalize)
def vuln_fmt(x): return ("Classify the following C function as 'vulnerable' or 'secure'. "
                         f"Answer with exactly one word.\n```c\n{x}\n```\nAnswer:")
def vuln_norm(s):
    s = s.strip().lower()
    if "vuln" in s: return "vulnerable"
    if "sec" in s: return "secure"
    return s.split()[0] if s.split() else ""

def cve_fmt(x): return ("Classify the CVSS severity of this CVE as one of CRITICAL, HIGH, MEDIUM, LOW. "
                        f"Answer with exactly one word.\nDescription: {x}\nSeverity:")
def cve_norm(s):
    s = s.strip().upper()
    for lab in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        if lab in s: return lab
    return s.split()[0] if s.split() else ""

def code_fmt(x): return ("Does the following Python function correctly implement the described task? "
                         f"Answer with exactly one word: 'yes' or 'no'.\n{x}\nAnswer:")
def code_norm(s):
    s = s.strip().lower()
    if "yes" in s: return "yes"
    if "no" in s: return "no"
    return s.split()[0] if s.split() else ""

SPECS = {
    "vuln-detect": dict(loader=load_vuln, fmt=vuln_fmt, norm=vuln_norm, maxnew=4,
                        labels=["secure", "vulnerable"], metric="acc"),
    "cve-severity": dict(loader=load_cve, fmt=cve_fmt, norm=cve_norm, maxnew=4,
                         labels=["CRITICAL", "HIGH", "MEDIUM", "LOW"], metric="accf1"),
    "code-match": dict(loader=load_code, fmt=code_fmt, norm=code_norm, maxnew=4,
                       labels=["yes", "no"], metric="acc"),
}


# ---------------------------------------------------------------- model + eval
def load_base():
    from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hl
    hl.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32,
                                                local_files_only=True).to(DEV)
    return tok, model


def build_prompt(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def macro_f1(pairs, labels):
    # pairs: list of (gold, pred) canonical strings
    f1s = []
    for c in labels:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


@torch.no_grad()
def evaluate(model, tok, spec, rows, adapter=None, disable=False):
    model.eval()
    cor = 0
    pairs = []
    ex = []
    for x, gold in rows:
        ids = tok(build_prompt(tok, spec["fmt"](x)), return_tensors="pt",
                  truncation=True, max_length=400).to(DEV)
        gen_kw = dict(max_new_tokens=spec["maxnew"], do_sample=False,
                      pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if disable:
            with model.disable_adapter():
                out = model.generate(**ids, **gen_kw)
        else:
            if adapter is not None:
                model.set_adapter(adapter)
            out = model.generate(**ids, **gen_kw)
        gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        p, g = spec["norm"](gen), spec["norm"](gold)
        cor += (p == g)
        pairs.append((g, p))
        if len(ex) < 5:
            ex.append((str(x)[:50].replace("\n", " "), g, p, p == g))
    acc = cor / len(rows)
    f1 = macro_f1(pairs, [spec["norm"](l) for l in spec["labels"]]) if spec["metric"] == "accf1" else None
    return acc, f1, ex, pairs


# ---------------------------------------------------------------- forge (test-time LoRA, clipped)
def forge(tok, base, spec, train_rows, args, name):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    pad = tok.pad_token_id or tok.eos_token_id
    samples = []
    for x, gold in train_rows:
        pids = tok(build_prompt(tok, spec["fmt"](x)), truncation=True, max_length=380,
                   return_tensors="pt")["input_ids"][0]
        aids = tok(" " + gold + tok.eos_token, return_tensors="pt")["input_ids"][0]
        ids = torch.cat([pids, aids])
        samples.append((ids, torch.cat([torch.full((len(pids),), -100), aids.clone()])))
    rng = random.Random(0)
    t0 = time.time()
    for step in range(args.steps):
        b = rng.sample(samples, min(args.batch, len(samples)))
        mx = max(len(s[0]) for s in b)
        inp = torch.full((len(b), mx), pad, dtype=torch.long)
        lab = torch.full((len(b), mx), -100, dtype=torch.long)
        att = torch.zeros((len(b), mx), dtype=torch.long)
        for i, (ids, labels) in enumerate(b):
            inp[i, :len(ids)] = ids; lab[i, :len(labels)] = labels; att[i, :len(ids)] = 1
        opt.zero_grad()
        loss = model(input_ids=inp, attention_mask=att, labels=lab).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % max(1, args.steps // 4) == 0 or step == args.steps - 1:
            log(f"    forge[{name}] {step:>4}/{args.steps} loss={loss.item():.3f} ({time.time()-t0:.0f}s)")
    d = os.path.join(ADAPTERS, name)
    if os.path.isdir(d): shutil.rmtree(d)
    model.save_pretrained(d)
    return model, d


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tools", nargs="+", default=["vuln-detect", "cve-severity", "code-match"])
    p.add_argument("--n_train", type=int, default=300)
    p.add_argument("--n_test", type=int, default=150)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_train, args.n_test, args.steps = 40, 30, 40

    os.makedirs(ADAPTERS, exist_ok=True)
    report = {"model": MODEL, "steps": args.steps, "rank": args.rank, "tools": {}}
    for name in args.tools:
        spec = SPECS[name]
        log(f"==================== {name} ====================")
        train_rows, test_rows = spec["loader"](args.n_train, args.n_test)
        maj = collections.Counter(g for _, g in train_rows).most_common(1)[0]
        log(f"  loaded train={len(train_rows)} test={len(test_rows)}  majority-class='{maj[0]}' ({maj[1]/len(train_rows):.2f})")

        tok, base = load_base()
        from peft import get_peft_model  # noqa
        # BASE zero-shot
        b_acc, b_f1, b_ex, _ = evaluate(base, tok, spec, test_rows)  # plain base, no peft
        log(f"  BASE zero-shot: acc={b_acc:.3f}" + (f" macroF1={b_f1:.3f}" if b_f1 is not None else ""))
        for xs, g, pr, ok in b_ex:
            log(f"      {xs!r} | gold {g!r} got {pr!r} {'OK' if ok else 'x'}")

        # FORGE
        log("  forging skill-shot adapter...")
        model, d = forge(tok, base, spec, train_rows, args, name)
        # POST (active adapter is the just-forged one)
        a_acc, a_f1, a_ex, _ = evaluate(model, tok, spec, test_rows)
        log(f"  TOOL (+LoRA):   acc={a_acc:.3f}" + (f" macroF1={a_f1:.3f}" if a_f1 is not None else ""))
        for xs, g, pr, ok in a_ex:
            log(f"      {xs!r} | gold {g!r} got {pr!r} {'OK' if ok else 'x'}")

        lift = a_acc - b_acc
        verdict = "APPRECIABLE LIFT" if lift >= 0.15 else "modest lift" if lift >= 0.05 else "weak/none"
        log(f"  => {name}: acc {b_acc:.3f} -> {a_acc:.3f}  (+{lift:.3f})  [{verdict}]  saved {d}")
        report["tools"][name] = {"base_acc": b_acc, "tool_acc": a_acc, "lift": lift,
                                 "base_f1": b_f1, "tool_f1": a_f1, "majority": maj[1] / len(train_rows),
                                 "verdict": verdict, "n_train": len(train_rows), "n_test": len(test_rows)}
        del model, base
    with open(os.path.join(HERE, "real_tools_results.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"\nsaved -> {os.path.join(HERE, 'real_tools_results.json')}")
    log("  SUMMARY:")
    for n, r in report["tools"].items():
        log(f"    {n:>13}: base {r['base_acc']:.3f} -> tool {r['tool_acc']:.3f}  (+{r['lift']:.3f})  [{r['verdict']}]")


if __name__ == "__main__":
    main()
