#!/usr/bin/env python
"""
SKILLSHOT proof — Track A: skill-shot lift on a near-useless 270M base (CPU-only).

Thesis under test: gemma-3-270m alone is near-useless at a held-out procedural skill,
but a *skill-shot* adapter (a small LoRA forged at test time on a handful of examples —
the SKILLSHOT 'project -> test -> memorialize' forge) lifts it appreciably, and the lift
GENERALIZES to held-out instances (skill, not memorization).

Two skills (train/test inputs always DISJOINT, so test = genuine transfer):
  reverse : reverse the letters of a word. HARD for exact-match (subword tokenizer fights
            char-level output) -> shows lift on a graded char-similarity metric.
  cipher  : apply a fixed secret digit->digit substitution elementwise. Tokenization-friendly
            (space-separated single-token digits) -> shows a CLEAN exact-match jump.

Metrics: exact-match + graded char-similarity, base vs +skill-shot LoRA.
Kill criterion: no appreciable generalizing lift on either metric.

CPU-only; uses the model already in the HF cache (offline, no download).
"""
from __future__ import annotations
import argparse, json, os, time, random, string, difflib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never touch the GPU
os.environ.setdefault("HF_HUB_OFFLINE", "1")       # use local cache only (gemma is gated)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch

MODEL = "unsloth/gemma-3-270m-it"  # fully cached, weights present, non-gated mirror
DEV = "cpu"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- skills
def build_skill(name, rng):
    if name == "reverse":
        def gen_input():
            return "".join(rng.choice(list(string.ascii_lowercase)) for _ in range(rng.randint(4, 8)))
        def apply(x): return x[::-1]
        instr = "Reverse the letters of this word. Output only the reversed word, nothing else."
        def fmt(x): return f"{instr}\nWord: {x}"
        def parse(gen, gold):
            return gen.split()[0].strip(string.punctuation) if gen.split() else ""
        return dict(name=name, gen_input=gen_input, apply=apply, fmt=fmt, parse=parse, maxnew=16)
    if name == "cipher":
        perm = list(range(10)); rng.shuffle(perm)   # the fixed secret mapping = the skill
        log(f"  cipher secret perm: {perm}")
        def gen_input():
            return " ".join(str(rng.randint(0, 9)) for _ in range(rng.randint(4, 7)))
        def apply(x): return " ".join(str(perm[int(d)]) for d in x.split())
        instr = ("Apply the secret digit substitution to each digit, in order. "
                 "Output only the resulting digits separated by single spaces.")
        def fmt(x): return f"{instr}\nInput: {x}"
        def parse(gen, gold):
            k = len(gold.split())
            return " ".join(gen.split()[:k])
        return dict(name=name, gen_input=gen_input, apply=apply, fmt=fmt, parse=parse, maxnew=24)
    raise ValueError(name)


def gen_unique(skill, n):
    out, seen = [], set()
    guard = 0
    while len(out) < n and guard < n * 200:
        x = skill["gen_input"](); guard += 1
        if x in seen:
            continue
        seen.add(x); out.append(x)
    return out


# ---------------------------------------------------------------- prompting / eval
def build_prompt(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def evaluate(model, tok, skill, inputs):
    model.eval()
    cor = 0
    sims = []
    preds = []
    for x in inputs:
        gold = skill["apply"](x)
        ids = tok(build_prompt(tok, skill["fmt"](x)), return_tensors="pt").to(DEV)
        out = model.generate(**ids, max_new_tokens=skill["maxnew"], do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred = skill["parse"](gen, gold)
        cor += (pred == gold)
        sims.append(difflib.SequenceMatcher(None, pred, gold).ratio())
        preds.append((x, gold, pred, pred == gold))
    return cor / len(inputs), sum(sims) / len(sims), preds


# ---------------------------------------------------------------- skill-shot forge (test-time LoRA)
def forge_skill(model, tok, skill, train_inputs, args):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    samples = []  # prompt masked out of the loss; only the answer is learned
    for x in train_inputs:
        gold = skill["apply"](x)
        pids = tok(build_prompt(tok, skill["fmt"](x)), return_tensors="pt")["input_ids"][0]
        aids = tok(gold + tok.eos_token, return_tensors="pt")["input_ids"][0]
        ids = torch.cat([pids, aids])
        labels = torch.cat([torch.full((len(pids),), -100), aids.clone()])
        samples.append((ids, labels))

    rng = random.Random(0)
    pad = tok.pad_token_id or tok.eos_token_id
    t0 = time.time()
    for step in range(args.steps):
        batch = rng.sample(samples, min(args.batch, len(samples)))
        maxlen = max(len(s[0]) for s in batch)
        inp = torch.full((len(batch), maxlen), pad, dtype=torch.long)
        lab = torch.full((len(batch), maxlen), -100, dtype=torch.long)
        att = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, (ids, labels) in enumerate(batch):
            inp[i, :len(ids)] = ids; lab[i, :len(labels)] = labels; att[i, :len(ids)] = 1
        opt.zero_grad()
        loss = model(input_ids=inp.to(DEV), attention_mask=att.to(DEV), labels=lab.to(DEV)).loss
        loss.backward(); opt.step()
        if step % max(1, args.steps // 8) == 0 or step == args.steps - 1:
            log(f"  forge step {step:>4}/{args.steps}  loss={loss.item():.3f}  ({time.time()-t0:.0f}s)")
    return model


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skill", choices=["reverse", "cipher"], default="reverse")
    p.add_argument("--n_train", type=int, default=200)
    p.add_argument("--n_test", type=int, default=80)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_train, args.n_test, args.steps = 80, 20, 80

    from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging
    hf_logging.set_verbosity_error()  # silence generate max_length warning spam
    log(f"loading {MODEL} on CPU (float32)...")
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32,
                                                local_files_only=True).to(DEV)
    log(f"loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    rng = random.Random(0)
    skill = build_skill(args.skill, rng)
    pool = gen_unique(skill, args.n_train + args.n_test)
    train_inputs, test_inputs = pool[:args.n_train], pool[args.n_train:]
    log(f"skill={args.skill!r}  train={len(train_inputs)} test={len(test_inputs)} (disjoint)")

    base_em, base_sim, base_preds = evaluate(model, tok, skill, test_inputs)
    log(f"BASE  gemma-3-270m-it   exact={base_em:.3f}  char-sim={base_sim:.3f}")
    for x, g, pr, ok in base_preds[:5]:
        log(f"    {x!r:>14} -> gold {g!r:>14} | got {pr!r:>14} {'OK' if ok else 'x'}")

    log("forging skill-shot LoRA (test-time)...")
    model = forge_skill(model, tok, skill, train_inputs, args)

    sk_em, sk_sim, sk_preds = evaluate(model, tok, skill, test_inputs)
    log(f"SKILL-SHOT (+LoRA)      exact={sk_em:.3f}  char-sim={sk_sim:.3f}")
    for x, g, pr, ok in sk_preds[:5]:
        log(f"    {x!r:>14} -> gold {g!r:>14} | got {pr!r:>14} {'OK' if ok else 'x'}")

    lift_em, lift_sim = sk_em - base_em, sk_sim - base_sim
    verdict = ("APPRECIABLE LIFT" if (lift_em >= 0.3 or lift_sim >= 0.25) else
               "modest lift" if (lift_em >= 0.1 or lift_sim >= 0.1) else "NO appreciable lift (kill)")
    log(f"\n  exact-match: {base_em:.3f} -> {sk_em:.3f}  (+{lift_em:.3f})")
    log(f"  char-sim:    {base_sim:.3f} -> {sk_sim:.3f}  (+{lift_sim:.3f})  => {verdict}")
    out = {"model": MODEL, "skill": args.skill,
           "base_exact": base_em, "skill_exact": sk_em, "lift_exact": lift_em,
           "base_charsim": base_sim, "skill_charsim": sk_sim, "lift_charsim": lift_sim,
           "verdict": verdict, "n_train": len(train_inputs), "n_test": len(test_inputs),
           "steps": args.steps, "rank": args.rank, "lr": args.lr}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"skillshot_lift_{args.skill}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"saved -> {path}")


if __name__ == "__main__":
    main()
