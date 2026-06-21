#!/usr/bin/env python
"""
SKILLSHOT proof — the full TOOL-CHANGER POC (CPU-only).

One frozen machine (gemma-3-270m) + a magazine of LoRA "tools" (skill-adapters) +
a changer (router) that swaps in the right tool per job -- and forges a NEW tool at
runtime on a cache miss (project -> test -> memorialize).

Demonstrates end-to-end on the REAL model:
  1. magazine   : forge N distinct skill-adapters (the tools).
  2. base-alone : machine with no tool fails every skill (~0).
  3. tool-change: route query -> set_adapter -> answer. routing acc + answer acc.
  4. wrong-tool : force the wrong adapter -> fails (proves the *change* matters).
  5. miss->forge: an unseen skill -> router miss -> forge at runtime -> test-gate ->
                  memorialize into the magazine -> now served.

CPU-only, offline, uses the cached unsloth/gemma-3-270m-it mirror.
"""
from __future__ import annotations
import argparse, json, os, time, random, shutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch

MODEL = "unsloth/gemma-3-270m-it"
DEV = "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTERS = os.path.join(HERE, "tool_adapters")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- the tools (skills)
def build_skills(rng):
    perm = list(range(10)); rng.shuffle(perm)
    def seq(): return " ".join(str(rng.randint(0, 9)) for _ in range(rng.randint(4, 7)))
    def mk(instr, kw, apply):
        def fmt(x): return f"{instr}\nInput: {x}"
        def parse(gen, gold): return " ".join(gen.split()[:len(gold.split())])
        return dict(instr=instr, kw=kw, gen_input=seq, apply=apply, fmt=fmt, parse=parse, maxnew=24)
    skills = {
        "cipher": mk("Apply the secret digit substitution to each digit, in order. "
                     "Output only the resulting digits separated by single spaces.",
                     ["substitution", "substitute"],
                     lambda x: " ".join(str(perm[int(d)]) for d in x.split())),
        "sort":   mk("Sort these digits into ascending order. "
                     "Output only the sorted digits separated by single spaces.",
                     ["sort", "ascending"],
                     lambda x: " ".join(sorted(x.split(), key=int))),
        "addmod": mk("Add 3 to each digit, wrapping around after 9 (mod 10). "
                     "Output only the resulting digits separated by single spaces.",
                     ["add", "wrapping", "mod"],
                     lambda x: " ".join(str((int(d) + 3) % 10) for d in x.split())),
        "reverse": mk("Reverse the order of these digits. "
                      "Output only the reversed digits separated by single spaces.",
                      ["reverse", "reversed", "backwards"],
                      lambda x: " ".join(x.split()[::-1])),
    }
    log(f"  cipher secret perm: {perm}")
    return skills


def route(instr_text, magazine, skills):
    low = instr_text.lower()
    for name in magazine:
        if any(kw in low for kw in skills[name]["kw"]):
            return name
    return None  # cache miss


def gen_unique(skill, n):
    out, seen = [], set()
    g = 0
    while len(out) < n and g < n * 200:
        x = skill["gen_input"](); g += 1
        if x not in seen:
            seen.add(x); out.append(x)
    return out


# ---------------------------------------------------------------- model io
def load_base(tok_only=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hl
    hl.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tok_only:
        return tok, None
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32,
                                                local_files_only=True).to(DEV)
    return tok, model


def build_prompt(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def gen_one(model, tok, skill, x):
    ids = tok(build_prompt(tok, skill["fmt"](x)), return_tensors="pt").to(DEV)
    out = model.generate(**ids, max_new_tokens=skill["maxnew"], do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------- forge a tool (test-time LoRA)
def forge_tool(skill_name, skill, train_inputs, args):
    """Fresh base -> LoRA on this skill -> save adapter to the magazine dir."""
    from peft import LoraConfig, get_peft_model
    tok, base = load_base()
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    samples = []
    for x in train_inputs:
        gold = skill["apply"](x)
        pids = tok(build_prompt(tok, skill["fmt"](x)), return_tensors="pt")["input_ids"][0]
        aids = tok(gold + tok.eos_token, return_tensors="pt")["input_ids"][0]
        samples.append((torch.cat([pids, aids]),
                        torch.cat([torch.full((len(pids),), -100), aids.clone()])))
    rng = random.Random(0)
    pad = tok.pad_token_id or tok.eos_token_id
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
            log(f"    forge[{skill_name}] step {step:>4}/{args.steps} loss={loss.item():.3f} ({time.time()-t0:.0f}s)")
    out_dir = os.path.join(ADAPTERS, skill_name)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    model.save_pretrained(out_dir)
    del model, base
    return out_dir


# ---------------------------------------------------------------- evaluation
def acc_on(model, tok, skill, inputs, set_adapter=None, disable=False):
    cor = 0
    sample = None
    for x in inputs:
        gold = skill["apply"](x)
        if disable:
            with model.disable_adapter():
                gen = gen_one(model, tok, skill, x)
        else:
            model.set_adapter(set_adapter)
            gen = gen_one(model, tok, skill, x)
        pred = skill["parse"](gen, gold)
        cor += (pred == gold)
        if sample is None:
            sample = (x, gold, pred)
    return cor / len(inputs), sample


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=120)
    p.add_argument("--n_test", type=int, default=40)
    p.add_argument("--steps", type=int, default=350)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gate", type=float, default=0.5, help="test-gate: keep a forged tool iff held-out acc >= gate")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_train, args.n_test, args.steps = 40, 10, 40

    os.makedirs(ADAPTERS, exist_ok=True)
    rng = random.Random(0)
    skills = build_skills(rng)
    magazine_init = ["cipher", "sort", "addmod"]  # pre-forged tools
    miss_skill = "reverse"                          # arrives unseen -> forge at runtime
    all_used = magazine_init + [miss_skill]

    # disjoint train/test per skill
    data = {}
    for s in all_used:
        pool = gen_unique(skills[s], args.n_train + args.n_test)
        data[s] = (pool[:args.n_train], pool[args.n_train:])

    report = {"model": MODEL, "magazine_init": magazine_init, "miss_skill": miss_skill,
              "steps": args.steps, "rank": args.rank, "phases": {}}

    # ---- Phase 1: forge the magazine -------------------------------------------------
    log("=== PHASE 1: forge the magazine (the tools) ===")
    dirs = {}
    for s in magazine_init:
        dirs[s] = forge_tool(s, skills[s], data[s][0], args)
        log(f"  tool '{s}' forged -> {dirs[s]}")

    # ---- build the serving machine: base + all magazine adapters loaded --------------
    from peft import PeftModel
    tok, base = load_base()
    model = PeftModel.from_pretrained(base, dirs[magazine_init[0]], adapter_name=magazine_init[0])
    for s in magazine_init[1:]:
        model.load_adapter(dirs[s], adapter_name=s)
    model.eval()
    magazine = list(magazine_init)
    log(f"serving machine ready; magazine = {magazine}")

    # ---- Phase 2: base alone (no tool) fails everything ------------------------------
    log("=== PHASE 2: base alone (no tool) ===")
    base_acc = {}
    for s in magazine:
        a, ex = acc_on(model, tok, skills[s], data[s][1], disable=True)
        base_acc[s] = a
        log(f"  base no-tool  {s:>8}: exact={a:.3f}   e.g. {ex[0]!r}->got {ex[2]!r} (gold {ex[1]!r})")
    report["phases"]["base_no_tool"] = base_acc

    # ---- Phase 3: route -> hot-swap -> answer (the tool-changer) ----------------------
    log("=== PHASE 3: tool-changer (route -> set_adapter -> answer) ===")
    stream = []
    for s in magazine:
        for x in data[s][1]:
            stream.append((s, x))
    random.Random(1).shuffle(stream)
    route_ok = ans_ok = 0
    swap_t = []
    per_skill = {s: [0, 0] for s in magazine}
    for true_s, x in stream:
        predicted = route(skills[true_s]["fmt"](x), magazine, skills)
        route_ok += (predicted == true_s)
        per_skill[true_s][1] += 1
        if predicted is None:
            continue
        t0 = time.perf_counter(); model.set_adapter(predicted); swap_t.append((time.perf_counter() - t0) * 1e3)
        gen = gen_one(model, tok, skills[true_s], x)
        pred = skills[true_s]["parse"](gen, skills[true_s]["apply"](x))
        ok = (pred == skills[true_s]["apply"](x))
        ans_ok += ok; per_skill[true_s][0] += ok
    n = len(stream)
    log(f"  routing accuracy : {route_ok}/{n} = {route_ok/n:.3f}")
    log(f"  answer accuracy  : {ans_ok}/{n} = {ans_ok/n:.3f}   (mean tool-swap {sum(swap_t)/max(1,len(swap_t)):.2f} ms)")
    for s in magazine:
        log(f"     {s:>8}: {per_skill[s][0]}/{per_skill[s][1]} = {per_skill[s][0]/max(1,per_skill[s][1]):.3f}")
    report["phases"]["tool_changer"] = {"routing_acc": route_ok / n, "answer_acc": ans_ok / n,
                                        "per_skill": {s: per_skill[s][0] / max(1, per_skill[s][1]) for s in magazine},
                                        "mean_swap_ms": sum(swap_t) / max(1, len(swap_t))}

    # ---- Phase 4: wrong-tool control (proves the *change* matters) --------------------
    log("=== PHASE 4: wrong-tool control ===")
    wrong = {}
    for s in magazine:
        bad = next(o for o in magazine if o != s)
        a, ex = acc_on(model, tok, skills[s], data[s][1][:max(5, args.n_test // 2)], set_adapter=bad)
        wrong[s] = {"wrong_tool": bad, "acc": a}
        log(f"  {s:>8} answered with '{bad}' tool: exact={a:.3f}")
    report["phases"]["wrong_tool"] = wrong

    # ---- Phase 5: miss -> forge -> test-gate -> memorialize --------------------------
    log("=== PHASE 5: unseen skill -> miss -> forge at runtime -> memorialize ===")
    ms = miss_skill
    predicted = route(skills[ms]["fmt"](data[ms][1][0]), magazine, skills)
    log(f"  query for '{ms}' routes to: {predicted}  ({'MISS' if predicted is None else 'hit'})")
    assert predicted is None, "expected a cache miss for the unseen skill"
    log(f"  forging new tool '{ms}' on the fly...")
    d = forge_tool(ms, skills[ms], data[ms][0], args)
    model.load_adapter(d, adapter_name=ms)                       # load into the machine
    gate_acc, ex = acc_on(model, tok, skills[ms], data[ms][1], set_adapter=ms)  # TEST gate
    keep = gate_acc >= args.gate
    log(f"  test-gate: held-out exact={gate_acc:.3f} (threshold {args.gate}) -> {'MEMORIALIZE' if keep else 'reject'}")
    if keep:
        magazine.append(ms)
    log(f"  e.g. {ex[0]!r} -> got {ex[2]!r} (gold {ex[1]!r})")
    log(f"  magazine now = {magazine}")
    report["phases"]["miss_forge"] = {"routed_before": predicted, "gate_acc": gate_acc,
                                      "memorialized": keep, "magazine_after": magazine}

    # ---- verdict --------------------------------------------------------------------
    base_mean = sum(base_acc.values()) / len(base_acc)
    tc = report["phases"]["tool_changer"]
    log("\n  ================= TOOL-CHANGER VERDICT =================")
    log(f"  base (no tool) mean exact : {base_mean:.3f}")
    log(f"  tool-changer answer acc   : {tc['answer_acc']:.3f}  (routing {tc['routing_acc']:.3f})")
    log(f"  wrong-tool mean exact     : {sum(w['acc'] for w in wrong.values())/len(wrong):.3f}")
    log(f"  runtime-forged '{ms}'      : {gate_acc:.3f} -> {'added to magazine' if keep else 'rejected'}")
    ok = (base_mean < 0.1 and tc["answer_acc"] >= 0.6 and keep)
    log(f"  => {'TOOL-CHANGER WORKS' if ok else 'INCOMPLETE / see numbers'}")
    report["verdict"] = {"base_mean": base_mean, "tool_changer_acc": tc["answer_acc"],
                         "wrong_tool_mean": sum(w['acc'] for w in wrong.values()) / len(wrong),
                         "miss_forge_acc": gate_acc, "pass": bool(ok)}
    with open(os.path.join(HERE, "tool_changer_results.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"saved -> {os.path.join(HERE, 'tool_changer_results.json')}")


if __name__ == "__main__":
    main()
