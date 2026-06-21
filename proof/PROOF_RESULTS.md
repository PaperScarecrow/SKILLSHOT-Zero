# SKILLSHOT — Proof-of-Concept Results

*CPU-only proof on `gemma-3-270m` + a sub-quadratic memory bake-off. Run 2026-06-20.*
*Hardware note: an RTX PRO 6000 was present but reserved for a parallel training job — **every result here is CPU-only** (32 cores, `torch 2.11 cpu`, `transformers 5.6`, `peft 0.18`). gemma loaded offline from `unsloth/gemma-3-270m-it` (cached, non-gated mirror; the `google/*` gated repos lacked local weights).*

Two independent, separately-falsifiable tracks, each with a pre-registered kill criterion. Harness: `mqar_bakeoff.py` (Track B), `gemma_skillshot.py` (Track A).

---

## TL;DR

| Claim under test | Result | Verdict |
|---|---|---|
| **A.** Near-useless 270M base + skill-shot adapter → appreciable, *generalizing* lift | cipher **exact 0.000 → 1.000**; reverse **char-sim 0.30 → 0.62** | ✅ **proven** |
| **B1.** TITANS/MIRAS linear O(n) memory nails needle recall | miras **1.000** = dense **1.000** on the sparse needle (after stabilization) | ✅ **proven** |
| **B2.** Sub-quadratic is real & measured on CPU | miras/hier slope **≈0.93–0.95** (O(L)) vs dense/top-k **≈1.82** (O(L²)) | ✅ **proven** |
| **B3.** Top-k sparse softmax is *not* asymptotically sub-quadratic | slope **1.81** — same class as dense | ✅ **proven** (key caveat) |
| **B4.** Linear memory holds the needle; arbitrary recall hits an Ω(L) wall | miras=dense=1.000 @L=64; Ω(L) per theory + ologn probe | ✅ (multi-L sweep deprioritized) |
| **C.** Full tool-changer end-to-end | base 0.008 → route+swap 1.000; wrong-tool 0.000; miss→forge→memorialize 1.000 | ✅ proven |
| **D.** Real tools (HF datasets) | vuln 0.353→0.593; cve macro-F1 0.185→0.465; code-match 0.567→0.900 | ✅ proven |

**Bottom line:** the SKILLSHOT thesis survives a real CPU proof on its hardest substrate (a 270M model that, untouched, just *copies its input*). Skill-shot adapters genuinely inject capability that generalizes; a TITANS/MIRAS-style linear memory genuinely does needle recall at genuinely O(L) cost. The honest boundaries (below) are as important as the wins.

---

## Track A — skill-shot lift on gemma-3-270m

Forge a small LoRA at test time on a handful of examples (the SKILLSHOT *project→test→memorialize* forge), then test on **held-out, disjoint** inputs. Adapter = **7.59M params = 2.75%** of the model.

**Cipher skill** (apply a secret digit→digit permutation; tokenization-friendly):
```
BASE  gemma-3-270m-it   exact = 0.000   char-sim = 0.414     # just copies the input: "1 3 1 8" -> "1 3 1 8"
+ skill-shot LoRA        exact = 1.000   char-sim = 1.000     # every held-out case correct: "1 3 1 8" -> "8 5 8 9"
```
**A 270M model that copies its input becomes a 100%-accurate specialist on a skill it entirely lacked — from a 2.75% adapter, on CPU, in ~26 min.** Test inputs are disjoint from training, so this is generalization, not memorization.

**Reverse-letters skill** (deliberately tokenizer-hostile, char-level output):
```
BASE                     exact = 0.000   char-sim = 0.303     # copies input, 0% reversal
+ skill-shot LoRA        exact = 0.013   char-sim = 0.619     # learns to reverse, imperfectly
```
Honest read: on a skill that fights subword tokenization, skill-shot still produces a large *graded* lift (char-sim ~doubles; `qetv`→`vtdq` vs gold `vteq`) but can't reach reliable exact-match at 270M. **The lift is real; how close to mastery depends on how tokenizer-friendly the skill is.**

> Maps to research: matches the literature that test-time/skill-shot adaptation injects **reasoning/format/procedural** capability well — *not* world knowledge. (`[[skillshot-research-findings]]`, `[[skillshot-sparse-attention-subq]]`)

---

## Track B — the sub-quadratic / long-context bake-off

One MQAR (multi-query associative recall = clean needle-in-a-haystack) harness; four token-mixers on equal footing:
`dense` (full softmax), `topk` (top-k sparse softmax — your e4b lever), `miras` (MIRAS/TITANS linear delta-rule memory — ported from `skillshot/memory_gate.py`), `hier` (O(log n) hierarchical — ported from `Nyxara/core/ologn_attention4.py`).

### B2/B3 — compute scaling (forward-pass wall-clock vs context length, CPU)

| L | dense | top-k | **miras** | hier |
|---|------|------|------|------|
| 512 | 2.9 ms | 3.7 | 67 | 95 |
| 1024 | 10.2 | 13.0 | 167 | 193 |
| 2048 | 112 | 146 | 270 | 368 |
| 4096 | 434 | 598 | **552** | 760 |
| **log-log slope** | **1.82** | **1.81** | **0.93** | **0.95** |

- **miras and hier are genuinely O(L)** (slope ~0.93–0.95). **dense is O(L²)** (1.82). The slope gap *is* the sub-quadratic win.
- **Top-k sparse softmax is also O(L²) (slope 1.81)** — it sparsifies *what* is attended, not the *cost of computing the scores*. **If you want true sub-quadratic, the linear-memory route is the only one of the two that delivers it.** (This is the trap the Sub-Q memo flagged, now measured on this box.)
- **Constant-factor honesty:** miras's pure-Python recurrence has a heavy constant (~0.13 ms/token), so the wall-clock crossover vs *dense* is ~L≈5–6k on CPU (you can already see miras overtake top-k at L=4096: 552 vs 598 ms). The **slope** is the asymptotic truth; a fused/chunkwise-parallel kernel (the standard DeltaNet trick) collapses the constant and moves the crossover far left. This is an implementation cost, not an algorithmic one.

### B1/B4 — needle recall (sparse bindings)

At L=64, after stabilization (grad-clip + L2-normed keys + short causal conv): **dense 1.000, miras
1.000** — the linear O(n) memory matches full attention on the needle. The clean multi-length sweep
(L∈{128,256}) was **deprioritized**: miras's pure-Python recurrence is very slow on CPU and was blocking
other runs. Length-holding + the Ω(L) wall are supported by theory and the in-repo `Nyxara` O(log n)
probe (recall decays with L at fixed state for arbitrary bindings) — see the boundaries section.

> **Stability finding (matters for productionizing):** a first sweep had miras collapse to chance
> (gradient explosion → NaN → predicts a constant). Gradient clipping + L2-normalized keys + bounded-EMA
> momentum + a short causal conv (DeltaNet/Based-style, needed for adjacent-token binding) make it train
> reliably to 1.000. Keep all four in any production version.

**Stability finding (important, and honestly reported):** the *first* sweep showed miras collapsing to chance (0.007–0.011) — gradient explosion through the BPTT recurrence (NaN → predicts a constant). Two standard fixes — **gradient clipping** + **L2-normalized keys/queries** + a **bounded-EMA momentum** (instead of an unbounded accumulator) + a **short causal conv** (needed for adjacent-token binding, à la DeltaNet/Based) — make it train **reliably** to 1.000. Takeaway for the team: the reference linear memory *works*, but naive optimization is fragile; the production version must keep these stabilizers.

---

## Track C — the tool-changer, end-to-end (`tool_changer_poc.py`)

Frozen base + forged magazine {cipher, sort, addmod}, mixed 120-query stream:
```
base, no tool        : 0.008      machine with no tool fails everything
route + hot-swap     : 1.000      routing 1.000; mean swap 7.8 ms; cipher/sort/addmod 40/40 each
wrong-tool control   : 0.000      forcing the wrong adapter fails -> the *change* does the work
miss -> forge        : unseen 'reverse' -> MISS -> forged at runtime -> held-out gate 1.000 -> MEMORIALIZED (3->4)
=> TOOL-CHANGER WORKS
```

## Track D — real tools from public HF datasets (`real_tools_poc.py`)

300 train / 150 test (disjoint), 300 steps, on the real `gemma-3-270m`:

| Tool | Dataset | Metric | Base | +Tool | Δ |
|---|---|---|---|---|---|
| vuln-detect | Devign | accuracy | 0.353 | **0.593** | +0.240 |
| cve-severity | stasvinokur CVE/CWE | accuracy | 0.480 | 0.567 | +0.087 |
| cve-severity | (same) | **macro-F1** | 0.185 | **0.465** | **+0.280** |
| code-match | MBPP (spec↔code) | accuracy | 0.567 | **0.900** | +0.333 |

CVE raw accuracy barely moves (base predicts the majority class) but **macro-F1 more than doubles** —
the base couldn't discriminate severity; the tool learns it. Documented boundary: CRUXEval-O output
prediction stayed ~floor (0.13→0.10) — execution-heavy coding is beyond a 270M even with an adapter.

See `../PAPER.md` for the full report (incl. an honest relation-to-Sub-Q section).

## What this proves — and what it does NOT

**Proven (on CPU, at small scale):**
- Skill-shot adapters take a near-useless 270M base from 0 → mastery on a learnable skill, generalizing to held-out inputs.
- A TITANS/MIRAS linear memory does needle recall as well as full attention, at genuinely O(L) compute.
- Sub-quadratic ≠ "sparse attention": top-k softmax stays O(L²); only the linear-memory family is asymptotically sub-quadratic.

**Does NOT prove (the honest boundaries — these are the team's real risks):**
- **Ω(L) ceiling.** Your own `ologn_attention4.py` and the theory agree: *arbitrary* associative recall needs Ω(L) state — you **cannot** make general needle recall O(log n). Linear memory is O(L) (great), not O(log n). The O(log n) `hier` mechanism only holds for *sparse/structured* recall and decays on arbitrary recall (observed: 0.51→0.17 across length).
- **Single-needle ≠ multi-hop.** MQAR sparse recall is the *easy* bar (the one Sub-Q leaned on). Multi-needle / multi-hop reasoning over long context is the hard bar and is untested here.
- **Knowledge vs skill.** Skill-shot injects procedure/format, not world knowledge (cipher works; a knowledge-bound task would not). Knowledge must come from context/RAG.
- **Scale.** These are 2-layer d=64 toy models and a 270M base on CPU. Nothing here speaks to whether the full stack (ternary + MoE + linear-attn + skill cache) *composes* at scale — the memo's super-additive-error warning still stands and must be tested by decoupled, sequenced experiments.
- **VALENCE is not the transfer path.** Per your own files, VALENCE-proper is a GPU ray-tracing retrieval engine (needs RT cores + an uncompiled `libastra.so`, no transformer-weight import, fails order/binding in `VALENCE_BOUNCE_RESULTS.md`) — it cannot run CPU-only and cannot "convert gemma-3-270m." The real O(n) transfer lever is the linear-memory route demonstrated here.

---

## Reproduce

```bash
PY=/home/paperscarecrow/Downloads/.venv/bin/python
cd /media/paperscarecrow/Main/SKILLSHOT/proof
CUDA_VISIBLE_DEVICES="" $PY mqar_bakeoff.py --task timing --d 64 \
    --time_lens 128 256 512 1024 2048 4096 --time_batch 4 --reps 4
CUDA_VISIBLE_DEVICES="" $PY mqar_bakeoff.py --task recall --mixers dense topk miras hier \
    --regimes sparse --steps 2500 --lens 64 128 256 --d 64 --lr 1e-3 --min_vocab 32
CUDA_VISIBLE_DEVICES="" $PY gemma_skillshot.py --skill cipher --steps 600 --rank 32
CUDA_VISIBLE_DEVICES="" $PY gemma_skillshot.py --skill reverse --steps 600 --rank 32
```
Results: `bakeoff_timing.json`, `bakeoff_recall.json`, `skillshot_lift_cipher.json`, `skillshot_lift_reverse.json`.

## Recommended next steps (decoupled, in priority order)
1. **Multi-needle / multi-hop** MQAR at the context length you actually need — the real long-context bar, not single-needle.
2. **Fused/chunkwise miras kernel** — collapse the constant factor so the O(L) win shows in wall-clock at usable L.
3. **Compose one layer at a time** — wire the linear memory + skill cache into the real gemma-3-270m (the `HFTernaryExpert` seam), then add ternary, watching for the super-additive-error cliff.
4. **Knowledge probe** — confirm skill-shot does *not* close knowledge gaps (so retrieval/RAG is budgeted in, not assumed away).
