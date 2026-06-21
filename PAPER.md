# SKILLSHOT: A Tool-Changer for Small Language Models

*Proof-of-concept report. 2026-06. All experiments CPU-only on `gemma-3-270m`.*

## Abstract

We treat a small, frozen language model as a **machine** and a library of low-rank adapters (LoRA)
as interchangeable **tools**: a router selects the right tool per request, hot-swaps it onto the base,
and — when no tool fits — **forges a new one at runtime** and admits it to the library after a held-out
test gate (a *project → test → memorialize* loop). On `gemma-3-270m` (a 270M model that, untouched,
largely copies its input) we show that (1) test-time "skill-shot" adapters take the base from **0.000
to 1.000** exact-match on a held-out skill; (2) a TITANS/MIRAS-style **linear O(n) memory** matches dense
attention on needle recall at genuinely sub-quadratic compute, while top-k sparse softmax does *not*;
(3) the full tool-changer runs end-to-end — routing 1.000, hot-swap ~8 ms, wrong-tool 0.000, and a new
tool forged + memorialized at runtime to 1.000; and (4) the same mechanism produces real, measurable
specialists from public datasets for **vulnerability detection, CVE-severity, and code-comprehension**.
We are deliberate about boundaries: arbitrary recall is Ω(L) (no O(log n)), skill-shot injects procedure
not world knowledge, and execution-heavy coding is beyond a 270M even with an adapter.

---

## 1. Introduction

Capability density is rising fast, but a sub-1B base is still near-useless as a generalist. SKILLSHOT
asks a different question: *can a near-useless base become a competent multi-skill system by swapping
small, cheap, hot-loadable tools?* The unit of capability is the adapter, not the base. This decouples
"what the machine can do" from "what's loaded right now," and makes new skills a runtime operation
(forge a tool) rather than a retraining event.

This report is a falsification-oriented proof-of-concept: each claim has a metric and a pre-registered
kill criterion, and we report the honest boundaries as prominently as the wins.

## 2. Method

- **Machine:** a frozen base model (here `gemma-3-270m`). Never updated.
- **Tool:** a LoRA adapter (rank 16–32; ~2.75% of params) over the attention + MLP projections.
- **Magazine:** a set of named adapters loadable into one base via PEFT multi-adapter; `set_adapter`
  is the tool change (~8 ms).
- **Changer (router):** maps a request to a tool (here keyword routing over the instruction; a learned
  router is future work). A no-match is a *miss*.
- **Forge (skill-shot):** on a miss, train a LoRA at test time on a handful of examples for the needed
  skill, prompt-masked so only the answer is learned. Stabilized with gradient clipping.
- **Test → Memorialize:** a freshly forged tool is admitted to the magazine only if it clears a
  held-out accuracy gate.
- **Long context:** for sequence skills we additionally study a **MIRAS/TITANS linear memory** — a
  gated delta-rule matrix memory (O(n), O(d²) state) with data-dependent retention/write gates,
  bounded-EMA momentum, L2-normalized keys, and a short causal conv (for adjacent-token binding).

## 3. Setup

CPU-only (32 cores), `torch 2.11 (cpu)`, `transformers 5.6`, `peft 0.18`. Base = `gemma-3-270m`
(loaded offline from the cached `unsloth/gemma-3-270m-it` mirror). Datasets pulled from the HF Hub.
Harness: `proof/` (`mqar_bakeoff.py`, `gemma_skillshot.py`, `tool_changer_poc.py`, `real_tools_poc.py`).

## 4. Results

### 4.1 Skill-shot lift (does a forged tool rescue the base?)
A test-time LoRA (7.59M params = 2.75%) on disjoint held-out inputs:

| Skill | Base exact | +Tool exact | Base char-sim | +Tool char-sim |
|---|---|---|---|---|
| cipher (digit permutation) | 0.000 | **1.000** | 0.414 | 1.000 |
| reverse-letters (tokenizer-hostile) | 0.000 | 0.013 | 0.303 | 0.619 |

The base *copies its input* (0% on both); the cipher tool reaches **100% on held-out inputs**
(generalization, not memorization). On a char-level skill that fights subword tokenization, the lift is
large but graded (char-sim ~doubles) — honest evidence that *how much* a skill-shot helps depends on
how tokenizer-friendly the skill is.

### 4.2 Sub-quadratic long-context memory (Track B bake-off)
One MQAR (needle-in-a-haystack) harness; four token-mixers on equal footing.

**Recall (sparse needle, L=64):** dense **1.000**, MIRAS/TITANS linear memory **1.000** — the linear
memory matches full attention on the needle. (A first run revealed the recurrence is training-*unstable*
without gradient clipping + normed keys + a short conv; with those, it trains reliably.)

**Compute scaling (CPU wall-clock vs context length L):**

| L | dense | top-k | miras | hier |
|---|---|---|---|---|
| 512 | 2.9 ms | 3.7 | 67 | 95 |
| 2048 | 112 | 146 | 270 | 368 |
| 4096 | 434 | 598 | **552** | 760 |
| **log-log slope** | **1.82** | **1.81** | **0.93** | **0.95** |

miras/hier are genuinely **O(L)**; dense is **O(L²)**. The decisive, often-missed point: **top-k sparse
softmax is also O(L²) (slope 1.81)** — it sparsifies *which* keys are read, not the cost of computing the
scores. True sub-quadratic needs the linear-recurrence family. (miras's pure-Python recurrence has a
heavy constant, so the wall-clock crossover vs dense is ~L≈5–6k on CPU; a fused/chunkwise kernel removes
the constant. The slope is the asymptotic truth.)

### 4.3 The tool-changer, end-to-end
A frozen base + a forged magazine {cipher, sort, addmod}, on a mixed 120-query stream:

```
base, no tool        : 0.008      (machine with no tool fails everything)
route + hot-swap     : 1.000      (routing 1.000; mean swap 7.8 ms; cipher/sort/addmod 40/40 each)
wrong-tool control   : 0.000      (forcing the wrong adapter fails -> the *change* does the work)
miss -> forge        : an unseen 'reverse' skill routes to MISS -> forged at runtime ->
                       held-out test gate 1.000 -> MEMORIALIZED -> magazine 3->4
```
Verdict: the full project→test→memorialize loop works on the real model.

### 4.4 Real tools from public datasets
Forged from HF datasets (300 train / 150 test, disjoint; 300 steps):

| Tool | Dataset | Metric | Base | +Tool | Δ |
|---|---|---|---|---|---|
| vuln-detect | Devign (`code_x_glue_cc_defect_detection`) | accuracy | 0.353 | **0.593** | +0.240 |
| cve-severity | `stasvinokur/cve-and-cwe-dataset-1999-2025` | accuracy | 0.480 | 0.567 | +0.087 |
| cve-severity | (same) | **macro-F1** | 0.185 | **0.465** | **+0.280** |
| code-match | MBPP (spec↔code, yes/no) | accuracy | 0.567 | **0.900** | +0.333 |

vuln-detect approaches fine-tuned-CodeBERT territory (~0.62) from a 270M base. For cve-severity, raw
accuracy barely moves because the base predicts the majority class; **macro-F1 more than doubles**, i.e.
the base couldn't discriminate severity and the tool learns to. code-match is a clean, large lift.

**Documented boundary:** the same pipeline on CRUXEval-O (predict a function's exact output) gave
**0.13 → 0.10** — execution-heavy coding requires mentally running Python, which a 270M cannot acquire
from an adapter (it learns output *format*, not execution). We replaced it with the learnable
comprehension task above and report the negative result honestly.

### 4.5 The recall set — beyond single-needle (multi-hop & capacity)

Single-needle recall (§4.2) is the *easy* bar. We tested the harder regimes (3-layer models, 2000 steps).
The cleanly-trained mixers (dense, top-k) give the trustworthy curves:

**Multi-hop** (chained recall: query k₀ → the node H hops away):

| hops | dense | top-k |
|---|---|---|
| 1 | 0.997 | 1.000 |
| 2 | 0.159 | 0.159 |
| 3 | 0.141 | 0.148 |

**Multi-needle** (capacity: one query, N bindings in L=64):

| N bindings | dense | top-k |
|---|---|---|
| 4 | 0.997 | 0.997 |
| 16 | 1.000 | 0.948 |
| 32 | 0.060 | 0.086 |

Two honest findings: (1) **compositional/multi-hop recall is the wall** — even *full attention* nails
1-hop (1.0) but collapses past it (~0.15 at 2–3 hops) at this 270M-toy scale, i.e. single-needle success
does **not** imply multi-hop reasoning; (2) **capacity is finite** — recall holds to ~16 bindings then
collapses by 32 (the Ω(L)/state wall), as theory predicts.

*Methodological caveat:* the linear-memory (`miras`) and `hier` mixers trained **erratically** in this
3-layer setup — they failed easy cases yet partially solved hard ones (e.g. miras 0.03 at 1-hop but 0.44
at N=32), the signature of an under-converged deep delta-rule recurrence (BPTT through stacked recurrences
is hard to optimize), **not** a capability ranking. Their clean single-needle parity (§4.2, 2 layers)
stands; a reliable deep/compositional comparison for the linear memory needs more tuning (more steps,
lower LR, multi-seed) and is future work. Full numbers in `proof/recall_set_results.json`.

## 5. Edge footprint & quantization

The machine is small, and — importantly — the **embedding table, not the transformer, dominates its
size**.

**Parameters (gemma-3-270m, ~268M):** embedding 167.8M (**63%**, = 262,144 vocab × 640 dim); transformer
(18 layers) 100.3M (37%). So aggressive quantization of the transformer alone helps less than expected —
the vocabulary table is the wall.

**On-disk now:** repo (code+docs) ~0.3 MB · base model 511 MB (bf16) + tokenizer 32 MB · each forged tool
30 MB (rank-32 LoRA, fp32) — or ~2 MB at rank-8/int8.

**Quantized model size:**

| Precision | bytes/param | model |
|---|---|---|
| bf16 (now) | 2.0 | 536 MB |
| int8 | 1.0 | 268 MB |
| 4-bit (NF4) | 0.5 | 134 MB |
| ternary transformer + 4-bit embedding (BitNet b1.58 style) | — | **~104 MB** |
| ternary transformer + 2-bit embedding | — | **~62 MB** |
| 1-bit everything (theoretical floor; below the quality cliff) | 0.125 | ~34 MB |

Realistic aggressive quant lands a **~60–100 MB machine**; tools are **~1–2 MB each** (int8, rank-8), so
the *magazine* — not the machine — eventually dominates (≈50 tools ≈ 100 MB). A full "machine + a handful
of tools" plausibly fits in **~100–150 MB**; with a domain-pruned vocabulary the ternary model could fall
toward ~30–40 MB.

**Why this matters — edge / defensive deployment.** SKILLSHOT is CPU-only by construction and ~100 MB
quantized, which puts it on hardware a frontier model can't reach: a **Raspberry Pi Zero**, a
microcontroller with enough flash (e.g. via a CPython/`pyduino`-style bridge), or — most consequentially —
**natively on a router or gateway**. A self-contained tool-changer running in-place could forge and
hot-swap narrow *defensive* skills at the network edge — the vuln / CVE-severity / log-classification tools
demonstrated here — fully offline, no cloud dependency, and able to grow new tools at runtime. This is
forward-looking and contingent on the §6 open questions holding up at the relevant context lengths and task
mix; but the footprint arithmetic says the deployment target is real, not hypothetical.

## 6. Limitations & honest boundaries

- **Ω(L) ceiling.** Arbitrary associative recall needs Ω(L) state — general needle recall is O(n) at
  best, *not* O(log n). The O(log n) `hier` mixer holds only for sparse/structured recall.
- **Single-needle ≠ multi-hop (now measured, §4.5).** Single-needle recall is the easy bar; even full
  attention collapses past 1 hop and past ~16 bindings at this 270M-toy scale. Compositional/long-context
  reasoning is the hard, unsolved regime here — budget for it (it does not come free with the base).
- **Skill, not knowledge.** Skill-shot injects procedure/format; world knowledge must come from
  context/RAG (consistent with the small-model knowledge floor).
- **Execution-heavy coding** (CRUXEval) is beyond a 270M even with an adapter.
- **Scale.** Toy 2-layer mixers + a 270M base on CPU. Whether the *full* stack (linear-attn + ternary +
  MoE + skill cache) composes at scale is untested; prior evidence warns the errors are super-additive,
  so layers should be added one at a time and measured.
- **miras constant factor.** The reference recurrence is Python; a fused/chunkwise kernel is needed for
  the O(L) win to show in wall-clock at usable L.
- **Routing** here is keyword-based; a learned/embedding router is future work.

## 7. Conclusion & next steps

A near-useless 270M base becomes a 100%-accurate specialist from a 2.75% adapter, swaps tools in ~8 ms,
forges new tools at runtime, and yields real security/coding specialists — all CPU-only and reproducible.
The linear-memory route is the only measured path to genuine sub-quadratic long context; sparse softmax
is not it. Next: multi-needle/multi-hop recall; a fused miras kernel; compose one stack layer at a time
on the real base; and a learned router. See `proof/PROOF_RESULTS.md` for the raw log and `README.md` to
reproduce.
