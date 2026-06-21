# SKILLSHOT

**A tool-changer for language models.** One small, frozen base model (a "machine"), a magazine of
hot-swappable LoRA skill-adapters ("tools"), and a router ("the changer") that selects the right
tool per job — and *forges a new tool at runtime* when none fits (project → test → memorialize).

This repo contains the orchestration scaffold and a **CPU-only proof-of-concept** that validates the
core claims on the hardest possible substrate: `gemma-3-270m`, a model so small it is near-useless on
its own.

> Status: research proof-of-concept (2026-06). All POC results are CPU-only.

---

## Headline results (CPU, gemma-3-270m)

| Result | Number |
|---|---|
| **Tool-changer, end-to-end** | base-no-tool **0.008** → route+swap **1.000** (routing 1.000, swap **7.8 ms**); wrong-tool **0.000**; **runtime-forged** a new tool to **1.000** and memorialized it |
| **Skill-shot lift** (cipher skill) | base **0.000** → forged adapter **1.000** exact-match, generalizing to held-out inputs (adapter = 2.75% of params) |
| **Sub-quadratic memory** | a MIRAS/TITANS linear memory matches dense attention on needle recall (**1.000 = 1.000**) at genuinely **O(L)** cost (slope 0.93) vs dense **O(L²)** (slope 1.82) |
| **Real tools** (HF datasets) | CVE-severity, vuln-detection (Devign), code-comprehension — base fails zero-shot, forged tools lift (see `proof/PROOF_RESULTS.md`) |

**Edge footprint:** quantized (BitNet-b1.58 transformer + 4-bit embedding) the machine is **~100 MB** and
each tool **~1–2 MB**; CPU-only by construction — target hardware is a Raspberry Pi Zero / microcontroller
/ **router-native** deployment for offline defensive tooling (see `PAPER.md` §5).

Honest boundary, documented: top-k sparse softmax is **not** sub-quadratic (O(L²)); arbitrary recall
is Ω(L) (no O(log n) free lunch); skill-shot injects *procedure*, not *world knowledge* (use RAG for
that); execution-heavy coding (CRUXEval) is beyond a 270M even with an adapter.

---

## Layout

```
skillshot/              orchestration scaffold (mothership/router, registry, adapter cache,
                        memorialize loop, ortho-LoRA, MIRAS/TITANS memory gate, ternary)
demo_control_loop.py    runnable mock orchestration demo
proof/                  the CPU proof-of-concept (the evidence)
  mqar_bakeoff.py         Track B: needle-recall + compute-scaling bake-off (4 mixers)
  gemma_skillshot.py      Track A: skill-shot lift on gemma-3-270m
  tool_changer_poc.py     the full tool-changer loop (magazine + route + miss→forge→memorialize)
  real_tools_poc.py       real HF-dataset tools (CVE / vuln / code)
  PROOF_RESULTS.md        the writeup (numbers + honest caveats)
  *_results.json          raw measured results
DESIGN.md, *_PLAN.md    architecture & plans
```

## Reproduce (CPU-only)

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch transformers peft datasets accelerate safetensors
cd proof
CUDA_VISIBLE_DEVICES="" python mqar_bakeoff.py   --task timing
CUDA_VISIBLE_DEVICES="" python mqar_bakeoff.py   --task recall --mixers dense topk miras hier --regimes sparse --steps 2500 --lens 64 128 256
CUDA_VISIBLE_DEVICES="" python gemma_skillshot.py --skill cipher --steps 600 --rank 32
CUDA_VISIBLE_DEVICES="" python tool_changer_poc.py --steps 500
CUDA_VISIBLE_DEVICES="" python real_tools_poc.py  --steps 300
```
gemma-3-270m loads from the local HF cache (`unsloth/gemma-3-270m-it`, offline). Datasets download
from the Hub on first run.

See `PAPER.md` for the full report.
