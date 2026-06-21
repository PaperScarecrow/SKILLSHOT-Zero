# SKILLSHOT — Resource & Compute Plan

Planned hardware and training-time budget for the **~222M, 1.58-bit, weight-tied
recurrent-depth (looped) swarm drone** and the swarm around it. Covers three scenarios:
the **current rig**, an **array of DGX Sparks**, and an **array of A100s**. All GPU specs
are verified from primary sources (citations at the end); all time estimates state their
assumptions so you can recompute.

---

## 0. The single most important number: you train **one** base, not 200

The instinct "200 specialist 200M models = 200 training runs" is the expensive trap, and
the whole architecture exists to avoid it.

| Swarm model | What you actually train | Cost |
|---|---|---|
| **Homogeneous swarm (default)** | **1** ternary looped base + **K** Ortho-LoRA skills + **1** T2L hypernet | ~1 base run + K cheap forges |
| Heterogeneous fleet (later) | a *handful* of distinct bases (e.g. code-drone, recall-drone) × the above | few × base run |

A "drone" = the shared base **+ an adapter loadout from the cache + a sample seed/persona**.
200 drones is **1 base × many adapter loadouts**, not 200 pretrains. So the resource plan
is dominated by exactly three line items:

1. **Base pretrain/warm-start** (once) — the big one, estimated below.
2. **Per-skill LoRA forge** (K times) — minutes–hours each, trivial.
3. **T2L hypernetwork train** (once) — ~1 GPU-day.

Everything else (firing, routing, memorialize) is inference. **Plan the base run; the rest
is rounding error.**

---

## 1. Workload characterization

**Drone:** ~222M params, ternary {−1,0,+1} (BitNet b1.58 QAT), Prelude→looped-MoE-core→Coda
(OpenMythos/Geiping template). Optional sparse MoE inside the core (we'll likely start
**dense** for the proof, add MoE in Track-A polish).

**Training-compute model** (state your assumptions, then it's just arithmetic):

```
C  ≈  6 · N · D · M            FLOPs
  N = 222e6      (active params per token; dense case. MoE cuts this ~2–3×)
  D = tokens
  M = (prelude + n_loops·core + coda) / (prelude + core + coda)   loop multiplier
      OpenMythos-small layout (2,1,2):  M = (4 + n_loops)/5
      → n_loops=4 → M=1.6 · n_loops=8 → M=2.4 · n_loops=16 → M=4.0
time = C / (eff_TFLOPS · 1e12)         add ~1.3× for QAT (STE + fp master weights)
```

Two facts that pull in opposite directions and roughly cancel:
- **Looping multiplies compute** by `M` (the cost of "depth for free").
- **MoE divides active-FLOPs** by ~2–3× (only top-k experts fire), and **ternary shrinks
  weight traffic**, and **weight-tied looping reuses a resident block** (bandwidth-favorable
  → *better* MFU on bandwidth-starved hardware like the Spark).

**Memory (tiny — never the constraint here):** 222M ⇒ ~0.44 GB bf16 weights + ~1.8 GB Adam
states + grads + activations. With gradient-checkpointing through the loop, **< 10 GB even
at n_loops=16.** Every card below holds many drones at once — so memory is a *serving*
advantage, not a training limit.

---

## 2. Hardware inventory (verified specs)

| | RTX PRO 6000 Blackwell | DGX Spark (GB10) | A100 80GB SXM |
|---|---|---|---|
| BF16 dense TFLOPS | ~126 | ~125 (peak; bandwidth-bound) | **312** |
| FP8 / FP4 | 503 / ~2000 dense | — / 1000 *sparse* | — (no FP8/FP4) |
| Memory | 96 GB GDDR7 | **128 GB** LPDDR5x unified | 80 GB HBM2e |
| **Bandwidth** | **1.79 TB/s** | **0.273 TB/s** ⚠️ | **2.0 TB/s** |
| Interconnect | PCIe Gen5 (**no NVLink**) | ConnectX-7 200 GbE (**max 2 units**) | NVLink/NVSwitch 600 GB/s (≤16) |
| TDP | 600 W | 140 W chip / 240 W | 400 W |
| ≈ Price | ~$8.5–13 k | ~$4 k | ~$15–20 k (used/cloud) |

**The Spark's headline "1 PFLOP" is *sparse FP4* — irrelevant to bf16 QAT training.** Its
real training speed is governed by **273 GB/s** (LMSYS measured ~4× slower than the RTX 6000
on a 20B model, attributing it entirely to bandwidth). Treat the Spark as a **memory-rich
inference/serving box**, not a trainer.

**Effective sustained throughput I use below** (peak × realistic small-model MFU 25–40%,
bandwidth-adjusted):

| Device | eff. bf16 TFLOPS (assumed) |
|---|---|
| RTX PRO 6000 | ~40 |
| A100 (single) | ~120 |
| 8× A100 node (near-linear DP for 222M) | ~900 |
| DGX Spark (single) | ~10–15 (bandwidth-bound; a bit better for looped models) |
| 2× Spark (200GbE, comms-limited) | ~18–22 |

---

## 3. Base-train time by scenario

**From-scratch, D = 5B tokens** (a solid small base; Bonsai used <5B for 500M ternary).
QAT overhead included.

| Hardware | n_loops=8 (M=2.4) | n_loops=16 (M=4.0) |
|---|---|---|
| **RTX PRO 6000 (your card)** | **~6 days** | ~10 days |
| DGX Spark ×1 | ~20 days ⚠️ | ~33 days ⚠️ |
| DGX Spark ×2 | ~13 days | ~21 days |
| A100 ×1 | ~2 days | ~3.3 days |
| **A100 ×8 (DGX node)** | **~6–8 hours** | ~11 hours |

**Warm-start instead — the recommended path.** Don't train 222M ternary from zero. Either
(a) continue-train **Bonsai** (already a trained 500M ternary — distill/prune toward ~222M),
or (b) distill from a stronger teacher into the looped student. Adaptation budget ~0.5–1B
tokens ⇒ **5–10× cheaper**:

| Hardware | warm-start (≈0.75B tok, n_loops=8) |
|---|---|
| RTX PRO 6000 | **~0.7–1 day** |
| DGX Spark ×1 | ~3 days |
| A100 ×1 | ~6 hours |
| A100 ×8 | **~1 hour** |

**Per-skill LoRA forge** (the memorialize path, r=8 on q/v): minutes–~1 h each on any card.
**T2L hypernetwork:** ~1 GPU-day (matches the ~10 h/H100 reference) on RTX 6000 or 1× A100.

---

## 4. Role assignment (what each box is *for*)

The three platforms are complementary, not interchangeable — assign by their bottleneck:

- **RTX PRO 6000 Blackwell (96 GB) — the workshop.** Your primary single-drone trainer,
  T2L trainer, and skill-forge. High bandwidth (1.79 TB/s) makes it a real trainer; 96 GB
  holds the base + many adapters for dev. No NVLink, but you're single-card so it doesn't
  matter. **Does the base warm-start in ~1 day.**
- **7900 XTX (24 GB, ROCm) — the sidecar.** Not a co-trainer (CUDA/ROCm can't DP one model;
  QAT/BitNet kernels are CUDA-only). Use for **eval harness, inference, and serving a few
  drones** while the Blackwell trains. Keep it on its own jobs.
- **DGX Spark array — the swarm hangar (serving + embarrassingly-parallel jobs).** 128 GB
  unified per unit = *tons* of resident drones + adapters; the looped model's resident-block
  reuse is bandwidth-friendly, so **inference/serving is the Spark's sweet spot.** For
  training, use Sparks as **independent nodes** (one Spark = one drone's warm-start or one
  skill-forge in parallel), NOT a tight DP cluster — only 2 cluster natively and 200 GbE
  (25 GB/s) is comms-bound for small models. A rack of Sparks **forges K skills in parallel**
  beautifully.
- **A100 array — the heavy lifter.** NVLink/NVSwitch → near-linear DP for the 222M base.
  An 8× node does a from-scratch 5B-token base in **~6–8 h**, or the whole base+T2L+skills in
  a day. This is the platform to reach for when you want a genuinely **new substrate** (Track
  A from scratch, or a heterogeneous fleet of distinct bases) rather than a warm-start.

**Decision heuristic:** warm-start + iterate → **RTX 6000**. Serve/scale the swarm + parallel
forges → **DGX Sparks**. From-scratch base or a fleet of distinct bases → **A100 array**.

---

## 5. OpenMythos as the reference for the looped-MoE drone

Mirror these (verified coherent in `open_mythos/main.py`); skip the rest.

**Mirror:**
- `RecurrentBlock`: one tied `TransformerBlock(use_moe=True)`, `for t in range(n_loops)` with
  the **encoded input `e` re-injected every iteration** and a sinusoidal **loop-index
  embedding** into the first `dim//8` channels (lets the block know which iteration it's on).
- `LTIInjection`: stability **by construction** — `A = exp(-exp(clamp(log_dt+log_A)))` forces
  each diagonal element into (0,1) (ρ<1), update `A·h + B·e + block_out`. Cheap, no penalty
  to tune. **Adopt this** — it's the un-sexy reason looped training doesn't diverge.
- `ACTHalting`: Graves remainder-trick halting → **learned per-token loop count** capped by
  `max_loop_iters`. This is the test-time compute dial, made adaptive.
- Aux-loss-free MoE (DeepSeek-V3 `router_bias` trick) — *if* we add MoE inside the core.

**Hand-author a ~222M config** (OpenMythos's smallest is 1B): start from `dim≈768–1024`,
`prelude=coda=2`, one tied core block, `max_loop_iters≈8–16`, and either **dense** core (proof)
or a **small MoE** (`n_experts≈8–16, top-2, 1 shared`) to keep active-params low. Our
`skillshot/looped_expert.py` already has the Prelude→tied-core→Coda skeleton with the live
`r` dial; the OpenMythos additions to fold in are **LTI injection + ACT halting + loop-index
embedding** (and optional MoE).

**Do NOT copy:** `moda.py` (dead/unintegrated), the ≥10B/1T configs (aspirational, untrained),
or assume the MoE balances itself — the `router_bias` update isn't wired in their trainer, so
**we must actually update it** (or add a light load-balance loss).

---

## 6. Recommended path (ties to PRODUCTION_PLAN phases)

1. **Phase 1–3 on the RTX 6000**, using an **fp16** 222M looped drone (defer ternary) +
   MockExpert→real swap + forge 2–3 skills + stand up the eval harness. Cheap, fast iteration.
2. **Warm-start the real ternary drone from Bonsai on the RTX 6000 (~1 day)** for Phase 4,
   rather than frying Gemma-270m or training from scratch.
3. **Forge the skill cache + train T2L** — parallelize forges across DGX Sparks if available.
4. **Serve the swarm on DGX Sparks** (128 GB unified holds the fleet + adapter cache; looped
   reuse is bandwidth-friendly) with the 7900 XTX as an eval sidecar.
5. **Reach for an A100 array only** when you want a from-scratch base, a bigger base (≥500M
   to clear the ternary quality floor), or a heterogeneous fleet — then an 8× node trains the
   base in hours.

**Reality check to keep posted:** a 222M ternary drone is *weak alone* (BitNet shows the gap
widens below 3B). The plan deliberately spends its compute on **looping (depth-for-free) +
the shared skill cache + two-step firing**, not on a bigger lonely base. If single-drone
quality disappoints at the Phase-4 gate, the lever is **base size → ~500M (Bonsai's size)**,
which the A100 array makes affordable.

---

## Sources
RTX PRO 6000 Blackwell — [NVIDIA datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf), [TechPowerUp](https://www.techpowerup.com/333802/) ·
DGX Spark — [NVIDIA](https://www.nvidia.com/en-us/products/workstations/dgx-spark/), [LMSYS review](https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/) (273 GB/s, ~4× slower than RTX 6000), [Tom's Hardware price](https://www.tomshardware.com/desktops/mini-pcs/nvidia-dgx-spark-gets-18-percent-price-increase) ·
A100 — [NVIDIA datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) ·
MFU / DP scaling — [Databricks](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices), [arXiv 2012.01839](https://arxiv.org/pdf/2012.01839) ·
OpenMythos — `github.com/kyegomez/OpenMythos` (`open_mythos/main.py`) ·
Looped/ternary primaries — Geiping [2502.05171](https://arxiv.org/abs/2502.05171), BitNet b1.58 [2402.17764](https://arxiv.org/abs/2402.17764), Bonsai (`github.com/deepgrove-ai/Bonsai`).

> All throughput/time figures are **estimates** built on assumed 25–40% MFU and the stated
> compute model — treat as planning ballparks (±~2×), not benchmarks. Re-measure on the real
> drone config before committing a schedule.
