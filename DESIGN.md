# SKILLSHOT — Mothership / 1-bit Expert Swarm — Design Document

> An orchestration **mothership** model directs a **swarm of ~200M-param 1-bit/ternary specialist experts**
> that share a **cache of orthogonal (Ortho) LoRA skill adapters**. When a needed skill is missing, a
> **projected LoRA** (hypernetwork-generated) is the contingency, triggering a **project → test → memorialize** loop.
>
> Status: design + scaffold. Substrate built fresh; the `~/Downloads/Polymath` + `otitans` code is **inspiration only**.

---

## 0. TL;DR of the verified research (mid-2026, primary sources)

| Pillar | What it gives us | Primary source(s) |
|---|---|---|
| **TITANS** | Test-time "surprise" memory write (delta-rule + momentum + adaptive forget); MAC/MAG/MAL | arXiv 2501.00663 |
| **MIRAS** | The unifying lens: memory = associative module with **4 knobs** (arch, attentional-bias objective, **retention gate**, learning algo). Moneta/Yaad/Memora | arXiv 2504.13173 |
| **HOPE / Nested Learning** | Self-modifying recurrence + **Continuum Memory System** (MLP chain, block *l* updates every C^(l) steps) → tiered memorialization | arXiv 2512.24695 (NeurIPS'25) |
| **Projected LoRA = Text-to-LoRA** | NL task description → rank-8 LoRA on q/v in **one forward pass**, ~97% of trained-LoRA quality, standard PEFT format | arXiv 2506.06105, github.com/SakanaAI/text-to-lora |
| **Ternary / 1-bit** | {−1,0,+1} absmean weights, 8-bit activations, **QAT (not PTQ)**; LoRA-on-frozen-ternary is **published-feasible** | BitNet b1.58 2402.17764; Bonsai (deepgrove-ai); QVAC Fabric; BitLoRA |
| **"Turbocache"** | Not a real system. Use **LRU adapter cache in VRAM + SGMV/MBGMV batched kernels + paged KV** | S-LoRA 2311.03285, Punica 2310.18547, dLoRA OSDI'24, vLLM 2309.06180 |
| **Looped / recurrent-depth** | Weight-tied block looped *r* times → "depth for free" (8 layers → 132 effective @ r=32). Track-A substrate | Geiping 2502.05171; Mamba-2 2405.21060; DeltaNet 2406.06484 |

**Three corrections that shape the build:**
1. **Bonsai is 500M and runs in fp16** (ternary-valued but unpacked) — it proves *trainability*, not yet a memory/speed win.
2. **You cannot zero-shot PTQ Gemma-3-270m to ternary** — it collapses. "Frying" = a QAT continued-pretraining job (extra-RMSNorm trick + continual QAT), ~1–3 pt quality drop.
3. **MIRAS has four knobs, not three** (a three-knob phrasing was explicitly refuted in verification).

---

## 1. The core idea, in one diagram

```
                          ┌──────────────────────────────────────────────┐
   user turn  ───────────▶│                 MOTHERSHIP                    │
                          │  (router + planner + recurrent memory gate)   │
                          │  TITANS/MIRAS memory over the conversation     │
                          └───────────────┬──────────────────────────────┘
                                          │  decompose → required skill keys
                                          ▼
                          ┌──────────────────────────────────────────────┐
                          │            SKILL RESOLVER                     │
                          │  for each skill key:                          │
                          │   ├─ in Ortho-LoRA cache?  ──yes──▶ load it    │
                          │   └─ miss ──▶ PROJECT (T2L) ──▶ TEST ──┐       │
                          │                                        │       │
                          │              pass eval? ──yes──▶ MEMORIALIZE   │
                          │                            └─no──▶ escalate    │
                          └───────────────┬──────────────────────────────┘
                                          ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │                      EXPERT SWARM (serving plane)                      │
   │   N × ~200M ternary expert bases (frozen)   +   shared Ortho-LoRA cache│
   │   ┌────────┐ ┌────────┐ ┌────────┐ ...      LRU adapter cache (VRAM)    │
   │   │expert 0│ │expert 1│ │expert k│          SGMV/MBGMV batched kernels  │
   │   └────────┘ └────────┘ └────────┘          paged KV cache              │
   └──────────────────────────────────────────────────────────────────────┘
```

**Key separation of concerns** (this is the part the old Polymath swarm conflated):
- **Base weights** = the ternary expert (frozen, cheap, many copies/one shared base + many KV slots).
- **Skill** = an Ortho-LoRA adapter in a parallel fp16 path, *added* to the frozen base output.
- **Memory** = a recurrent gate on the residual stream (TITANS/MIRAS), state, not weights.
- **Contingency** = a projected LoRA generated on demand, then promoted to a real adapter.

---

## 2. The planes

### 2.1 The Mothership (orchestration + memory)

**Role:** decompose a turn into skill keys, choose expert(s), carry conversational long-term memory, and run the project→test→memorialize loop on a cache miss. It does **not** answer directly (the Polymath "brainstem" pattern, kept).

**Memory = MIRAS associative module.** The mothership owns a small recurrent **memory gate** (`memory_gate.py`) — the modern, verified version of your existing `OTitansMemoryGate`:
- **Write rule (TITANS):** `M_t = (1−α_t)·diag-decay ⊙ M_{t-1} + θ_t·(v_t − M_{t-1}k_t)⊗k_t` — surprise = `v − Mk` (delta rule), `α_t` = **retention gate** (MIRAS knob 3), `θ_t` = surprise learning rate.
- **Read:** `r_t = M_t q_t`, blended into the stream by a learned gate (your existing pattern).
- Why upgrade: your current gate uses a fixed `memory_momentum` scalar and L2-ish update. MIRAS says the retention gate and the attentional-bias objective are *independent knobs* — making `α_t` and `θ_t` **data-dependent** (small Linear heads) is the single highest-value change.

**Routing.** Keep the deterministic `[ROUTE: key1, key2]` contract from `polymath_swarm_*.py` — it works and it's debuggable. The router is itself a small model (Track B: the 270m; later: the mothership's own head). Output is validated against the skill registry (`engrams`-style), exactly as you already do.

**HOPE tie-in (later):** the **Continuum Memory System** maps cleanly onto our memory tiers — conversation memory (fast, every step) → skill cache (slow, every memorialize) → base weights (frozen, never). That *is* "memory as a spectrum of modules updating at different frequencies." We don't need full self-modification on day one, but the tiering is the design north star.

### 2.2 The Expert Swarm (ternary bases)

**What an "expert" is:** a ~200M ternary-weight model, **frozen**, serving as the base for a family of skills. Two substrate tracks:

- **Track B (test-fire, now):** `gemma-3-270m` → ternary via **QAT continued-pretraining** (not PTQ — see §0). One shared base, many adapters. This validates the *entire* control loop cheaply. Reality check: this is a small training run, not a quantize-and-go.
- **Track A (target):** train our **own looped / recurrent-depth** ~200M expert (`looped_expert.py`). Weight-tied block (Geiping: prelude → recurrent core → coda), looped `r` times for "depth for free" + test-time-tunable compute. This is where the differentiation lives; a tiny tied core is naturally parameter-frugal.

**Ternary + LoRA composition (the load-bearing feasibility claim):** verified feasible. Forward is
`y = ternary_matmul(x, W_tern) + (α/r)·B(A x)`, with `W_tern` frozen and `A,B` in fp16. Published in QVAC Fabric (ternary forward / fp16 backward, adapter kept as a separate `--lora` file) and BitLoRA. **Implication:** the Ortho-LoRA cache stays fp16 and is *base-agnostic in format*, which is what makes a shared cache across many experts tractable.

### 2.3 The Ortho-LoRA Skill Cache (shared memory of skills)

**What's cached:** per-skill `{A, B}` low-rank matrices on the attention projections (q/v, optionally k/o), trained **orthogonally** so skills compose without destructive interference — your `get_orthogonal_penalty()` = |cos(W_base, ΔW)| added to the loss is exactly right and is **kept**.

**Why orthogonal matters more here than in Polymath:** in a swarm you will stack *several* skills onto one expert per turn. Orthogonality (skills live in different subspaces) is what lets `load_state_dict(..., strict=False)` additive stacking not blow up — the assumption your `execute_hot_swap` already relies on, now made principled.

**Cache as MIRAS associative memory:** keys = skill/task descriptors (embeddings), values = adapter handles. A lookup is nearest-neighbor in descriptor space; a *miss* is "no key within threshold τ" → triggers projection. This reframes the registry from a hard string match (`engrams.json`) to a **soft semantic memory**, which is what lets new/unseen skills be detected.

**Serving ("turbocache"):** since the term isn't real, we implement what the literature proves:
1. **LRU adapter cache in VRAM**, backed by host RAM, backed by disk (`adapter_cache.py`). Hundreds of 3–20 MB adapters → host RAM is fine; VRAM holds the hot set.
2. **Batched heterogeneous-adapter kernel** (SGMV / MBGMV style) so different skills run in one GPU call. Scaffold uses a correct-but-slow gather-loop with a clear seam to drop in a real kernel.
3. **Paged KV cache** for the expert(s) (vLLM-style) once we serve concurrently.

### 2.4 The Projected-LoRA Contingency (project → test → memorialize)

The contingency path when the cache misses. Verified primitive: **Text-to-LoRA (T2L)**.

```
PROJECT      hypernetwork( embed(skill_description) , layer/module embeddings )  →  {A,B} rank-8 on q/v
                (one forward pass, ~seconds, standard PEFT LoRA out)

TEST         run a held-out / synthesized eval for the skill on (expert + projected adapter)
                gate = score ≥ θ_skill ?   (T2L is very sensitive to description quality → this gate is mandatory)

MEMORIALIZE  if pass:  write adapter .safetensors + config + descriptor-key into the Ortho-LoRA cache,
                       register the skill in the registry (your forge/save_pure_adapter machinery, reused)
             if fail:  (a) refine description and re-project, (b) fall back to a real forge() training job,
                       or (c) escalate to a larger model and log the gap for offline training.
```

**Quality expectation (T2L, author-reported):** a projected adapter beats the base model and a multi-task LoRA zero-shot, recovers ~97% of a per-task trained LoRA — **a genuinely useful cold-start**, not a toy. The honest caveats: self-reported/single-lab numbers, and sharp degradation on out-of-distribution skills or poor descriptions. That's *why* the TEST gate exists and why MEMORIALIZE keeps the option to replace a projected adapter with a properly forged one later (HOPE's "fast in-context test → slower persistent memory" tiering).

---

### 2.5 The Collaboration Plane — "two-step firing" (draft → sync → vote)

LLMs get better with collaboration, so the swarm does not just pick one expert and answer.
The selected experts **fire twice** with a sync in between:

```
Step 1  DRAFT   each selected expert answers independently      → diversity of view
        SYNC    experts exchange drafts + rationales, then revise → coherence (collaboration)
Step 2  VOTE    experts vote/score the synced candidates         → potency (best survives)
        AGGREGATE  majority | judge-scored | borda  → final answer
```

**Why it works (established multi-agent results, general knowledge — not part of the
mid-2026 verified pass):** this fuses two effects that repeatedly help — **multi-agent
debate / "society of minds"** (agents revise after seeing peers → fewer errors, better
reasoning) and **self-consistency** (vote over multiple candidates beats greedy single-pass).
The SYNC round is the coherence boost; the VOTE is the potency boost.

**Where diversity comes from:** every expert fires with the *same* resolved skill adapters,
so the candidates differ by expert (persona/seed/substrate), not by skill. Later, pick the
collaborating set to span *complementary* specialties so the sync round has something to
trade — that's when collaboration pays the most.

**Cost knob:** `sync_rounds` and `swarm_width` directly trade latency for quality. With
ternary experts being cheap, firing 3–5 in parallel + 1 sync round is the sweet spot to
start. This is also a natural fit for the looped expert: "think longer" (more loops) and
"confer longer" (more sync rounds) are two dials on the same compute-for-quality tradeoff.

Implemented in `consensus.py` (`two_step_firing`), driven by `mothership.py`.

## 3. Mapping old → new (what we keep, change, drop)

| Polymath / O-TITANS asset | Verdict | New home |
|---|---|---|
| `OLoRALinear` + `get_orthogonal_penalty()` | **Keep** (it's correct) | `skillshot/ortho_lora.py` |
| `OTitansTrainer` (ortho_lambda penalty in loss) | **Keep** | training scripts |
| `OTitansMemoryGate` (fixed momentum delta-rule) | **Upgrade** → data-dependent retention/surprise gates | `skillshot/memory_gate.py` |
| `OTitansTriArchRouter` (alpha blend) | **Generalize** → N-skill additive stack | mothership / expert forward |
| `polymath_swarm_*` hot-swap-adapters-onto-one-base | **Reframe** → swarm of ternary bases + shared cache | `skillshot/mothership.py` |
| `engrams.json` hard string match | **Upgrade** → semantic descriptor keys (miss detection) | `skillshot/registry.py` |
| `forge_*` + `save_pure_adapter` | **Keep** as the "real training" memorialize path | `skillshot/memorialize.py` |
| ledger.json | **Keep** as conversation store; back the memory gate | mothership |
| Single fp16 4B/12B synthesizer | **Drop** | replaced by ternary expert swarm |

---

## 4. Risks & open decisions (be honest)

1. **Ternary quality floor.** At 200–500M, ternary costs ~1–3 pts vs fp16, and BitNet only reaches fp parity ~3B. A 270M ternary expert may be *too weak* to be a useful specialist even with a good adapter. **Mitigation / decision:** Track B is explicitly a *de-risking* run — if quality is unacceptable, the swarm base moves up to ~500M–1B ternary, or Track A's looped-depth recovers effective capacity without param growth.
2. **Ternary speed needs kernels.** Storage shrinks immediately; throughput does **not** until bitnet.cpp-class GEMM. Until then the swarm's win is memory density (many experts resident), not latency.
3. **Projected-LoRA OOD failure.** T2L degrades on skills far from its training distribution. The TEST gate catches it, but a hard miss means falling back to real training — so the loop must never *block* on projection. Always have the escalate path.
4. **Adapter-on-ternary numerics.** Published feasible, but fp16 adapter + ternary base means the adapter carries disproportionate signal; watch for the adapter "doing all the work" (i.e., the base contributing little). Eval base-only vs base+adapter to quantify.
5. **Orthogonality at scale.** |cos| penalty is a soft constraint; stacking many skills may still interfere. May need true subspace allocation (assign each skill a reserved rank block) if interference shows up.
6. **HOPE self-modification is research-grade.** We adopt its *tiered-memory* framing now; full self-referential update rules are a later research bet, not a v1 commitment.

---

## 5. Where the scaffold sits

See `skillshot/` for runnable skeletons (pure-torch definitions; heavy model loads behind functions), and `PRODUCTION_PLAN.md` for the phased path from test-fire to production. Each module's docstring cites the specific paper/section it implements.
