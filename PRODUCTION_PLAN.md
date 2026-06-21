# SKILLSHOT — Production Plan

From the verified scaffold to a shippable system. Phases are ordered to **de-risk early**:
prove the control loop with cheap parts, then upgrade substrate and serving. Each phase has
an explicit **gate** — do not advance until it's met.

Legend: 🅑 = Track B (test-fire / fried Gemma-3-270m) · 🅐 = Track A (own looped model) ·
⚙️ = serving/infra.

---

## Phase 0 — Foundations (this scaffold) ✅
- `skillshot/` package + `demo_control_loop.py` run; primitives verified (ternary+LoRA
  composes, memory gate recurs, looped expert depth-for-free).
- **Gate:** control loop runs end-to-end with mock experts. **(met)**

---

## Phase 1 — Real single expert, real adapters 🅑
Goal: replace `MockExpert` with one real (fp16 first) Gemma-3-270m and forge 2–3 real
Ortho-LoRA skills, so the cache/registry/firing planes run on real generations.

- Wire `expert.HFTernaryExpert` for an **fp16** Gemma-3-270m (defer ternary to Phase 3) —
  load base, `inject_ortho_lora(q_proj, v_proj)`, implement `answer/revise/vote/score`.
- Port the user's `forge_*.py` + `save_pure_adapter` as the **memorialize "real training"**
  path; forge `code_python`, `cysec`, one more, into the `AdapterStore`.
- Replace `keyword_router` with the **270m `[ROUTE: ...]` router** (their existing
  `nano_router_270m_v2`).
- Swap `TextEncoder` to a real sentence-transformer (e.g. `gte-large`).
- **Gate:** on a 20-prompt eval, swarm (1 expert, real skills) beats base-no-skill; the
  `[ROUTE]` router picks the right skill ≥90%; a forged adapter loads from cache and changes
  outputs measurably.

## Phase 2 — Swarm + two-step firing 🅑
Goal: many experts collaborating beats one.

- Run **3–5 expert instances** (same base, different sample seeds/personas to start; one
  base copy + per-request adapters via the cache).
- Turn on `two_step_firing` with `sync_rounds=1`, `aggregate="judge"`.
- Build an **eval harness**: A/B {single expert} vs {swarm, no sync} vs {swarm, 1 sync} on
  reasoning + factuality sets (GSM8K subset, a recall set, a domain set).
- **Gate:** swarm+sync beats single-expert by a real margin on ≥2 of 3 eval sets, at an
  acceptable latency multiple (target ≤ 3× single-pass). If sync doesn't help, fix the
  expert-diversity (Phase 2.5) before blaming the method.
- **2.5 (if needed):** diversify experts by **skill affinity** (assign complementary
  specialties) so sync has something to trade.

## Phase 3 — Projected-LoRA contingency, end to end 🅑
Goal: a missing skill is handled live, tested, and persisted.

- Train the **Text-to-LoRA hypernetwork** (`projected_lora.py`) on the forged-adapter
  corpus from Phase 1 + public instruction-task LoRAs (input = skill description embedding,
  target = LoRA weights). ~single-GPU-day per T2L (matches the ~10h/H100 reference).
- Implement the **TEST evaluator** for `Memorializer`: a per-skill held-out / synthesized
  eval returning a real score (mandatory gate — T2L is description-sensitive).
- Wire the **escalation hook** to queue a real `forge()` job when projection fails.
- **Gate:** for ≥5 held-out skills, project→test→memorialize promotes a *useful* adapter
  (beats base; recovers a meaningful fraction of a forged adapter), and a bad/OOD skill is
  correctly **rejected/escalated**, never silently shipped.

## Phase 4 — Ternary substrate (the actual 1-bit experts) 🅑→🅐
Goal: the experts are genuinely 1.58-bit.

- **Fry Gemma-3-270m to ternary** via **QAT continued-pretraining** (NOT PTQ): swap Linears
  for `TernaryLinear` (extra-RMSNorm on), continual-QAT from the fp checkpoint over
  millions–billions of tokens. Expect ~1–3 pt drop.
- Re-validate Phases 1–3 on the ternary base. **Re-forge adapters on the ternary base**
  (an fp16 adapter trained against the ternary forward), per QVAC Fabric/BitLoRA.
- Add **packed 1.58-bit storage + kernel** (`quantized_weight()` → bitnet.cpp-class GEMM)
  for the real memory/throughput win; until then ternary only buys storage density.
- **Gate:** ternary expert + adapter stays within ~1–3 pts of the fp16 expert on the
  Phase-2 evals; many experts fit resident in VRAM. **Decision point:** if 270m-ternary is
  too weak, move base to ~500M–1B ternary, or commit to Track A.

## Phase 5 — Memory + Track-A looped expert 🅐
Goal: the differentiation — own substrate + real long-term memory.

- Train the **`LoopedExpert`** (~200M, ternary, weight-tied recurrent-depth) with loop-count
  `r` sampled during training (Geiping); expose test-time `r` as a quality dial.
- Integrate the **`MirasMemoryGate`** as the mothership's conversation memory; implement the
  **chunkwise-parallel** training form (replace the sequential reference); make retention/
  surprise data-dependent.
- Tie memory into HOPE's **tiered** view: fast gate (per step) · skill cache (per memorialize)
  · base (frozen). Optional research bet: self-modifying update rule.
- **Gate:** looped expert at higher `r` improves reasoning with no param growth; the memory
  gate measurably helps multi-turn recall vs a stateless baseline.

## Phase 6 — Serving at scale ⚙️
Goal: hundreds of experts/adapters, fast.

- **LRU adapter cache** (Phase 0) + **SGMV/MBGMV batched kernel** (`batched_delta` seam) +
  **paged KV cache** (vLLM). Add **dLoRA-style** merge/unmerge + request/adapter migration
  once multi-GPU.
- Load-test: adapter swap latency, batched throughput across heterogeneous skills, VRAM
  headroom vs resident-adapter count.
- **Gate:** sustained throughput serving N≫expert-count adapters with swap latency hidden by
  batching; no OOM under the target concurrent-skill mix.

---

## Cross-cutting

- **Eval harness is the spine** — stand it up in Phase 2 and reuse it as every later gate.
  Without it, "boosts coherence/potency" stays a vibe.
- **Always-non-blocking contingency** — projection failure must escalate, never stall a turn.
- **Honest substrate expectations** — ternary < fp at <3B; the swarm + collaboration + memory
  are what recover quality, not the base alone.
- **Reuse, don't rebuild** — forge/ortho-penalty/router/ledger all port directly from
  Polymath; only the substrate and serving are net-new.

## Risk → phase that retires it

| Risk | Retired by |
|---|---|
| Ternary too weak at 270M | Phase 4 gate (fallback to 500M–1B / Track A) |
| Two-step firing doesn't help | Phase 2 gate (+ 2.5 diversity fix) |
| Projected-LoRA ships garbage | Phase 3 TEST gate (mandatory) |
| No real ternary speedup | Phase 4 packed kernel / Phase 6 serving |
| Memory gate unstable in training | Phase 5 chunkwise form + gating stabilization |
