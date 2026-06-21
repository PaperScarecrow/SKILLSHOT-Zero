# `skillshot/` — scaffold package

Runnable skeletons for the mothership / 1-bit-expert-swarm architecture. See top-level
`DESIGN.md` for the why and `PRODUCTION_PLAN.md` for the how-to-ship.

> Every module's docstring cites the specific paper/section it implements.
> Pure-torch primitives run on CPU; heavy model loads sit behind functions.

## Module map

| Module | Plane | Status |
|---|---|---|
| `ortho_lora.py` | Orthogonal LoRA skill primitive (`get_orthogonal_penalty`) | ✅ runs |
| `ternary.py` | 1.58-bit BitNet expert substrate (QAT/STE, extra-RMSNorm) | ✅ runs |
| `looped_expert.py` | Track-A weight-tied recurrent-depth expert (Geiping) | ✅ runs |
| `memory_gate.py` | MIRAS/TITANS recurrent memory (data-dependent retention+surprise) | ✅ runs (seq ref) |
| `projected_lora.py` | Text-to-LoRA hypernetwork (cache-miss contingency) | ✅ runs (hash-embed fallback) |
| `adapter_cache.py` | LRU VRAM cache + disk store (the real "turbocache") | ✅ runs; ⚠️ batched kernel = seam |
| `registry.py` | Semantic skill registry (miss detection) | ✅ runs |
| `memorialize.py` | project → test → memorialize loop | ✅ runs |
| `consensus.py` | Two-step firing: draft → sync → vote | ✅ runs |
| `expert.py` | Expert interfaces + `MockExpert` | ✅ mock runs; ⚠️ HF expert = sketch |
| `mothership.py` | Orchestrator | ✅ runs |

## Try it

```bash
# any python with torch (e.g. /home/paperscarecrow/Downloads/.venv)
python demo_control_loop.py
```

Exercises both the cache-HIT path and the MISS → project → test → memorialize path,
then two-step firing across three mock experts.

## The seams that need real work (not stubs forever)

1. **Batched heterogeneous-adapter kernel** — `LRUAdapterCache.batched_delta` raises
   `NotImplementedError`. Drop in SGMV (Punica) / MBGMV (S-LoRA).
2. **Real experts** — `expert.HFTernaryExpert` is a sketch; needs a fried ternary base.
3. **Packed 1.58-bit storage/kernel** — `ternary.TernaryLinear` simulates ternary in fp;
   `quantized_weight()` is the packing seam.
4. **Real description encoder** — `TextEncoder` falls back to hash embeddings offline;
   pass `model_name="thenlper/gte-large"` (or similar) for real semantics.
5. **Chunkwise-parallel memory** — `MirasMemoryGate.forward` is the sequential reference;
   training wants the parallel form.
