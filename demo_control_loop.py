"""
End-to-end demo of the SKILLSHOT control loop with MockExperts (no model downloads).

Exercises: route -> semantic resolve -> (cache HIT path) AND (cache MISS -> project ->
test -> memorialize) -> LRU load -> two-step firing (draft -> sync -> vote) -> answer.

Run (needs torch, e.g. the otitans venv):
    python demo_control_loop.py
"""
import os
import tempfile

import torch  # noqa: F401  (pulled in by the skillshot modules)

from skillshot.adapter_cache import AdapterStore, LRUAdapterCache, SkillAdapter
from skillshot.consensus import two_step_firing  # noqa: F401 (used via mothership)
from skillshot.expert import MockExpert
from skillshot.memorialize import Memorializer
from skillshot.mothership import Mothership, keyword_router
from skillshot.projected_lora import ProjectedLoRAHypernet, TargetSpec, TextEncoder
from skillshot.registry import SkillRegistry


def build_targets(n_layers=4, dim=256):
    specs = []
    for li in range(n_layers):
        for mod in ("q_proj", "v_proj"):
            specs.append(TargetSpec(li, mod, dim, dim))
    return specs


def fake_forged_adapter(targets, key, desc):
    sd = {}
    for t in targets:
        sd[f"{t.layer_idx}.{t.module}.lora_A"] = torch.zeros(8, t.in_features)
        sd[f"{t.layer_idx}.{t.module}.lora_B"] = torch.zeros(t.out_features, 8)
    return SkillAdapter(key=key, description=desc, state_dict=sd, source="forged", score=1.0)


def main():
    tmp = tempfile.mkdtemp(prefix="skillshot_")
    targets = build_targets()
    encoder = TextEncoder(dim=256)                      # hash-embedding fallback (offline)
    hypernet = ProjectedLoRAHypernet(targets, rank=8, desc_dim=256)

    # --- registry + store seeded with ONE real skill; a second will be projected live ---
    registry = SkillRegistry(encoder, threshold=0.45)
    store = AdapterStore(os.path.join(tmp, "adapters"))
    skills = {"code_python": "Python programming, async logic and scripting."}
    for k, d in skills.items():
        store.put(fake_forged_adapter(targets, k, d))
        registry.register(k, d)
    cache = LRUAdapterCache(store, device="cpu", capacity_bytes=512 * 1024 ** 2)

    # --- TEST gate evaluator: accept the projected adapter (stub returns a passing score) ---
    def evaluator(key, desc, sd):
        # Real: run a held-out eval on (expert + sd). Stub: pass if the hypernet produced
        # finite weights for every target.
        ok = all(torch.isfinite(v).all() for v in sd.values())
        return 0.8 if ok else 0.0

    escalated = []
    memorializer = Memorializer(hypernet, encoder, store, registry, evaluator,
                                theta=0.6, escalate=lambda k, d: escalated.append((k, d)))

    experts = [MockExpert("expert-0", "concise"), MockExpert("expert-1", "rigorous"),
               MockExpert("expert-2", "creative")]

    router = keyword_router({**skills, "threat modeling": "security threat modeling"})
    ship = Mothership(experts, router, registry, cache, memorializer,
                      aggregate="judge", sync_rounds=1, swarm_width=3)

    print(f"[demo] workspace: {tmp}\n")
    for prompt in [
        "write python to poll three apis concurrently",     # HIT: code_python
        "do a threat modeling pass on this login flow",     # MISS: project->test->memorialize
    ]:
        res = ship.handle(prompt)
        print(f"USER: {prompt}")
        print(f"  route        : {res.route}")
        print(f"  skills_used  : {res.skills_used}")
        print(f"  projected    : {res.projected}")
        if res.firing:
            print(f"  drafts       : {list(res.firing.drafts.values())}")
            print(f"  vote tally   : {res.firing.tally}")
            print(f"  winner       : [{res.firing.winner_expert}] {res.firing.winner}")
        print()

    print(f"[demo] registry now: {registry._keys}")
    print(f"[demo] escalated   : {escalated}")


if __name__ == "__main__":
    main()
