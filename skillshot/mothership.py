"""
The Mothership — orchestration over the expert swarm.

Per turn:
  1. ROUTE      decompose the turn into required skill descriptions (deterministic
                [ROUTE: ...] contract from the Polymath router; pluggable).
  2. RESOLVE    semantic registry lookup -> (hits, misses).
  3. CONTINGENCY for each miss: project -> test -> memorialize (turns a miss into a hit).
  4. LOAD       pull hit adapters from the LRU AdapterCache (host/disk backed).
  5. FIRE       two-step firing across the selected swarm experts: draft -> sync -> vote.
  6. REMEMBER   update the conversation memory gate / ledger.

The mothership itself does not answer; it directs. This keeps the brainstem/frontal-lobe
split from Polymath while replacing the single synthesizer with a collaborating swarm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .adapter_cache import AdapterStore, LRUAdapterCache
from .consensus import two_step_firing, FiringTrace
from .memorialize import Memorializer
from .registry import SkillRegistry


# A router: prompt -> list of required skill descriptions/keys.
Router = Callable[[str], list[str]]


@dataclass
class TurnResult:
    answer: str
    skills_used: list[str]
    projected: list[str]
    firing: Optional[FiringTrace]
    route: list[str]


class Mothership:
    def __init__(self, experts: list, router: Router, registry: SkillRegistry,
                 cache: LRUAdapterCache, memorializer: Memorializer,
                 memory_gate=None, aggregate: str = "judge", sync_rounds: int = 1,
                 swarm_width: int = 3):
        self.experts = experts
        self.router = router
        self.registry = registry
        self.cache = cache
        self.memorializer = memorializer
        self.memory_gate = memory_gate
        self.aggregate = aggregate
        self.sync_rounds = sync_rounds
        self.swarm_width = swarm_width
        self.ledger: list[dict] = []

    def _select_experts(self, skills) -> list:
        # Day 1: fire the first `swarm_width` experts. Later: pick by skill affinity so the
        # collaborating set spans complementary specialties (boosts the sync round's value).
        return self.experts[: self.swarm_width] if self.swarm_width else self.experts

    def handle(self, prompt: str) -> TurnResult:
        self.ledger.append({"role": "user", "content": prompt})

        # 1. ROUTE
        route = self.router(prompt)

        # 2. RESOLVE (semantic; a miss = no skill within threshold)
        res = self.registry.resolve(route)
        projected: list[str] = []

        # 3. CONTINGENCY: project -> test -> memorialize each miss
        for miss in res.misses:
            outcome = self.memorializer.handle_miss(miss)
            if outcome.status == "memorialized":
                res.hits.append(outcome.key)
                projected.append(outcome.key)
            # escalated/rejected misses simply proceed without that skill (never block)

        # 4. LOAD adapters from the cache
        adapters = [a for a in (self.cache.get(k) for k in dict.fromkeys(res.hits)) if a]

        # 5. FIRE: two-step firing across the swarm
        experts = self._select_experts(res.hits)
        firing = two_step_firing(experts, prompt, adapters,
                                 aggregate=self.aggregate, sync_rounds=self.sync_rounds)
        answer = firing.winner

        # 6. REMEMBER
        self.ledger.append({"role": "assistant", "content": answer})
        # memory_gate would consume embedded (prompt, answer) here to update M (slow tier:
        # the cache; fast tier: this gate) — wired once experts are real.

        return TurnResult(answer=answer,
                          skills_used=[getattr(a, "key", "?") for a in adapters],
                          projected=projected, firing=firing, route=route)


def keyword_router(skill_descriptions: dict[str, str]) -> Router:
    """Trivial offline router: returns skill keys whose name appears in the prompt.

    Real router = the fried 270m emitting [ROUTE: ...] (polymath_swarm pattern), or the
    mothership's own routing head.
    """
    def _route(prompt: str) -> list[str]:
        p = prompt.lower()
        hits = [k for k in skill_descriptions if k.replace("_", " ") in p or k in p]
        return hits or list(skill_descriptions)[:1]
    return _route
