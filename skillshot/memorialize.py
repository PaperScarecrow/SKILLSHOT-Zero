"""
The project -> test -> memorialize loop (the contingency control flow).

When the registry reports a MISS for a needed skill:
  PROJECT     T2L hypernetwork emits a rank-r LoRA from the skill description.
  TEST        run the skill's eval on (expert + projected adapter); gate on score >= theta.
              (T2L is sensitive to description quality, so this gate is non-negotiable.)
  MEMORIALIZE on pass -> persist to the AdapterStore + register the skill (now a cache hit
              forever). on fail -> escalate: refine description / queue a real forge() job /
              hand off to a bigger model. The loop must never BLOCK on projection.

This mirrors HOPE's tiering: a fast in-context test promotes a skill into slower, persistent
memory. The "real forge() job" is the user's existing forge_*.py + save_pure_adapter path,
unchanged — projected adapters are placeholders that a trained adapter can later replace.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .adapter_cache import AdapterStore, SkillAdapter
from .projected_lora import ProjectedLoRAHypernet, TextEncoder, project_adapter
from .registry import SkillRegistry


@dataclass
class MemorializeResult:
    key: str
    status: str                 # "memorialized" | "rejected" | "escalated"
    source: str                 # "projected" | "forged"
    score: float
    detail: str = ""


# An evaluator: given (skill_key, description, adapter_state_dict) -> score in [0,1].
SkillEvaluator = Callable[[str, str, dict], float]
# An escalation hook: called when projection fails the gate (e.g. queue a forge job).
EscalationHook = Callable[[str, str], None]


class Memorializer:
    def __init__(self, hypernet: ProjectedLoRAHypernet, encoder: TextEncoder,
                 store: AdapterStore, registry: SkillRegistry,
                 evaluator: SkillEvaluator, theta: float = 0.6,
                 escalate: Optional[EscalationHook] = None, max_refine: int = 2):
        self.hypernet = hypernet
        self.encoder = encoder
        self.store = store
        self.registry = registry
        self.evaluator = evaluator
        self.theta = theta
        self.escalate = escalate
        self.max_refine = max_refine

    def _key_from_desc(self, description: str) -> str:
        return "proj_" + "_".join(description.lower().split()[:4])

    def handle_miss(self, description: str) -> MemorializeResult:
        key = self._key_from_desc(description)
        desc = description
        best_score = -1.0
        for attempt in range(self.max_refine + 1):
            # --- PROJECT ---
            sd = project_adapter(self.hypernet, self.encoder, desc)
            # --- TEST ---
            score = self.evaluator(key, desc, sd)
            best_score = max(best_score, score)
            if score >= self.theta:
                # --- MEMORIALIZE ---
                adapter = SkillAdapter(key=key, description=description, state_dict=sd,
                                       source="projected", score=score)
                self.store.put(adapter)
                self.registry.register(key, description)
                return MemorializeResult(key, "memorialized", "projected", score,
                                         f"passed on attempt {attempt}")
            # refine the description and retry (cheap; one more forward pass)
            desc = f"{description} (be precise and task-specific; attempt {attempt + 2})"

        # --- ESCALATE ---
        if self.escalate is not None:
            self.escalate(key, description)
        return MemorializeResult(key, "escalated", "projected", best_score,
                                 "projection below threshold; queued for real forge()/bigger model")
