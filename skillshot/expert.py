"""
Expert substrate interfaces + a mock for wiring the control loop offline.

An Expert = a frozen ~200M ternary base (ternary.py) with Ortho-LoRA skill slots
(ortho_lora.py) and access to the mothership's memory read. The swarm runs many experts
(or one base in many KV slots) behind the AdapterCache.

`MockExpert` lets you exercise the whole mothership + consensus + project/test/memorialize
loop with NO model downloads, so the orchestration is testable in CI.

`HFTernaryExpert` is the real-substrate sketch (Track B: a fried Gemma-3-270m).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MockExpert:
    """Deterministic stand-in. `persona` biases its drafts so votes are non-trivial."""
    id: str
    persona: str = "balanced"
    skill_bias: dict = field(default_factory=dict)

    def answer(self, prompt: str, adapters: list) -> str:
        skills = "+".join(getattr(a, "key", str(a)) for a in adapters) or "base"
        return f"[{self.id}/{self.persona}|{skills}] draft for: {prompt[:60]}"

    def revise(self, prompt: str, own_draft: str, peer_drafts: dict, adapters: list) -> str:
        # "Collaboration": fold in the highest-agreement peer signal.
        peer_note = f" (synced with {len(peer_drafts)} peers)" if peer_drafts else ""
        return own_draft + peer_note

    def vote(self, prompt: str, candidates: list[str]) -> int:
        # Prefer the longest synced candidate as a crude "most complete" heuristic.
        return max(range(len(candidates)), key=lambda i: len(candidates[i]))

    def score(self, prompt: str, candidate: str) -> float:
        # Reward overlap with the prompt + this expert's skill bias keywords.
        toks = set(candidate.lower().split())
        overlap = len(toks & set(prompt.lower().split()))
        bias = sum(1 for kw in self.skill_bias if kw in candidate.lower())
        return overlap + 0.5 * bias + 1e-3 * len(candidate)


class HFTernaryExpert:
    """Real substrate sketch (Track B). Loads a ternary base + applies cached adapters.

    Filled in once a fried ternary checkpoint exists; kept thin so the interface is the
    contract the rest of the system codes against.
    """

    def __init__(self, id: str, model_path: str, memory_gate=None, device: str = "cuda"):
        self.id = id
        self.model_path = model_path
        self.memory_gate = memory_gate
        self.device = device
        self._model = None
        self._tok = None

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .ternary import TernaryLinear  # noqa: F401  (used during fry/conversion)
        self._tok = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)
        self._model.eval()
        return self

    def apply_adapters(self, adapters: list) -> None:
        # Additive load of orthogonal adapters onto wrapped OrthoLoRALinear modules.
        from .ortho_lora import load_adapter
        for a in adapters:
            load_adapter(self._model, getattr(a, "state_dict", a))

    def clear_adapters(self) -> None:
        raise NotImplementedError("reset OrthoLoRALinear lora_* to the sterile state (see swarm wipe)")

    def answer(self, prompt: str, adapters: list) -> str:
        raise NotImplementedError("apply_adapters + generate; thread memory_gate read into hidden states")

    def revise(self, prompt, own_draft, peer_drafts, adapters) -> str:
        peers = "\n".join(f"- peer {k}: {v}" for k, v in peer_drafts.items())
        return self.answer(f"{prompt}\n\nYour draft:\n{own_draft}\n\nPeers:\n{peers}\n"
                           "Revise to a stronger, coherent answer.", adapters)

    def vote(self, prompt, candidates) -> int:
        raise NotImplementedError("ask the expert to pick the best candidate index")

    def score(self, prompt, candidate) -> float:
        raise NotImplementedError("logprob- or rubric-based scoring of candidate")
