"""
Semantic skill registry — upgrade of `engrams.json` hard string matching.

A skill is a (key, description, embedding). Resolving a request returns the matching
keys; a MISS is "best cosine similarity < threshold τ" -> hand off to the projected-LoRA
contingency. This is the MIRAS "associative memory" view of the cache: keys = task
descriptors, values = adapter handles, lookup = nearest neighbour.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ResolveResult:
    hits: list[str]                 # registered skill keys to load from cache
    misses: list[str]               # requested skills with no close match -> project
    scores: dict[str, float]


class SkillRegistry:
    def __init__(self, encoder, threshold: float = 0.55):
        self.encoder = encoder
        self.threshold = threshold
        self._keys: list[str] = []
        self._desc: dict[str, str] = {}
        self._emb: dict[str, torch.Tensor] = {}

    def register(self, key: str, description: str) -> None:
        self._keys.append(key)
        self._desc[key] = description
        self._emb[key] = self.encoder.encode(description)

    def has(self, key: str) -> bool:
        return key in self._desc

    def _nearest(self, query_emb: torch.Tensor) -> tuple[str | None, float]:
        best_key, best = None, -1.0
        for k in self._keys:
            sim = torch.nn.functional.cosine_similarity(
                query_emb, self._emb[k], dim=0).item()
            if sim > best:
                best_key, best = k, sim
        return best_key, best

    def resolve(self, requested) -> ResolveResult:
        """`requested` is a list of skill descriptions (or keys) from the router.

        Each is matched semantically; close enough -> hit (existing skill), else miss.
        """
        hits, misses, scores = [], [], {}
        for req in requested:
            if self.has(req):                       # exact key from router
                hits.append(req); scores[req] = 1.0; continue
            emb = self.encoder.encode(req)
            key, sim = self._nearest(emb)
            scores[req] = sim
            if key is not None and sim >= self.threshold:
                hits.append(key)
            else:
                misses.append(req)
        return ResolveResult(hits=hits, misses=misses, scores=scores)
