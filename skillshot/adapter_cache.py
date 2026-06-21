"""
Adapter cache + skill registry — the shared "memory of skills".

"Turbocache" is not a real published system (verified). What the literature actually
proves for serving hundreds of small adapters is the tiered design below:

  * LRU adapter cache in VRAM, backed by host RAM, backed by disk  (S-LoRA Unified Paging)
  * one batched heterogeneous-adapter kernel (SGMV / MBGMV)        (Punica / S-LoRA)
  * paged KV cache for the expert(s)                               (vLLM / PagedAttention)

This module implements tier 1 (the LRU cache + the on-disk store) and a SEMANTIC
registry (descriptor embeddings -> skill), so a "miss" is "no key within threshold τ"
rather than a hard string-mismatch. The batched kernel (tier 2) is a documented seam.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch


@dataclass
class SkillAdapter:
    key: str                                   # stable id, e.g. "code_python"
    description: str                           # NL descriptor (the semantic key)
    state_dict: dict                           # {f"{layer}.{module}.lora_A/B": tensor}
    source: str = "forged"                     # "forged" | "projected"
    score: float = 0.0                         # last eval score
    embedding: Optional[torch.Tensor] = field(default=None, repr=False)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.state_dict.values())


class AdapterStore:
    """Disk-backed store of forged/memorialized adapters (the slow tier)."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.index_path = os.path.join(root, "registry.json")
        self.index: dict[str, dict] = self._load_index()

    def _load_index(self) -> dict:
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)

    def put(self, adapter: SkillAdapter) -> None:
        path = os.path.join(self.root, f"{adapter.key}.pt")
        torch.save(adapter.state_dict, path)
        self.index[adapter.key] = {
            "description": adapter.description, "source": adapter.source,
            "score": adapter.score, "path": path,
        }
        self._save_index()

    def get(self, key: str) -> Optional[SkillAdapter]:
        meta = self.index.get(key)
        if not meta:
            return None
        sd = torch.load(meta["path"], map_location="cpu", weights_only=True)
        return SkillAdapter(key, meta["description"], sd, meta["source"], meta["score"])

    def keys(self):
        return list(self.index.keys())


class LRUAdapterCache:
    """VRAM-resident LRU hot set (the fast tier). Evicts to host/disk on pressure."""

    def __init__(self, store: AdapterStore, device: str = "cuda",
                 capacity_bytes: int = 2 * 1024 ** 3):
        self.store = store
        self.device = device
        self.capacity = capacity_bytes
        self._hot: "OrderedDict[str, SkillAdapter]" = OrderedDict()
        self._bytes = 0

    def _evict_until_fits(self, incoming: int) -> None:
        while self._hot and self._bytes + incoming > self.capacity:
            _, victim = self._hot.popitem(last=False)   # LRU end
            self._bytes -= victim.nbytes()

    def get(self, key: str) -> Optional[SkillAdapter]:
        if key in self._hot:
            self._hot.move_to_end(key)
            return self._hot[key]
        adapter = self.store.get(key)                    # host/disk -> VRAM
        if adapter is None:
            return None
        adapter.state_dict = {k: v.to(self.device) for k, v in adapter.state_dict.items()}
        self._evict_until_fits(adapter.nbytes())
        self._hot[key] = adapter
        self._bytes += adapter.nbytes()
        return adapter

    def warm(self, keys: list[str]) -> None:
        for k in keys:
            self.get(k)

    # SEAM: batched heterogeneous-adapter application across many requests/skills.
    # Real impl = SGMV (Punica) / MBGMV (S-LoRA) CUDA kernel. Scaffold does a loop.
    def batched_delta(self, x: torch.Tensor, keys: list[str], target: str, layer: int):
        raise NotImplementedError("Drop in SGMV/MBGMV kernel here (PRODUCTION_PLAN Phase 4).")
