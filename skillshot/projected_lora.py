"""
Projected LoRA — the cache-miss contingency.

Text-to-LoRA / T2L (Sakana, arXiv 2506.06105): a hypernetwork takes an embedded natural-
language SKILL DESCRIPTION plus learnable per-(layer, module) embeddings and emits a
standard rank-r LoRA {A, B} on q/v in ONE forward pass. Output is an ordinary PEFT
adapter, so it drops straight into the Ortho-LoRA cache.

Verified expectation: a projected adapter beats base + multi-task LoRA zero-shot and
recovers ~97% of a per-task trained LoRA — a real cold-start, NOT a toy. But it is sharply
sensitive to description quality and degrades out-of-distribution, which is exactly why
`memorialize.py` makes the TEST gate mandatory before persisting.

This is the "L"-style joint head variant. The text encoder is pluggable (default: a
sentence-transformer; a hash-embedding fallback keeps the module importable offline).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TargetSpec:
    """Which (layer, module) projections the hypernet writes, and their shapes."""
    layer_idx: int
    module: str            # "q_proj" | "v_proj" | ...
    in_features: int
    out_features: int


class TextEncoder(nn.Module):
    """Pluggable description encoder. Real: gte-large/sentence-transformers.
    Fallback: deterministic hash embedding so the package imports with no downloads.
    """

    def __init__(self, dim: int = 384, model_name: str | None = None):
        super().__init__()
        self.dim = dim
        self.model_name = model_name
        self._st = None
        if model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(model_name)
                self.dim = self._st.get_sentence_embedding_dimension()
            except Exception:
                self._st = None  # fall back silently

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        if self._st is not None:
            return torch.tensor(self._st.encode(text), dtype=torch.float32)
        # Deterministic bag-of-hashed-tokens fallback (good enough to wire the loop).
        vec = torch.zeros(self.dim)
        for tok in text.lower().split():
            vec[hash(tok) % self.dim] += 1.0
        return vec / (vec.norm() + 1e-6)


class ProjectedLoRAHypernet(nn.Module):
    """description embedding (+ target embeddings) -> {A,B} per target, one pass."""

    def __init__(self, targets: list[TargetSpec], rank: int = 8, alpha: float = 16.0,
                 desc_dim: int = 384, hidden: int = 512, n_layers: int = 8,
                 n_modules: int = 4):
        super().__init__()
        self.targets = targets
        self.rank = rank
        self.alpha = alpha
        self.layer_emb = nn.Embedding(n_layers, 64)
        self.module_ids = {m: i for i, m in enumerate(sorted({t.module for t in targets}))}
        self.module_emb = nn.Embedding(max(n_modules, len(self.module_ids)), 64)

        self.trunk = nn.Sequential(
            nn.Linear(desc_dim + 128, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        # One small head per distinct output width; here a generic factored head that
        # produces A (rank x in) and B (out x rank) given the target's dims.
        self.a_head = nn.Linear(hidden, rank)   # expanded against an input basis
        self.b_head = nn.Linear(hidden, rank)
        # Learned bases let a fixed-width head emit variable-width matrices.
        self.in_basis = nn.ParameterDict()
        self.out_basis = nn.ParameterDict()
        for t in targets:
            key = f"{t.layer_idx}.{t.module}"
            self.in_basis[key.replace('.', '_')] = nn.Parameter(torch.randn(t.in_features, rank) * 0.02)
            self.out_basis[key.replace('.', '_')] = nn.Parameter(torch.randn(t.out_features, rank) * 0.02)

    def forward(self, desc_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns a LoRA state_dict keyed to match OrthoLoRALinear param names."""
        out: dict[str, torch.Tensor] = {}
        for t in self.targets:
            key = f"{t.layer_idx}.{t.module}"
            bkey = key.replace('.', '_')
            cond = torch.cat([
                desc_emb,
                self.layer_emb.weight[t.layer_idx % self.layer_emb.num_embeddings],
                self.module_emb.weight[self.module_ids[t.module]],
            ], dim=-1)
            h = self.trunk(cond)
            # A: (rank, in) = (rank-coeffs) outer-projected through in_basis^T
            a_coeff = self.a_head(h)                                  # (rank,)
            b_coeff = self.b_head(h)                                  # (rank,)
            A = (self.in_basis[bkey] * a_coeff).T                    # (rank, in)
            B = (self.out_basis[bkey] * b_coeff)                     # (out, rank)
            out[f"{key}.lora_A"] = A.contiguous()
            out[f"{key}.lora_B"] = B.contiguous()
        return out


def project_adapter(hypernet: ProjectedLoRAHypernet, encoder: TextEncoder,
                    description: str) -> dict[str, torch.Tensor]:
    """One-call cold-start: NL skill description -> LoRA state_dict."""
    emb = encoder.encode(description).to(next(hypernet.parameters()).dtype)
    with torch.no_grad():
        return hypernet(emb)
