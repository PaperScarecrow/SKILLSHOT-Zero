"""
Track A substrate — weight-tied recurrent-depth expert (skeleton).

Geiping et al. (arXiv 2502.05171): a small weight-tied block looped r times gives "depth
for free" — 8 physical layers unrolled to ~132 effective at r=32 — and lets reasoning
scale at TEST TIME by choosing r, with no extra parameters. Ideal for a ~200M expert.

Structure: Prelude (embed into latent) -> Recurrent core (looped r times) -> Coda (read out).
The recurrent core is where a delta-rule / MIRAS memory can live so "thinking longer"
(more loops) and "remembering" (memory writes) share the same recurrence — the unifying
view MIRAS argues for. Layers here use TernaryLinear so the looped expert is also 1-bit.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .ternary import TernaryLinear


class TiedBlock(nn.Module):
    """One transformer-ish block whose weights are REUSED across loop iterations."""

    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: int = 4, ternary: bool = True):
        super().__init__()
        Lin = TernaryLinear if ternary else nn.Linear
        self.norm1 = nn.RMSNorm(dim)
        self.qkv = Lin(dim, dim * 3)
        self.proj = Lin(dim, dim)
        self.n_heads = n_heads
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = nn.Sequential(Lin(dim, dim * mlp_ratio), nn.SiLU(), Lin(dim * mlp_ratio, dim))

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, C // self.n_heads)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))         # (B,h,T,d)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))

    def forward(self, x: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        # injection = prelude output, re-added each loop (Geiping: keeps input visible).
        x = x + injection
        x = x + self._attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LoopedExpert(nn.Module):
    def __init__(self, vocab: int, dim: int = 768, prelude_layers: int = 2,
                 coda_layers: int = 2, core_loops: int = 4, n_heads: int = 8,
                 ternary: bool = True):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.prelude = nn.ModuleList([TiedBlock(dim, n_heads, ternary=ternary) for _ in range(prelude_layers)])
        self.core = TiedBlock(dim, n_heads, ternary=ternary)     # the ONE tied block
        self.coda = nn.ModuleList([TiedBlock(dim, n_heads, ternary=ternary) for _ in range(coda_layers)])
        self.norm_f = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.default_loops = core_loops

    def forward(self, input_ids: torch.Tensor, r: int | None = None) -> torch.Tensor:
        r = r or self.default_loops                              # test-time-tunable depth
        x = self.embed(input_ids)
        zero = torch.zeros_like(x)
        for blk in self.prelude:
            x = blk(x, zero)
        injection = x                                           # keep prelude state visible
        s = torch.randn_like(x) * 0.02                          # latent state init (Geiping)
        for _ in range(r):                                      # <-- the loop = depth-for-free
            s = self.core(s, injection)
        x = s
        for blk in self.coda:
            x = blk(x, zero)
        return self.lm_head(self.norm_f(x))
