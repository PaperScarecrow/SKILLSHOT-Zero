"""
Ternary (1.58-bit) expert substrate.

Implements BitNet b1.58 (arXiv 2402.17764) absmean weight quantization {-1,0,+1} with
8-bit activation quantization, plus a Straight-Through Estimator so the layer is
QAT-trainable. This is the *expert base* the swarm runs on.

Two verified facts drive this file:
  * You cannot zero-shot PTQ a model to ternary (it collapses; PTQTP 2509.16989) — so
    BitLinear is built for QAT continued-pretraining (the "fry" job), with the optional
    extra-RMSNorm trick (2505.08823) before quantization.
  * LoRA composes with a FROZEN ternary base in a higher-precision parallel path
    (QVAC Fabric, BitLoRA) — so OrthoLoRALinear wraps TernaryLinear unchanged.

NOTE: this is reference/QAT code in fp16-simulated ternary (like Bonsai today). Real
memory/throughput wins need a packed 1.58-bit kernel (bitnet.cpp class) — see the
`dense_weight()` seam and PRODUCTION_PLAN.md Phase 4.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def absmean_quantize_weight(w: torch.Tensor, eps: float = 1e-5):
    """BitNet b1.58 weight quant: scale by mean|W|, round-clip to {-1,0,1}."""
    scale = w.abs().mean().clamp_(min=eps)
    q = (w / scale).round().clamp_(-1, 1)
    return q, scale


def absmax_quantize_act(x: torch.Tensor, bits: int = 8, eps: float = 1e-5):
    """Per-token absmax activation quant to `bits` int (8-bit per BitNet)."""
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_(min=eps) / qmax
    q = (x / scale).round().clamp_(-qmax, qmax)
    return q, scale


class TernaryLinear(nn.Module):
    """BitLinear: ternary weights + 8-bit activations, QAT via STE.

    Stores a full-precision shadow weight (`weight`) for training; quantizes on the
    fly. After training you would pack `quantized_weight()` into 1.58-bit + a scale.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 act_bits: int = 8, pre_rmsnorm: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_bits = act_bits
        # FP shadow weight (the QAT master copy). Frozen at serve time.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # Extra RMSNorm before quant (2505.08823) — stabilizes FP->ternary fine-tuning.
        self.pre_norm = nn.RMSNorm(in_features) if pre_rmsnorm else nn.Identity()

    @staticmethod
    def _ste(real: torch.Tensor, quant: torch.Tensor) -> torch.Tensor:
        """Straight-through: forward uses quant, backward flows to real."""
        return real + (quant - real).detach()

    def dense_weight(self) -> torch.Tensor:
        """Materialize the effective ternary weight (for ortho-penalty / packing)."""
        q, scale = absmean_quantize_weight(self.weight)
        return q * scale

    def quantized_weight(self):
        """Return (ternary {-1,0,1}, scale) for packing into 1.58-bit storage."""
        return absmean_quantize_weight(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        # Weights: STE ternary
        q_w, w_scale = absmean_quantize_weight(self.weight)
        w_eff = self._ste(self.weight / w_scale.clamp(min=1e-5), q_w) * w_scale
        # Activations: STE 8-bit
        q_x, x_scale = absmax_quantize_act(x, self.act_bits)
        x_eff = self._ste(x / x_scale.clamp(min=1e-5), q_x) * x_scale
        return F.linear(x_eff, w_eff, self.bias)


def ternarize_linear(layer: nn.Linear, **kw) -> TernaryLinear:
    """Seed a TernaryLinear from an existing fp Linear (start of a fry job)."""
    t = TernaryLinear(layer.in_features, layer.out_features,
                      bias=layer.bias is not None, **kw)
    with torch.no_grad():
        t.weight.copy_(layer.weight)
        if layer.bias is not None and t.bias is not None:
            t.bias.copy_(layer.bias)
    return t


def freeze_ternary_base(model: nn.Module) -> None:
    """Freeze every TernaryLinear so only LoRA adapters + memory gate train at serve."""
    for m in model.modules():
        if isinstance(m, TernaryLinear):
            for p in m.parameters():
                p.requires_grad = False
