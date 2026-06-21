"""
Orthogonal LoRA primitive (the shared "skill" representation).

Direct descendant of the user's `otitans_core.OLoRALinear` — kept because it is correct.
A frozen base layer runs in parallel with a low-rank A->B highway; the orthogonal
penalty |cos(W_base, dW)| is added to the training loss to push each skill into a
subspace that minimally overlaps the base (so multiple skills stack additively).

Reference framing: MIRAS (arXiv 2504.13173) treats such adapters as associative-memory
modules; orthogonality is our cheap stand-in for the "reserved subspace per skill".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthoLoRALinear(nn.Module):
    """Frozen base + parallel orthogonal LoRA highway.

    Works over ANY base layer exposing .in_features/.out_features and a callable
    forward, including the ternary `TernaryLinear` in ternary.py — that base-agnostic
    property is what lets one fp16 adapter cache serve many expert substrates.
    """

    def __init__(self, base_layer: nn.Module, rank: int = 8, alpha: float = 16.0,
                 adapter_dtype: torch.dtype = torch.float16):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.adapter_dtype = adapter_dtype

        # The only trainable params. Named lora_A/lora_B so existing extraction
        # (`save_pure_adapter`, `{k: v for ... if "lora" in k}`) keeps working.
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=adapter_dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=adapter_dtype))
        self._enabled = True
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # orthogonal_ needs float32 (QR unsupported for fp16 on CPU); init then cast.
        with torch.no_grad():
            tmp = torch.empty_like(self.lora_A, dtype=torch.float32)
            nn.init.orthogonal_(tmp)       # strict orthogonal init (attentional-bias geometry)
            self.lora_A.copy_(tmp.to(self.lora_A.dtype))
            nn.init.zeros_(self.lora_B)    # start invisible to the network

    @property
    def delta_w(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def get_orthogonal_penalty(self) -> torch.Tensor:
        """|cos(W_base, dW)| — 0 == orthogonal, 1 == full overlap. Add to loss."""
        base_w = getattr(self.base_layer, "weight", None)
        if base_w is None:                      # e.g. ternary base stores packed weights
            base_w = self.base_layer.dense_weight()
        base_flat = base_w.reshape(-1).float()
        delta_flat = self.delta_w.reshape(-1).float()
        return torch.abs(F.cosine_similarity(base_flat, delta_flat, dim=0))

    def set_enabled(self, flag: bool) -> None:
        self._enabled = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if not self._enabled:
            return base_out
        a = self.lora_A.to(x.dtype)
        b = self.lora_B.to(x.dtype)
        lora_out = (x @ a.T) @ b.T
        return base_out + lora_out * self.scaling


def inject_ortho_lora(model: nn.Module, target_suffixes=("q_proj", "v_proj"),
                      rank: int = 8, alpha: float = 16.0) -> int:
    """Surgically wrap target Linear layers (mirror of otitans_surgery)."""
    count = 0
    for name, module in list(model.named_modules()):
        if any(name.endswith(t) for t in target_suffixes) and hasattr(module, "in_features"):
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            child = name.rsplit(".", 1)[-1]
            setattr(parent, child, OrthoLoRALinear(module, rank=rank, alpha=alpha))
            count += 1
    return count


def extract_adapter(model: nn.Module) -> dict[str, torch.Tensor]:
    """Pull only LoRA tensors -> the persistable adapter state_dict."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def load_adapter(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Additive load (orthogonality is what makes strict=False stacking safe)."""
    model.load_state_dict(state_dict, strict=False)
