"""
Recurrent memory gate (the mothership's long-term memory).

Upgrade of the user's `OTitansMemoryGate`. Two changes, both motivated by verified
research:

  * TITANS (2501.00663): the write is a delta-rule "surprise" update with momentum and
    adaptive forgetting. Keep this.
  * MIRAS (2504.13173): the *retention gate* (forgetting) and the *surprise learning
    rate* are INDEPENDENT knobs, not a single fixed momentum scalar. Make them
    data-dependent (small Linear heads). This is the single highest-value upgrade.

State is a matrix memory M (d_k x d_v). Read r = M^T q. This is the "fast" tier of
HOPE's Continuum Memory System (updates every step); the skill cache is the "slow" tier.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MirasMemoryGate(nn.Module):
    def __init__(self, hidden_size: int, key_dim: int | None = None,
                 value_dim: int | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_k = key_dim or hidden_size
        self.d_v = value_dim or hidden_size

        self.q_proj = nn.Linear(hidden_size, self.d_k, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.d_k, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.d_v, bias=False)

        # MIRAS knobs as data-dependent heads:
        #   alpha_t in (0,1) = retention gate (how much past memory to keep)
        #   theta_t > 0      = surprise learning rate (how hard to write)
        self.retention_head = nn.Linear(hidden_size, 1)
        self.surprise_head = nn.Linear(hidden_size, 1)
        self.momentum = nn.Parameter(torch.tensor(0.9))  # TITANS momentum on the surprise

        # Output blend gate (kept from the original design).
        self.out_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.Sigmoid(),
        )
        self.read_proj = nn.Linear(self.d_v, hidden_size, bias=False)
        self.register_buffer("memory", torch.zeros(self.d_k, self.d_v), persistent=False)
        self.register_buffer("surprise_mom", torch.zeros(self.d_k, self.d_v), persistent=False)

    def reset(self) -> None:
        self.memory.zero_()
        self.surprise_mom.zero_()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, T, H). Sequential recurrence (inference path).

        Training would use the chunkwise-parallel form (see PRODUCTION_PLAN Phase 5);
        the sequential loop here is the readable reference.
        """
        B, T, H = hidden.shape
        assert B == 1, "reference gate is single-stream; batch via chunkwise form later"
        q = self.q_proj(hidden)
        k = self.k_proj(hidden)
        v = self.v_proj(hidden)
        alpha = torch.sigmoid(self.retention_head(hidden))      # (1,T,1) retention
        theta = torch.nn.functional.softplus(self.surprise_head(hidden))  # (1,T,1) lr

        M = self.memory.clone()
        S = self.surprise_mom.clone()
        reads = []
        for t in range(T):
            q_t, k_t, v_t = q[0, t], k[0, t], v[0, t]
            reads.append(M.t() @ q_t)                            # (d_v,) read before write
            surprise = torch.outer(k_t, v_t - (M.t() @ k_t))     # (d_k,d_v) delta rule
            S = self.momentum * S + theta[0, t] * surprise       # momentum on surprise
            M = alpha[0, t] * M + S                               # retention-gated write
        self.memory.copy_(M.detach())
        self.surprise_mom.copy_(S.detach())

        read_v = torch.stack(reads).unsqueeze(0)                 # (1,T,d_v)
        read = self.read_proj(read_v)                            # (1,T,H) back to model dim
        g = self.out_gate(torch.cat([hidden, read], dim=-1))     # (1,T,H) blend gate
        return hidden + g * read
