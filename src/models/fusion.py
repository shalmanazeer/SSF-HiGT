"""
Adaptive cross-stream fusion module.

Implements Eq. (15)-(16):
    g   = sigma( W_g [Z_T || Z_G || Z_T⊙Z_G || |Z_T - Z_G|] + b )
    Z_f = g⊙Z_T + (1-g)⊙Z_G + alpha * (Z_T ⊙ Z_G)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveCrossStreamFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 4, dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, z_t: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: (B, dim) Transformer-stream embedding
            z_g: (B, dim) GNN-stream embedding
        Returns:
            (B, dim) fused embedding Z_f
        """
        interaction = z_t * z_g
        diff = torch.abs(z_t - z_g)
        gate_input = torch.cat([z_t, z_g, interaction, diff], dim=-1)
        g = torch.sigmoid(self.gate(gate_input))  # Eq. (15)
        z_f = g * z_t + (1.0 - g) * z_g + self.alpha * interaction  # Eq. (16)
        return z_f
