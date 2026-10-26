"""
Transformer encoder/decoder for temporal flow-sequence modelling.

Implements E_Theta / D_Theta from Eq. (9) of the paper:
    L_Mask = (1/|M|) * sum_{t in M} || x_t - D_Theta(E_Theta(M_rho(x))) ||_2^2

Backbone follows the standard multi-head self-attention Transformer
(Vaswani et al., "Attention Is All You Need", NeurIPS 2017) [24].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class FlowTransformerEncoder(nn.Module):
    """Encoder E_Theta: maps a (possibly masked) flow-feature sequence to
    a contextual latent representation Z_T.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.15,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, num_features) raw or masked flow-feature sequence
            attn_mask: optional (B, T) padding mask, True = pad
        Returns:
            (B, T, d_model) contextual embeddings; caller may pool over T.
        """
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h, src_key_padding_mask=attn_mask)
        return self.norm(h)

    def pooled(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Mean-pool over the temporal dimension to get a per-flow embedding Z_T."""
        h = self.forward(x, attn_mask)
        if attn_mask is not None:
            valid = (~attn_mask).unsqueeze(-1).float()
            return (h * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        return h.mean(dim=1)


class FlowReconstructionDecoder(nn.Module):
    """Decoder D_Theta: reconstructs original per-timestep features from the
    latent representation produced by E_Theta, used for L_Mask in Eq. (9)."""

    def __init__(self, d_model: int, num_features: int, num_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        layers = []
        in_dim = d_model
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, d_model), nn.GELU(), nn.Dropout(dropout)]
            in_dim = d_model
        layers.append(nn.Linear(in_dim, num_features))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, T, d_model) -> (B, T, num_features)
        return self.decoder(z)
