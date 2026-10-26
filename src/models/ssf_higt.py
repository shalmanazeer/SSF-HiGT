"""
Full SSF-HiGT model f_Theta: (x, G) -> y_hat in Delta^C   -- Eq. (4)-(5)

Parameter groups Theta = {Theta_T, Theta_G, Theta_F, Theta_C}:
    Theta_T -> FlowTransformerEncoder
    Theta_G -> HeterogeneousGATEncoder
    Theta_F -> AdaptiveCrossStreamFusion
    Theta_C -> classification head
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .transformer_encoder import FlowTransformerEncoder, FlowReconstructionDecoder
from .gat_encoder import HeterogeneousGATEncoder
from .fusion import AdaptiveCrossStreamFusion


class SSFHiGT(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_transformer_layers: int = 6,
        num_gat_layers: int = 3,
        gat_heads: int = 4,
        edge_dim: int = 16,
        dropout: float = 0.15,
        max_seq_len: int = 32,
    ):
        super().__init__()

        # Theta_T
        self.transformer_encoder = FlowTransformerEncoder(
            num_features=num_features,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        # decoder D_Theta only used during SSL pre-training (Eq. 9)
        self.reconstruction_decoder = FlowReconstructionDecoder(d_model, num_features, dropout=dropout)

        # Theta_G
        self.gat_encoder = HeterogeneousGATEncoder(
            in_dim=num_features,
            hidden_dim=d_model,
            edge_dim=edge_dim,
            num_layers=num_gat_layers,
            num_heads=gat_heads,
            dropout=dropout,
        )

        # Theta_F
        self.fusion = AdaptiveCrossStreamFusion(dim=d_model)

        # Theta_C
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    # ---- self-supervised pre-training forward (masked reconstruction) ----
    def forward_pretrain(self, masked_seq: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """Returns reconstructed features for L_Mask (Eq. 9)."""
        h = self.transformer_encoder(masked_seq, attn_mask)   # (B, T, d_model)
        recon = self.reconstruction_decoder(h)                 # (B, T, num_features)
        return recon

    # ---- encoder-only forward, used for contrastive views ----
    def encode_streams(
        self,
        flow_seq: torch.Tensor,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ):
        z_t = self.transformer_encoder.pooled(flow_seq, attn_mask)         # (B, d_model)
        z_g_nodes = self.gat_encoder(node_features, adj, edge_attr)        # (N, d_model)
        z_g = z_g_nodes.mean(dim=0, keepdim=True).expand(z_t.size(0), -1)  # (B, d_model)
        return z_t, z_g, z_g_nodes

    # ---- full supervised forward: f_Theta(x, G) -> y_hat ----
    def forward(
        self,
        flow_seq: torch.Tensor,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        edge_attr: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ):
        z_t, z_g, z_g_nodes = self.encode_streams(flow_seq, node_features, adj, edge_attr, attn_mask)
        z_f = self.fusion(z_t, z_g)          # Eq. (15)-(16)
        logits = self.classifier(z_f)        # pre-softmax; use Eq. (7)-(8) loss externally
        return {
            "logits": logits,
            "z_t": z_t,
            "z_g": z_g,
            "z_g_nodes": z_g_nodes,
            "z_f": z_f,
        }
