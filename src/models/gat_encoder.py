"""
Heterogeneous, edge-aware Graph Attention Network encoder.

Implements Eq. (10)-(12):

    h_v^(l+1) = sigma( sum_{u in N(v)} alpha_vu^(l) * W^(l) h_u^(l) )         (10)
    alpha_vu  = softmax_u( e_vu )                                            (11)
    e_vu      = LeakyReLU( a^T [ W h_v || W h_u || x_vu^e ] )                (12)

This extends the original GAT (Velickovic et al., ICLR 2018) [16] by
concatenating edge attributes x_vu^e into the attention-coefficient
computation, so device-communication metadata (protocol, port, byte-rate,
etc.) directly informs the attention weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareGATLayer(nn.Module):
    """A single heterogeneous GAT layer with multi-head attention and
    edge-attribute conditioning, implemented with dense adjacency for
    clarity. For large graphs, swap for torch_geometric.nn.GATv2Conv
    with edge_dim set, using the same attention formulation.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.edge_proj = nn.Linear(edge_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(num_heads, 3 * self.head_dim))
        nn.init.xavier_uniform_(self.attn)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,           # (N, in_dim) node features
        adj: torch.Tensor,         # (N, N) binary adjacency (1 = edge u->v)
        edge_attr: torch.Tensor,   # (N, N, edge_dim) edge attribute matrix
    ) -> torch.Tensor:
        N = h.size(0)
        Wh = self.W(h)                                    # (N, out_dim)
        We = self.edge_proj(edge_attr)                    # (N, N, out_dim)

        Wh_heads = Wh.view(N, self.num_heads, self.head_dim)     # (N, H, d)
        We_heads = We.view(N, N, self.num_heads, self.head_dim)  # (N, N, H, d)

        # e_vu = LeakyReLU(a^T [Wh_v || Wh_u || x_vu^e])   -- Eq. (12)
        Wh_v = Wh_heads.unsqueeze(1).expand(N, N, self.num_heads, self.head_dim)
        Wh_u = Wh_heads.unsqueeze(0).expand(N, N, self.num_heads, self.head_dim)
        triplet = torch.cat([Wh_v, Wh_u, We_heads], dim=-1)      # (N, N, H, 3d)

        e = torch.einsum("nmhd,hd->nmh", triplet, self.attn)     # (N, N, H)
        e = self.leaky_relu(e)

        mask = (adj > 0).unsqueeze(-1)                            # (N, N, 1)
        e = e.masked_fill(~mask, float("-inf"))

        # alpha_vu = softmax_u(e_vu)   -- Eq. (11)
        alpha = F.softmax(e, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)

        # h_v = sigma(sum_u alpha_vu * W h_u)   -- Eq. (10)
        out = torch.einsum("nmh,mhd->nhd", alpha, Wh_heads)
        out = out.reshape(N, self.num_heads * self.head_dim)
        return F.elu(out)


class HeterogeneousGATEncoder(nn.Module):
    """Stacks L_G EdgeAwareGATLayer blocks to produce topology-aware node
    embeddings Z_G, then mean-pools over relevant nodes for a per-flow
    graph embedding."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        edge_dim: int = 16,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            [
                EdgeAwareGATLayer(dims[i], dims[i + 1], edge_dim, num_heads, dropout)
                for i in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i + 1]) for i in range(num_layers)])

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = node_features
        for layer, norm in zip(self.layers, self.norms):
            h = norm(layer(h, adj, edge_attr))
        return h  # (N, hidden_dim) node embeddings

    def graph_pooled(self, node_features: torch.Tensor, adj: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.forward(node_features, adj, edge_attr)
        return h.mean(dim=0, keepdim=True)  # (1, hidden_dim)
