"""Lightweight shape/smoke tests (no dataset required). Run with `pytest`."""
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.ssf_higt import SSFHiGT
from src.losses import total_loss, contrastive_loss, graph_smoothness_loss
from src.utils.masking import mask_features


def _build_model(num_features=20, num_classes=5, d_model=32, seq_len=8, num_nodes=15):
    model = SSFHiGT(
        num_features=num_features,
        num_classes=num_classes,
        d_model=d_model,
        num_heads=4,
        num_transformer_layers=2,
        num_gat_layers=2,
        gat_heads=2,
        edge_dim=num_features,
        dropout=0.1,
        max_seq_len=seq_len,
    )
    return model


def test_forward_shapes():
    B, T, F, N, C = 4, 8, 20, 15, 5
    model = _build_model(num_features=F, num_classes=C, seq_len=T, num_nodes=N)

    x_seq = torch.randn(B, T, F)
    node_features = torch.randn(N, F)
    adj = (torch.rand(N, N) > 0.5).float()
    edge_attr = torch.randn(N, N, F)

    out = model(x_seq, node_features, adj, edge_attr)
    assert out["logits"].shape == (B, C)
    assert out["z_t"].shape[0] == B
    assert out["z_f"].shape[0] == B


def test_masked_reconstruction_shapes():
    B, T, F = 4, 8, 20
    x = torch.randn(B, T, F)
    masked_x, mask = mask_features(x, ratio=0.3)
    assert masked_x.shape == x.shape
    assert mask.shape == (B, T)


def test_losses_run():
    B, T, F, N, C, D = 4, 8, 20, 15, 5, 32
    model = _build_model(num_features=F, num_classes=C, d_model=D, seq_len=T, num_nodes=N)
    x_seq = torch.randn(B, T, F)
    node_features = torch.randn(N, F)
    adj = (torch.rand(N, N) > 0.5).float()
    edge_attr = torch.randn(N, N, F)
    y = torch.randint(0, C, (B,))

    out = model(x_seq, node_features, adj, edge_attr)
    masked_x, mask = mask_features(x_seq, ratio=0.3)
    recon = model.forward_pretrain(masked_x)

    loss, components = total_loss(
        logits=out["logits"],
        targets=y,
        reconstructed=recon,
        original=x_seq,
        recon_mask=mask,
        z_anchor=out["z_t"],
        z_positive=out["z_t"].detach() + 0.01,
        node_embeddings=out["z_g_nodes"],
        adj=adj,
        model=model,
    )
    assert loss.dim() == 0
    assert "L_CE" in components


if __name__ == "__main__":
    test_forward_shapes()
    test_masked_reconstruction_shapes()
    test_losses_run()
    print("All smoke tests passed.")
