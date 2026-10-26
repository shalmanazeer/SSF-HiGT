"""
All loss terms composing the unified multi-objective L(Theta) -- Eq. (6):

    L(Theta) = L_CE + lambda1*L_Mask + lambda2*L_Contrast
               + lambda3*L_Graph + lambda4*L_Reg + lambda5*L_Fed
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """L_CE -- Eq. (7)-(8). `targets` are integer class indices;
    F.cross_entropy internally applies log-softmax, matching Eq. (8)."""
    return F.cross_entropy(logits, targets)


def masked_reconstruction_loss(
    reconstructed: torch.Tensor, original: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """L_Mask -- Eq. (9).

    Args:
        reconstructed: (B, T, F) decoder output D_Theta(E_Theta(M_rho(x)))
        original:      (B, T, F) unmasked ground-truth features x
        mask:          (B, T) boolean, True where the timestep was masked (t in M)
    """
    mask = mask.unsqueeze(-1).float()
    sq_err = (reconstructed - original) ** 2 * mask
    denom = mask.sum().clamp(min=1.0)
    return sq_err.sum() / denom


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """sim(a, b) -- Eq. (18)."""
    return F.cosine_similarity(a, b, dim=-1)


def contrastive_loss(z: torch.Tensor, z_pos: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """L_Contrast -- Eq. (17), NT-Xent style topology-aware contrastive loss.

    Args:
        z:     (B, d) anchor embeddings
        z_pos: (B, d) positive-pair embeddings (e.g. augmented graph view)
        temperature: tau
    """
    z = F.normalize(z, dim=-1)
    z_pos = F.normalize(z_pos, dim=-1)

    representations = torch.cat([z, z_pos], dim=0)  # (2B, d)
    sim_matrix = representations @ representations.t()  # (2B, 2B)
    sim_matrix = sim_matrix / temperature

    B = z.size(0)
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)

    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim_matrix.masked_fill_(mask, float("-inf"))

    return F.cross_entropy(sim_matrix, labels)


def graph_smoothness_loss(z: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """L_Graph = Tr(Z^T L Z), L = D - A -- Eq. (13)-(14).

    Args:
        z:   (N, d) node latent embeddings
        adj: (N, N) adjacency matrix A
    """
    degree = adj.sum(dim=1)
    laplacian = torch.diag(degree) - adj
    return torch.trace(z.t() @ laplacian @ z) / z.size(0)


def weight_regularization_loss(model: nn.Module) -> torch.Tensor:
    """L_Reg = ||Theta||_2^2, standard L2 weight decay term."""
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for p in model.parameters():
        reg = reg + p.pow(2).sum()
    return reg


def federated_consistency_loss(local_params: dict, global_params: dict) -> torch.Tensor:
    """L_Fed = sum_i || Theta_i - Theta_g ||_2^2, encourages local models to
    stay close to the current global model between aggregation rounds."""
    loss = None
    for name, local_p in local_params.items():
        if name not in global_params:
            continue
        global_p = global_params[name]
        term = (local_p - global_p).pow(2).sum()
        loss = term if loss is None else loss + term
    if loss is None:
        return torch.tensor(0.0)
    return loss


def total_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reconstructed: torch.Tensor | None = None,
    original: torch.Tensor | None = None,
    recon_mask: torch.Tensor | None = None,
    z_anchor: torch.Tensor | None = None,
    z_positive: torch.Tensor | None = None,
    node_embeddings: torch.Tensor | None = None,
    adj: torch.Tensor | None = None,
    model: nn.Module | None = None,
    local_params: dict | None = None,
    global_params: dict | None = None,
    weights: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    """Composes Eq. (6)/(21). Any component can be omitted (e.g. during
    pure SSL pre-training, `targets`/`logits` and L_Fed are skipped by the caller)."""
    weights = weights or {}
    lambda1 = weights.get("lambda_mask", 1.0)
    lambda2 = weights.get("lambda_contrast", 0.5)
    lambda3 = weights.get("lambda_graph", 0.1)
    lambda4 = weights.get("lambda_reg", 1e-5)
    lambda5 = weights.get("lambda_fed", 0.01)
    temperature = weights.get("nt_xent_temperature", 0.07)

    components = {}
    loss = torch.tensor(0.0, device=logits.device if logits is not None else "cpu")

    if logits is not None and targets is not None:
        l_ce = classification_loss(logits, targets)
        components["L_CE"] = l_ce.item()
        loss = loss + l_ce

    if reconstructed is not None and original is not None and recon_mask is not None:
        l_mask = masked_reconstruction_loss(reconstructed, original, recon_mask)
        components["L_Mask"] = l_mask.item()
        loss = loss + lambda1 * l_mask

    if z_anchor is not None and z_positive is not None:
        l_contrast = contrastive_loss(z_anchor, z_positive, temperature)
        components["L_Contrast"] = l_contrast.item()
        loss = loss + lambda2 * l_contrast

    if node_embeddings is not None and adj is not None:
        l_graph = graph_smoothness_loss(node_embeddings, adj)
        components["L_Graph"] = l_graph.item()
        loss = loss + lambda3 * l_graph

    if model is not None:
        l_reg = weight_regularization_loss(model)
        components["L_Reg"] = l_reg.item()
        loss = loss + lambda4 * l_reg

    if local_params is not None and global_params is not None:
        l_fed = federated_consistency_loss(local_params, global_params)
        components["L_Fed"] = float(l_fed.item()) if torch.is_tensor(l_fed) else l_fed
        loss = loss + lambda5 * l_fed

    return loss, components
