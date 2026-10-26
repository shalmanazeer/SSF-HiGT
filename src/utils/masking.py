"""Feature masking operator M_rho(.) used in Eq. (9)."""
from __future__ import annotations

import torch


def mask_features(x: torch.Tensor, ratio: float = 0.30, mask_value: float = 0.0):
    """Randomly masks a proportion `ratio` of timesteps in a flow-feature
    sequence, replacing them with `mask_value` (a learnable [MASK] token
    could be substituted here for a stronger baseline).

    Args:
        x: (B, T, F) input sequence
        ratio: rho, proportion of timesteps to mask
    Returns:
        masked_x: (B, T, F) with masked timesteps zeroed
        mask: (B, T) boolean, True where masked (the set M)
    """
    B, T, _ = x.shape
    mask = torch.rand(B, T, device=x.device) < ratio
    masked_x = x.clone()
    masked_x[mask] = mask_value
    return masked_x, mask
