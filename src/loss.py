"""
Constraint losses for the named-dimension training.

Two sparse losses apply only at color-anchor positions:
  - anchor_loss : MSE between the named R, G, B dims and the target RGB
                  values, masked per-channel. Each anchor record specifies
                  which channels to supervise via `anchor_channel`:
                    -1 -> all three channels active (named colors, rgb(...))
                     0 -> only R (first digit pair of a hex code)
                     1 -> only G (second pair)
                     2 -> only B (third pair)
  - sparsity_loss : L2 penalty on all non-RGB dims at anchor positions
                    (drives the model to localize color info to dims 0/1/2).

The standard LM cross-entropy is computed by the HuggingFace model when
labels are passed; we just combine.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    lambda_anchor: float = 1.0
    lambda_sparse: float = 0.1


@dataclass
class LossOutputs:
    lm_loss: torch.Tensor
    anchor_loss: torch.Tensor
    sparse_loss: torch.Tensor
    total: torch.Tensor
    n_anchors: int


def _build_channel_mask(anchor_channel: torch.Tensor, dtype, device) -> torch.Tensor:
    """Convert (N,) int channel codes to an (N, 3) float mask.

    channel == -1 -> [1, 1, 1]
    channel == 0  -> [1, 0, 0]
    channel == 1  -> [0, 1, 0]
    channel == 2  -> [0, 0, 1]
    """
    N = anchor_channel.numel()
    mask = torch.zeros((N, 3), dtype=dtype, device=device)
    full = (anchor_channel == -1)
    if full.any():
        mask[full] = 1.0
    for k in range(3):
        kmask = (anchor_channel == k)
        if kmask.any():
            mask[kmask, k] = 1.0
    return mask


def constraint_loss(
    hidden_states: torch.Tensor,         # (B, T, H)
    anchor_batch: torch.Tensor,          # (N,) batch indices
    anchor_pos: torch.Tensor,            # (N,) ID-space positions
    anchor_rgb: torch.Tensor,            # (N, 3) target RGB in [0, 1]
    anchor_channel: torch.Tensor,        # (N,) int8/int64; -1 = full, 0/1/2 = per-channel
    named_dim_indices: tuple[int, int, int] = (0, 1, 2),
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute anchor + sparsity losses at the given positions.

    Returns (anchor_loss, sparse_loss, n_anchors). If no anchors in this
    batch, returns zero losses that still carry grad-fn so the optimizer
    step is happy.
    """
    n = anchor_batch.numel()
    if n == 0:
        zero = hidden_states.sum() * 0.0
        return zero, zero.clone(), 0

    selected = hidden_states[anchor_batch, anchor_pos, :]    # (N, H)

    r_idx, g_idx, b_idx = named_dim_indices
    rgb_pred = torch.stack(
        [selected[:, r_idx], selected[:, g_idx], selected[:, b_idx]], dim=1,
    )                                                         # (N, 3)

    mask = _build_channel_mask(anchor_channel, dtype=selected.dtype,
                               device=selected.device)        # (N, 3)
    sq_err = (rgb_pred - anchor_rgb) ** 2
    n_active = mask.sum().clamp(min=1.0)
    anchor_loss = (sq_err * mask).sum() / n_active

    # Sparsity: every OTHER dimension at these positions should be near zero
    H = selected.size(-1)
    sparse_mask = torch.ones(H, dtype=torch.bool, device=selected.device)
    for idx in named_dim_indices:
        sparse_mask[idx] = False
    other = selected[:, sparse_mask]
    sparse_loss = (other ** 2).mean()

    return anchor_loss, sparse_loss, n


def combined_loss(
    model_outputs,
    hidden_states: torch.Tensor,
    anchor_batch: torch.Tensor,
    anchor_pos: torch.Tensor,
    anchor_rgb: torch.Tensor,
    anchor_channel: torch.Tensor,
    weights: LossWeights,
) -> LossOutputs:
    """Combine LM loss (from HuggingFace output) with the constraint losses."""
    lm_loss = model_outputs.loss
    anchor_loss, sparse_loss, n = constraint_loss(
        hidden_states, anchor_batch, anchor_pos, anchor_rgb, anchor_channel,
    )
    total = lm_loss + weights.lambda_anchor * anchor_loss + weights.lambda_sparse * sparse_loss
    return LossOutputs(
        lm_loss=lm_loss,
        anchor_loss=anchor_loss,
        sparse_loss=sparse_loss,
        total=total,
        n_anchors=n,
    )


@torch.no_grad()
def rgb_accuracy_stats(
    hidden_states: torch.Tensor,
    anchor_batch: torch.Tensor,
    anchor_pos: torch.Tensor,
    anchor_rgb: torch.Tensor,
    anchor_channel: torch.Tensor,
    named_dim_indices: tuple[int, int, int] = (0, 1, 2),
) -> dict[str, float]:
    """Diagnostic stats on RGB encoding quality at anchor positions.

    Per-channel masking is honored: an anchor with channel=0 contributes
    only to the R-channel statistics.
    """
    if anchor_batch.numel() == 0:
        return {
            "rgb_mae": float("nan"),
            "rgb_max_err": float("nan"),
            "rgb_corr_r": float("nan"),
            "rgb_corr_g": float("nan"),
            "rgb_corr_b": float("nan"),
            "n_anchors": 0,
        }
    selected = hidden_states[anchor_batch, anchor_pos, :]
    r_idx, g_idx, b_idx = named_dim_indices
    pred = torch.stack(
        [selected[:, r_idx], selected[:, g_idx], selected[:, b_idx]], dim=1,
    )
    mask = _build_channel_mask(anchor_channel, dtype=pred.dtype, device=pred.device)
    err = (pred - anchor_rgb).abs() * mask
    n_active = mask.sum().clamp(min=1.0)
    stats = {
        "rgb_mae": (err.sum() / n_active).item(),
        "rgb_max_err": err.max().item(),
        "n_anchors": int(anchor_batch.numel()),
    }
    for j, name in enumerate(("r", "g", "b")):
        active = mask[:, j] > 0
        if active.sum() < 2:
            stats[f"rgb_corr_{name}"] = float("nan")
            continue
        p = pred[active, j]
        t = anchor_rgb[active, j]
        if p.std() < 1e-6 or t.std() < 1e-6:
            stats[f"rgb_corr_{name}"] = float("nan")
        else:
            stats[f"rgb_corr_{name}"] = torch.corrcoef(
                torch.stack([p, t])
            )[0, 1].item()
    return stats
