"""chip_loss_helpers.py

Helpers for the observed-over-expected (OE) phase-similarity loss term.

This module implements chip_oe_similarity_loss, which supervises the
ChIP-seq auxiliary head predictions to match the pairwise absolute
differences in log-OE contact maps across cell-cycle phases.

Loss intuition
--------------
Given the log-normalised true phase maps x0_true (B, 5, N, N):

  1. Undo log normalisation  → raw contacts via exp(matrix)
  2. Compute differentiable O/E: divide each element X_ij by the mean
     contact value Ex(s) = mean_{|i-j|=s} X_ij along diagonal s.
  3. Take log → log_OE(phase)  (B, N, N)
  4. The target for each phase pair is the element-wise absolute
     difference:  target_{a,b}[i,j] = |log_OE_a[i,j] - log_OE_b[i,j]|
  5. The prediction is the element-wise absolute difference of the
     chip-aux-head outputs for those two phases:
         pred_{a,b}[i,j] = |chip_pred_a[i,j] - chip_pred_b[i,j]|
  6. MSE over all three pairs (prometa/earlyG1, prometa/anatelo,
     anatelo/earlyG1) gives the final scalar loss.

Note on formulation
-------------------
A Hadamard product (chip_pred_a * chip_pred_b) was tried but suffers from
a dead-gradient problem at zero initialisation: each factor's gradient is
gated by the other factor, so if both start at zero neither can bootstrap.
The absolute difference has a direct, non-multiplicative gradient path.

Phase channel indices in x0_true (and chip_pred):
    0 = earlyG1,  1 = midG1,  2 = lateG1,  3 = anatelo,  4 = prometa
"""

import torch
import torch.nn.functional as F

# Phase indices (matching x0_current channel order in the training loop)
_IDX_EARLYG1 = 0
_IDX_ANATELO = 3
_IDX_PROMETA = 4

# Numerical stability floor for log and division
_EPS = 1e-6


def _compute_log_oe(raw_contacts: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """
    Compute log(Observed / Expected) for a batch of contact maps.

    Expected contact at genomic separation s is the mean of all matrix
    entries X_ij with |i - j| = s, computed as a vectorised scatter-add
    so the operation is fully differentiable and GPU-friendly.

    Args:
        raw_contacts: (B, N, N)  positive-valued raw contact counts
        eps:          small additive floor for numerical stability
    Returns:
        (B, N, N)  log-OE values
    """
    B, N, _ = raw_contacts.shape
    device  = raw_contacts.device
    dtype   = raw_contacts.dtype

    # Integer genomic-distance for every (i, j) entry  → (N, N)
    dist_mat  = (
        torch.arange(N, device=device).unsqueeze(1)
        - torch.arange(N, device=device).unsqueeze(0)
    ).abs()
    dist_flat = dist_mat.reshape(-1).long()              # (N*N,)

    contacts_flat = raw_contacts.reshape(B, -1)          # (B, N*N)
    idx           = dist_flat.unsqueeze(0).expand(B, -1) # (B, N*N)

    # Sum contacts that fall on each diagonal distance
    sums = torch.zeros(B, N, device=device, dtype=dtype)
    sums.scatter_add_(1, idx, contacts_flat)             # (B, N)

    # Count how many (i, j) pairs have each distance (same for every batch)
    counts = torch.zeros(N, device=device, dtype=dtype)
    counts.scatter_add_(
        0, dist_flat,
        torch.ones(N * N, device=device, dtype=dtype),
    )                                                    # (N,)

    # Mean per diagonal distance, broadcast-divided
    mean_per_diag = sums / counts.unsqueeze(0).clamp(min=1.0)   # (B, N)

    # Look up the expected value for every (i, j)  → (B, N, N)
    expected = mean_per_diag.gather(1, idx).reshape(B, N, N)

    oe = raw_contacts / (expected + eps)
    return torch.log(oe + eps)                           # (B, N, N)


def chip_oe_similarity_loss(
    chip_pred: torch.Tensor,
    x0_true:   torch.Tensor,
    eps:       float = _EPS,
) -> torch.Tensor:
    """
    OE-based pairwise phase-similarity loss.

    For three pairs of phases — (prometa, earlyG1), (prometa, anatelo),
    (anatelo, earlyG1) — we compute:

        target_{a,b}  = |log_OE_a - log_OE_b|           (B, N, N)
        pred_{a,b}    = |chip_pred_a - chip_pred_b|      (B, N, N)  abs diff
        pair_loss     = F.mse_loss(pred_{a,b}, target_{a,b})

    The returned loss is the mean over the three pairs.

    Targets are detached from the computational graph (they serve as
    fixed supervision; gradients flow only through chip_pred).

    Args:
        chip_pred: (B, 5, N, N)  chip-aux-head phase predictions
                   (output of raw_model.chip_aux_pred(h_chip))
        x0_true:   (B, 5, N, N)  ground-truth log-normalised phase maps
        eps:       numerical stability floor

    Returns:
        scalar tensor — average MSE across the three phase pairs
    """
    # 1. Undo log normalisation to recover raw contact values
    raw_early = x0_true[:, _IDX_EARLYG1].exp()   # (B, N, N)
    raw_ana   = x0_true[:, _IDX_ANATELO].exp()
    raw_pro   = x0_true[:, _IDX_PROMETA].exp()

    # 2+3. Differentiable log-OE; detach so no gradient flows through targets
    log_oe_early = _compute_log_oe(raw_early, eps).detach()
    log_oe_ana   = _compute_log_oe(raw_ana,   eps).detach()
    log_oe_pro   = _compute_log_oe(raw_pro,   eps).detach()

    # 4. Pairwise absolute log-OE differences (targets)
    target_pro_early  = (log_oe_pro - log_oe_early).abs()
    target_pro_ana    = (log_oe_pro - log_oe_ana  ).abs()
    target_ana_early  = (log_oe_ana - log_oe_early).abs()

    # 5. Chip-aux predictions per phase
    pred_early = chip_pred[:, _IDX_EARLYG1]   # (B, N, N)
    pred_ana   = chip_pred[:, _IDX_ANATELO]
    pred_pro   = chip_pred[:, _IDX_PROMETA]

    # 6. Predicted pairwise difference (absolute difference, not Hadamard product).
    # This avoids the dead-gradient problem: gradient is ±1 * upstream_grad,
    # independent of the other factor's magnitude.
    pred_pro_early  = (pred_pro - pred_early).abs()
    pred_pro_ana    = (pred_pro - pred_ana  ).abs()
    pred_ana_early  = (pred_ana - pred_early).abs()

    # 7. MSE for each pair, averaged
    return (
        F.mse_loss(pred_pro_early,  target_pro_early)
        + F.mse_loss(pred_pro_ana,  target_pro_ana)
        + F.mse_loss(pred_ana_early, target_ana_early)
    ) / 3.0
