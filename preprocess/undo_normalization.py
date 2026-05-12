"""
Utility to reverse the normalization applied in the dataloader.

The dataloader normalization process:
1. (possibly, depending on whether we use observed or observed/expected) log1p(x) - log transform raw counts
2. Clip at 99.9th percentile
3. Min-max normalization to [0, 1]
4. Scale to [-1, 1]: x * 2 - 1

To reverse (approximately):
1. [-1, 1] -> [0, 1]: (x + 1) / 2
2. [0, 1] -> [0, ~max_log]: estimate original (possibly) log-transformed scale
3. (depending on whether we use observed or observed/expected) expm1(x) to get back to raw counts
4. Clamp to visualization range (e.g., [0, 40])

Note: If you save normalization stats using the dataloader's save_normalization_stats=True
parameter, you can use reverse_normalization_exact() for perfect reversal.
"""

import numpy as np
from scipy.stats import rankdata
from typing import Tuple


def reverse_normalization_to_log_scale(
    normalized_matrix: np.ndarray,
    max_log_value: float = 40.0,
    clip_min: float = 0.0,
    clip_max: float = 40.0
) -> np.ndarray:
    """
    Reverse the dataloader normalization to approximate log-transformed Hi-C values.

    In papers, the [0, 40] scale is applied to LOG-TRANSFORMED contact counts,
    not raw counts. This function reverses the [-1, 1] normalization back to
    approximate log-transformed values for visualization.

    Args:
        normalized_matrix: Matrix with values in [-1, 1] (model output)
        max_log_value: Maximum log-transformed value to scale to (default 40)
                       This represents the range used in papers for visualization
        clip_min: Minimum value for visualization (default 0)
        clip_max: Maximum value for visualization (default 40)

    Returns:
        Matrix with approximate log-transformed values, clipped to [clip_min, clip_max]
    """
    # Step 1: Reverse [-1, 1] scaling to [0, 1]
    matrix_01 = (normalized_matrix + 1.0) / 2.0

    # Step 2: Scale to [0, max_log_value] - this is in log-transformed space
    # The dataloader normalized log1p(counts) to [-1, 1]
    # We're reversing that to approximate the log-transformed values
    matrix_log = matrix_01 * max_log_value

    # Step 3: Clip to visualization range (still in log space)
    matrix_clipped = np.clip(matrix_log, clip_min, clip_max)

    return matrix_clipped


def quantile_normalize_across_samples(
    matrices: list,
    output_min: float = 0.0,
    output_max: float = 40.0
) -> list:
    """
    Apply quantile normalization across multiple matrices using pooled ranks.

    This allows fair comparison across phases by ensuring the same quantile
    scale is used for all matrices from the same genomic region. Instead of
    normalizing each phase independently, this function computes ranks across
    ALL values from all phases together.

    Example:
        For region chr1:10Mb-10.064Mb, if you have earlyG1, midG1, lateG1, anatelo:
        - Pool all values from all 4 phases together
        - Compute global ranks across this pool
        - Map each value to [output_min, output_max] based on its global rank

        This way, colors represent the same relative contact frequency across
        all phases, making phase-to-phase comparison meaningful.

    Args:
        matrices: List of matrices to normalize together (e.g., [early, mid, late, anatelo, bulk])
        output_min: Minimum value in output (default 0)
        output_max: Maximum value in output (default 40)

    Returns:
        List of quantile-normalized matrices, same order as input
    """
    if not matrices:
        return []

    # Flatten all matrices and concatenate
    all_values = np.concatenate([m.flatten() for m in matrices])

    # Get ranks across ALL values from all matrices
    # method='average' handles ties by averaging their ranks
    all_ranks = rankdata(all_values, method='average')

    # Normalize ranks to [0, 1]
    ranks_normalized = (all_ranks - 1) / (len(all_ranks) - 1)

    # Scale to [output_min, output_max]
    quantile_normalized = ranks_normalized * (output_max - output_min) + output_min

    # Split back into separate matrices
    result_matrices = []
    offset = 0
    for m in matrices:
        n_elements = m.size
        result_matrices.append(quantile_normalized[offset:offset + n_elements].reshape(m.shape))
        offset += n_elements

    return result_matrices


def reverse_normalization_exact(
    normalized_matrix: np.ndarray,
    phase_min: float,
    phase_max: float,
    clip_min: float = 0.0,
    clip_max: float = 40.0
) -> np.ndarray:
    """
    Exactly reverse the dataloader normalization using saved min/max values.

    This function performs perfect reversal of the normalization when you have
    the original min/max values that were used during normalization. These values
    can be saved by setting save_normalization_stats=True in the dataloader.

    The reversal process:
    1. Reverse [-1, 1] scaling: (x + 1) / 2 -> [0, 1]
    2. Reverse min-max normalization: x * (max - min) + min -> original log scale
    3. Clip to visualization range

    Args:
        normalized_matrix: Matrix with values in [-1, 1] (model output or dataloader output)
        phase_min: The minimum value that was used during normalization (from stats file)
        phase_max: The maximum value that was used during normalization (from stats file)
        clip_min: Minimum value for visualization (default 0)
        clip_max: Maximum value for visualization (default 40)

    Returns:
        Matrix with exact log-transformed values, clipped to [clip_min, clip_max]
    """
    # Step 1: Reverse [-1, 1] scaling to [0, 1]
    matrix_01 = (normalized_matrix + 1.0) / 2.0

    # Step 2: Reverse min-max normalization to original log scale
    # Original normalization: (x - min) / (max - min) * 2 - 1
    # So: x_01 = (x - min) / (max - min)
    # Therefore: x = x_01 * (max - min) + min
    matrix_log = matrix_01 * (phase_max - phase_min) + phase_min

    # Step 3: Clip to visualization range (in log space)
    matrix_clipped = np.clip(matrix_log, clip_min, clip_max)

    return matrix_clipped


