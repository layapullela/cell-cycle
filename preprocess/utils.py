import numpy as np
import torch


############################################
# NUMPY VARIANTS (used in Dataloader)
############################################
def matrix_to_upper_tri_vec_np(matrix: np.ndarray) -> np.ndarray:
    """
    Extract upper triangular vector from a single symmetric numpy matrix.

    Args:
        matrix: (n, n) symmetric numpy array

    Returns:
        vec: (n*(n+1)/2,) numpy array, row-major order
    """
    idx = np.triu_indices(matrix.shape[0])
    return matrix[idx[0], idx[1]]


def upper_tri_vec_to_matrix_np(vec: np.ndarray, n: int) -> np.ndarray:
    """
    Reconstruct a symmetric numpy matrix from an upper triangular vector.

    Args:
        vec: (n*(n+1)/2,) numpy array
        n: matrix size

    Returns:
        matrix: (n, n) symmetric numpy array
    """
    matrix = np.zeros((n, n), dtype=vec.dtype)
    idx = np.triu_indices(n)
    matrix[idx[0], idx[1]] = vec
    matrix[idx[1], idx[0]] = vec  # Make symmetric
    return matrix


############################################
# TORCH VARIANTS (used in model / training)
############################################
def matrix_to_upper_tri_vec(matrix):
    """
    Convert batch of symmetric matrices to upper triangular vectors.

    Args:
        matrix: (batch, n, n) symmetric matrices

    Returns:
        vec: (batch, n*(n+1)/2) upper triangular vectors
    """
    batch_size, n, _ = matrix.shape
    indices = torch.triu_indices(n, n, device=matrix.device)
    vec = matrix[:, indices[0], indices[1]]  # (batch, n*(n+1)/2)
    return vec


def upper_tri_vec_to_matrix(vec, n):
    """
    Convert batch of upper triangular vectors to symmetric matrices.

    Args:
        vec: (batch, n*(n+1)/2) upper triangular vectors
        n: matrix size

    Returns:
        matrix: (batch, n, n) symmetric matrices
    """
    batch_size = vec.shape[0]
    device = vec.device
    matrix = torch.zeros(batch_size, n, n, device=device)

    indices = torch.triu_indices(n, n, device=device)
    matrix[:, indices[0], indices[1]] = vec
    matrix[:, indices[1], indices[0]] = vec  # Make symmetric

    return matrix
