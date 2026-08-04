"""
loop_calling.py — helper functions for k-means procedure downstream.

compute_E(S, valid_bin) implements the Mustache/HICCUPS donut background model:
  - D[d]   : geometric-mean expected contact at genomic distance d (1-D baseline)
  - E[i,j] : locally corrected expected value at pixel (i,j), computed from
              the donut neighbourhood (w=16 bins, inner exclusion p=4 bins).
              Short-distance pixels use the upper-triangle subset of the donut;
              long-distance pixels take the max of the full donut and the
              lower-left quadrant.

compute_E_at_pixels(S, valid_bin, row_idx, col_idx) is a focused variant that
  computes E only at the requested (row_idx[k], col_idx[k]) coordinates.  It
  shares the same D-computation (vectorised) but skips the full n×n scan,
  making it orders of magnitude faster when only a handful of summit pixels
  are needed (e.g., one loop summit per loop per chromosome).
"""

import numpy as np

def compute_E(S: np.ndarray, valid_bin: np.ndarray) -> np.ndarray:
    """
    Compute the expected contact matrix E using the donut/lower-left background model.

    Parameters
    ----------
    S : np.ndarray, shape (n, n)
        Observed KR-balanced contact matrix (upper triangle populated; NaN for missing).
    valid_bin : np.ndarray of bool, shape (n,)
        True for bins with sufficient sequencing coverage.

    Returns
    -------
    E : np.ndarray, shape (n, n)
        Expected contact matrix; np.nan where the estimate cannot be computed.
    """
    P                 = 4      # inner-square half-width to exclude from donut
    W                 = 16     # outer donut half-width
    SHORT_DISTANCE_BINS = 10
    MAX_DISTANCE_BINS   = 1000

    n = S.shape[0]
    E = np.full((n, n), np.nan, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # is_valid_pixel — vectorised where possible, scalar fallback below    #
    # ------------------------------------------------------------------ #
    def is_valid_pixel(a: int, b: int) -> bool:
        if a < 0 or b < 0:
            return False
        if a >= n or b >= n:
            return False
        if b < a:
            return False
        if b - a > MAX_DISTANCE_BINS:
            return False
        if not valid_bin[a] or not valid_bin[b]:
            return False
        v = S[a, b]
        return bool(np.isfinite(v))

    # ------------------------------------------------------------------ #
    # Compute 1-D distance expectation D[d]                               #
    #   D[d] = exp( mean( log(S[a, a+d] + 1) ) ) - 1                     #
    # ------------------------------------------------------------------ #
    D = np.full(MAX_DISTANCE_BINS + 1, np.nan, dtype=np.float64)

    for d in range(MAX_DISTANCE_BINS + 1):
        a_idx = np.arange(n - d)
        b_idx = a_idx + d

        # keep valid bins
        vb = valid_bin[a_idx] & valid_bin[b_idx]
        a_idx, b_idx = a_idx[vb], b_idx[vb]
        if a_idx.size == 0:
            continue

        vals = S[a_idx, b_idx]
        finite = np.isfinite(vals)
        vals = vals[finite]
        if vals.size == 0:
            continue

        D[d] = np.exp(np.mean(np.log(vals + 1.0))) - 1.0

    # ------------------------------------------------------------------ #
    # Precompute donut relative offsets (da, db)                          #
    #   Excluded: row i (da == 0), column j (db == 0),                   #
    #             inner P×P square (|da| <= P AND |db| <= P)             #
    # ------------------------------------------------------------------ #
    da_range = np.arange(-W, W + 1)
    db_range = np.arange(-W, W + 1)
    da_grid, db_grid = np.meshgrid(da_range, db_range, indexing="ij")
    da_flat = da_grid.ravel()
    db_flat = db_grid.ravel()

    exclude = (
        (da_flat == 0)                                      # same row as target
        | (db_flat == 0)                                    # same col as target
        | ((np.abs(da_flat) <= P) & (np.abs(db_flat) <= P))  # inner square
    )
    da_offsets = da_flat[~exclude]  # shape (N_donut,)
    db_offsets = db_flat[~exclude]

    # ------------------------------------------------------------------ #
    # footprint_expected(i, j, a_fp, b_fp)                               #
    #   Returns D[j-i] * (sum_observed / sum_D_expected) over footprint  #
    # ------------------------------------------------------------------ #
    def footprint_expected(i: int, j: int,
                           a_fp: np.ndarray, b_fp: np.ndarray) -> float:
        if a_fp.size == 0:
            return np.nan
        d_target = j - i
        if np.isnan(D[d_target]):
            return np.nan

        numerator   = float(np.sum(S[a_fp, b_fp]))
        dist_fp     = b_fp - a_fp
        denominator = float(np.sum(D[dist_fp]))    # D already NaN-filtered below

        if denominator <= 0.0:
            return np.nan

        correction = numerator / denominator
        return float(D[d_target]) * correction

    # ------------------------------------------------------------------ #
    # Main loop: compute E[i, j] for every valid upper-triangle pixel     #
    # ------------------------------------------------------------------ #
    for i in range(n):
        j_max = min(n - 1, i + MAX_DISTANCE_BINS)

        for j in range(i, j_max + 1):

            if not is_valid_pixel(i, j):
                continue

            target_distance = j - i

            # ---- build the full donut --------------------------------- #
            a_cands = i + da_offsets
            b_cands = j + db_offsets

            # (1) in-bounds and upper-triangle (b >= a)
            mask = (
                (a_cands >= 0) & (a_cands < n)
                & (b_cands >= 0) & (b_cands < n)
                & (b_cands >= a_cands)
            )
            a_cands, b_cands = a_cands[mask], b_cands[mask]

            # (2) distance within range and D[d] defined
            # Clip before indexing D to avoid out-of-bounds when dist_c == MAX_DISTANCE_BINS+1;
            # the in-range guard (dist_c <= MAX_DISTANCE_BINS) removes those entries anyway.
            dist_c = b_cands - a_cands
            dist_safe = np.minimum(dist_c, MAX_DISTANCE_BINS)
            mask = (dist_c <= MAX_DISTANCE_BINS) & np.isfinite(D[dist_safe])
            a_cands, b_cands = a_cands[mask], b_cands[mask]

            # (3) both bins valid
            mask = valid_bin[a_cands] & valid_bin[b_cands]
            a_cands, b_cands = a_cands[mask], b_cands[mask]

            # (4) S value present
            mask = np.isfinite(S[a_cands, b_cands])
            a_donut = a_cands[mask]
            b_donut = b_cands[mask]

            # ---- choose background model ------------------------------ #
            if target_distance <= SHORT_DISTANCE_BINS:
                # Upper-triangle subset: genomic distance >= target distance
                ut_mask = (b_donut - a_donut) >= target_distance
                E[i, j] = footprint_expected(i, j, a_donut[ut_mask], b_donut[ut_mask])

            else:
                # Full donut expected
                E_donut = footprint_expected(i, j, a_donut, b_donut)

                # Lower-left quadrant: a < i AND b < j
                ll_mask = (a_donut < i) & (b_donut < j)
                E_ll    = footprint_expected(i, j, a_donut[ll_mask], b_donut[ll_mask])

                # Take the larger of the two valid estimates
                both_nan = np.isnan(E_donut) and np.isnan(E_ll)
                if both_nan:
                    E[i, j] = np.nan
                elif np.isnan(E_donut):
                    E[i, j] = E_ll
                elif np.isnan(E_ll):
                    E[i, j] = E_donut
                else:
                    E[i, j] = max(E_donut, E_ll)

    return E


def compute_E_at_pixels(
    S: np.ndarray,
    valid_bin: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
) -> np.ndarray:
    """
    Compute the donut/lower-left expected value E[i,j] only at the requested
    pixel coordinates (row_idx[k], col_idx[k]).

    Parameters
    ----------
    S : np.ndarray, shape (n, n)
        Observed KR-balanced contact matrix (NaN for missing bins).
    valid_bin : np.ndarray of bool, shape (n,)
        True for bins with sufficient sequencing coverage.
    row_idx, col_idx : array-like of int
        Pixel coordinates to evaluate (must satisfy row_idx[k] <= col_idx[k]).

    Returns
    -------
    E_vals : np.ndarray, shape (len(row_idx),)
        Expected contact at each requested pixel; np.nan where not computable.
    """
    P                   = 4
    W                   = 16
    SHORT_DISTANCE_BINS = 10
    MAX_DISTANCE_BINS   = 1000

    n       = S.shape[0]
    row_idx = np.asarray(row_idx, dtype=int)
    col_idx = np.asarray(col_idx, dtype=int)
    E_vals  = np.full(len(row_idx), np.nan, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # 1-D distance expectation D[d]  (vectorised — same as compute_E)    #
    # ------------------------------------------------------------------ #
    D = np.full(MAX_DISTANCE_BINS + 1, np.nan, dtype=np.float64)
    for d in range(MAX_DISTANCE_BINS + 1):
        a_idx = np.arange(n - d)
        b_idx = a_idx + d
        vb    = valid_bin[a_idx] & valid_bin[b_idx]
        a_idx, b_idx = a_idx[vb], b_idx[vb]
        if a_idx.size == 0:
            continue
        vals   = S[a_idx, b_idx]
        finite = np.isfinite(vals)
        vals   = vals[finite]
        if vals.size == 0:
            continue
        D[d] = np.exp(np.mean(np.log(vals + 1.0))) - 1.0

    # ------------------------------------------------------------------ #
    # Precompute donut relative offsets                                    #
    # ------------------------------------------------------------------ #
    da_range = np.arange(-W, W + 1)
    db_range = np.arange(-W, W + 1)
    da_grid, db_grid = np.meshgrid(da_range, db_range, indexing="ij")
    da_flat  = da_grid.ravel()
    db_flat  = db_grid.ravel()
    exclude  = (
        (da_flat == 0)
        | (db_flat == 0)
        | ((np.abs(da_flat) <= P) & (np.abs(db_flat) <= P))
    )
    da_offsets = da_flat[~exclude]
    db_offsets = db_flat[~exclude]

    def footprint_expected(i: int, j: int,
                           a_fp: np.ndarray, b_fp: np.ndarray) -> float:
        if a_fp.size == 0:
            return np.nan
        d_target = j - i
        if np.isnan(D[d_target]):
            return np.nan
        numerator   = float(np.sum(S[a_fp, b_fp]))
        dist_fp     = b_fp - a_fp
        denominator = float(np.sum(D[dist_fp]))
        if denominator <= 0.0:
            return np.nan
        return float(D[d_target]) * (numerator / denominator)

    # ------------------------------------------------------------------ #
    # Per-pixel donut computation (only at requested coordinates)         #
    # ------------------------------------------------------------------ #
    for k in range(len(row_idx)):
        i, j = int(row_idx[k]), int(col_idx[k])

        # Enforce upper-triangle convention
        if i > j:
            i, j = j, i

        if not (0 <= i < n and 0 <= j < n):
            continue
        if not (valid_bin[i] and valid_bin[j]):
            continue
        if j - i > MAX_DISTANCE_BINS:
            continue
        if not np.isfinite(S[i, j]):
            continue

        target_distance = j - i

        # Build donut candidates
        a_cands = i + da_offsets
        b_cands = j + db_offsets

        mask = (
            (a_cands >= 0) & (a_cands < n)
            & (b_cands >= 0) & (b_cands < n)
            & (b_cands >= a_cands)
        )
        a_cands, b_cands = a_cands[mask], b_cands[mask]

        dist_c    = b_cands - a_cands
        dist_safe = np.minimum(dist_c, MAX_DISTANCE_BINS)
        mask      = (dist_c <= MAX_DISTANCE_BINS) & np.isfinite(D[dist_safe])
        a_cands, b_cands = a_cands[mask], b_cands[mask]

        mask      = valid_bin[a_cands] & valid_bin[b_cands]
        a_cands, b_cands = a_cands[mask], b_cands[mask]

        mask    = np.isfinite(S[a_cands, b_cands])
        a_donut = a_cands[mask]
        b_donut = b_cands[mask]

        if target_distance <= SHORT_DISTANCE_BINS:
            ut_mask    = (b_donut - a_donut) >= target_distance
            E_vals[k]  = footprint_expected(i, j,
                                            a_donut[ut_mask], b_donut[ut_mask])
        else:
            E_donut = footprint_expected(i, j, a_donut, b_donut)
            ll_mask = (a_donut < i) & (b_donut < j)
            E_ll    = footprint_expected(i, j,
                                         a_donut[ll_mask], b_donut[ll_mask])

            if np.isnan(E_donut) and np.isnan(E_ll):
                E_vals[k] = np.nan
            elif np.isnan(E_donut):
                E_vals[k] = E_ll
            elif np.isnan(E_ll):
                E_vals[k] = E_donut
            else:
                E_vals[k] = max(E_donut, E_ll)

    return E_vals
