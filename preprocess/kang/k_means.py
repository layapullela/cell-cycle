"""
k_means.py — K-means clustering of loop strength dynamics across cell-cycle phases.

Algorithm (Kang et al. method):
  1. Load annotated loops (output of loop_classify_kang.R).
  2. For each cell-cycle phase, load the KR-normalised Hi-C matrix and the
     juicer/straw distance-dependent expected vector (OE expected; no donut).
  3. Per loop, compute the average log2(S_ij / E_ij) over the 3×3 bin summit
     neighbourhood at each phase → one scalar strength value per (loop, phase),
     with E_ij = expected[|j-i|] from straw.
  4. Assemble matrix A  : shape (n_loops, n_phases).
  5. Drop loops whose summit center pixel has KR observed ≤0 in any phase
     (avoids log2(S/E) floor artefacts under sparse maps).
  6. Row-normalise → A_hat_ij = (A_ij - mean_i) / std_i.
  7. Run k-means (k=3 by default) on rows of A_hat.
  8. Write cluster assignments TSV and a heatmap PNG sorted by cluster.

Usage:
  python preprocess/kang/k_means.py \\
      --loops  preprocess/kang/loop_annotation/loop_classify_kang_async_wig10000.csv \\
      --hic_dir raw_data/kang \\
      --output_dir preprocess/kang/loop_clusters \\
      [--loop_class structural_loop|regulatory|all] \\
      [--k 3] \\
      [--norm KR]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: F401

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2, whiten  # no sklearn needed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



# ──────────────────────────────────────────────────────────────────────────────
# Constants — must match loop_classify_kang.R
# ──────────────────────────────────────────────────────────────────────────────

RESOLUTION  = 10_000
SUMMIT_RADIUS = 1          # 3×3 neighbourhood = ±1 bin on each anchor
K_DEFAULT   = 3
PHASES: Tuple[str, ...] = ("prometa", "anatelo", "earlyG1", "midG1", "lateG1")

PHASE_HIC_FILES: Dict[str, List[str]] = {
    "prometa": ["prometa.hic"],
    "anatelo": ["anatelo.hic"],
    "earlyG1": ["earlyG1.hic"],
    "midG1":   ["midG12.hic"],
    "lateG1":  ["lateG1.hic"],
}

# Map loop `type` values → broad class labels
REGULATORY_TYPES = {"e_e_loop", "e_p_loop", "p_p_loop"}

CLUSTER_COLORS = ["#E64B35", "#4DBBD5", "#00A087"]  # red, blue, green


# ──────────────────────────────────────────────────────────────────────────────
# Hi-C loading and O/E
# ──────────────────────────────────────────────────────────────────────────────

def _bare_chrom(chrom: str) -> str:
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def load_chrom_matrix(
    hic_files: Sequence[Path],
    chrom: str,
    *,
    norm: str = "KR",
) -> np.ndarray:
    """Sum KR-normalised observed contacts from one or more .hic files for one chromosome."""
    try:
        import hicstraw as straw
    except ImportError:
        raise SystemExit("hicstraw is not installed — run: pip install hic-straw")

    bare = _bare_chrom(chrom)
    # derive chromosome size from the .hic file itself
    first_hic = straw.HiCFile(str(hic_files[0]))
    chrom_sizes = {c.name: c.length for c in first_hic.getChromosomes()}

    # hicstraw may or may not use 'chr' prefix
    chrom_key = bare if bare in chrom_sizes else f"chr{bare}"
    if chrom_key not in chrom_sizes:
        raise KeyError(f"Chromosome '{bare}' not found in {hic_files[0]}.  "
                       f"Available: {list(chrom_sizes)[:10]}")
    chrom_size = chrom_sizes[chrom_key]
    n_bins = (chrom_size + RESOLUTION - 1) // RESOLUTION
    mat = np.zeros((n_bins, n_bins), dtype=np.float64)

    for hic_path in hic_files:
        records = straw.straw(
            "observed", norm, str(hic_path),
            f"{chrom_key}:0:{chrom_size}",
            f"{chrom_key}:0:{chrom_size}",
            "BP", RESOLUTION,
        )
        for rec in records:
            i = rec.binX // RESOLUTION
            j = rec.binY // RESOLUTION
            if 0 <= i < n_bins and 0 <= j < n_bins:
                mat[i, j] += float(rec.counts)
                if i != j:
                    mat[j, i] += float(rec.counts)

    return mat


def load_straw_expected(
    hic_path: Path,
    chrom: str,
    *,
    norm: str = "KR",
) -> np.ndarray:
    """
    Load juicer/straw distance-dependent expected vector D[d] for one chromosome.

    Uses MatrixZoomData with matrix type ``oe`` — that is where .hic files store
    the normalised expected values used by built-in observed/expected.
    """
    try:
        import hicstraw as straw
    except ImportError:
        raise SystemExit("hicstraw is not installed — run: pip install hic-straw")

    hf = straw.HiCFile(str(hic_path))
    chrom_sizes = {c.name: c.length for c in hf.getChromosomes()}
    bare = _bare_chrom(chrom)
    chrom_key = bare if bare in chrom_sizes else f"chr{bare}"
    if chrom_key not in chrom_sizes:
        raise KeyError(f"Chromosome '{bare}' not found in {hic_path}")

    mzd = hf.getMatrixZoomData(chrom_key, chrom_key, "oe", norm, "BP", RESOLUTION)
    expected = np.asarray(mzd.getExpectedValues(), dtype=np.float64)
    if expected.size == 0:
        raise RuntimeError(
            f"No OE expected vector in {hic_path} for {chrom_key} "
            f"@ {RESOLUTION} bp ({norm})."
        )
    return expected


def expected_at_pixels(
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    expected: np.ndarray,
) -> np.ndarray:
    """E_k = expected[|col_k - row_k|] from the straw OE distance vector."""
    row_idx = np.asarray(row_idx, dtype=int)
    col_idx = np.asarray(col_idx, dtype=int)
    d = np.abs(col_idx - row_idx)
    E_vals = np.full(len(d), np.nan, dtype=np.float64)
    ok = d < len(expected)
    if not np.any(ok):
        return E_vals
    e = expected[d[ok]]
    finite = np.isfinite(e) & (e > 0)
    idx = np.flatnonzero(ok)
    E_vals[idx[finite]] = e[finite]
    return E_vals


def valid_bins_from_matrix(mat: np.ndarray) -> np.ndarray:
    """Bins with ≥1 finite, non-zero contact (KR-coverage proxy)."""
    row_has_data = np.any(np.isfinite(mat) & (mat > 0), axis=1)
    col_has_data = np.any(np.isfinite(mat) & (mat > 0), axis=0)
    return row_has_data & col_has_data


def neighbourhood_pixels(
    bin1: int,
    bin2: int,
    n_bins: int,
    *,
    radius: int = SUMMIT_RADIUS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (row_idx, col_idx) for the (2*radius+1)² summit neighbourhood."""
    rows: List[int] = []
    cols: List[int] = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            ii, jj = bin1 + di, bin2 + dj
            if 0 <= ii < n_bins and 0 <= jj < n_bins:
                rows.append(ii)
                cols.append(jj)
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def loop_log2_enrichment_at_pixels(
    S: np.ndarray,
    E_vals: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
) -> float:
    """
    Average log2(S_ij / E_ij) over the precomputed neighbourhood pixels.
    Returns NaN if no valid pixels exist.
    """
    vals: List[float] = []
    for ii, jj, e_val in zip(row_idx, col_idx, E_vals):
        s_val = S[ii, jj]
        if e_val > 0 and np.isfinite(s_val) and np.isfinite(e_val):
            vals.append(float(np.log2(s_val / e_val + 1e-6)))
    return float(np.mean(vals)) if vals else np.nan


def summit_has_zero_observed(
    S: np.ndarray,
    bin1: int,
    bin2: int,
) -> bool:
    """True if the summit center pixel has non-positive / non-finite KR observed."""
    if not (0 <= bin1 < S.shape[0] and 0 <= bin2 < S.shape[1]):
        return True
    s_val = S[bin1, bin2]
    return (not np.isfinite(s_val)) or s_val <= 0


# ──────────────────────────────────────────────────────────────────────────────
# Loop table normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalize_loop_columns(loops: pd.DataFrame) -> pd.DataFrame:
    """
    Accept loop_classify outputs in either legacy or HiCCUPS column naming.

    Legacy (async FitHiC): BIN1_CHR, BIN1_START, BIN2_CHROMOSOME, BIN2_START, ...
    HiCCUPS (merged_loops): chr1, x1, x2, chr2, y1, y2, summit_x1, summit_y1, ...
    """
    df = loops.copy()

    if "BIN1_CHR" not in df.columns and "chr1" in df.columns:
        df = df.rename(
            columns={
                "chr1": "BIN1_CHR",
                "chr2": "BIN2_CHROMOSOME",
                "x2": "BIN1_END",
                "y2": "BIN2_END",
            }
        )
        df["BIN1_START"] = df["summit_x1"] if "summit_x1" in df.columns else df["x1"]
        df["BIN2_START"] = df["summit_y1"] if "summit_y1" in df.columns else df["y1"]

    required = ("BIN1_CHR", "BIN1_START", "BIN2_START")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Loop file is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Building matrix A
# ──────────────────────────────────────────────────────────────────────────────

def build_strength_matrix(
    loops: pd.DataFrame,
    hic_dir: Path,
    *,
    norm: str = "KR",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute A  : shape (n_loops, n_phases).
    A[k, l] = average log2(S_ij / E_ij) for loop k at phase l.

    Also returns zero_obs : bool array (n_loops, n_phases). True when the
    summit center pixel has S <= 0 (or non-finite) at that phase — these
    drive extreme log2(S/E) under sparse maps.
    """
    n_loops  = len(loops)
    n_phases = len(PHASES)
    A = np.full((n_loops, n_phases), np.nan, dtype=np.float64)
    zero_obs = np.ones((n_loops, n_phases), dtype=bool)  # True until proven otherwise

    # Group loops by chromosome so we load each Hi-C matrix once per (chrom, phase)
    chroms = loops["BIN1_CHR"].unique()

    for chrom in chroms:
        chrom_mask = loops["BIN1_CHR"] == chrom
        chrom_loop_idx = np.where(chrom_mask)[0]
        chrom_loops = loops.iloc[chrom_loop_idx]

        for p_idx, phase in enumerate(PHASES):
            print(f"  loading {chrom} × {phase} ...", flush=True)
            hic_paths = [hic_dir / f for f in PHASE_HIC_FILES[phase]]
            missing = [p for p in hic_paths if not p.exists()]
            if missing:
                warnings.warn(f"Missing HiC files for {phase}: {missing}; skipping.")
                continue

            S = load_chrom_matrix(hic_paths, chrom, norm=norm)
            n_bins = S.shape[0]
            # Juicer/straw OE expected vector (distance-dependent D[d]; no donut).
            # When multiple .hic files are summed, use the first file's expected.
            expected = load_straw_expected(hic_paths[0], chrom, norm=norm)

            # Collect every 3×3 summit neighbourhood pixel once, look up
            # E = expected[|j-i|], then average log2(S/E) per loop.
            pix_rows: List[np.ndarray] = []
            pix_cols: List[np.ndarray] = []
            summit_bins: List[Tuple[int, int]] = []
            for _, row in chrom_loops.iterrows():
                bin1 = int(row["BIN1_START"]) // RESOLUTION
                bin2 = int(row["BIN2_START"]) // RESOLUTION
                r_idx, c_idx = neighbourhood_pixels(bin1, bin2, n_bins)
                pix_rows.append(r_idx)
                pix_cols.append(c_idx)
                summit_bins.append((bin1, bin2))

            all_rows = np.concatenate(pix_rows) if pix_rows else np.array([], dtype=int)
            all_cols = np.concatenate(pix_cols) if pix_cols else np.array([], dtype=int)
            E_all = (
                expected_at_pixels(all_rows, all_cols, expected)
                if len(all_rows)
                else np.array([], dtype=np.float64)
            )

            offset = 0
            for row_i, r_idx, c_idx, (bin1, bin2) in zip(
                chrom_loop_idx, pix_rows, pix_cols, summit_bins
            ):
                n_pix = len(r_idx)
                E_vals = E_all[offset: offset + n_pix]
                offset += n_pix
                zero_obs[row_i, p_idx] = summit_has_zero_observed(S, bin1, bin2)
                A[row_i, p_idx] = loop_log2_enrichment_at_pixels(
                    S, E_vals, r_idx, c_idx,
                )

    return A, zero_obs


# ──────────────────────────────────────────────────────────────────────────────
# Row normalisation → A_hat
# ──────────────────────────────────────────────────────────────────────────────

def row_normalise(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    A_hat_il = (A_il - mean_i) / std_i   (row-wise z-score).

    Returns (A_hat, valid_row_mask) where valid rows have finite values at
    every phase and non-zero row standard deviation.
    """
    row_mean = np.nanmean(A, axis=1, keepdims=True)
    row_std  = np.nanstd(A,  axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        A_hat = (A - row_mean) / np.where(row_std > 0, row_std, np.nan)

    valid = np.all(np.isfinite(A_hat), axis=1)
    return A_hat, valid


# ──────────────────────────────────────────────────────────────────────────────
# K-means and output
# ──────────────────────────────────────────────────────────────────────────────

def run_kmeans(A_hat: np.ndarray, k: int, *, seed: int = 42) -> np.ndarray:
    """
    Fit k-means on rows of A_hat; returns cluster labels (0-indexed).
    Runs 20 random restarts and keeps the result with the lowest inertia.
    Uses scipy.cluster.vq.kmeans2 — no sklearn dependency.
    """
    rng = np.random.default_rng(seed)
    best_labels: Optional[np.ndarray] = None
    best_inertia = np.inf

    for _ in range(20):
        # random initialisation by drawing k rows as centroids
        init_idx = rng.choice(len(A_hat), size=k, replace=False)
        try:
            centroids, labels = kmeans2(A_hat, A_hat[init_idx], iter=300, minit="matrix")
        except Exception:
            continue

        # compute inertia (sum of squared distances to assigned centroid)
        inertia = float(np.sum((A_hat - centroids[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    if best_labels is None:
        raise RuntimeError("k-means failed to converge on any restart")
    return best_labels


def plot_heatmap(
    A_hat: np.ndarray,
    labels: np.ndarray,
    *,
    output_path: Path,
    loop_class: str,
    k: int,
) -> None:
    """Heatmap of A_hat rows sorted by cluster, with a cluster colour bar."""
    sort_idx = np.argsort(labels, kind="stable")
    A_sorted = A_hat[sort_idx]
    L_sorted = labels[sort_idx]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(8, max(4, len(A_hat) * 0.07 + 1)),
        gridspec_kw={"width_ratios": [0.05, 1]},
    )

    # cluster colour bar
    cmap_cat = mcolors.ListedColormap(CLUSTER_COLORS[:k])
    axes[0].imshow(
        L_sorted[:, None], aspect="auto",
        cmap=cmap_cat, vmin=0, vmax=k - 1,
    )
    axes[0].set_xticks([])
    axes[0].set_ylabel("Loops")
    axes[0].set_title("Cluster", fontsize=8)

    # A_hat heatmap
    vmax = np.nanpercentile(np.abs(A_hat), 98)
    im = axes[1].imshow(
        A_sorted, aspect="auto",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    axes[1].set_xticks(range(len(PHASES)))
    axes[1].set_xticklabels(PHASES, rotation=40, ha="right", fontsize=8)
    axes[1].set_yticks([])
    axes[1].set_title(
        f"Loop strength dynamics — {loop_class}  (k={k})", fontsize=9,
    )
    plt.colorbar(im, ax=axes[1], label="Row-normalised log₂(S/E)", fraction=0.02)

    # cluster boundary lines
    boundaries = np.where(np.diff(L_sorted))[0] + 0.5
    for b in boundaries:
        axes[0].axhline(b, color="white", lw=0.5)
        axes[1].axhline(b, color="white", lw=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  heatmap → {output_path}")


def cluster_one_class(
    loops: pd.DataFrame,
    hic_dir: Path,
    output_dir: Path,
    loop_class: str,
    *,
    k: int,
    norm: str,
    drop_zero_obs: bool = True,
) -> pd.DataFrame:
    """Run the full pipeline for one loop class; returns a DataFrame with cluster labels."""
    print(f"\n=== {loop_class} : {len(loops)} loops ===")

    A, zero_obs = build_strength_matrix(loops, hic_dir, norm=norm)

    A_hat, valid = row_normalise(A)
    n_dropped_nan = int((~valid).sum())
    if n_dropped_nan:
        print(f"  dropped {n_dropped_nan} loops with missing/constant strength profile")

    if drop_zero_obs:
        # Drop any loop whose summit center pixel has S<=0 in ANY phase.
        has_zero = zero_obs.any(axis=1)
        n_zero = int((valid & has_zero).sum())
        per_phase = {
            ph: int((valid & zero_obs[:, i]).sum())
            for i, ph in enumerate(PHASES)
        }
        print(
            f"  dropped {n_zero} loops with zero KR summit observed in ≥1 phase "
            f"(by phase among previously-valid: {per_phase})"
        )
        valid = valid & ~has_zero

    A_hat_valid = A_hat[valid]
    if len(A_hat_valid) < k:
        print(f"  not enough valid loops ({len(A_hat_valid)}) for k={k}; skipping")
        return pd.DataFrame()

    print(f"  clustering {len(A_hat_valid)} loops")
    labels = run_kmeans(A_hat_valid, k)

    # cluster sizes
    for c in range(k):
        print(f"  cluster {c}: {(labels == c).sum()} loops")

    # assemble results
    result_loops = loops.iloc[np.where(valid)[0]].copy()
    result_loops = result_loops.assign(
        cluster=labels,
        **{f"strength_{ph}": A[valid, p_idx] for p_idx, ph in enumerate(PHASES)},
        **{f"znorm_{ph}": A_hat_valid[:, p_idx] for p_idx, ph in enumerate(PHASES)},
    )

    # save TSV
    tsv_path = output_dir / f"clusters_{loop_class}_k{k}.tsv"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_loops.to_csv(tsv_path, sep="\t", index=False, float_format="%.6f")
    print(f"  assignments → {tsv_path}")

    # heatmap
    png_path = output_dir / f"heatmap_{loop_class}_k{k}.png"
    plot_heatmap(A_hat_valid, labels, output_path=png_path, loop_class=loop_class, k=k)

    return result_loops


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--loops", type=Path, required=True,
                   help="Annotated loop CSV from loop_classify_kang.R "
                        "(e.g. loop_classify_kang_hiccups_wig10000.csv or "
                        "loop_classify_kang_async_wig10000.csv)")
    p.add_argument("--hic_dir", type=Path, default=Path("raw_data/kang"),
                   help="Directory containing per-phase .hic files (default: raw_data/kang)")
    p.add_argument("--output_dir", type=Path, default=Path("preprocess/kang/loop_clusters"),
                   help="Output directory for TSVs and heatmaps")
    p.add_argument("--loop_class", default="all",
                   choices=["structural_loop", "regulatory", "all"],
                   help="Which loop class to cluster (default: all — runs both)")
    p.add_argument("--k", type=int, default=K_DEFAULT,
                   help=f"Number of k-means clusters (default: {K_DEFAULT})")
    p.add_argument("--norm", default="KR",
                   help="Hi-C normalisation for hicstraw (default: KR)")
    p.add_argument(
        "--drop-zero-obs",
        dest="drop_zero_obs",
        action="store_true",
        default=True,
        help="Exclude loops whose summit center pixel has KR observed ≤0 "
             "in any phase (default: on)",
    )
    p.add_argument(
        "--keep-zero-obs",
        dest="drop_zero_obs",
        action="store_false",
        help="Disable the zero-observed filter (previous behaviour)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    loops_path = args.loops.resolve()
    if not loops_path.exists():
        raise SystemExit(f"Loop file not found: {loops_path}")

    loops_all = normalize_loop_columns(pd.read_csv(loops_path))
    print(f"Loaded {len(loops_all)} loops from {loops_path.name}")
    print(f"Type breakdown:\n{loops_all['type'].value_counts().to_string()}\n")
    print(f"drop_zero_obs={args.drop_zero_obs}")

    classes_to_run: List[Tuple[str, pd.DataFrame]] = []

    if args.loop_class in ("structural_loop", "all"):
        df = loops_all[loops_all["type"] == "structural_loop"].copy()
        classes_to_run.append(("structural_loop", df))

    if args.loop_class in ("regulatory", "all"):
        df = loops_all[loops_all["type"].isin(REGULATORY_TYPES)].copy()
        classes_to_run.append(("regulatory", df))

    if not classes_to_run:
        raise SystemExit(f"No loops matched loop_class='{args.loop_class}'")

    for loop_class, df in classes_to_run:
        if df.empty:
            print(f"No loops for class '{loop_class}', skipping.")
            continue
        cluster_one_class(
            df,
            args.hic_dir.resolve(),
            args.output_dir.resolve(),
            loop_class,
            k=args.k,
            norm=args.norm,
            drop_zero_obs=args.drop_zero_obs,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
