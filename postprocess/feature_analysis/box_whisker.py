#!/usr/bin/env python3
"""
Box-and-whisker plots of Hi-C contact strength at loop coordinates,
grouped by feature label (class × cluster_id), one plot per cell-cycle phase.

Loop intensity is computed on quantile-normalized (QN) contact matrices:
  1. KR-balance (or NONE for predicted) matrices per phase.
  2. QN-normalize full chromosome matrices across phases.
  3. Extract a 130 kb × 130 kb window (13 bins at 10 kb) per loop.
  4. Use the mean QN value over that window as the y-axis metric.

Usage
-----
python box_whisker.py <hic_dir> <excel_path> [options]

Example
-------
python box_whisker.py \
    train/full_chr_outputs/chr2-4-17/hic \
    raw_data/zhang_4dn/41586_2019_1778_MOESM5_ESM_split.xlsx \
    --output_dir results/box_whisker
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hicstraw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qn_hic import (
    RESOLUTION_BP,
    load_qn_phase_matrices,
    mean_qn_window_intensity,
    qn_window_ylabel,
)

# Loop-intensity window: 3 bins → 30 kb × 30 kb at 10 kb resolution
WINDOW_PIXELS = 3


# ── phase / file discovery ─────────────────────────────────────────────────────

def discover_chr_hic_dirs(base_dir: str) -> Dict[str, str]:
    """
    Scan *base_dir* for per-chromosome subdirectories structured as
    <base_dir>/chr<N>/hic/ and return a dict mapping the bare chromosome
    name (e.g. '2', 'X') to the absolute path of the ``hic`` subdirectory.

    Expected layout (produced by the full-chromosome training pipeline)::

        base_dir/
          chr1/hic/chr1_prometa_10kb.hic
          chr2/hic/chr2_earlyG1_10kb.hic
          ...
    """
    base = Path(base_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Base directory not found: {base}")
    chr_dirs: Dict[str, str] = {}
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name.startswith("chr"):
            hic_sub = d / "hic"
            if hic_sub.is_dir():
                bare = d.name[3:]  # strip 'chr' prefix
                chr_dirs[bare] = str(hic_sub.resolve())
    if not chr_dirs:
        raise RuntimeError(
            f"No chr*/hic/ subdirectories found in {base}. "
            "Expected layout: <base_dir>/<chrN>/hic/*.hic"
        )
    return chr_dirs


def extract_phases_from_hic_dir(hic_dir: str) -> Dict[str, str]:
    """
    Scan *hic_dir* for *.hic files and return a dict mapping
      phase_name -> absolute file path.

    Expected file-name pattern:  <prefix>_<phase>_<resolution>.hic
    e.g. chr2_earlyG1_10kb.hic  ->  phase = "earlyG1"

    If a file has only two underscore-separated tokens the second is used as
    the phase name; if there is only one token the whole stem is used.
    """
    hic_dir = Path(hic_dir)
    if not hic_dir.is_dir():
        raise FileNotFoundError(f"HiC directory not found: {hic_dir}")

    phase_files: Dict[str, str] = {}
    for f in sorted(hic_dir.glob("*.hic")):
        parts = f.stem.split("_")
        if len(parts) >= 3:
            phase = "_".join(parts[1:-1])   # drop first (chrom) and last (res)
        elif len(parts) == 2:
            phase = parts[1]
        else:
            phase = f.stem
        phase_files[phase] = str(f)

    if not phase_files:
        raise RuntimeError(f"No *.hic files found in {hic_dir}")
    return phase_files


# ── loop feature loading ───────────────────────────────────────────────────────

def load_loop_features(excel_path: str) -> pd.DataFrame:
    """
    Read the loop-feature Excel file and return a DataFrame with columns:
      loop_coordinate_row_mm10, loop_coordinate_col_mm10, class, cluster_id, feature_label

    feature_label is simply "Cluster 1", "Cluster 2", or "Cluster 3".
    Only enhancer-promoter (E/P) loops are kept (class contains "E/P").
    Rows without a cluster assignment (NaN cluster_id, i.e. "others") are dropped.
    """
    df = pd.read_excel(
        excel_path,
        usecols=["loop_coordinate_row_mm10", "loop_coordinate_col_mm10", "class", "cluster_id"],
    )

    # Keep only enhancer-promoter loops
    df = df[df["class"].astype(str).str.contains("E/P", na=False)].copy()

    # Keep only loops that belong to a numbered cluster (drop "others" with NaN cluster_id)
    df = df.dropna(subset=["cluster_id"]).copy()
    df["cluster_id"] = df["cluster_id"].astype(int)
    df["feature_label"] = df["cluster_id"].apply(lambda cid: f"Cluster {cid}")
    return df


def unique_feature_labels(loop_df: pd.DataFrame) -> List[str]:
    """Return sorted list of unique feature labels."""
    return sorted(loop_df["feature_label"].unique())


# ── coordinate parsing ─────────────────────────────────────────────────────────

_ANCHOR_RE = re.compile(r"^(\w+):(\d+)-(\d+)$")

def parse_anchor(coord: str) -> Tuple[str, int, int]:
    """
    Parse a single anchor coordinate 'chr1:74990000-75010000' into
    (chrom, start, end).
    """
    m = _ANCHOR_RE.match(coord.strip())
    if not m:
        raise ValueError(f"Cannot parse anchor coordinate: {coord!r}")
    chrom, start, end = m.groups()
    return chrom, int(start), int(end)


def _anchor_midpoint(coord: str) -> int:
    _, start, end = parse_anchor(coord)
    return (start + end) // 2


def _chrom_bare_from_coord(coord: str) -> str:
    chrom, _, _ = parse_anchor(coord)
    return chrom.lstrip("chr")


def _chroms_in_loops(loops: pd.DataFrame) -> List[str]:
    chroms = {_chrom_bare_from_coord(c) for c in loops["loop_coordinate_row_mm10"]}

    def _sort_key(chrom: str) -> Tuple[int, object]:
        if chrom.isdigit():
            return (0, int(chrom))
        return (1, chrom)

    return sorted(chroms, key=_sort_key)


# ── HiC querying ───────────────────────────────────────────────────────────────

def probe_normalization(
    hic_obj,
    resolution: int,
    preferred: str = "KR",
    fallback: str = "NONE",
) -> str:
    """
    Check whether *preferred* normalization is usable in *hic_obj* by attempting
    a live getRecords call on the first non-'All' chromosome.

    hicstraw does not raise when KR vectors are absent — it only crashes at
    getRecords time (std::bad_alloc).  This probe catches that crash and
    returns *fallback* instead.

    Returns the normalization string that should be used for all subsequent
    queries on this file.
    """
    if preferred == fallback:
        return preferred

    test_chrom = next(
        (c.name for c in hic_obj.getChromosomes() if c.name.lower() not in ("all", "mt", "chrm")),
        None,
    )
    if test_chrom is None:
        return fallback

    try:
        mzd = hic_obj.getMatrixZoomData(
            test_chrom, test_chrom, "observed", preferred, "BP", resolution
        )
        # probe a wide region near the start of the chromosome
        mzd.getRecords(0, resolution * 100, resolution * 100, resolution * 200)
        return preferred
    except Exception:
        return fallback


def query_contact_from_mzd(
    mzd,
    start1: int,
    end1: int,
    start2: int,
    end2: int,
    score: str = "max",
) -> Optional[float]:
    """
    Query a pre-opened MatrixZoomData object for the contact value in the
    window (start1-end1) x (start2-end2).

    Parameters
    ----------
    score : {'max', 'mean'}
        How to summarise multiple bins within the window.

    Returns None if no positive-count records are found.
    """
    try:
        records = mzd.getRecords(start1, end1, start2, end2)
        if not records:
            return None
        values = [r.counts for r in records if r.counts > 0]
        return float(max(values) if score == "max" else np.mean(values)) if values else None
    except Exception:
        return None


def log_normalize_contact(value: float, log_scale: bool) -> float:
    """Apply log1p to a raw Hi-C contact count when *log_scale* is enabled."""
    if not log_scale:
        return value
    return float(np.log1p(value))


def _resolve_active_norm(
    hic_path: str,
    resolution: int,
    preferred: str,
) -> str:
    """Probe *preferred* normalization on a Hi-C file; fall back to NONE."""
    try:
        hic_obj = hicstraw.HiCFile(hic_path)
        return probe_normalization(hic_obj, resolution, preferred, "NONE")
    except Exception:
        return "NONE"


# ── main data-building pipeline ────────────────────────────────────────────────

def build_contact_table(
    hic_dir: str,
    excel_path: str,
    normalization: str = "NONE",
    chrom: Optional[str] = None,
    resolution: int = RESOLUTION_BP,
    window_pixels: int = WINDOW_PIXELS,
) -> pd.DataFrame:
    """
    For every (phase, loop) combination, compute mean QN contact in a loop-centered
    APA window and return a tidy DataFrame with columns:
      phase, loop_coordinate_row_mm10, loop_coordinate_col_mm10,
      class, cluster_id, feature_label, contact_value

    Parameters
    ----------
    chrom : str or None
        If given (e.g. 'chr2'), only loops on that chromosome are processed.
        The 'chr' prefix is normalised automatically.  Pass None for all chromosomes.
    resolution : int
        Hi-C bin size in bp (default: 10 000).
    window_pixels : int
        Side length of the square APA window in bins (default: 13 → 130 kb).
    """
    phase_files = extract_phases_from_hic_dir(hic_dir)
    print(f"Found {len(phase_files)} phase(s): {sorted(phase_files)}")

    loops = load_loop_features(excel_path)

    # filter to requested chromosome — require BOTH anchors on the same chrom
    chrom_norm: Optional[str] = None
    if chrom is not None:
        chrom_norm = chrom if chrom.startswith("chr") else f"chr{chrom}"
        before = len(loops)
        prefix = chrom_norm + ":"
        loops = loops[
            loops["loop_coordinate_row_mm10"].str.startswith(prefix)
            & loops["loop_coordinate_col_mm10"].str.startswith(prefix)
        ].copy()
        print(f"Chromosome filter '{chrom_norm}': {len(loops)} / {before} loops kept")

    labels = unique_feature_labels(loops)
    print(f"Loaded {len(loops)} loops  |  {len(labels)} unique feature labels:")
    for lbl in labels:
        print(f"    {lbl}")

    chrom_bares = (
        [chrom_norm.lstrip("chr")]
        if chrom_norm is not None
        else _chroms_in_loops(loops)
    )

    rows = []
    for chrom_bare in chrom_bares:
        chrom_tag = f"chr{chrom_bare}"
        chrom_prefix = f"{chrom_tag}:"
        chrom_loops = loops[
            loops["loop_coordinate_row_mm10"].str.startswith(chrom_prefix)
            & loops["loop_coordinate_col_mm10"].str.startswith(chrom_prefix)
        ]
        if chrom_loops.empty:
            continue

        print(f"\nChromosome {chrom_tag}: {len(chrom_loops)} loops", flush=True)

        sample_hic = next(iter(phase_files.values()))
        active_norm = _resolve_active_norm(sample_hic, resolution, normalization)
        if active_norm != normalization:
            print(f"  {normalization} not available — using {active_norm}")
        else:
            print(f"  Normalization: {active_norm}")
        print(f"  QN-normalizing matrices across phases …", flush=True)

        try:
            qn_dicts, n_bins, zero_qn = load_qn_phase_matrices(
                phase_files, chrom_bare, resolution, norm=active_norm
            )
        except RuntimeError as exc:
            print(f"  Skipping {chrom_tag}: {exc}")
            continue

        n_ok = n_skip = 0
        for phase, qn_dict in sorted(qn_dicts.items()):
            print(f"  Phase: {phase}")
            for _, row in chrom_loops.iterrows():
                mid1 = _anchor_midpoint(row["loop_coordinate_row_mm10"])
                mid2 = _anchor_midpoint(row["loop_coordinate_col_mm10"])
                val = mean_qn_window_intensity(
                    qn_dict, mid1, mid2, window_pixels, resolution, n_bins, zero_qn
                )
                if val is None:
                    n_skip += 1
                    continue
                n_ok += 1
                rows.append(
                    {
                        "phase": phase,
                        "loop_coordinate_row_mm10": row["loop_coordinate_row_mm10"],
                        "loop_coordinate_col_mm10": row["loop_coordinate_col_mm10"],
                        "class": row["class"],
                        "cluster_id": row["cluster_id"],
                        "feature_label": row["feature_label"],
                        "contact_value": val,
                    }
                )
            print(f"    Queried {n_ok} loop-phase values so far, skipped {n_skip}")

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["contact_value"])
    print(f"\nTotal data points after dropping NaN: {len(df)}")
    return df


# ── plotting ───────────────────────────────────────────────────────────────────

# Preferred phase display order (earlier phases left, later right)
_PHASE_ORDER = ["prometa", "anatelo", "earlyG1", "midG1", "lateG1"]


def _ordered(items: List[str], preferred: List[str]) -> List[str]:
    """Return *items* sorted by position in *preferred*, unknowns appended alphabetically."""
    order = {p: i for i, p in enumerate(preferred)}
    return sorted(items, key=lambda x: (order.get(x, len(preferred)), x))


def _make_boxplot(
    ax,
    data_list: List[np.ndarray],
    labels: List[str],
    colors,
    ylabel: str,
) -> None:
    """Draw a box-and-whisker plot on *ax* (no individual data points)."""
    bp = ax.boxplot(
        data_list,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        showfliers=False,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # sample-size annotation below each box
    for i, values in enumerate(data_list, start=1):
        ax.text(
            i,
            ax.get_ylim()[0],
            f"n={len(values)}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="grey",
        )


def plot_box_whisker(
    contact_df: pd.DataFrame,
    output_dir: str = "box_whisker_plots",
    resolution: int = RESOLUTION_BP,
    window_pixels: int = WINDOW_PIXELS,
    chrom_tag: str = "",
) -> None:
    """
    Produce one box-and-whisker plot per cluster.
    Each plot has one box per cell-cycle phase (x-axis), ordered by cell-cycle
    progression.  Plots are saved as PNG files in *output_dir*.

    Parameters
    ----------
    chrom_tag : str
        Optional tag appended to the output filename and shown in the title
        (e.g. ``'chr2'`` or ``'all_chr'``).  Empty string → no tag.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters = sorted(contact_df["feature_label"].unique())
    phases = _ordered(sorted(contact_df["phase"].unique()), _PHASE_ORDER)
    colors = plt.cm.tab10(np.linspace(0, 1, len(phases)))
    ylabel = qn_window_ylabel(window_pixels, resolution)

    print(f"\nGenerating {len(clusters)} plot(s) in '{output_dir}' ...")

    for cluster in clusters:
        cluster_df = contact_df[contact_df["feature_label"] == cluster]

        data_per_phase = [
            cluster_df.loc[cluster_df["phase"] == ph, "contact_value"].values
            for ph in phases
        ]

        fig, ax = plt.subplots(figsize=(max(6, len(phases) * 1.8), 5))
        _make_boxplot(ax, data_per_phase, phases, colors, ylabel)
        chrom_label = f"  [{chrom_tag}]" if chrom_tag else ""
        ax.set_title(
            f"QN-normalized loop contact across cell-cycle phases\n{cluster}{chrom_label}",
            fontsize=12,
        )

        plt.tight_layout()
        safe_name = cluster.lower().replace(" ", "_")
        tag_suffix = f"_{chrom_tag}" if chrom_tag else ""
        out_path = output_dir / f"box_whisker_{safe_name}{tag_suffix}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Box-and-whisker plots of QN-normalized loop contact (mean over "
            "130 kb APA window), grouped by cluster."
        )
    )
    parser.add_argument(
        "hic_dir",
        help="Directory containing *.hic files (one per phase, named <prefix>_<phase>_<res>.hic).",
    )
    parser.add_argument(
        "excel_path",
        help="Loop-feature Excel file with columns: loop_coordinate_row_mm10, loop_coordinate_col_mm10, class, cluster_id.",
    )
    parser.add_argument(
        "--output_dir",
        default="box_whisker_plots",
        help="Directory for output PNG files (default: box_whisker_plots).",
    )
    parser.add_argument(
        "--normalization",
        default="NONE",
        choices=["KR", "VC", "VC_SQRT", "NONE"],
        help="Preferred Hi-C normalization before QN (default: NONE for predicted). "
             "Automatically falls back to NONE if KR vectors are absent.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=RESOLUTION_BP,
        help="Hi-C resolution in bp (default: 10000).",
    )
    parser.add_argument(
        "--window-pixels",
        type=int,
        default=WINDOW_PIXELS,
        help="Window side length in bins (default: 3 → 30 kb at 10 kb).",
    )
    parser.add_argument(
        "--chrom",
        default="chr2",
        help="Only process loops on this chromosome (default: chr2). "
             "Pass 'all' to include every chromosome from a single hic_dir.",
    )
    parser.add_argument(
        "--all_chr",
        action="store_true",
        help=(
            "Discover all chromosomes under <hic_dir>/chr*/hic/, process each "
            "one individually, pool the contact values, and produce averaged "
            "box-whisker plots.  When this flag is set, <hic_dir> is treated as "
            "the base directory (e.g. train/full_chr_outputs_zhang/final/iw_ssim)."
        ),
    )
    args = parser.parse_args()

    if args.all_chr:
        chr_dirs = discover_chr_hic_dirs(args.hic_dir)

        def _chrom_sort_key(c: str):
            return (0, int(c)) if c.isdigit() else (1, c)

        sorted_chroms = sorted(chr_dirs, key=_chrom_sort_key)
        print(f"Found {len(sorted_chroms)} chromosome(s): {sorted_chroms}")

        all_dfs = []
        for bare_chrom in sorted_chroms:
            chr_hic_dir = chr_dirs[bare_chrom]
            print(f"\n{'='*60}\nProcessing chr{bare_chrom}\n{'='*60}")
            try:
                df = build_contact_table(
                    chr_hic_dir,
                    args.excel_path,
                    args.normalization,
                    chrom=f"chr{bare_chrom}",
                    resolution=args.resolution,
                    window_pixels=args.window_pixels,
                )
                if not df.empty:
                    all_dfs.append(df)
            except (RuntimeError, FileNotFoundError) as exc:
                print(f"Skipping chr{bare_chrom}: {exc}")

        if not all_dfs:
            print("No data collected for any chromosome.")
            return

        contact_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\nTotal data points across {len(all_dfs)} chromosome(s): {len(contact_df)}")
        plot_box_whisker(
            contact_df,
            args.output_dir,
            resolution=args.resolution,
            window_pixels=args.window_pixels,
            chrom_tag="all_chr",
        )
    else:
        chrom_filter = None if args.chrom.lower() == "all" else args.chrom
        contact_df = build_contact_table(
            args.hic_dir,
            args.excel_path,
            args.normalization,
            chrom_filter,
            resolution=args.resolution,
            window_pixels=args.window_pixels,
        )
        plot_box_whisker(
            contact_df,
            args.output_dir,
            resolution=args.resolution,
            window_pixels=args.window_pixels,
        )
    print("\nDone.")


if __name__ == "__main__":
    main()


# /home/lpullela/miniconda3/envs/cell-cycle/bin/python postprocess/feature_analysis/box_whisker.py     train/full_chr_outputs/chr2-4-17/hic     raw_data/zhang_4dn/41586_2019_1778_MOESM5_ESM.xlsx     --output_dir results/box_whisker     --score max