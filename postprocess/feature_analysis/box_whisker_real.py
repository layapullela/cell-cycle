#!/usr/bin/env python3
"""
Box-and-whisker plots of Hi-C contact strength at loop coordinates using
real (experimental) Hi-C data, grouped by cluster, one plot per cluster.

Loop intensity is computed on quantile-normalized (QN) contact matrices:
  1. KR-balance matrices per phase.
  2. QN-normalize full chromosome matrices across phases.
  3. Extract a 130 kb × 130 kb window (13 bins at 10 kb) per loop.
  4. Use the mean QN value over that window as the y-axis metric.

Differences from box_whisker.py
--------------------------------
* HiC files are named <phase>.hic (no chromosome prefix or resolution token).
* Chromosomes inside the files are stored WITHOUT the 'chr' prefix (e.g. '2').
* KR normalization is used by default.
* Resolution is fixed at 10 000 bp.

Usage
-----
python box_whisker_real.py <hic_dir> <excel_path> --chrom 2 [options]

Example
-------
python box_whisker_real.py \
    raw_data/zhang_4dn \
    raw_data/zhang_4dn/41586_2019_1778_MOESM5_ESM_split.xlsx \
    --chrom 2 \
    --output_dir results/box_whisker_real
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

def extract_phases_from_hic_dir(hic_dir: str) -> Dict[str, str]:
    """
    Scan *hic_dir* for *.hic files named <phase>.hic and return a dict
    mapping phase_name -> absolute file path.
    """
    hic_dir = Path(hic_dir)
    if not hic_dir.is_dir():
        raise FileNotFoundError(f"HiC directory not found: {hic_dir}")

    phase_files: Dict[str, str] = {}
    for f in sorted(hic_dir.glob("*.hic")):
        phase_files[f.stem] = str(f.resolve())

    if not phase_files:
        raise RuntimeError(f"No *.hic files found in {hic_dir}")
    return phase_files


# ── loop feature loading ───────────────────────────────────────────────────────

def load_loop_features(excel_path: str) -> pd.DataFrame:
    """
    Read the loop-feature Excel file and return a DataFrame with columns:
      loop_coordinate_row_mm10, loop_coordinate_col_mm10, class, cluster_id,
      feature_label

    feature_label is "Cluster 1", "Cluster 2", or "Cluster 3".
    Only enhancer-promoter (E/P) loops are kept (class contains "E/P").
    Rows without a cluster assignment (NaN cluster_id) are dropped.
    """
    df = pd.read_excel(
        excel_path,
        usecols=["loop_coordinate_row_mm10", "loop_coordinate_col_mm10", "class", "cluster_id"],
    )

    # Keep only enhancer-promoter loops
    df = df[df["class"].astype(str).str.contains("E/P", na=False)].copy()

    df = df.dropna(subset=["cluster_id"]).copy()
    df["cluster_id"] = df["cluster_id"].astype(int)
    df["feature_label"] = df["cluster_id"].apply(lambda cid: f"Cluster {cid}")
    return df


def unique_feature_labels(loop_df: pd.DataFrame) -> List[str]:
    return sorted(loop_df["feature_label"].unique())


# ── coordinate parsing ─────────────────────────────────────────────────────────

_ANCHOR_RE = re.compile(r"^(chr)?(\w+):(\d+)-(\d+)$")


def parse_anchor(coord: str) -> Tuple[str, int, int]:
    """
    Parse 'chr2:74990000-75010000' or '2:74990000-75010000' into
    (raw_chrom_without_chr_prefix, start, end).

    Returns the chromosome name stripped of any 'chr' prefix so it matches
    the naming convention used in the experimental .hic files.
    """
    m = _ANCHOR_RE.match(coord.strip())
    if not m:
        raise ValueError(f"Cannot parse anchor coordinate: {coord!r}")
    _, chrom, start, end = m.groups()
    return chrom, int(start), int(end)


def _anchor_midpoint(coord: str) -> int:
    _, start, end = parse_anchor(coord)
    return (start + end) // 2


def _chroms_in_loops(loops: pd.DataFrame) -> List[str]:
    chroms = {parse_anchor(c)[0] for c in loops["loop_coordinate_row_mm10"]}

    def _sort_key(chrom: str) -> Tuple[int, object]:
        if chrom.isdigit():
            return (0, int(chrom))
        return (1, chrom)

    return sorted(chroms, key=_sort_key)


# ── main data-building pipeline ────────────────────────────────────────────────

def build_contact_table(
    hic_dir: str,
    excel_path: str,
    chrom: Optional[str] = None,
    normalization: str = "KR",
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
        Chromosome to restrict to (e.g. '2' or 'chr2').  Both anchors must
        be on this chromosome.  Pass None to include all chromosomes.
    resolution : int
        Hi-C bin size in bp (default: 10 000).
    window_pixels : int
        Side length of the square APA window in bins (default: 13 → 130 kb).
    """
    phase_files = extract_phases_from_hic_dir(hic_dir)
    print(f"Found {len(phase_files)} phase(s): {sorted(phase_files)}")

    loops = load_loop_features(excel_path)

    chrom_raw: Optional[str] = None
    if chrom is not None:
        chrom_raw = chrom.lstrip("chr")
        chrom_prefix = f"chr{chrom_raw}:"
        before = len(loops)
        loops = loops[
            loops["loop_coordinate_row_mm10"].str.startswith(chrom_prefix)
            & loops["loop_coordinate_col_mm10"].str.startswith(chrom_prefix)
        ].copy()
        print(f"Chromosome filter 'chr{chrom_raw}': {len(loops)} / {before} loops kept")

    labels = unique_feature_labels(loops)
    print(f"Loaded {len(loops)} loops  |  {len(labels)} unique feature labels:")
    for lbl in labels:
        print(f"    {lbl}")

    chrom_bares = [chrom_raw] if chrom_raw is not None else _chroms_in_loops(loops)

    rows = []
    for chrom_bare in chrom_bares:
        chrom_prefix = f"chr{chrom_bare}:"
        chrom_loops = loops[
            loops["loop_coordinate_row_mm10"].str.startswith(chrom_prefix)
            & loops["loop_coordinate_col_mm10"].str.startswith(chrom_prefix)
        ]
        if chrom_loops.empty:
            continue

        print(f"\nChromosome chr{chrom_bare}: {len(chrom_loops)} loops", flush=True)
        print(f"  Normalization: {normalization}")
        print(f"  QN-normalizing matrices across phases …", flush=True)

        try:
            qn_dicts, n_bins, zero_qn = load_qn_phase_matrices(
                phase_files, chrom_bare, resolution, norm=normalization
            )
        except RuntimeError as exc:
            print(f"  Skipping chr{chrom_bare}: {exc}")
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

_PHASE_ORDER = ["prometa", "anatelo", "earlyG1", "midG1", "lateG1"]


def _ordered(items: List[str], preferred: List[str]) -> List[str]:
    order = {p: i for i, p in enumerate(preferred)}
    return sorted(items, key=lambda x: (order.get(x, len(preferred)), x))


def plot_box_whisker(
    contact_df: pd.DataFrame,
    output_dir: str = "box_whisker_real_plots",
    chrom: Optional[str] = None,
    resolution: int = RESOLUTION_BP,
    window_pixels: int = WINDOW_PIXELS,
) -> None:
    """
    Produce one box-and-whisker plot per cluster.
    X-axis = phases ordered by cell-cycle progression.
    Y-axis = mean QN contact over the APA window.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters = sorted(contact_df["feature_label"].unique())
    phases = _ordered(sorted(contact_df["phase"].unique()), _PHASE_ORDER)
    colors = plt.cm.tab10(np.linspace(0, 1, len(phases)))
    ylabel = qn_window_ylabel(window_pixels, resolution)

    if not chrom:
        chrom_tag = "all"
    elif chrom == "all_chr":
        chrom_tag = "all_chr"
    else:
        chrom_tag = f"chr{chrom.lstrip('chr')}"
    print(f"\nGenerating {len(clusters)} plot(s) in '{output_dir}' ...")

    for cluster in clusters:
        cluster_df = contact_df[contact_df["feature_label"] == cluster]

        data_per_phase = [
            cluster_df.loc[cluster_df["phase"] == ph, "contact_value"].values
            for ph in phases
        ]

        fig, ax = plt.subplots(figsize=(max(6, len(phases) * 1.8), 5))

        bp = ax.boxplot(
            data_per_phase,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.4),
            capprops=dict(linewidth=1.4),
            showfliers=False,
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(phases) + 1))
        ax.set_xticklabels(phases, rotation=25, ha="right", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(
            f"QN-normalized loop contact — {cluster}  ({chrom_tag}, real data)",
            fontsize=12,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for i, values in enumerate(data_per_phase, start=1):
            ax.text(
                i,
                ax.get_ylim()[0],
                f"n={len(values)}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="grey",
            )

        plt.tight_layout()
        safe_name = cluster.lower().replace(" ", "_")
        out_path = output_dir / f"box_whisker_real_{safe_name}_{chrom_tag}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Box-and-whisker plots of QN-normalized real Hi-C loop contact "
            "(mean over 130 kb APA window), grouped by cluster."
        )
    )
    parser.add_argument(
        "hic_dir",
        help="Directory containing <phase>.hic files.",
    )
    parser.add_argument(
        "excel_path",
        help="Loop-feature Excel file with columns: loop_coordinate_row_mm10, "
             "loop_coordinate_col_mm10, class, cluster_id.",
    )
    parser.add_argument(
        "--output_dir",
        default="box_whisker_real_plots",
        help="Directory for output PNG files (default: box_whisker_real_plots).",
    )
    parser.add_argument(
        "--chrom",
        default="2",
        help="Chromosome to process — no 'chr' prefix needed (default: 2). "
             "Pass 'all' for every chromosome in the .hic files.",
    )
    parser.add_argument(
        "--all_chr",
        action="store_true",
        help=(
            "Process all chromosomes present in the .hic files and pool "
            "contact values before plotting.  Equivalent to --chrom all, "
            "but also labels output files with 'all_chr'."
        ),
    )
    parser.add_argument(
        "--normalization",
        default="KR",
        choices=["KR", "VC", "VC_SQRT", "NONE"],
        help="Hi-C normalization before QN (default: KR).",
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
    args = parser.parse_args()

    # --all_chr overrides --chrom: process every chromosome and pool results
    if args.all_chr or args.chrom.lower() == "all":
        chrom_filter = None
    else:
        chrom_filter = args.chrom

    contact_df = build_contact_table(
        args.hic_dir,
        args.excel_path,
        chrom=chrom_filter,
        normalization=args.normalization,
        resolution=args.resolution,
        window_pixels=args.window_pixels,
    )
    # When --all_chr is used, label outputs with 'all_chr' rather than the
    # generic 'all' tag that the existing --chrom all path would produce.
    plot_chrom = "all_chr" if args.all_chr else chrom_filter
    plot_box_whisker(
        contact_df,
        args.output_dir,
        chrom=plot_chrom,
        resolution=args.resolution,
        window_pixels=args.window_pixels,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
