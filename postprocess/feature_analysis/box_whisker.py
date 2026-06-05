#!/usr/bin/env python3
"""
Box-and-whisker plots of Hi-C contact strength at loop coordinates,
grouped by feature label (class × cluster_id), one plot per cell-cycle phase.

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


# ── phase / file discovery ─────────────────────────────────────────────────────

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


# ── main data-building pipeline ────────────────────────────────────────────────

def build_contact_table(
    hic_dir: str,
    excel_path: str,
    normalization: str = "NONE",
    score: str = "max",
    chrom: Optional[str] = None,
    log_scale: bool = False,
) -> pd.DataFrame:
    """
    For every (phase, loop) combination, query the Hi-C contact value and
    return a tidy DataFrame with columns:
      phase, loop_coordinate_row_mm10, loop_coordinate_col_mm10,
      class, cluster_id, feature_label, contact_value

    Parameters
    ----------
    chrom : str or None
        If given (e.g. 'chr2'), only loops on that chromosome are processed.
        The 'chr' prefix is normalised automatically.
    log_scale : bool
        If True, store log1p(raw contact count) for each loop query.
    """
    phase_files = extract_phases_from_hic_dir(hic_dir)
    print(f"Found {len(phase_files)} phase(s): {sorted(phase_files)}")

    loops = load_loop_features(excel_path)

    # filter to requested chromosome — require BOTH anchors on the same chrom
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

    rows = []
    for phase, hic_file in sorted(phase_files.items()):
        # resolve to absolute path once so hicstraw never gets a stale relative path
        hic_abs = str(Path(hic_file).resolve())
        print(f"\n  Phase: {phase}")

        # Open the HiC file once and keep it alive for the entire phase so that
        # mzd objects (which hold a reference to the underlying file handle)
        # remain valid when getRecords() is called later.
        try:
            hic_obj = hicstraw.HiCFile(hic_abs)
            resolution = hic_obj.getResolutions()[0]
        except Exception as exc:
            print(f"    Could not open {hic_abs}: {exc} — skipping phase")
            continue

        # KR vectors may be absent even though getMatrixZoomData doesn't raise;
        # probe with a live getRecords call and fall back to NONE if needed.
        active_norm = probe_normalization(hic_obj, resolution, normalization, "NONE")
        if active_norm != normalization:
            print(f"    {normalization} not available — falling back to {active_norm}")
        else:
            print(f"    Normalization: {active_norm}")

        mzd_cache: Dict[Tuple[str, str], Optional[object]] = {}

        def get_mzd(c1: str, c2: str) -> Optional[object]:
            key = (c1, c2)
            if key not in mzd_cache:
                try:
                    mzd_cache[key] = hic_obj.getMatrixZoomData(
                        c1, c2, "observed", active_norm, "BP", resolution
                    )
                except Exception:
                    mzd_cache[key] = None
            return mzd_cache[key]

        n_ok = n_skip = 0
        for _, row in loops.iterrows():
            try:
                chrom1, s1, e1 = parse_anchor(row["loop_coordinate_row_mm10"])
                chrom2, s2, e2 = parse_anchor(row["loop_coordinate_col_mm10"])
            except ValueError:
                n_skip += 1
                continue

            mzd = get_mzd(chrom1, chrom2)
            if mzd is None:
                n_skip += 1
                continue

            val = query_contact_from_mzd(mzd, s1, e1, s2, e2, score)
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
                    "contact_value": log_normalize_contact(val, log_scale),
                }
            )

        print(f"    Queried {n_ok} loops, skipped {n_skip}")

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
    log_scale: bool,
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
    ax.set_ylabel(
        "log1p(Hi-C contact count)" if log_scale else "Hi-C contact value",
        fontsize=11,
    )
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
    log_scale: bool = False,
) -> None:
    """
    Produce one box-and-whisker plot per cluster.
    Each plot has one box per cell-cycle phase (x-axis), ordered by cell-cycle
    progression.  Plots are saved as PNG files in *output_dir*.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters = sorted(contact_df["feature_label"].unique())
    phases = _ordered(sorted(contact_df["phase"].unique()), _PHASE_ORDER)
    colors = plt.cm.tab10(np.linspace(0, 1, len(phases)))

    print(f"\nGenerating {len(clusters)} plot(s) in '{output_dir}' ...")

    for cluster in clusters:
        cluster_df = contact_df[contact_df["feature_label"] == cluster]

        data_per_phase = [
            cluster_df.loc[cluster_df["phase"] == ph, "contact_value"].values
            for ph in phases
        ]

        fig, ax = plt.subplots(figsize=(max(6, len(phases) * 1.8), 5))
        _make_boxplot(ax, data_per_phase, phases, colors, log_scale)
        ax.set_title(
            f"Hi-C contact strength across cell-cycle phases\n{cluster}",
            fontsize=12,
        )

        plt.tight_layout()
        safe_name = cluster.lower().replace(" ", "_")
        out_path = output_dir / f"box_whisker_{safe_name}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Box-and-whisker plots of Hi-C contact values at loop coordinates, "
            "grouped by feature class × cluster_id, one plot per cell-cycle phase."
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
        default="KR",
        choices=["KR", "VC", "VC_SQRT", "NONE"],
        help="Preferred Hi-C normalization (default: KR). "
             "Automatically falls back to NONE if KR vectors are absent.",
    )
    parser.add_argument(
        "--score",
        default="max",
        choices=["max", "mean"],
        help="How to summarise multiple bins within a loop window (default: max).",
    )
    parser.add_argument(
        "--chrom",
        default="chr2",
        help="Only process loops on this chromosome (default: chr2). "
             "Pass 'all' to include every chromosome.",
    )
    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Apply log1p to each contact value read from the Hi-C matrix (linear y-axis).",
    )
    args = parser.parse_args()

    chrom_filter = None if args.chrom.lower() == "all" else args.chrom
    contact_df = build_contact_table(
        args.hic_dir,
        args.excel_path,
        args.normalization,
        args.score,
        chrom_filter,
        log_scale=args.log_scale,
    )
    plot_box_whisker(contact_df, args.output_dir, args.log_scale)
    print("\nDone.")


if __name__ == "__main__":
    main()


# /home/lpullela/miniconda3/envs/cell-cycle/bin/python postprocess/feature_analysis/box_whisker.py     train/full_chr_outputs/chr2-4-17/hic     raw_data/zhang_4dn/41586_2019_1778_MOESM5_ESM.xlsx     --output_dir results/box_whisker     --score max