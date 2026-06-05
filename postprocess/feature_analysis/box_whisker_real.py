#!/usr/bin/env python3
"""
Box-and-whisker plots of Hi-C contact strength at loop coordinates using
real (experimental) Hi-C data, grouped by cluster, one plot per cluster.

Differences from box_whisker.py
--------------------------------
* HiC files are named <phase>.hic (no chromosome prefix or resolution token).
* Chromosomes inside the files are stored WITHOUT the 'chr' prefix (e.g. '2').
* KR normalization is used by default.
* Resolution is fixed at 10 000 bp.
* Contact values are log1p-transformed before plotting.

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
    --score mean \
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


# ── HiC querying ───────────────────────────────────────────────────────────────

def query_contact_from_mzd(
    mzd,
    start1: int,
    end1: int,
    start2: int,
    end2: int,
    score: str = "max",
) -> Optional[float]:
    """
    Query a pre-opened MatrixZoomData for the contact value in the window
    (start1-end1) × (start2-end2).  Returns None if no positive records found.
    """
    try:
        records = mzd.getRecords(start1, end1, start2, end2)
        if not records:
            return None
        values = [r.counts for r in records if r.counts > 0]
        return float(max(values) if score == "max" else np.mean(values)) if values else None
    except Exception:
        return None


# ── main data-building pipeline ────────────────────────────────────────────────

def build_contact_table(
    hic_dir: str,
    excel_path: str,
    chrom: Optional[str] = None,
    normalization: str = "KR",
    resolution: int = 10_000,
    score: str = "max",
) -> pd.DataFrame:
    """
    For every (phase, loop) combination, query the Hi-C contact value and
    return a tidy DataFrame with columns:
      phase, loop_coordinate_row_mm10, loop_coordinate_col_mm10,
      class, cluster_id, feature_label, contact_value

    contact_value is log1p-transformed.

    Parameters
    ----------
    chrom : str or None
        Chromosome to restrict to (e.g. '2' or 'chr2').  Both anchors must
        be on this chromosome.  Pass None to include all chromosomes.
    """
    phase_files = extract_phases_from_hic_dir(hic_dir)
    print(f"Found {len(phase_files)} phase(s): {sorted(phase_files)}")

    loops = load_loop_features(excel_path)

    # normalise chrom: strip 'chr' to match file convention
    chrom_raw: Optional[str] = None
    if chrom is not None:
        chrom_raw = chrom.lstrip("chr")  # '2', not 'chr2'
        chrom_prefix = f"chr{chrom_raw}:"   # Excel coords use 'chr' prefix
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

    rows = []
    for phase, hic_file in sorted(phase_files.items()):
        print(f"\n  Phase: {phase}")

        try:
            hic_obj = hicstraw.HiCFile(hic_file)
        except Exception as exc:
            print(f"    Could not open {hic_file}: {exc} — skipping")
            continue

        available_chroms = {c.name for c in hic_obj.getChromosomes()}
        mzd_cache: Dict[Tuple[str, str], Optional[object]] = {}

        def get_mzd(c1: str, c2: str) -> Optional[object]:
            key = (c1, c2)
            if key not in mzd_cache:
                if c1 not in available_chroms or c2 not in available_chroms:
                    mzd_cache[key] = None
                else:
                    try:
                        mzd_cache[key] = hic_obj.getMatrixZoomData(
                            c1, c2, "observed", normalization, "BP", resolution
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
            if val is not None:
                val = float(np.log1p(val))   # log1p transform
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

        print(f"    Queried {n_ok} loops, skipped {n_skip}")

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
) -> None:
    """
    Produce one box-and-whisker plot per cluster.
    X-axis = phases ordered by cell-cycle progression.
    Y-axis = log1p(Hi-C contact value, KR-normalised).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters = sorted(contact_df["feature_label"].unique())
    phases = _ordered(sorted(contact_df["phase"].unique()), _PHASE_ORDER)
    colors = plt.cm.tab10(np.linspace(0, 1, len(phases)))

    chrom_tag = f"chr{chrom.lstrip('chr')}" if chrom else "all"
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
        ax.set_ylabel("log1p(Hi-C contact, KR)", fontsize=11)
        ax.set_title(
            f"Hi-C contact strength — {cluster}  ({chrom_tag}, real data)",
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
            "Box-and-whisker plots of real Hi-C contact values at loop "
            "coordinates, log1p-transformed, KR-normalised, grouped by cluster."
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
             "Pass 'all' for every chromosome.",
    )
    parser.add_argument(
        "--normalization",
        default="KR",
        choices=["KR", "VC", "VC_SQRT", "NONE"],
        help="Hi-C normalization to apply (default: KR).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=10_000,
        help="Hi-C resolution in bp (default: 10000).",
    )
    parser.add_argument(
        "--score",
        default="max",
        choices=["max", "mean"],
        help="How to summarise multiple bins within a loop window (default: max).",
    )
    args = parser.parse_args()

    chrom_filter = None if args.chrom.lower() == "all" else args.chrom
    contact_df = build_contact_table(
        args.hic_dir,
        args.excel_path,
        chrom=chrom_filter,
        normalization=args.normalization,
        resolution=args.resolution,
        score=args.score,
    )
    plot_box_whisker(contact_df, args.output_dir, chrom=chrom_filter)
    print("\nDone.")


if __name__ == "__main__":
    main()
