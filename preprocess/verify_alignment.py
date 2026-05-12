#!/usr/bin/env python3
"""
Verify HiC and ChIP-seq alignment for a specific genomic region.

Queries one 640kb region through the CellCycleDataLoader and plots:
  - HiC contact matrix (anatelo phase, OE-normalized) with genomic-coordinate tick labels
  - All three chip-seq tracks alongside the matrix at the same genomic scale

The tick labels let you directly compare pixel positions to Juicebox coordinates,
confirming that bin i in the plot = genomic position (region_start + i * 10000).

Also runs a direct-query cross-check: reads the same bins from the raw bigWig
files without going through the dataloader and prints any discrepancy.

Usage:
    python verify_alignment.py
    python verify_alignment.py --region "2:18400000-19040000" --phase anatelo
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from Dataloader import CellCycleDataLoader


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_matrix(upper_tri_vec, n=64):
    """Reconstruct symmetric n×n matrix from upper-triangular vector (row-major)."""
    matrix = np.zeros((n, n), dtype=np.float32)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            matrix[i, j] = upper_tri_vec[idx]
            matrix[j, i] = upper_tri_vec[idx]
            idx += 1
    return matrix


def genomic_ticks(start, n_bins, resolution, n_ticks=8):
    """
    Return (tick_positions, tick_labels) for axis labels in Mb.
    tick_positions: bin indices (0-based)
    tick_labels:    strings like '18.40', '18.48', ...
    """
    step = max(1, n_bins // n_ticks)
    positions = list(range(0, n_bins, step))
    labels = [f"{(start + p * resolution) / 1e6:.2f}" for p in positions]
    return positions, labels


def direct_chip_query(bw_file, chrom_name, start, n_bins, bin_size):
    """
    Query bigWig directly (bypassing the dataloader object) using identical logic
    to CellCycleDataLoader._extract_chipseq_signal.

    Returns log1p(max) signal array of shape (n_bins,).
    """
    signal = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        bin_start = start + i * bin_size
        bin_end   = start + (i + 1) * bin_size
        try:
            result  = bw_file.stats(chrom_name, bin_start, bin_end, type="max", exact=True)
            max_val = result[0] if result[0] is not None else 0.0
        except Exception:
            max_val = 0.0
        #signal[i] = np.log1p(max_val)
        signal[i] = max_val
    return signal


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify HiC + ChIP-seq alignment")
    parser.add_argument(
        "--region",
        default="2:18400000-19040000",
        help="Region to inspect, e.g. '2:18400000-19040000' (default: %(default)s)",
    )
    parser.add_argument(
        "--phase",
        default="anatelo",
        choices=["earlyG1", "midG1", "lateG1", "anatelo"],
        help="HiC phase to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--data_dir",
        default=str(Path(__file__).parent.parent / "raw_data" / "zhang_4dn"),
        help="Path to data dir containing .hic and bigWig files",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "alignment_verification.png"),
        help="Output plot path (default: preprocess/alignment_verification.png)",
    )
    args = parser.parse_args()

    region   = args.region
    phase    = args.phase
    data_dir = Path(args.data_dir)
    output   = Path(args.output)

    # parse region
    chrom, coords = region.split(":")
    start, end = map(int, coords.split("-"))
    n_bins     = (end - start) // 10000   # should be 64
    resolution = 10000
    chrom_name = f"chr{chrom}"

    print(f"Region:     {chrom_name}:{start:,}-{end:,}")
    print(f"Phase:      {phase}")
    print(f"Bins:       {n_bins} × {resolution//1000}kb")

    # ── load dataloader ────────────────────────────────────────────────────
    print(f"\nInitializing CellCycleDataLoader ...")
    dl = CellCycleDataLoader(
        data_dir=data_dir,
        hold_out_chromosome=chrom,   # treat as holdout so region is indexed
        augment=False,
    )

    print(f"\nFetching region '{region}' from dataloader ...")
    sample = dl[region]

    # ── HiC matrix ────────────────────────────────────────────────────────
    hic_vec    = sample[phase]                 # (2080,) upper-tri vector
    hic_matrix = reconstruct_matrix(hic_vec)   # (64, 64)

    # ── chip-seq from dataloader ───────────────────────────────────────────
    chip_ctcf  = sample["chip_seq_ctcf"]   # (64,)
    chip_hac   = sample["chip_seq_hac"]    # (64,)
    chip_rad21 = sample["chip_seq_rad21"]  # (64,)

    # ── direct cross-check (same logic, bypasses dataloader object) ────────
    print("\nCross-check: querying bigWig directly ...")
    cross_checks = {
        "ctcf":  ("ctcf",  chip_ctcf),
        "hac":   ("hac",   chip_hac),
        "rad21": ("rad21", chip_rad21),
    }
    for label, (key, dl_signal) in cross_checks.items():
        bw = dl.chipseq_files.get(key)
        if bw is None:
            print(f"  [{label}] bigWig not available, skipping cross-check")
            continue
        direct = direct_chip_query(bw, chrom_name, start, n_bins, resolution)
        max_diff = np.abs(direct - dl_signal).max()
        print(f"  [{label}] max |direct - dataloader| = {max_diff:.6f}  "
              f"({'OK' if max_diff < 1e-5 else 'MISMATCH!'})")

    # ── genomic tick labels ────────────────────────────────────────────────
    tick_pos, tick_labels = genomic_ticks(start, n_bins, resolution, n_ticks=8)

    # ─────────────────────────────────────────────────────────────────────
    # Juicebox-style layout:
    #
    #   ┌──────────────────────┬──────────┐
    #   │  H3K27ac (horiz.)    │  corner  │  ← chip above heatmap
    #   ├──────────────────────┼──────────┤
    #   │  HiC heatmap         │ H3K27ac  │  ← chip to right (rotated 90°)
    #   │                      │  (vert.) │
    #   └──────────────────────┴──────────┘
    #
    # sharex / sharey locks chip tracks to heatmap axes → pixel-perfect alignment.
    # ─────────────────────────────────────────────────────────────────────
    chip_color    = "#2ca02c"
    bin_positions = np.arange(n_bins)
    # Use 99th-percentile cap so outlier peaks don't compress all other features,
    # matching Juicebox's autoscale behaviour more closely.
    hac_max = float(np.percentile(chip_hac[chip_hac > 0], 99)) if chip_hac.max() > 0 else 1.0

    fig = plt.figure(figsize=(9, 9))
    gs  = fig.add_gridspec(
        2, 2,
        width_ratios=[8, 1.5],
        height_ratios=[1.5, 8],
        hspace=0.02,
        wspace=0.02,
    )

    # Heatmap first so the chip axes can share its limits
    ax_hic        = fig.add_subplot(gs[1, 0])
    ax_chip_top   = fig.add_subplot(gs[0, 0], sharex=ax_hic)  # columns align with heatmap
    ax_chip_right = fig.add_subplot(gs[1, 1], sharey=ax_hic)  # rows align with heatmap
    ax_corner     = fig.add_subplot(gs[0, 1])
    ax_corner.axis("off")

    # ── HiC heatmap ───────────────────────────────────────────────────────
    im = ax_hic.imshow(hic_matrix, cmap="Reds", aspect="auto")

    ax_hic.set_xticks(tick_pos)
    ax_hic.set_xticklabels(tick_labels, fontsize=7, rotation=45)
    ax_hic.set_yticks(tick_pos)
    ax_hic.set_yticklabels(tick_labels, fontsize=7)
    ax_hic.set_xlabel("Genomic position (Mb)", fontsize=9)
    ax_hic.set_ylabel("Genomic position (Mb)", fontsize=9)

    # colorbar in the corner so it doesn't disturb the shared-axis layout
    fig.colorbar(im, ax=ax_corner, fraction=0.8, pad=0.05,
                 label="OE [-1, 1]", orientation="vertical")

    # ── Horizontal chip track above heatmap ───────────────────────────────
    # x shared with heatmap: bin 0 → left, bin 63 → right
    ax_chip_top.fill_between(bin_positions, chip_hac, alpha=0.7, color=chip_color)
    ax_chip_top.plot(bin_positions, chip_hac, color=chip_color, linewidth=0.8)
    ax_chip_top.set_ylim(0, hac_max * 1.15)
    ax_chip_top.set_ylabel("H3K27ac\n(max)", fontsize=8)
    ax_chip_top.tick_params(axis="x", labelbottom=False)
    ax_chip_top.tick_params(axis="y", labelsize=7)
    ax_chip_top.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax_chip_top.set_title(
        f"H3K27ac ChIP-seq  |  HiC {phase}  |  "
        f"{chrom_name}:{start//1000}kb–{end//1000}kb",
        fontsize=9,
    )

    # ── Vertical chip track to the right of heatmap ───────────────────────
    # y shared with heatmap (inverted by imshow: bin 0 at top, bin 63 at bottom)
    # fill_betweenx(y, x1, x2): signal extends to the right
    ax_chip_right.fill_betweenx(bin_positions, 0, chip_hac, alpha=0.7, color=chip_color)
    ax_chip_right.plot(chip_hac, bin_positions, color=chip_color, linewidth=0.8)
    ax_chip_right.set_xlim(0, hac_max * 1.15)
    ax_chip_right.tick_params(axis="y", labelleft=False)
    ax_chip_right.tick_params(axis="x", labelsize=7, rotation=45)
    ax_chip_right.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax_chip_right.set_xlabel("H3K27ac\n(max)", fontsize=8)

    # ── peak annotation: print top-5 bins per track ───────────────────────
    print(f"\nTop-5 bins by max signal (bin → genomic position):")
    for label, sig in [("CTCF", chip_ctcf), ("H3K4me1", chip_hac), ("RAD21", chip_rad21)]:
        top = np.argsort(sig)[::-1][:5]
        coords_str = "  ".join(
            f"bin{b}={start + b * resolution:,}bp (raw_max={sig[b]:.3f})" for b in top
        )
        print(f"  [{label}] {coords_str}")

    plt.suptitle(
        f"Dataloader alignment check — {chrom_name}:{start:,}–{end:,}\n"
        f"Bin i → genomic position {start:,} + i × {resolution:,}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(str(output), dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {output}")
    plt.close()

    dl.close()


if __name__ == "__main__":
    main()
