"""
Export a dense chr contact matrix to a Juicer `pre`-compatible sparse text file.

Default format is Juicer **"short with score"** (9 columns; see Juicer wiki "Pre"):

  <str1> <chr1> <pos1> <frag1> <str2> <chr2> <pos2> <frag2> <score>

We use dummy fragment ids **0 / 1** (required by `pre` when frag is ignored) and strand **0**.

Also supports `--format extra_short` for the older 5-column layout:

  <chr1> <pos1> <chr2> <pos2> <score>

Note: some `juicer_tools.jar` builds reject `extra_short` (they throw "Unexpected column count").
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

# Keep this pipeline self-contained (avoid import-path issues under Slurm).
RESOLUTION_BP = 10_000


def main() -> None:
    p = argparse.ArgumentParser(description="Export dense .npy matrix to sparse upper-tri text for Juicer.")
    p.add_argument("--chrom", default="2")
    p.add_argument("--matrix_npy", required=True)
    p.add_argument("--output_txt", required=True)
    p.add_argument(
        "--format",
        default="short_with_score",
        choices=("short_with_score", "extra_short"),
        help="Juicer `pre` input format (default: short_with_score).",
    )
    p.add_argument("--threshold", type=float, default=0.0, help="Skip |value| <= threshold (default 0).")
    p.add_argument(
        "--round_int",
        action="store_true",
        help="Round scores to int before writing. WARNING: for O/E maps many small values become 0 and vanish from export.",
    )
    p.add_argument(
        "--stats",
        action="store_true",
        help="Print quick diagnostics (nonzero fraction along a mid-chromosome row) and exit without writing.",
    )
    args = p.parse_args()

    chrom = str(args.chrom)
    M = np.load(args.matrix_npy, mmap_mode="r")
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got shape {M.shape}")
    L = int(M.shape[0])

    out = Path(args.output_txt)
    out.parent.mkdir(parents=True, exist_ok=True)

    thr = float(args.threshold)
    chrname = "chr" + chrom

    if args.stats:
        mid = L // 2
        row = np.asarray(M[mid, :], dtype=np.float64)
        j = np.arange(L)
        far = np.abs(j - mid) > 64  # genomic separation > 64 bins along this row
        print(
            f"[stats] {args.matrix_npy} shape={M.shape} dtype={M.dtype}\n"
            f"  mid_row i={mid}: nonzero_frac={np.mean(row != 0):.6f} min={row.min():.6g} max={row.max():.6g} mean={row.mean():.6g}\n"
            f"  same row with |j-i|>64 bins: nonzero_frac={np.mean(row[far] != 0):.6f} mean={row[far].mean():.6g}"
        )
        return

    with open(out, "w") as f:
        for i in range(L):
            row = np.asarray(M[i, i:], dtype=np.float32)
            if thr > 0:
                nz = np.where(np.abs(row) > thr)[0]
            else:
                nz = np.where(row != 0)[0]
            if nz.size == 0:
                continue
            pos1 = i * RESOLUTION_BP
            for k in nz:
                j = i + int(k)
                pos2 = j * RESOLUTION_BP
                v = float(row[k])
                if args.round_int:
                    v = int(math.floor(v + 0.5))
                if args.format == "extra_short":
                    f.write(f"{chrname}\t{pos1}\t{chrname}\t{pos2}\t{v}\n")
                else:
                    # short_with_score: str chr pos frag str chr pos frag score
                    # Dummy frags must differ (Juicer note), even if -f is not used.
                    f.write(f"0\t{chrname}\t{pos1}\t0\t0\t{chrname}\t{pos2}\t1\t{v}\n")

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

