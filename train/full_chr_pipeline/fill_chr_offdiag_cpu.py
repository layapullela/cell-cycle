"""
CPU: build full L×L final matrices for each phase.

Every bin (i, j) is filled:

  - If near-diagonal diffusion wrote that bin (near_cnt > 0), use
        near_sum / near_cnt
  - Else use the cheap approximation (same as before):
        phase_map[i,j] = 0.25 * bulk_raw[i,j]
    where bulk_raw = 0.25 * (early + mid + late + anatelo) in **raw** space.

The previous version only wrote 64×64 windows on a sparse patch grid (diagonal
tiles + sampled off-diagonals from prestore), so almost all off-diagonal bins
stayed zero and disappeared from sparse Juicer export. This version covers the
**entire** chromosome grid.

Reads:
  - raw phase arrays from extract_chr_numpy.py
  - near-diagonal accumulators from infer_chr_near_diag_gpu.py

Writes:
  - chr{chrom}_{phase}_final_raw.npy  float32 (L,L)
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_TRAIN_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _TRAIN_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "preprocess"))

from prestore_hic import CHROMOSOME_SIZES

# Keep this pipeline self-contained (avoid import-path issues under Slurm).
RESOLUTION_BP = 10_000

PHASES = ("earlyG1", "midG1", "lateG1", "anatelo")


def chrom_bins(chrom: str) -> int:
    return int(math.ceil(CHROMOSOME_SIZES[str(chrom)] / RESOLUTION_BP))


def open_memmap(path: Path, shape: tuple[int, ...], dtype) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    mm[:] = 0
    mm.flush()
    return mm


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fill full chromosome matrices (diffusion where available, else 0.25*bulk)."
    )
    p.add_argument("--chrom", default="2")
    p.add_argument("--arrays_dir", required=True)
    p.add_argument("--near_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--near_band_bp",
        type=float,
        default=0.0,
        help="Deprecated (ignored). Kept so sbatch / old CLIs still parse.",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=512,
        help="Tile size for streaming over L×L (default 512). Lower if RAM is tight.",
    )
    args = p.parse_args()

    chrom = str(args.chrom)
    arrays_dir = Path(args.arrays_dir)
    near_dir = Path(args.near_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    L = chrom_bins(chrom)
    chunk = max(int(args.chunk), 32)

    raw = {ph: np.load(arrays_dir / f"chr{chrom}_{ph}_raw.npy", mmap_mode="r") for ph in PHASES}
    near_sum = {ph: np.load(near_dir / f"chr{chrom}_{ph}_pred_raw.npy", mmap_mode="r") for ph in PHASES}
    near_cnt = {ph: np.load(near_dir / f"chr{chrom}_{ph}_pred_cnt.npy", mmap_mode="r") for ph in PHASES}

    final = {ph: open_memmap(out_dir / f"chr{chrom}_{ph}_final_raw.npy", (L, L), np.float32) for ph in PHASES}

    n_i = (L + chunk - 1) // chunk
    n_j = (L + chunk - 1) // chunk
    total_blocks = n_i * n_j

    for bi in tqdm(range(n_i), desc="rows (chunked)"):
        i0 = bi * chunk
        i1 = min(i0 + chunk, L)
        for bj in range(n_j):
            j0 = bj * chunk
            j1 = min(j0 + chunk, L)

            bulk_raw = (
                np.asarray(raw["earlyG1"][i0:i1, j0:j1], dtype=np.float32)
                + np.asarray(raw["midG1"][i0:i1, j0:j1], dtype=np.float32)
                + np.asarray(raw["lateG1"][i0:i1, j0:j1], dtype=np.float32)
                + np.asarray(raw["anatelo"][i0:i1, j0:j1], dtype=np.float32)
            )
            bulk_raw *= 0.25  # mean of four phases
            # Each phase ≈ (1/4) × total = (1/4) × (4 × mean) = mean.
            # The diffusion output is also in the same per-phase raw scale,
            # so using bulk_raw here gives a scale-compatible boundary.
            fallback = bulk_raw.astype(np.float32)

            for ph in PHASES:
                sm = np.asarray(near_sum[ph][i0:i1, j0:j1], dtype=np.float32)
                cnt = np.asarray(near_cnt[ph][i0:i1, j0:j1], dtype=np.float32)
                mask = cnt > 0.0
                out = fallback.copy()
                out[mask] = (sm[mask] / cnt[mask]).astype(np.float32)
                final[ph][i0:i1, j0:j1] = out

    for ph in PHASES:
        final[ph].flush()
    print(f"Done. Wrote full {L}×{L} matrices for {len(PHASES)} phases -> {out_dir}")


if __name__ == "__main__":
    main()
