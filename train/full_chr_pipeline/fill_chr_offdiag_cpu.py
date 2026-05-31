"""
CPU: build full L×L final matrices for each phase.

Every bin (i, j) is filled:

  - If near-diagonal diffusion wrote that bin (pred_wsum > 0):
        pred_raw = pred_count / pred_wsum   (Hanning-weighted arithmetic mean in count space)

    Each tile was denormalized to count space with its own bulk (lo, hi) during inference,
    so this is the posterior mean of the overlapping diffusion samples.  Arithmetic
    count-space averaging (rather than a log-space / geometric mean) preserves the sparse,
    probabilistic off-diagonal contacts instead of collapsing them toward zero.

  - Else: fallback = bulk_raw = 0.2 * (early + mid + late + anatelo + prometa).

Reads:
  - raw phase arrays from extract_chr_numpy.py
  - chr{chrom}_{phase}_pred_count.npy  }  from infer_chr_near_diag_gpu.py
  - chr{chrom}_{phase}_pred_wsum.npy   }    (merged by merge_near_diag_shards.py)

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

PHASES = ("earlyG1", "midG1", "lateG1", "anatelo", "prometa")


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
        description="Fill full chromosome matrices (diffusion where available, else bulk fallback)."
    )
    p.add_argument("--chrom", default="2")
    p.add_argument("--arrays_dir", required=True)
    p.add_argument("--near_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--near_band_bp",
        type=float,
        default=0.0,
        help="Deprecated, not used.",
    )
    p.add_argument(
        "--no_log1p",
        action="store_true",
        help="Deprecated/no-op: denormalization (incl. expm1) now happens during inference.",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=512,
        help="Tile size for streaming over L×L (default 512). Lower if RAM is tight.",
    )
    p.add_argument(
        "--test_frac",
        type=float,
        default=0.0,
        help=(
            "Testing shortcut: if > 0, only the TRAILING test_frac block of the chromosome "
            "(rows and cols >= (1-test_frac)*L) is filled; every bin outside that block is set "
            "to 0 so the exported .hic only contains the tested region. Default 0 = whole matrix."
        ),
    )
    args = p.parse_args()

    chrom = str(args.chrom)
    arrays_dir = Path(args.arrays_dir)
    near_dir = Path(args.near_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    L = chrom_bins(chrom)
    chunk = max(int(args.chunk), 32)

    # Testing shortcut: bins before L_start (in either axis) are forced to zero, keeping only
    # the trailing block [L_start, L).
    test_frac = float(args.test_frac)
    L_start = int(math.floor((1.0 - test_frac) * L)) if test_frac > 0.0 else 0
    if test_frac > 0.0:
        print(f"[TEST MODE] test_frac={test_frac} -> zeroing everything outside the trailing [{L_start}, {L}) bins.")

    raw = {ph: np.load(arrays_dir / f"chr{chrom}_{ph}_raw.npy", mmap_mode="r") for ph in PHASES}

    near_count = {ph: np.load(near_dir / f"chr{chrom}_{ph}_pred_count.npy", mmap_mode="r") for ph in PHASES}
    near_wsum  = {ph: np.load(near_dir / f"chr{chrom}_{ph}_pred_wsum.npy",  mmap_mode="r") for ph in PHASES}

    final = {ph: open_memmap(out_dir / f"chr{chrom}_{ph}_final_raw.npy", (L, L), np.float32) for ph in PHASES}

    n_i = (L + chunk - 1) // chunk
    n_j = (L + chunk - 1) // chunk

    for bi in tqdm(range(n_i), desc="rows (chunked)"):
        i0 = bi * chunk
        i1 = min(i0 + chunk, L)
        for bj in range(n_j):
            j0 = bj * chunk
            j1 = min(j0 + chunk, L)

            # In test mode, any chunk entirely outside the trailing [L_start, L) block is zero.
            if test_frac > 0.0 and (i1 <= L_start or j1 <= L_start):
                for ph in PHASES:
                    final[ph][i0:i1, j0:j1] = 0.0
                continue

            bulk_raw = (
                np.asarray(raw["earlyG1"][i0:i1, j0:j1], dtype=np.float32)
                + np.asarray(raw["midG1"][i0:i1, j0:j1], dtype=np.float32)
                + np.asarray(raw["lateG1"][i0:i1, j0:j1], dtype=np.float32)
                + np.asarray(raw["anatelo"][i0:i1, j0:j1], dtype=np.float32)
                + np.asarray(raw["prometa"][i0:i1, j0:j1], dtype=np.float32)
            )
            bulk_raw *= 0.2
            fallback = bulk_raw.astype(np.float32)

            for ph in PHASES:
                wsum = np.asarray(near_wsum[ph][i0:i1, j0:j1], dtype=np.float32)
                mask = wsum > 0.0
                safe_wsum = np.where(mask, wsum, 1.0)

                # Hanning-weighted arithmetic mean in count space (posterior mean of the
                # overlapping diffusion samples).  Predictions were already converted to
                # counts during inference, so no expm1 is applied here.
                pred_raw_diffusion = (
                    np.asarray(near_count[ph][i0:i1, j0:j1], dtype=np.float32) / safe_wsum
                ).astype(np.float32)

                out = fallback.copy()
                out[mask] = pred_raw_diffusion[mask]

                # Zero the portion of a boundary-straddling chunk that lies outside the
                # trailing [L_start, L) block (global row/col < L_start).
                if test_frac > 0.0:
                    if i0 < L_start:
                        out[:min(L_start - i0, i1 - i0), :] = 0.0
                    if j0 < L_start:
                        out[:, :min(L_start - j0, j1 - j0)] = 0.0

                final[ph][i0:i1, j0:j1] = out

    for ph in PHASES:
        final[ph].flush()
    print(f"Done. Wrote full {L}×{L} matrices for {len(PHASES)} phases -> {out_dir}")


if __name__ == "__main__":
    main()
