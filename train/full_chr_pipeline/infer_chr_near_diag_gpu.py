"""
GPU: run diffusion inference on near-diagonal patches only.

Inputs: arrays_dir from extract_chr_numpy.py (raw phase matrices + chip tracks).
Outputs:
  - chr{chrom}_{phase}_pred_raw.npy   float32 (L,L)  accumulated predicted counts (sum over overlaps)
  - chr{chrom}_{phase}_pred_cnt.npy   int32   (L,L)  overlap counts per entry
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_TRAIN_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _TRAIN_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "preprocess"))
sys.path.insert(0, str(_TRAIN_DIR))

from inference import Inference, region_is_symmetric
from model import SR3UNet, NoiseEmbedding
from prestore_hic import (
    CHROMOSOME_SIZES,
    MIN_START,
    OFFDIAG_NEAR_BAND_BP,
    REGION_SIZE,
    STEP_BP,
)

# Keep this pipeline self-contained (avoid import-path issues under Slurm).
N = 64
RESOLUTION_BP = 10_000

PHASES = ("earlyG1", "midG1", "lateG1", "anatelo")


def chrom_bins(chrom: str) -> int:
    return int(math.ceil(CHROMOSOME_SIZES[str(chrom)] / RESOLUTION_BP))


def regions_for_chrom(chrom: str, diag_step_bp: int) -> list[str]:
    size_bp = CHROMOSOME_SIZES[str(chrom)]
    diag_step_bp = int(diag_step_bp)
    if diag_step_bp <= 0:
        raise ValueError(f"diag_step_bp must be > 0, got {diag_step_bp}")
    diag_pos = list(range(MIN_START, size_bp - REGION_SIZE + 1, diag_step_bp))
    diag = [f"{chrom}:{s}-{s + REGION_SIZE}:{s}-{s + REGION_SIZE}" for s in diag_pos]
    # Generate every off-diagonal tile within OFFDIAG_NEAR_BAND_BP of the diagonal.
    # Column offsets advance in STEP_BP (100 kb) increments so coverage matches the
    # training distribution (midpoint_gap = k * STEP_BP ≤ OFFDIAG_NEAR_BAND_BP).
    off: list[str] = []
    n_offdiag_steps = OFFDIAG_NEAR_BAND_BP // STEP_BP
    for rs in diag_pos:
        for k in range(1, n_offdiag_steps + 1):
            cs = rs + k * STEP_BP
            if cs + REGION_SIZE <= size_bp:
                off.append(f"{chrom}:{rs}-{rs + REGION_SIZE}:{cs}-{cs + REGION_SIZE}")
    return diag + off


def parse_region(region: str) -> tuple[str, int, int, int, int]:
    parts = region.split(":")
    chrom = parts[0]
    rs, re = map(int, parts[1].split("-"))
    if len(parts) == 3:
        cs, ce = map(int, parts[2].split("-"))
    else:
        cs, ce = rs, re
    return chrom, rs, re, cs, ce


def midpoint_gap(rs: int, re: int, cs: int, ce: int) -> float:
    return abs(0.5 * (rs + re) - 0.5 * (cs + ce))


def normalize_patch(raw: np.ndarray, use_log1p: bool) -> tuple[np.ndarray, float, float]:
    x = raw.astype(np.float32, copy=False)
    if use_log1p:
        x = np.log1p(x)
    thr = np.percentile(x, 99.9)
    x = np.where(x > thr, thr, x).astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-10:
        return np.zeros_like(x, dtype=np.float32), lo, hi
    norm = ((x - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)
    return norm, lo, hi


def denorm_to_raw(pred_norm: np.ndarray, lo: float, hi: float, use_log1p: bool) -> np.ndarray:
    if hi - lo < 1e-10:
        pred_log = np.full_like(pred_norm, lo, dtype=np.float32)
    else:
        pred_log = ((pred_norm + 1.0) * 0.5 * (hi - lo) + lo).astype(np.float32)
    return np.expm1(pred_log).astype(np.float32) if use_log1p else pred_log.astype(np.float32)


def denorm_batch_to_raw(
    pred_norm: np.ndarray,  # (B, N, N)
    lo: np.ndarray,         # (B,)
    hi: np.ndarray,         # (B,)
    use_log1p: bool,
) -> np.ndarray:
    """
    Vectorized denormalization back to raw space per sample.
    """
    lo3 = lo.astype(np.float32, copy=False)[:, None, None]
    hi3 = hi.astype(np.float32, copy=False)[:, None, None]
    span = (hi3 - lo3).astype(np.float32)
    pred_log = np.where(
        span < 1e-10,
        lo3,
        ((pred_norm + 1.0) * 0.5 * span + lo3).astype(np.float32),
    ).astype(np.float32)
    return np.expm1(pred_log).astype(np.float32) if use_log1p else pred_log.astype(np.float32)


def load_checkpoint(path: Path, device: torch.device) -> SR3UNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    noise = NoiseEmbedding(256, max_value=1000)
    model = SR3UNet(n=N, noise_embed_module=noise, base_ch=64).to(device)
    state = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def open_memmap(path: Path, shape: tuple[int, ...], dtype) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    mm[:] = 0
    mm.flush()
    return mm


def main() -> None:
    p = argparse.ArgumentParser(description="Near-diagonal diffusion inference (GPU).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--arrays_dir", required=True)
    p.add_argument("--chrom", default="2")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--near_band_bp", type=float, default=float(OFFDIAG_NEAR_BAND_BP))
    p.add_argument("--no_log1p", action="store_true")
    p.add_argument("--batch_size", type=int, default=8, help="Number of patches per diffusion call (default: 8).")
    p.add_argument(
        "--diag_step_bp",
        type=int,
        default=int(REGION_SIZE),
        help="Step size for diagonal tiles in base pairs (default: REGION_SIZE = 640000 for non-overlap).",
    )
    args = p.parse_args()

    chrom = str(args.chrom)
    arrays_dir = Path(args.arrays_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    L = chrom_bins(chrom)
    use_log1p = not args.no_log1p

    raw_phase = {ph: np.load(arrays_dir / f"chr{chrom}_{ph}_raw.npy", mmap_mode="r") for ph in PHASES}
    chip = {
        "ctcf": np.load(arrays_dir / f"chr{chrom}_chip_ctcf.npy", mmap_mode="r"),
        "hac": np.load(arrays_dir / f"chr{chrom}_chip_hac.npy", mmap_mode="r"),
        "h3k4me1": np.load(arrays_dir / f"chr{chrom}_chip_h3k4me1.npy", mmap_mode="r"),
        "h3k4me3": np.load(arrays_dir / f"chr{chrom}_chip_h3k4me3.npy", mmap_mode="r"),
    }

    pred_sum = {ph: open_memmap(out_dir / f"chr{chrom}_{ph}_pred_raw.npy", (L, L), np.float32) for ph in PHASES}
    pred_cnt = {ph: open_memmap(out_dir / f"chr{chrom}_{ph}_pred_cnt.npy", (L, L), np.int32) for ph in PHASES}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(Path(args.checkpoint), device)
    infer = Inference(model, device, T=1000)

    regs_all = regions_for_chrom(chrom, diag_step_bp=args.diag_step_bp)
    regs = [r for r in regs_all if midpoint_gap(*parse_region(r)[1:]) <= args.near_band_bp]
    print(f"Near-diagonal patches: {len(regs)} / {len(regs_all)}")

    with torch.no_grad():
        bs = max(int(args.batch_size), 1)
        for b0 in tqdm(range(0, len(regs), bs), desc="near-diag batches"):
            batch_regs = regs[b0:b0 + bs]

            # Split by symmetry so enforce_symmetry is correct.
            diag_regs = [r for r in batch_regs if region_is_symmetric([r])]
            off_regs = [r for r in batch_regs if not region_is_symmetric([r])]

            for regs_group, enforce_sym in ((diag_regs, True), (off_regs, False)):
                if not regs_group:
                    continue

                B = len(regs_group)
                i0s = np.empty((B,), dtype=np.int64)
                j0s = np.empty((B,), dtype=np.int64)

                bulk = np.zeros((B, N, N), dtype=np.float32)
                chip_row = np.zeros((B, 4, N), dtype=np.float32)
                chip_col = np.zeros((B, 4, N), dtype=np.float32)
                lo = np.zeros((B, 4), dtype=np.float32)
                hi = np.zeros((B, 4), dtype=np.float32)
                lo_bulk = np.zeros((B,), dtype=np.float32)
                hi_bulk = np.zeros((B,), dtype=np.float32)

                for bi, region in enumerate(regs_group):
                    _, rs, re, cs, ce = parse_region(region)
                    i0 = rs // RESOLUTION_BP
                    j0 = cs // RESOLUTION_BP
                    i0s[bi] = i0
                    j0s[bi] = j0

                    norm_ph = {}
                    for pi, ph in enumerate(PHASES):
                        raw_patch = np.asarray(raw_phase[ph][i0:i0 + N, j0:j0 + N], dtype=np.float32)
                        norm, lo_i, hi_i = normalize_patch(raw_patch, use_log1p)
                        norm_ph[ph] = norm
                        lo[bi, pi] = lo_i
                        hi[bi, pi] = hi_i

                    bulk[bi] = 0.25 * (norm_ph["earlyG1"] + norm_ph["midG1"] + norm_ph["lateG1"] + norm_ph["anatelo"])

                    # Bulk lo/hi for denormalization: derived from the raw bulk patch so that
                    # all phase outputs are rescaled on a common bulk-anchored scale, matching
                    # the inference scenario where only bulk data is available.
                    raw_bulk = 0.25 * (
                        np.asarray(raw_phase["earlyG1"][i0:i0 + N, j0:j0 + N], dtype=np.float32)
                        + np.asarray(raw_phase["midG1"][i0:i0 + N, j0:j0 + N], dtype=np.float32)
                        + np.asarray(raw_phase["lateG1"][i0:i0 + N, j0:j0 + N], dtype=np.float32)
                        + np.asarray(raw_phase["anatelo"][i0:i0 + N, j0:j0 + N], dtype=np.float32)
                    )
                    _, lo_bulk[bi], hi_bulk[bi] = normalize_patch(raw_bulk, use_log1p)

                    chip_row[bi, 0] = np.asarray(chip["ctcf"][i0:i0 + N], dtype=np.float32)
                    chip_row[bi, 1] = np.asarray(chip["hac"][i0:i0 + N], dtype=np.float32)
                    chip_row[bi, 2] = np.asarray(chip["h3k4me1"][i0:i0 + N], dtype=np.float32)
                    chip_row[bi, 3] = np.asarray(chip["h3k4me3"][i0:i0 + N], dtype=np.float32)

                    chip_col[bi, 0] = np.asarray(chip["ctcf"][j0:j0 + N], dtype=np.float32)
                    chip_col[bi, 1] = np.asarray(chip["hac"][j0:j0 + N], dtype=np.float32)
                    chip_col[bi, 2] = np.asarray(chip["h3k4me1"][j0:j0 + N], dtype=np.float32)
                    chip_col[bi, 3] = np.asarray(chip["h3k4me3"][j0:j0 + N], dtype=np.float32)

                bulk_t = torch.from_numpy(bulk).to(device).unsqueeze(1)  # (B,1,N,N)
                row_t = [torch.from_numpy(chip_row[:, k, :]).to(device) for k in range(4)]
                col_t = [torch.from_numpy(chip_col[:, k, :]).to(device) for k in range(4)]

                sampled = infer.sample(
                    bulk_t,
                    row_t[0], row_t[1], row_t[2], row_t[3],
                    col_t[0], col_t[1], col_t[2], col_t[3],
                    enforce_symmetry=enforce_sym,
                ).cpu().numpy().astype(np.float32)  # (B,4,N,N) normalized

                for pi, ph in enumerate(PHASES):
                    pred_raw_b = denorm_batch_to_raw(sampled[:, pi, :, :], lo_bulk, hi_bulk, use_log1p)
                    for bi in range(B):
                        i0 = int(i0s[bi])
                        j0 = int(j0s[bi])
                        pred_sum[ph][i0:i0 + N, j0:j0 + N] += pred_raw_b[bi]
                        pred_cnt[ph][i0:i0 + N, j0:j0 + N] += 1

    for ph in PHASES:
        pred_sum[ph].flush()
        pred_cnt[ph].flush()

    print("Done.")


if __name__ == "__main__":
    main()

