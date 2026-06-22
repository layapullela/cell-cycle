"""
GPU: run diffusion inference on near-diagonal patches only.

Inputs: arrays_dir from extract_chr_numpy.py (raw phase matrices + chromosome
z-scored ChIP tracks).

Bulk-only inference contract
----------------------------
Each tile reads the five raw phase patches exactly once.  Those patches are used ONLY to
build the bulk conditioning map; after the bulk is built the phase-specific maps are never
referenced again.  Concretely, per tile:
  1. norm_ph[p] = normalize(log1p(raw_p))               (per-phase [-1,1], matches training)
  2. bulk       = 0.2 * sum_p norm_ph[p]                (model input — average of normalized phases)
  3. raw_bulk   = 0.2 * sum_p raw_p                     (the bulk, in raw space)
  4. (lo, hi)   = log-space (min, 99.9pct) of raw_bulk  (the bulk's own scaling constants)
The model is conditioned on `bulk` only; phase outputs are rescaled with the bulk's (lo, hi).

Tile merging (non-patchy stitching)
-----------------------------------
The model output for each phase is in [-1,1].  We denormalize it to COUNT space *inside* the
loop using that tile's bulk (lo, hi), then accumulate with a 2-D Tukey weight so overlapping
tiles blend smoothly and each pixel is dominated by tiles centred near it.

Count-space (arithmetic) averaging is used on purpose.  Each diffusion sample is one
stochastic realization; off-diagonal contacts are present in some samples and absent in
others.  Averaging in log space would be a geometric mean in count space and collapses such
sparse-but-present contacts toward zero (off-diagonal looks far too sparse).  The arithmetic
count mean is the posterior mean and reproduces the off-diagonal density of the high-coverage
ground truth much better; the diagonal is consistent across samples and is unaffected.

Outputs (per phase, accumulated over overlapping tiles):
  - chr{chrom}_{phase}_pred_count.npy  float32 (L,L)  Tukey-weighted sum of count-space predictions
  - chr{chrom}_{phase}_pred_wsum.npy   float32 (L,L)  sum of Tukey weights

fill_chr_offdiag_cpu.py forms pred_count / pred_wsum to recover the per-pixel weighted mean.
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
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "preprocess"))
sys.path.insert(0, str(_TRAIN_DIR))

from inference import Inference, region_is_symmetric
from model import SR3UNet, NoiseEmbedding
from prestore_hic import (
    CHROMOSOME_SIZES,
    MIN_START,
    OFFDIAG_NEAR_BAND_BP, # this is imported because we need to match the training distribution
    REGION_SIZE,
    STEP_BP,
)

# Keep this pipeline self-contained (avoid import-path issues under Slurm).
N = 64
RESOLUTION_BP = 10_000

PHASES = ("earlyG1", "midG1", "lateG1", "anatelo", "prometa")
CHIP_MARKS = ("ctcf", "hac", "h3k4me1", "h3k4me3")

# Boundary separating "near" from "far" off-diagonal tiles.
# Tiles with midpoint_gap <= this use DIAG_STEP_NEAR_BP row spacing;
# tiles beyond it use DIAG_STEP_FAR_BP row spacing.
NEAR_FAR_THRESHOLD_BP = 1_000_000


def chrom_bins(chrom: str) -> int:
    return int(math.ceil(CHROMOSOME_SIZES[str(chrom)] / RESOLUTION_BP))


def regions_for_chrom(
    chrom: str,
    diag_step_near_bp: int,
    diag_step_far_bp: int,
    near_far_threshold_bp: int = NEAR_FAR_THRESHOLD_BP,
) -> list[str]:
    """Generate all near-diagonal tile regions for a chromosome.

    Diagonal row positions are sampled at two different densities:
      - diag_step_near_bp: step for tiles whose midpoint_gap ≤ near_far_threshold_bp
        (includes the diagonal tiles themselves, i.e. midpoint_gap = 0)
      - diag_step_far_bp:  step for tiles whose midpoint_gap > near_far_threshold_bp

    midpoint_gap of an off-diagonal tile is exactly k * STEP_BP, where k is the
    column offset index. The column offsets always advance in STEP_BP increments to
    match the training distribution (capped at OFFDIAG_NEAR_BAND_BP).
    """
    size_bp = CHROMOSOME_SIZES[str(chrom)]
    diag_step_near_bp = int(diag_step_near_bp)
    diag_step_far_bp  = int(diag_step_far_bp)
    if diag_step_near_bp <= 0 or diag_step_far_bp <= 0:
        raise ValueError(
            f"diag step sizes must be > 0, got near={diag_step_near_bp} far={diag_step_far_bp}"
        )

    near_diag_pos = list(range(MIN_START, size_bp - REGION_SIZE + 1, diag_step_near_bp))
    far_diag_pos  = list(range(MIN_START, size_bp - REGION_SIZE + 1, diag_step_far_bp))

    # k bounds: midpoint_gap = k * STEP_BP
    k_near_max = near_far_threshold_bp // STEP_BP   # last k that is "near"
    k_far_max  = OFFDIAG_NEAR_BAND_BP   // STEP_BP  # training-distribution cap

    regs: list[str] = []

    # Diagonal tiles (midpoint_gap = 0) and near off-diagonal (gap ≤ threshold)
    # both use the fine near step.
    for s in near_diag_pos:
        regs.append(f"{chrom}:{s}-{s + REGION_SIZE}:{s}-{s + REGION_SIZE}")
    for rs in near_diag_pos:
        for k in range(1, k_near_max + 1):
            cs = rs + k * STEP_BP
            if cs + REGION_SIZE <= size_bp:
                regs.append(f"{chrom}:{rs}-{rs + REGION_SIZE}:{cs}-{cs + REGION_SIZE}")

    # Far off-diagonal (gap > threshold) uses the coarse far step.
    for rs in far_diag_pos:
        for k in range(k_near_max + 1, k_far_max + 1):
            cs = rs + k * STEP_BP
            if cs + REGION_SIZE <= size_bp:
                regs.append(f"{chrom}:{rs}-{rs + REGION_SIZE}:{cs}-{cs + REGION_SIZE}")

    return regs


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


def load_chip_tracks(arrays_dir: Path, chrom: str) -> dict[str, np.ndarray]:
    """Load chromosome z-scored ChIP tracks written by extract_chr_numpy.py."""
    return {
        mark: np.load(arrays_dir / f"chr{chrom}_chip_{mark}.npy", mmap_mode="r")
        for mark in CHIP_MARKS
    }


def chip_patch(track: np.ndarray, i0: int) -> np.ndarray:
    """Slice a 64-bin window from a chromosome z-scored ChIP track."""
    return np.asarray(track[i0:i0 + N], dtype=np.float32)


def normalize_patch(raw: np.ndarray, use_log1p: bool) -> tuple[np.ndarray, float, float]:
    """Normalize a bulk patch: expects per-phase clipping already applied, then summed."""
    x = raw.astype(np.float32, copy=False)
    if use_log1p:
        x = np.log1p(x)  # note that this is undone in denorm_to_raw
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-10:
        return np.zeros_like(x, dtype=np.float32), lo, hi
    norm = ((x - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)
    return norm, lo, hi # norm is in [-1, 1]




def load_checkpoint(path: Path, device: torch.device) -> SR3UNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    noise = NoiseEmbedding(256, max_value=1000)
    model = SR3UNet(n=N, noise_embed_module=noise, base_ch=64).to(device)
    state = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def tukey_1d(m: int, taper_frac: float, min_weight: float) -> np.ndarray:
    """1-D Tukey window scaled to [min_weight, 1] (matches scipy.signal.windows.tukey)."""
    taper_frac = float(taper_frac)
    min_weight = float(min_weight)
    if m <= 1:
        return np.ones(m, dtype=np.float32)
    if taper_frac <= 0.0:
        w = np.ones(m, dtype=np.float32)
    elif taper_frac >= 1.0:
        w = np.hanning(m).astype(np.float32)
    else:
        n = np.arange(m, dtype=np.float64)
        width = int(np.floor(taper_frac * (m - 1) / 2.0))
        w = np.ones(m, dtype=np.float64)
        n1 = n[: width + 1]
        n2 = n[m - width - 1 :]
        w[: width + 1] = 0.5 * (
            1.0 + np.cos(np.pi * (-1.0 + 2.0 * n1 / (taper_frac * (m - 1))))
        )
        w[m - width - 1 :] = 0.5 * (
            1.0 + np.cos(np.pi * (-1.0 + 2.0 * (n2 - (m - 1)) / (taper_frac * (m - 1))))
        )
        w = w.astype(np.float32)
    return (min_weight + (1.0 - min_weight) * w).astype(np.float32)


def tile_weight_2d(
    n: int,
    taper_frac: float,
    min_weight: float,
) -> np.ndarray:
    """2-D Tukey weight for tile merging; outer product of scaled 1-D Tukey windows."""
    w1d = tukey_1d(n, taper_frac=taper_frac, min_weight=min_weight)
    return np.outer(w1d, w1d).astype(np.float32)


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
        "--test_frac",
        type=float,
        default=0.0,
        help=(
            "Testing shortcut: if > 0, only run inference on tiles fully contained in the "
            "LAST test_frac of the chromosome (e.g. 0.1 = final 10%%). The trailing region is "
            "used because it is less likely to contain unmappable regions. Tiles outside this "
            "trailing diagonal block are skipped entirely. Default 0 = whole chromosome."
        ),
    )
    p.add_argument(
        "--diag_step_near_bp",
        type=int,
        default=int(REGION_SIZE),
        help=(
            "Row-position step for tiles within NEAR_FAR_THRESHOLD_BP of the diagonal "
            "(default: REGION_SIZE = 640000, i.e. no overlap). "
            "Decrease for denser coverage near the diagonal (e.g. 80000 = 87.5%% overlap)."
        ),
    )
    p.add_argument(
        "--diag_step_far_bp",
        type=int,
        default=int(REGION_SIZE),
        help=(
            "Row-position step for tiles beyond NEAR_FAR_THRESHOLD_BP from the diagonal "
            "(default: REGION_SIZE = 640000, i.e. no overlap). "
            "Typically coarser than --diag_step_near_bp (e.g. 320000 = 50%% overlap)."
        ),
    )
    p.add_argument(
        "--near_far_threshold_bp",
        type=int,
        default=NEAR_FAR_THRESHOLD_BP,
        help=(
            "Midpoint-gap boundary (bp) separating near from far off-diagonal tiles "
            f"(default: {NEAR_FAR_THRESHOLD_BP} = 1 Mbp)."
        ),
    )
    p.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Index of this shard (0-based). Used to split regions across multiple GPUs.",
    )
    p.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards. Regions are distributed round-robin: shard k processes regs[k::num_shards].",
    )
    p.add_argument(
        "--tukey_min_weight",
        type=float,
        default=0.25,
        help="Minimum tile weight at patch edges when using the Tukey window (default: 0.25).",
    )
    p.add_argument(
        "--tukey_taper_frac",
        type=float,
        default=0.08,
        help=(
            "Fraction of each patch edge tapered by the Tukey window "
            "(default: 0.08 = 8%% cosine taper on each side)."
        ),
    )
    p.add_argument(
        "--no_hanning",
        action="store_true",
        help="Disable tile weighting when merging overlapping tiles (uniform weights).",
    )
    args = p.parse_args()

    chrom = str(args.chrom)
    arrays_dir = Path(args.arrays_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    L = chrom_bins(chrom)
    use_log1p = not args.no_log1p

    # Resolve sharding early so shard_suffix is available for output file names.
    num_shards = max(int(args.num_shards), 1)
    shard_id   = int(args.shard_id)
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id {shard_id} out of range [0, {num_shards})")
    shard_suffix = f"_shard{shard_id}" if num_shards > 1 else ""

    raw_phase = {ph: np.load(arrays_dir / f"chr{chrom}_{ph}_raw.npy", mmap_mode="r") for ph in PHASES}
    chip = load_chip_tracks(arrays_dir, chrom)

    # 2-D Tukey window for weighted accumulation (optional).
    # A scaled Tukey window keeps a broad flat plateau (weight = 1) over most of the tile
    # and only tapers the outer taper_frac fraction on each edge down to min_weight.
    # Every pixel that falls within any tile therefore gets a positive weight (no zeros).
    # Compared with Hanning, the plateau lets nearby tile centres contribute more evenly
    # while still down-weighting far-corner contributions from off-centre tiles.
    # With --no_hanning, use uniform weights (equivalent to a plain sum/mean over tiles).
    if args.no_hanning:
        tile_weight_2d_arr = np.ones((N, N), dtype=np.float32)
        weight_desc = "uniform"
    else:
        tile_weight_2d_arr = tile_weight_2d(
            N,
            taper_frac=args.tukey_taper_frac,
            min_weight=args.tukey_min_weight,
        )
        weight_desc = (
            f"tukey(min_weight={args.tukey_min_weight}, "
            f"taper_frac={args.tukey_taper_frac})"
        )

    # Accumulators — each tile's prediction is denormalized to COUNT space using that tile's
    # own bulk (lo, hi) and accumulated with a Tukey weight.  fill_chr_offdiag_cpu.py forms
    # pred_count / pred_wsum = the Tukey-weighted arithmetic (posterior) mean in count space.
    #   pred_count: Tukey-weighted sum of per-tile count-space predictions
    #   pred_wsum : sum of Tukey weights (float32, not an integer count)
    pred_count = {ph: open_memmap(out_dir / f"chr{chrom}_{ph}_pred_count{shard_suffix}.npy", (L, L), np.float32) for ph in PHASES}
    pred_wsum  = {ph: open_memmap(out_dir / f"chr{chrom}_{ph}_pred_wsum{shard_suffix}.npy",  (L, L), np.float32) for ph in PHASES}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(Path(args.checkpoint), device)
    infer = Inference(model, device, T=1000)

    regs_all = regions_for_chrom(
        chrom,
        diag_step_near_bp=args.diag_step_near_bp,
        diag_step_far_bp=args.diag_step_far_bp,
        near_far_threshold_bp=args.near_far_threshold_bp,
    )  # generate all regions to run (matches training dist)
    regs = [r for r in regs_all if midpoint_gap(*parse_region(r)[1:]) <= args.near_band_bp] # filter to only near-diag tiles

    # Testing shortcut: restrict to tiles fully inside the TRAILING test_frac block of the
    # chromosome so a partial result can be produced quickly.  The block is [L_start, L);
    # a tile at (i0, j0) bins has i0 as its smallest coordinate (j0 >= i0 for near-diagonal
    # tiles), so requiring i0 >= L_start keeps the whole tile within the trailing block.
    test_frac = float(args.test_frac)
    if test_frac > 0.0:
        L_start = int(math.floor((1.0 - test_frac) * L))
        def _tile_in_test_window(region: str) -> bool:
            _, rs, _re, cs, _ce = parse_region(region)
            i0 = rs // RESOLUTION_BP
            return i0 >= L_start
        n_before = len(regs)
        regs = [r for r in regs if _tile_in_test_window(r)]
        print(
            f"[TEST MODE] test_frac={test_frac} -> trailing block [{L_start}, {L}) bins; "
            f"kept {len(regs)}/{n_before} tiles."
        )

    # Contiguous split: shard k owns a spatially-localised block of regions so that
    # each worker covers a contiguous coordinate window of the chromosome.
    chunk = math.ceil(len(regs) / num_shards)
    regs = regs[shard_id * chunk : (shard_id + 1) * chunk]
    print(
        f"Near-diagonal patches (shard {shard_id}/{num_shards}): {len(regs)} "
        f"[near_step={args.diag_step_near_bp} far_step={args.diag_step_far_bp} "
        f"threshold={args.near_far_threshold_bp} tile_weight={weight_desc}]"
    )

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

                bulk     = np.zeros((B, N, N), dtype=np.float32)
                chip_row = np.zeros((B, 4, N), dtype=np.float32)
                chip_col = np.zeros((B, 4, N), dtype=np.float32)
                # Per-tile scalar (lo, hi) from the tile's own bulk — the single scale
                # used to denormalize this tile's prediction to log-count space.
                lo_bulk = np.zeros((B,), dtype=np.float32)
                hi_bulk = np.zeros((B,), dtype=np.float32)

                for bi, region in enumerate(regs_group):
                    _, rs, re, cs, ce = parse_region(region)
                    i0 = rs // RESOLUTION_BP
                    j0 = cs // RESOLUTION_BP
                    i0s[bi] = i0
                    j0s[bi] = j0

                    # Read each phase's raw patch exactly once.  These patches are used
                    # ONLY to build the bulk (input + scaling); after this the phase maps
                    # are never referenced again.
                    raw_patches = {
                        ph: np.asarray(raw_phase[ph][i0:i0 + N, j0:j0 + N], dtype=np.float32)
                        for ph in PHASES
                    }

                    # Model input: clip each phase on raw counts, average, log, normalize once
                    # to [-1, 1] (matches training).  lo_bulk/hi_bulk are the scaling
                    # constants used to map the model's [-1,1] output back to counts.
                    clipped_patches = {}
                    for ph in PHASES:
                        thr = np.percentile(raw_patches[ph], 99.5)
                        clipped_patches[ph] = np.where(
                            raw_patches[ph] > thr, thr, raw_patches[ph]
                        ).astype(np.float32)
                    raw_bulk_patch = 0.2 * sum(clipped_patches[ph] for ph in PHASES)
                    bulk[bi], lo_bulk[bi], hi_bulk[bi] = normalize_patch(raw_bulk_patch, use_log1p)

                    chip_row[bi, 0] = chip_patch(chip["ctcf"],    i0)
                    chip_row[bi, 1] = chip_patch(chip["hac"],     i0)
                    chip_row[bi, 2] = chip_patch(chip["h3k4me1"],  i0)
                    chip_row[bi, 3] = chip_patch(chip["h3k4me3"],  i0)

                    chip_col[bi, 0] = chip_patch(chip["ctcf"],    j0)
                    chip_col[bi, 1] = chip_patch(chip["hac"],     j0)
                    chip_col[bi, 2] = chip_patch(chip["h3k4me1"],  j0)
                    chip_col[bi, 3] = chip_patch(chip["h3k4me3"],  j0)

                bulk_t = torch.from_numpy(bulk).to(device).unsqueeze(1)  # (B,1,N,N)
                row_t = [torch.from_numpy(chip_row[:, k, :]).to(device) for k in range(4)]
                col_t = [torch.from_numpy(chip_col[:, k, :]).to(device) for k in range(4)]

                sampled = infer.sample(
                    bulk_t,
                    row_t[0], row_t[1], row_t[2], row_t[3],
                    col_t[0], col_t[1], col_t[2], col_t[3],
                    enforce_symmetry=enforce_sym,
                ).cpu().numpy().astype(np.float32)  # (B,5,N,N) in [-1,1]

                # Denormalize each tile to COUNT space using that tile's own bulk (lo, hi),
                # then accumulate the Tukey-weighted count.  fill_chr_offdiag_cpu.py forms
                # pred_count / pred_wsum = the Tukey-weighted arithmetic mean of the tile
                # predictions, i.e. the posterior mean in count space.
                #
                # Count-space (arithmetic) averaging is used deliberately instead of
                # log-space averaging: each diffusion sample is one stochastic realization,
                # and off-diagonal contacts appear in some samples but not others.  A log-space
                # mean is a geometric mean in count space, which drives such sparse-but-present
                # contacts toward zero and makes the off-diagonal look far too sparse.  The
                # arithmetic count mean accumulates that probabilistic signal and matches the
                # density of the (high-coverage) ground truth much better.  The diagonal, being
                # consistent across overlapping samples, is unaffected by the choice.
                #
                # The Tukey weight makes each pixel primarily reflect tiles centred near it,
                # suppressing the far-corner contribution of tiles that only barely reach the
                # pixel (the cause of the too-thick diagonal band).
                #
                # For off-diagonal tiles we also mirror into the lower triangle so both sides
                # of the diagonal are diffusion-predicted.  tile_weight_2d_arr is symmetric,
                # prediction needs transposing.
                for pi, ph in enumerate(PHASES):
                    for bi in range(B):
                        i0 = int(i0s[bi])
                        j0 = int(j0s[bi])
                        span = float(hi_bulk[bi] - lo_bulk[bi])
                        if span < 1e-10:
                            log_pred = np.full((N, N), float(lo_bulk[bi]), dtype=np.float32)
                        else:
                            log_pred = ((sampled[bi, pi] + 1.0) * 0.5 * span + lo_bulk[bi]).astype(np.float32)
                        count_pred = np.expm1(log_pred).astype(np.float32) if use_log1p else log_pred

                        pred_count[ph][i0:i0 + N, j0:j0 + N] += count_pred * tile_weight_2d_arr
                        pred_wsum[ph][i0:i0 + N, j0:j0 + N]  += tile_weight_2d_arr

                        if i0 != j0:  # off-diagonal: mirror into lower triangle
                            pred_count[ph][j0:j0 + N, i0:i0 + N] += count_pred.T * tile_weight_2d_arr
                            pred_wsum[ph][j0:j0 + N, i0:i0 + N]  += tile_weight_2d_arr

    for ph in PHASES:
        pred_count[ph].flush()
        pred_wsum[ph].flush()

    print("Done.")


if __name__ == "__main__":
    main()

