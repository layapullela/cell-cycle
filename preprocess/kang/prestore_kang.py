"""
prestore_kang.py — precompute cache files for Kang et al. Hi-C data.

Kang et al. arrest-release time course (U2OS human cells, hg38):
  0 min  → prometaphase
  35 min → anatelophase
  60/90 min   → early G1
  120/180 min → mid G1
  240/360 min → late G1

Phase .hic files (merged replicates, 10 kb) are read directly — one file per
phase, same pattern as preprocess/prestore_hic.py:
  prometa.hic, anatelo.hic, earlyG1.hic, midG1.hic, lateG1.hic

Output layout:
  <output_dir>/chr{chrom}/{row_start}-{row_end},{col_start}-{col_end}.npz

Each .npz contains:
  prometa, anatelo, earlyG1, midG1, lateG1       : float32 (N,N) KR-obs counts
  chip_ctcf_row/col, chip_hac_row/col,
  chip_h3k4me1_row/col, chip_h3k4me3_row/col    : float32 (N,) chrom z-scored log1p(sum-max-per-bin)

Keys match Zhang format exactly, so CellCycleDataLoader loads Kang patches
with no special-case logic.

ChIP-seq marks used (bulk = sum over interphase + prometaphase + anatelophase,
two replicates per phase):
  ctcf     : GSM4194671–676  (I/M/AT × r1/r2)
  hac      : H3K27ac proxy: GSM4194659–664
  h3k4me1  : GSM4194665–670
  h3k4me3  : GSM4194653–658

Usage:
  python preprocess/kang/prestore_kang.py \
      --data_dir raw_data/kang \
      --raw_data_dir raw_data/kang \
      --output_dir processed_data/kang \
      [--chrom 1] [--chrom_prefix chr|none] [--no_chr_prefix] [--dry_run]

  sbatch preprocess/kang/prestore_kang_all.sh
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preprocess.chip_signal import (
    apply_chip_zscore,
    compute_chip_chrom_stats,
)

PHASE_HIC_FILES: Dict[str, str] = {
    "earlyG1": "earlyG1.hic",
    "midG1": "midG12.hic", # this is a little less sparse
    "lateG1": "lateG1.hic",
    "anatelo": "anatelo.hic",
    "prometa": "prometa.hic",
}

# ChIP-seq bulk tracks: sum per-bin max across all phases and replicates.
# hac slot uses H3K27ac as a functional proxy.
CHIP_BW_FILES: Dict[str, List[str]] = {
    "ctcf": [
        "GSM4194671_HK81_CTCF_I_S10_both_hg38.bw",
        "GSM4194672_HK148_CTCF_I_S10_both_hg38.bw",
        "GSM4194673_HK82_CTCF_M_S11_both_hg38.bw",
        "GSM4194674_HK149_CTCF_M_S11_both_hg38.bw",
        "GSM4194675_HK83_CTCF_AT_S12_both_hg38.bw",
        "GSM4194676_HK150_CTCF_AT_S12_both_hg38.bw",
    ],
    "hac": [
        "GSM4194659_HK75_H3K27ac_I_S4_both_hg38.bw",
        "GSM4194660_HK158_H3K27ac_I1_S1_both_hg38.bw",
        "GSM4194661_HK76_H3K27ac_M_S5_both_hg38.bw",
        "GSM4194662_HK160_H3K27ac_M1_S3_both_hg38.bw",
        "GSM4194663_HK77_H3K27ac_AT_S6_both_hg38.bw",
        "GSM4194664_HK162_H3K27ac_AT1_S5_both_hg38.bw",
    ],
    "h3k4me1": [
        "GSM4194665_HK78_H3K4me1_I_S7_both_hg38.bw",
        "GSM4194666_HK145_H3K4me1_I_S7_both_hg38.bw",
        "GSM4194667_HK79_H3K4me1_M_S8_both_hg38.bw",
        "GSM4194668_HK146_H3K4me1_M_S8_both_hg38.bw",
        "GSM4194669_HK80_H3K4me1_AT_S9_both_hg38.bw",
        "GSM4194670_HK147_H3K4me1_AT_S9_both_hg38.bw",
    ],
    "h3k4me3": [
        "GSM4194653_HK72_H3K4me3_I_S1_both_hg38.bw",
        "GSM4194654_HK139_H3K4me3_I_S1_both_hg38.bw",
        "GSM4194655_HK73_H3K4me3_M_S2_both_hg38.bw",
        "GSM4194656_HK140_H3K4me3_M_S2_both_hg38.bw",
        "GSM4194657_HK74_H3K4me3_AT_S3_both_hg38.bw",
        "GSM4194658_HK141_H3K4me3_AT_S3_both_hg38.bw",
    ],
}

# hg38 chromosome sizes (from raw_data/kang/hg38.chrom.sizes)
CHROMOSOME_SIZES: Dict[str, int] = {
    "1":  248956422, "2":  242193529, "3":  198295559, "4":  190214555,
    "5":  181538259, "6":  170805979, "7":  159345973, "8":  145138636,
    "9":  138394717, "10": 133797422, "11": 135086622, "12": 133275309,
    "13": 114364328, "14": 107043718, "15": 101991189, "16":  90338345,
    "17":  83257441, "18":  80373285, "19":  58617616, "20":  64444167,
    "21":  46709983, "22":  50818468, "X":  156040895,
}

MIN_START           = 3_000_000
RESOLUTION          = 10_000
REGION_SIZE         = 640_000
STEP_PIXELS         = 10
STEP_BP             = STEP_PIXELS * RESOLUTION
OFFDIAG_NEAR_BAND   = 5_000_000
OFFDIAG_PER_DIAG    = 2
MIN_MEAN_COUNTS_PER_BIN = 0.2

# ---------------------------------------------------------------------------
# Process-level globals (set once in main)
# ---------------------------------------------------------------------------
_hic_paths: Dict[str, Optional[str]] = {}
_chip_paths: Dict[str, List[Optional[str]]] = {}
_bw_handles: Dict[str, List[object]] = {}
_chip_chrom_stats: Dict[Tuple[str, str], Tuple[float, float]] = {}
_resolution: int = RESOLUTION
_image_size: int = REGION_SIZE // RESOLUTION
_hic_data_type: str = "observed"
_normalization: str = "KR"
_chrom_prefix: str = "chr"


def _init(
    hic_paths: Dict[str, Optional[str]],
    chip_paths: Dict[str, List[Optional[str]]],
    hic_data_type: str,
    normalization: str,
    chrom_prefix: str,
    chromosomes: Optional[List[str]] = None,
) -> None:
    global _hic_paths, _chip_paths, _bw_handles, _chip_chrom_stats
    global _hic_data_type, _normalization, _chrom_prefix
    _hic_paths = hic_paths
    _chip_paths = chip_paths
    _hic_data_type = hic_data_type
    _normalization = normalization
    _chrom_prefix = chrom_prefix

    import pyBigWig
    _bw_handles = {}
    for mark, paths in chip_paths.items():
        handles = []
        for p in paths:
            if p is None:
                handles.append(None)
            else:
                handles.append(pyBigWig.open(p))
        _bw_handles[mark] = handles

    chroms = chromosomes if chromosomes is not None else list(CHROMOSOME_SIZES.keys())
    _chip_chrom_stats = {}
    print("\nChIP-seq chromosome stats (log1p, chromosome-wide z-score):")
    for chrom in chroms:
        for mark, bw_list in _bw_handles.items():
            mean, std = compute_chip_chrom_stats(
                bw_list, chrom, CHROMOSOME_SIZES, RESOLUTION, chrom_prefix=chrom_prefix or "chr",
            )
            _chip_chrom_stats[(mark, chrom)] = (mean, std)
            print(f"  {mark} chr{chrom}: mean={mean:.4f} std={std:.4f}")


def _parse_region(region: str) -> Tuple[str, int, int, int, int]:
    parts = region.split(":")
    chrom = parts[0]
    rs, re = map(int, parts[1].split("-"))
    cs, ce = map(int, parts[2].split("-")) if len(parts) == 3 else (rs, re)
    return chrom, rs, re, cs, ce


def _extract_matrix(hic_file: str, region: str) -> np.ndarray:
    """Extract one (N,N) contact matrix from a single phase .hic file."""
    import hicstraw as straw

    chrom, rs, re, cs, ce = _parse_region(region)
    qchrom = _chrom_prefix + chrom
    try:
        result = straw.straw(
            _hic_data_type, _normalization, hic_file,
            f"{qchrom}:{rs}:{re}", f"{qchrom}:{cs}:{ce}", "BP", _resolution,
        )
    except Exception as exc:
        alt_chrom = chrom if _chrom_prefix else ("chr" + chrom)
        try:
            result = straw.straw(
                _hic_data_type, _normalization, hic_file,
                f"{alt_chrom}:{rs}:{re}", f"{alt_chrom}:{cs}:{ce}", "BP", _resolution,
            )
        except Exception:
            raise RuntimeError(
                f"hicstraw failed for {hic_file} chrom={qchrom} (also tried {alt_chrom}): {exc}"
            ) from exc

    mat = np.zeros((_image_size, _image_size), dtype=np.float32)
    for rec in result:
        val = float(rec.counts)
        xi = int((rec.binX - rs) // _resolution)
        yj = int((rec.binY - cs) // _resolution)
        if 0 <= xi < _image_size and 0 <= yj < _image_size:
            mat[xi, yj] = val
        xi2 = int((rec.binY - rs) // _resolution)
        yj2 = int((rec.binX - cs) // _resolution)
        if 0 <= xi2 < _image_size and 0 <= yj2 < _image_size and (xi2, yj2) != (xi, yj):
            mat[xi2, yj2] = val
    return mat


def _mean_counts_per_bin(mat: np.ndarray) -> float:
    """Mean observed KR counts per 10 kb bin (same metric as visualize_npz.py)."""
    n = mat.size
    if n == 0:
        return 0.0
    return float(mat.sum()) / n


def _extract_chip_1d(
    chrom: str, start: int, end: int, mark: str, bw_list: List[object],
) -> np.ndarray:
    """Extract chromosome z-scored log1p(sum-max) bigWig signal per bin."""
    chrom_name = "chr" + chrom
    accum = np.zeros(_image_size, dtype=np.float64)
    for bw in bw_list:
        if bw is None:
            continue
        for i in range(_image_size):
            b0 = start + i * _resolution
            b1 = start + (i + 1) * _resolution
            vals = bw.stats(chrom_name, b0, b1, type="max")
            accum[i] += vals[0] if vals and vals[0] is not None else 0.0
    mean, std = _chip_chrom_stats[(mark, chrom)]
    return apply_chip_zscore(np.log1p(accum), mean, std)


def _process_region(args: Tuple[str, Path]) -> Optional[str]:
    """Compute and write one .npz file.

    Returns region string on success, None if skipped (already exists or low signal).
    """
    region, output_dir = args
    chrom, rs, re, cs, ce = _parse_region(region)
    is_diagonal = (rs == cs)

    chrom_dir = output_dir / f"chr{chrom}"
    out_path = chrom_dir / f"{rs}-{re},{cs}-{ce}.npz"
    if out_path.exists():
        return None

    arrays: Dict[str, np.ndarray] = {}
    for phase in ("earlyG1", "midG1", "lateG1", "anatelo", "prometa"):
        hic_file = _hic_paths.get(phase)
        if hic_file is None:
            arrays[phase] = np.zeros((_image_size, _image_size), dtype=np.float32)
        else:
            arrays[phase] = _extract_matrix(hic_file, region)

    for phase in ("earlyG1", "midG1", "lateG1", "anatelo", "prometa"):
        if _mean_counts_per_bin(arrays[phase]) < MIN_MEAN_COUNTS_PER_BIN:
            return None

    for mark, bw_list in _bw_handles.items():
        row_sig = _extract_chip_1d(chrom, rs, re, mark, bw_list)
        col_sig = row_sig.copy() if is_diagonal else _extract_chip_1d(chrom, cs, ce, mark, bw_list)
        arrays[f"chip_{mark}_row"] = row_sig
        arrays[f"chip_{mark}_col"] = col_sig

    chrom_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(out_path)
    return region


# ---------------------------------------------------------------------------
# Region generation (same sliding-window logic as prestore_hic.py)
# ---------------------------------------------------------------------------
def _sample_offdiag(chrom: str, diag_positions: List[int], rng: np.random.Generator) -> List[str]:
    pos = np.asarray(diag_positions, dtype=np.int64)
    n = pos.size
    if n < 2:
        return []
    near_max_steps = min(n - 1, OFFDIAG_NEAR_BAND // STEP_BP)
    near_steps = np.arange(1, near_max_steps + 1, dtype=np.int64)
    if near_steps.size == 0:
        return []
    split = near_max_steps // 4
    if split <= 0 or split >= near_max_steps:
        weights = np.full(near_steps.size, 1.0 / near_steps.size)
    else:
        weights = np.zeros(near_steps.size)
        low = near_steps <= split
        weights[low] = 0.70 / low.sum()
        weights[~low] = 0.30 / (~low).sum()
    regions: List[str] = []
    for ri in range(n - 1):
        k = min(OFFDIAG_PER_DIAG, near_steps.size)
        for ds in rng.choice(near_steps, size=k, replace=False, p=weights):
            ci = ri + int(ds)
            if ci >= n:
                continue
            regions.append(f"{chrom}:{pos[ri]}-{pos[ri]+REGION_SIZE}:{pos[ci]}-{pos[ci]+REGION_SIZE}")
    return regions


def generate_all_regions(chromosomes: Optional[List[str]] = None) -> List[str]:
    rng = np.random.default_rng(42)
    regions: List[str] = []
    chrom_items = CHROMOSOME_SIZES.items()
    if chromosomes is not None:
        chrom_items = [(c, CHROMOSOME_SIZES[c]) for c in chromosomes]
    for chrom, size in chrom_items:
        diag_pos = list(range(MIN_START, size - REGION_SIZE + 1, STEP_BP))
        regions.extend(f"{chrom}:{s}-{s+REGION_SIZE}:{s}-{s+REGION_SIZE}" for s in diag_pos)
        regions.extend(_sample_offdiag(chrom, diag_pos, rng))
    return regions


def _parse_chrom_prefix(value: str) -> str:
    """Map CLI value to hicstraw chrom name prefix. Use 'none' for bare names (e.g. '14')."""
    if value.lower() in ("", "none", "no", "false", "0"):
        return ""
    return value


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-store Kang et al. Hi-C data to .npz cache")
    default_kang = _REPO_ROOT / "raw_data" / "kang"
    default_hic = default_kang
    default_out = _REPO_ROOT / "processed_data" / "kang"
    parser.add_argument(
        "--data_dir",
        default=str(default_hic),
        help="Directory containing merged phase *.hic files (default: raw_data/kang)",
    )
    parser.add_argument(
        "--raw_data_dir",
        default=str(default_kang),
        help="Directory containing Kang ChIP-seq bigWig files (default: raw_data/kang)",
    )
    parser.add_argument("--output_dir", default=str(default_out))
    parser.add_argument("--hic_type", default="observed", help="hicstraw data type")
    parser.add_argument(
        "--norm",
        default="KR",
        help="Hi-C normalization (default: KR)",
    )
    parser.add_argument(
        "--chrom_prefix",
        type=_parse_chrom_prefix,
        default="chr",
        metavar="PREFIX",
        help=(
            "Prefix prepended to chrom names when querying .hic (default: 'chr'). "
            "Pass 'none' for bare names (e.g. '14' not 'chr14'). "
            "Shell empty string is awkward; use --no_chr_prefix instead."
        ),
    )
    parser.add_argument(
        "--no_chr_prefix",
        action="store_true",
        help="Query .hic with bare chrom names (same as --chrom_prefix none).",
    )
    parser.add_argument(
        "--chrom",
        default=None,
        help="Comma-separated chromosomes to process (e.g. '1' or '1,2'). Default: all.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    chrom_prefix = "" if args.no_chr_prefix else args.chrom_prefix

    data_dir = Path(args.data_dir)
    raw_dir = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hic_paths: Dict[str, Optional[str]] = {}
    print("Hi-C files:")
    for phase, fname in PHASE_HIC_FILES.items():
        p = data_dir / fname
        hic_paths[phase] = str(p) if p.exists() else None
        print(f"  {'✓' if p.exists() else '✗'} {phase}: {p}")

    chip_paths: Dict[str, List[Optional[str]]] = {}
    print("\nChIP-seq bigWig files:")
    for mark, fnames in CHIP_BW_FILES.items():
        resolved = []
        for fname in fnames:
            p = raw_dir / fname
            print(f"  {'✓' if p.exists() else '✗'} {mark}: {p}")
            resolved.append(str(p) if p.exists() else None)
        chip_paths[mark] = resolved

    chromosomes = None
    if args.chrom:
        chromosomes = [c.strip() for c in args.chrom.split(",")]
        unknown = [c for c in chromosomes if c not in CHROMOSOME_SIZES]
        if unknown:
            raise ValueError(f"Unknown chromosome(s): {unknown}. Valid: {list(CHROMOSOME_SIZES)}")

    chrom_label = ",".join(chromosomes) if chromosomes else "all"
    print(f"\nGenerating regions for human (hg38), chromosomes: {chrom_label}...")
    all_regions = generate_all_regions(chromosomes)

    def _npz_exists(region: str) -> bool:
        chrom, rs, re, cs, ce = _parse_region(region)
        return (output_dir / f"chr{chrom}" / f"{rs}-{re},{cs}-{ce}.npz").exists()

    pending = [r for r in all_regions if not _npz_exists(r)]
    print(f"Total regions  : {len(all_regions):,}")
    print(f"Already cached : {len(all_regions) - len(pending):,}")
    print(f"To process     : {len(pending):,}")
    print(f"Normalization  : {args.norm}")
    print(f"Min mean ct/bin: {MIN_MEAN_COUNTS_PER_BIN} (per phase; below → skip)")

    if args.dry_run or not pending:
        if not pending:
            print("All regions already cached.")
        return 0

    _init(
        hic_paths=hic_paths,
        chip_paths=chip_paths,
        hic_data_type=args.hic_type,
        normalization=args.norm,
        chrom_prefix=chrom_prefix,
        chromosomes=chromosomes,
    )

    n_written = 0
    n_filtered = 0
    for region in tqdm(pending, desc="Caching Kang"):
        if _process_region((region, output_dir)) is None:
            n_filtered += 1
        else:
            n_written += 1

    print(f"\nWritten        : {n_written:,}")
    print(f"Filtered (low) : {n_filtered:,}")
    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
