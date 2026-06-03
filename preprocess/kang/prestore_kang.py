"""
prestore_kang.py — precompute cache files for Kang et al. Hi-C data.

Kang et al. arrest-release time course (U2OS human cells, hg19):
  0 min  → prometaphase
  35 min → anatelophase
  60/90 min   → early G1
  120/180 min → mid G1
  240/360 min → late G1

Phase composition (type a and type b are a data-augmentation pair):
  prometa    = Rep1@0min   + Rep2@0min    (same for a and b)
  anatelo    = Rep1@35min  + Rep2@35min   (same for a and b)
  earlyG1_a  = Rep1@60min  + Rep2@90min
  earlyG1_b  = Rep2@60min  + Rep1@90min
  midG1_a    = Rep1@120min + Rep2@180min
  midG1_b    = Rep2@120min + Rep1@180min
  lateG1_a   = Rep1@240min + Rep2@360min
  lateG1_b   = Rep2@240min + Rep1@360min

Output layout:
  <output_dir>/chr{chrom}/{row_start}-{row_end},{col_start}-{col_end}.npz

Each .npz contains:
  prometa, anatelo                                : float32 (N,N) summed KR-obs counts
  earlyG1_a, earlyG1_b                           : float32 (N,N)
  midG1_a,   midG1_b                             : float32 (N,N)
  lateG1_a,  lateG1_b                            : float32 (N,N)
  chip_ctcf_row/col, chip_hac_row/col,
  chip_h3k4me1_row/col, chip_h3k4me3_row/col    : float32 (N,) log1p(avg-max-per-bin)

NOTE: The presence of the key 'earlyG1_a' signals Kang format to CellCycleDataLoader,
which then randomly picks type a or b at load time (50/50 augmentation).

ChIP-seq marks used (interphase tracks, averaged across two replicates):
  ctcf     : GSM4194671 + GSM4194672
  hac      : H3K27ac proxy: GSM4194659 + GSM4194660
  h3k4me1  : GSM4194665 + GSM4194666
  h3k4me3  : GSM4194653 + GSM4194654

Usage:
  python preprocess/kang/prestore_kang.py \
      --raw_data_dir raw_data/kang \
      --output_dir processed_data/kang \
      [--chrom 1] [--chrom_prefix chr|none] [--no_chr_prefix] [--dry_run]

  sbatch preprocess/kang/prestore_kang_chr1.sh
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Phase → list of .hic filenames to sum (within the raw_data_dir)
# ---------------------------------------------------------------------------
PHASE_HIC_FILES: Dict[str, List[str]] = {
    "prometa":   ["GSM4194449_U2OS_0min_Rep1.hic",   "GSM4194450_U2OS_0min_Rep2.hic"],
    "anatelo":   ["GSM4194451_U2OS_35min_Rep1.hic",  "GSM4194452_U2OS_35min_Rep2.hic"],
    "earlyG1_a": ["GSM4194453_U2OS_60min_Rep1.hic",  "GSM4194456_U2OS_90min_Rep2.hic"],
    "earlyG1_b": ["GSM4194454_U2OS_60min_Rep2.hic",  "GSM4194455_U2OS_90min_Rep1.hic"],
    "midG1_a":   ["GSM4194457_U2OS_120min_Rep1.hic", "GSM4194460_U2OS_180min_Rep2.hic"],
    "midG1_b":   ["GSM4194458_U2OS_120min_Rep2.hic", "GSM4194459_U2OS_180min_Rep1.hic"],
    "lateG1_a":  ["GSM4194461_U2OS_240min_Rep1.hic", "GSM4194464_U2OS_360min_Rep2.hic"],
    "lateG1_b":  ["GSM4194462_U2OS_240min_Rep2.hic", "GSM4194463_U2OS_360min_Rep1.hic"],
}

# ChIP-seq: two interphase replicates per mark (averaged at extraction time).
# hac slot uses H3K27ac as a functional proxy.
CHIP_BW_FILES: Dict[str, List[str]] = {
    "ctcf":    ["GSM4194671_HK81_CTCF_I_S10_both.bw",   "GSM4194672_HK148_CTCF_I_S10_both.bw"],
    "hac":     ["GSM4194659_HK75_H3K27ac_I_S4_both.bw", "GSM4194660_HK158_H3K27ac_I1_S1_both.bw"],
    "h3k4me1": ["GSM4194665_HK78_H3K4me1_I_S7_both.bw", "GSM4194666_HK145_H3K4me1_I_S7_both.bw"],
    "h3k4me3": ["GSM4194653_HK72_H3K4me3_I_S1_both.bw", "GSM4194654_HK139_H3K4me3_I_S1_both.bw"],
}

# hg19 chromosome sizes
CHROMOSOME_SIZES: Dict[str, int] = {
    "1":  249250621, "2":  243199373, "3":  198022430, "4":  191154276,
    "5":  180915260, "6":  171115067, "7":  159138663, "8":  146364022,
    "9":  141213431, "10": 135534747, "11": 135006516, "12": 133851895,
    "13": 115169878, "14": 107349540, "15": 102531392, "16":  90354753,
    "17":  81195210, "18":  78077248, "19":  59128983, "20":  63025520,
    "21":  48129895, "22":  51304566, "X":  155270560,
}

MIN_START           = 3_000_000
RESOLUTION          = 10_000
REGION_SIZE         = 640_000
STEP_PIXELS         = 10
STEP_BP             = STEP_PIXELS * RESOLUTION
OFFDIAG_NEAR_BAND   = 5_000_000
OFFDIAG_PER_DIAG    = 2

# ---------------------------------------------------------------------------
# Process-level globals (set once in main)
# ---------------------------------------------------------------------------
_hic_paths: Dict[str, List[str]] = {}
_chip_paths: Dict[str, List[Optional[str]]] = {}
_bw_handles: Dict[str, List[object]] = {}
_resolution: int = RESOLUTION
_image_size: int = REGION_SIZE // RESOLUTION
_hic_data_type: str = "observed"
_normalization: str = "KR"
_chrom_prefix: str = "chr"


def _init(
    hic_paths: Dict[str, List[str]],
    chip_paths: Dict[str, List[Optional[str]]],
    hic_data_type: str,
    normalization: str,
    chrom_prefix: str,
) -> None:
    global _hic_paths, _chip_paths, _bw_handles, _hic_data_type, _normalization, _chrom_prefix
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
                try:
                    handles.append(pyBigWig.open(p))
                except Exception:
                    handles.append(None)
        _bw_handles[mark] = handles


def _parse_region(region: str) -> Tuple[str, int, int, int, int]:
    parts = region.split(":")
    chrom = parts[0]
    rs, re = map(int, parts[1].split("-"))
    cs, ce = map(int, parts[2].split("-")) if len(parts) == 3 else (rs, re)
    return chrom, rs, re, cs, ce


def _extract_matrix(hic_file: str, chrom: str, rs: int, re: int, cs: int, ce: int) -> np.ndarray:
    """Extract one (N,N) contact matrix from a single .hic file."""
    import hicstraw as straw
    qchrom = _chrom_prefix + chrom
    try:
        result = straw.straw(
            _hic_data_type, _normalization, hic_file,
            f"{qchrom}:{rs}:{re}", f"{qchrom}:{cs}:{ce}", "BP", _resolution,
        )
    except Exception as exc:
        # Fallback: try the opposite prefix convention
        alt_chrom = chrom if _chrom_prefix else ("chr" + chrom)
        try:
            result = straw.straw(
                _hic_data_type, _normalization, hic_file,
                f"{alt_chrom}:{rs}:{re}", f"{alt_chrom}:{cs}:{ce}", "BP", _resolution,
            )
        except Exception:
            raise RuntimeError(
                f"hicstraw failed for {hic_file} chrom={qchrom} (also tried {alt_chrom}): {exc}"
            )

    mat = np.zeros((_image_size, _image_size), dtype=np.float32)
    for rec in result:
        val = float(rec.counts)
        xi  = int((rec.binX - rs) // _resolution)
        yj  = int((rec.binY - cs) // _resolution)
        if 0 <= xi < _image_size and 0 <= yj < _image_size:
            mat[xi, yj] = val
        xi2 = int((rec.binY - rs) // _resolution)
        yj2 = int((rec.binX - cs) // _resolution)
        if 0 <= xi2 < _image_size and 0 <= yj2 < _image_size and (xi2, yj2) != (xi, yj):
            mat[xi2, yj2] = val
    return mat


def _sum_matrices(hic_files: List[str], chrom: str, rs: int, re: int, cs: int, ce: int) -> np.ndarray:
    """Sum contact matrices across a list of .hic files (replicate merging)."""
    total = np.zeros((_image_size, _image_size), dtype=np.float32)
    for f in hic_files:
        if f is not None:
            total += _extract_matrix(f, chrom, rs, re, cs, ce)
    return total


def _extract_chip_1d(chrom: str, start: int, end: int, bw_list: List[object]) -> np.ndarray:
    """Extract log1p(avg-max) bigWig signal per bin, averaged over provided handles."""
    chrom_name = "chr" + chrom
    n_valid = 0
    accum = np.zeros(_image_size, dtype=np.float64)
    for bw in bw_list:
        if bw is None:
            continue
        n_valid += 1
        for i in range(_image_size):
            b0 = start + i * _resolution
            b1 = start + (i + 1) * _resolution
            try:
                vals = bw.stats(chrom_name, b0, b1, type="max")
                accum[i] += vals[0] if vals and vals[0] is not None else 0.0
            except Exception:
                pass
    if n_valid > 0:
        accum /= n_valid
    return np.log1p(accum).astype(np.float32)


def _process_region(args: Tuple[str, Path]) -> Optional[str]:
    """Compute and write one .npz file. Returns region string on success, None if skipped."""
    region, output_dir = args
    chrom, rs, re, cs, ce = _parse_region(region)

    chrom_dir = output_dir / f"chr{chrom}"
    out_path  = chrom_dir / f"{rs}-{re},{cs}-{ce}.npz"
    if out_path.exists():
        return None

    arrays: Dict[str, np.ndarray] = {}
    for phase_key, hic_files in _hic_paths.items():
        arrays[phase_key] = _sum_matrices(hic_files, chrom, rs, re, cs, ce)

    is_diagonal = (rs == cs)
    for mark, bw_list in _bw_handles.items():
        row_sig = _extract_chip_1d(chrom, rs, re, bw_list)
        col_sig = row_sig.copy() if is_diagonal else _extract_chip_1d(chrom, cs, ce, bw_list)
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
        weights[low]  = 0.70 / low.sum()
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-store Kang et al. Hi-C data to .npz cache")
    default_raw = Path(__file__).resolve().parent.parent.parent / "raw_data"  / "kang"
    default_out = Path(__file__).resolve().parent.parent.parent / "processed_data" / "kang"
    parser.add_argument("--raw_data_dir", default=str(default_raw))
    parser.add_argument("--output_dir",   default=str(default_out))
    parser.add_argument("--hic_type",     default="observed", help="hicstraw data type")
    parser.add_argument("--norm",         default="KR",       help="Hi-C normalization")
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
    parser.add_argument("--chrom", default=None,
                        help="Comma-separated chromosomes to process (e.g. '1' or '1,2'). Default: all.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    chrom_prefix = "" if args.no_chr_prefix else args.chrom_prefix

    raw_dir    = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve .hic paths
    hic_paths: Dict[str, List[str]] = {}
    print("Hi-C files:")
    for phase_key, fnames in PHASE_HIC_FILES.items():
        resolved = []
        for fname in fnames:
            p = raw_dir / fname
            print(f"  {'✓' if p.exists() else '✗'} {phase_key}: {p}")
            resolved.append(str(p) if p.exists() else None)
        hic_paths[phase_key] = resolved

    # Resolve bigWig paths
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
    print(f"\nGenerating regions for human (hg19), chromosomes: {chrom_label}...")
    all_regions = generate_all_regions(chromosomes)

    def _npz_exists(region: str) -> bool:
        chrom, rs, re, cs, ce = _parse_region(region)
        return (output_dir / f"chr{chrom}" / f"{rs}-{re},{cs}-{ce}.npz").exists()

    pending = [r for r in all_regions if not _npz_exists(r)]
    print(f"Total regions  : {len(all_regions):,}")
    print(f"Already cached : {len(all_regions) - len(pending):,}")
    print(f"To process     : {len(pending):,}")

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
    )

    for region in tqdm(pending, desc="Caching Kang"):
        try:
            _process_region((region, output_dir))
        except Exception as exc:
            print(f"\nWARN: skipping {region}: {exc}", file=sys.stderr)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)


# sbatch --job-name=prestore_kang_chr1 \
#   --partition=standard \
#   --account=minjilab99 \
#   --time=08:00:00 \
#   --mem=32G \
#   --cpus-per-task=4 \
#   --output=/nfs/turbo/umms-minjilab/lpullela/cell-cycle/preprocess/kang/logs/prestore_kang_chr1_%j.out \
#   --error=/nfs/turbo/umms-minjilab/lpullela/cell-cycle/preprocess/kang/logs/prestore_kang_chr1_%j.err \
#   --wrap="source ~/.bashrc && conda activate test_env && cd /nfs/turbo/umms-minjilab/lpullela/cell-cycle && python preprocess/kang/prestore_kang.py --raw_data_dir raw_data/kang --output_dir processed_data/kang --chrom_prefix chr --chrom 1"