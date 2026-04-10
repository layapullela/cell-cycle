"""
prestore_hic.py — precompute cache files for CellCycleDataLoader.

This script materializes Hi-C (per phase) and ChIP-seq (row/col tracks) into
compressed `.npz` files so training can be cache-only (no live hicstraw/pyBigWig
reads).

Output layout:
  <output_dir>/chr{chrom}/{row_start}-{row_end},{col_start}-{col_end}.npz

Each `.npz` contains:
  earlyG1, midG1, lateG1, anatelo : float32 (N, N) raw counts (no log transform)
  chip_ctcf_row/col, chip_hac_row/col, chip_h3k4me1_row/col, chip_h3k4me3_row/col
                                  float32 (N,) log1p(max-per-bin) tracks
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
# Single-process globals (initialized once in main)
# ---------------------------------------------------------------------------
_hic_paths: Dict[str, Optional[str]] = {}
_chip_paths: Dict[str, Optional[str]] = {}
_resolution: int = 10_000
_image_size: int = 64
_hic_data_type: str = "oe"
_normalization: str = "KR"
_bw_handles: Dict[str, object] = {}


def _init_single_process(
    hic_paths: Dict[str, Optional[str]],
    chip_paths: Dict[str, Optional[str]],
    resolution: int,
    image_size: int,
    hic_data_type: str,
    normalization: str,
) -> None:
    """Initialize globals and open bigWig handles once (single process)."""
    global _hic_paths, _chip_paths, _resolution, _image_size, _hic_data_type, _normalization, _bw_handles
    _hic_paths = hic_paths
    _chip_paths = chip_paths
    _resolution = int(resolution)
    _image_size = int(image_size)
    _hic_data_type = hic_data_type
    _normalization = normalization

    import pyBigWig  # local import

    _bw_handles = {}
    for key, path in chip_paths.items():
        if path is None:
            _bw_handles[key] = None
            continue
        try:
            _bw_handles[key] = pyBigWig.open(path)
        except Exception:
            _bw_handles[key] = None


def _parse_region(region: str) -> Tuple[str, int, int, int, int]:
    """Parse canonical region string 'chrom:rs-re:cs-ce' (or legacy diagonal)."""
    parts = region.split(":")
    chrom = parts[0]
    rs, re = map(int, parts[1].split("-"))
    if len(parts) == 3:
        cs, ce = map(int, parts[2].split("-"))
    else:
        cs, ce = rs, re
    return chrom, rs, re, cs, ce


def _extract_matrix(hic_file: str, region: str) -> np.ndarray:
    """Extract one (N,N) Hi-C matrix slice using hicstraw."""
    import hicstraw as straw

    chrom, rs, re, cs, ce = _parse_region(region)
    is_diagonal = (rs == cs)
    result = straw.straw(
        _hic_data_type,
        _normalization,
        hic_file,
        f"{chrom}:{rs}:{re}",
        f"{chrom}:{cs}:{ce}",
        "BP",
        _resolution,
    )

    mat = np.zeros((_image_size, _image_size), dtype=np.float32)
    for rec in result:
        xi = int((rec.binX - rs) // _resolution)
        yj = int((rec.binY - cs) // _resolution)
        if 0 <= xi < _image_size and 0 <= yj < _image_size:
            mat[xi, yj] = float(rec.counts)
            if is_diagonal and xi != yj:
                mat[yj, xi] = float(rec.counts)
    return mat


def _extract_chip_1d(chrom: str, start: int, end: int, bw) -> np.ndarray:
    """Extract log1p(max) bigWig signal per Hi-C bin across [start,end)."""
    signal = np.zeros(_image_size, dtype=np.float32)
    if bw is None:
        return signal
    chrom_name = "chr" + chrom
    for i in range(_image_size):
        b0 = start + i * _resolution
        b1 = start + (i + 1) * _resolution
        values = bw.stats(chrom_name, b0, b1, type="max")
        signal[i] = np.log1p(values[0] if values and values[0] is not None else 0.0)
    return signal


def _process_region(args: Tuple[str, Path]) -> Optional[str]:
    """
    Worker entrypoint: compute one region and write its `.npz`.

    Returns the region string on success, None if skipped because output exists.
    """
    region, output_dir = args
    chrom, rs, re, cs, ce = _parse_region(region)
    is_diagonal = (rs == cs)

    chrom_dir = output_dir / f"chr{chrom}"
    out_path = chrom_dir / f"{rs}-{re},{cs}-{ce}.npz"
    if out_path.exists():
        return None

    arrays: Dict[str, np.ndarray] = {}
    for phase in ("earlyG1", "midG1", "lateG1", "anatelo"):
        hic_file = _hic_paths.get(phase)
        if hic_file is None:
            arrays[phase] = np.zeros((_image_size, _image_size), dtype=np.float32)
        else:
            arrays[phase] = _extract_matrix(hic_file, region)

    for mark in ("ctcf", "hac", "h3k4me1", "h3k4me3"):
        bw = _bw_handles.get(mark)
        row_sig = _extract_chip_1d(chrom, rs, re, bw)
        col_sig = row_sig.copy() if is_diagonal else _extract_chip_1d(chrom, cs, ce, bw)
        arrays[f"chip_{mark}_row"] = row_sig
        arrays[f"chip_{mark}_col"] = col_sig

    chrom_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.rename(out_path)
    return region


# ---------------------------------------------------------------------------
# Region generation
# ---------------------------------------------------------------------------
CHROMOSOME_SIZES = {
    "1": 195471971,
    "2": 182113224,
    "3": 160039680,
    "4": 156508116,
    "5": 151834684,
    "6": 149736546,
    "7": 145441459,
    "8": 129401213,
    "9": 124595110,
    "10": 130694993,
    "11": 122082543,
    "12": 120129022,
    "13": 120421639,
    "14": 124902244,
    "15": 104043685,
    "16": 98207768,
    "17": 94987271,
    "18": 90702639,
    "19": 61431566,
    "X": 171031299,
}

MIN_START = 3_000_000
STEP_PIXELS = 10
RESOLUTION = 10_000
REGION_SIZE = 640_000
STEP_BP = STEP_PIXELS * RESOLUTION
OFFDIAG_NEAR_BAND_BP = 640_000


def _sample_offdiag(chrom: str, diag_positions: List[int], rng: np.random.Generator) -> List[str]:
    """
    Generate one upper-triangular off-diagonal crop per diagonal window.

    For each diagonal start position, draw an offset (in STEP_BP units) from
    near-diagonal steps up to OFFDIAG_NEAR_BAND_BP, then pair (row, col) windows.
    """
    pos = np.asarray(diag_positions, dtype=np.int64)
    n = int(pos.size)
    if n < 2:
        return []

    d_max = n - 1
    near_max_steps = min(d_max, OFFDIAG_NEAR_BAND_BP // STEP_BP)
    near_steps = np.arange(1, near_max_steps + 1, dtype=np.int64)
    if near_steps.size == 0:
        return []

    regions: List[str] = []
    for ri in range(n - 1):
        ds = int(rng.choice(near_steps))
        ci = ri + ds
        if ci >= n:
            continue
        rs = int(pos[ri])
        cs = int(pos[ci])
        regions.append(f"{chrom}:{rs}-{rs + REGION_SIZE}:{cs}-{cs + REGION_SIZE}")
    return regions


def generate_all_regions() -> List[str]:
    """Diagonal sliding windows + one off-diagonal crop per diagonal crop."""
    rng = np.random.default_rng(42)
    regions: List[str] = []
    for chrom, size in CHROMOSOME_SIZES.items():
        diag_pos = list(range(MIN_START, size - REGION_SIZE + 1, STEP_BP))
        diag_strs = [f"{chrom}:{s}-{s + REGION_SIZE}:{s}-{s + REGION_SIZE}" for s in diag_pos]
        regions.extend(diag_strs)
        regions.extend(_sample_offdiag(chrom, diag_pos, rng))
    return regions


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-store Hi-C + ChIP-seq to .npz cache files")
    parser.add_argument("--data_dir", required=True, help="Directory containing *.hic inputs")
    default_output_dir = Path(__file__).resolve().parent.parent / "processed_data" / "zhang" / "oe_kr2"
    parser.add_argument(
        "--output_dir",
        default=str(default_output_dir),
        help=(
            "Root output directory for .npz cache "
            f"(default: {default_output_dir})"
        ),
    )
    parser.add_argument("--hic_type", default="oe", help="hic_data_type (oe or observed)")
    parser.add_argument("--norm", default="KR", help="Normalization (KR, SCALE, NONE, ...)")
    parser.add_argument("--dry_run", action="store_true", help="Print counts without writing")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_files = {
        "earlyG1": "earlyG1.hic",
        "midG1": "midG1.hic",
        "lateG1": "lateG1.hic",
        "anatelo": "anatelo.hic",
    }
    hic_paths: Dict[str, Optional[str]] = {}
    for phase, fname in phase_files.items():
        p = data_dir / fname
        hic_paths[phase] = str(p) if p.exists() else None
        print(f"  {'✓' if p.exists() else '✗'} {phase}: {p}")

    bw_files = {
        "ctcf": "GSE129997_CTCF_asyn_mm10.bw",
        "hac": "GSM1502751_534.mm10.bigWig",
        "h3k4me1": "h3k04me1.mm10.bigWig",
        "h3k4me3": "G1eH3k04me3.mm10.bigWig",
    }
    raw_data_dir = data_dir.parent.parent / "raw_data" / "zhang_4dn"
    chip_paths: Dict[str, Optional[str]] = {}
    for mark, fname in bw_files.items():
        p = raw_data_dir / fname
        chip_paths[mark] = str(p) if p.exists() else None
        print(f"  {'✓' if p.exists() else '✗'} {mark}: {p}")

    print("\nGenerating regions (diagonal + 1 off-diagonal per diagonal)...")
    all_regions = generate_all_regions()

    def _npz_exists(region: str) -> bool:
        chrom, rs, re, cs, ce = _parse_region(region)
        return (output_dir / f"chr{chrom}" / f"{rs}-{re},{cs}-{ce}.npz").exists()

    pending = [r for r in all_regions if not _npz_exists(r)]
    print(f"Total regions  : {len(all_regions):,}")
    print(f"Already cached : {len(all_regions) - len(pending):,}")
    print(f"To process     : {len(pending):,}")

    if args.dry_run or not pending:
        if not pending:
            print("\nAll regions already cached. Nothing to do.")
        return 0

    _init_single_process(
        hic_paths=hic_paths,
        chip_paths=chip_paths,
        resolution=RESOLUTION,
        image_size=REGION_SIZE // RESOLUTION,
        hic_data_type=args.hic_type,
        normalization=args.norm,
    )

    for region in tqdm(pending, total=len(pending), desc="Caching"):
        _process_region((region, output_dir))

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

