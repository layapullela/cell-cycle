"""
Extract a full chromosome from per-phase .hic + ChIP bigWigs into NumPy arrays.

Writes (in output_dir):
  - chr{chrom}_{phase}_raw.npy        float32 (L, L)  raw matrix from hicstraw (symmetric filled)
  - chr{chrom}_chip_{mark}.npy        float32 (L,)    log1p(max per bin) track

This is intentionally simple: one-time extraction so later scripts only slice
NumPy arrays instead of calling hicstraw repeatedly.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_TRAIN_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _TRAIN_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "preprocess"))

import hicstraw as straw
import pyBigWig

from prestore_hic import CHROMOSOME_SIZES

# Keep this pipeline self-contained (avoid import-path issues under Slurm).
RESOLUTION_BP = 10_000

PHASES = ("earlyG1", "midG1", "lateG1", "anatelo")


def chrom_bins(chrom: str) -> int:
    return int(math.ceil(CHROMOSOME_SIZES[str(chrom)] / RESOLUTION_BP))


def open_memmap_npy(path: Path, shape: tuple[int, ...], dtype=np.float32) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    mm[:] = 0
    mm.flush()
    return mm


def extract_phase_matrix(hic_path: Path, chrom: str, hic_unit: str, norm: str, out_path: Path) -> None:
    L = chrom_bins(chrom)
    mm = open_memmap_npy(out_path, (L, L), dtype=np.float32)

    result = straw.straw(
        hic_unit,
        norm,
        str(hic_path),
        str(chrom),
        str(chrom),
        "BP",
        RESOLUTION_BP,
    )

    for rec in result:
        i = int(rec.binX // RESOLUTION_BP)
        j = int(rec.binY // RESOLUTION_BP)
        if 0 <= i < L and 0 <= j < L:
            v = float(rec.counts)
            mm[i, j] = v
            if i != j:
                mm[j, i] = v

    mm.flush()


def extract_chip_track(bw_path: Path, chrom: str, out_path: Path) -> None:
    L = chrom_bins(chrom)
    mm = open_memmap_npy(out_path, (L,), dtype=np.float32)

    bw = pyBigWig.open(str(bw_path))
    try:
        cname = "chr" + str(chrom)
        chrom_len = int(CHROMOSOME_SIZES[str(chrom)])
        for i in range(L):
            a = i * RESOLUTION_BP
            if a >= chrom_len:
                break
            b = min((i + 1) * RESOLUTION_BP, chrom_len)
            stats = bw.stats(cname, a, b, type="max")
            val = stats[0] if stats and stats[0] is not None else 0.0
            mm[i] = np.log1p(val)
    finally:
        try:
            bw.close()
        except Exception:
            pass
    mm.flush()


def main() -> None:
    p = argparse.ArgumentParser(description="Extract full chromosome arrays for later inference/stitching.")
    p.add_argument("--chrom", default="2")
    p.add_argument("--data_dir", default=None, help="Folder with {phase}.hic (default: raw_data/zhang_4dn)")
    p.add_argument("--chip_dir", default=None, help="Folder with bigWigs (default: raw_data/zhang_4dn)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--hic_unit", default="oe", choices=("oe", "observed"))
    p.add_argument("--norm", default="KR")
    args = p.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else _REPO_ROOT / "raw_data" / "zhang_4dn"
    chip_dir = Path(args.chip_dir) if args.chip_dir else data_dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chrom = str(args.chrom)
    L = chrom_bins(chrom)
    print(f"chr{chrom}: {CHROMOSOME_SIZES[chrom]} bp → {L} bins @ {RESOLUTION_BP} bp")

    for ph in PHASES:
        hic = data_dir / f"{ph}.hic"
        if not hic.is_file():
            raise FileNotFoundError(f"Missing {hic}")
        out = out_dir / f"chr{chrom}_{ph}_raw.npy"
        print(f"Extracting {ph}: {hic} -> {out}")
        extract_phase_matrix(hic, chrom, args.hic_unit, args.norm, out)

    chip_files = {
        "ctcf": "GSE129997_CTCF_asyn_mm10.bw",
        "hac": "GSM1502751_534.mm10.bigWig",
        "h3k4me1": "h3k04me1.mm10.bigWig",
        "h3k4me3": "G1eH3k04me3.mm10.bigWig",
    }
    for mark, fname in chip_files.items():
        bw = chip_dir / fname
        out = out_dir / f"chr{chrom}_chip_{mark}.npy"
        if not bw.is_file():
            print(f"Skipping chip {mark} (missing {bw}); writing zeros: {out}")
            open_memmap_npy(out, (L,), dtype=np.float32).flush()
            continue
        print(f"Extracting chip {mark}: {bw} -> {out}")
        extract_chip_track(bw, chrom, out)

    print("Done.")


if __name__ == "__main__":
    main()

