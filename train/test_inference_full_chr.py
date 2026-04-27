"""
Stitch SR3 inference over one chromosome into L×L maps (10 kb bins).

Regions match ``prestore_hic`` (diagonal tiles + one near off-diagonal per step).
Hi-C and ChIP are read with hicstraw / pyBigWig — no processed .npz cache.

If |mid(row) − mid(col)| > ``OFFDIAG_NEAR_BAND_BP``, each phase is set to
0.25 × bulk (normalized), same as before.

Outputs ``chr{chrom}_early.hic`` … (NumPy array bytes via ``numpy.save``; not Juicer .hic).
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
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_REPO_ROOT / "preprocess"))

import hicstraw as straw
import pyBigWig

from inference import Inference, region_is_symmetric
from model import SR3UNet, NoiseEmbedding
from prestore_hic import (
    CHROMOSOME_SIZES,
    MIN_START,
    OFFDIAG_NEAR_BAND_BP,
    REGION_SIZE,
    STEP_BP,
    _sample_offdiag,
)
from train_diffusion_alpha import N, RESOLUTION_BP

PHASE_FILES = ("earlyG1", "midG1", "lateG1", "anatelo")
OUTPUT_SUFFIX = ("early", "mid", "late", "anatelo")


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    noise_embed = NoiseEmbedding(256, max_value=1000)
    model = SR3UNet(n=N, noise_embed_module=noise_embed, base_ch=64).to(device)
    state = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def regions_for_chrom(chrom: str) -> list[str]:
    chrom = str(chrom)
    size_bp = CHROMOSOME_SIZES[chrom]
    diag_pos = list(range(MIN_START, size_bp - REGION_SIZE + 1, STEP_BP))
    diag = [f"{chrom}:{s}-{s + REGION_SIZE}:{s}-{s + REGION_SIZE}" for s in diag_pos]
    off = _sample_offdiag(chrom, diag_pos, np.random.default_rng(42))
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


def straw_submatrix(
    hic_path: str,
    norm: str,
    unit: str,
    region: str,
) -> np.ndarray:
    """One N×N float32 block (same layout as Dataloader / prestore_hic)."""
    chrom, rs, re, cs, ce = parse_region(region)
    result = straw.straw(
        unit,
        norm,
        hic_path,
        f"{chrom}:{rs}:{re}",
        f"{chrom}:{cs}:{ce}",
        "BP",
        RESOLUTION_BP,
    )
    mat = np.zeros((N, N), dtype=np.float32)
    for rec in result:
        v = float(rec.counts)
        xi = int((rec.binX - rs) // RESOLUTION_BP)
        yj = int((rec.binY - cs) // RESOLUTION_BP)
        if 0 <= xi < N and 0 <= yj < N:
            mat[xi, yj] = v
        xi2 = int((rec.binY - rs) // RESOLUTION_BP)
        yj2 = int((rec.binX - cs) // RESOLUTION_BP)
        if 0 <= xi2 < N and 0 <= yj2 < N and (xi2, yj2) != (xi, yj):
            mat[xi2, yj2] = v
    return mat


def normalize_hic(mat: np.ndarray, use_log: bool) -> np.ndarray:
    x = mat.astype(np.float32)
    if use_log:
        x = np.log1p(x)
    thr = np.percentile(x, 99.9)
    x = np.where(x > thr, thr, x)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-10:
        return np.zeros_like(x)
    return ((x - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)


def chip_track(bw, chrom: str, start: int, end: int) -> np.ndarray:
    out = np.zeros(N, dtype=np.float32)
    cname = "chr" + chrom
    for i in range(N):
        a = start + i * RESOLUTION_BP
        b = start + (i + 1) * RESOLUTION_BP
        stats = bw.stats(cname, a, b, type="max")
        out[i] = math.log1p(stats[0] if stats and stats[0] is not None else 0.0)
    return out


def load_patch_tensors(
    region: str,
    hic_paths: dict[str, str],
    chip_bw: dict[str, object | None],
    norm: str,
    unit: str,
    use_log: bool,
    device: torch.device,
) -> dict:
    chrom, rs, re, cs, ce = parse_region(region)
    row_1d = f"{chrom}:{rs}-{re}"
    col_1d = f"{chrom}:{cs}-{ce}"
    is_diag = rs == cs and re == ce

    phases = {}
    for ph in PHASE_FILES:
        phases[ph] = normalize_hic(
            straw_submatrix(hic_paths[ph], norm, unit, region),
            use_log,
        )

    def chip_pair(mark: str):
        bw = chip_bw.get(mark)
        if bw is None:
            z = np.zeros(N, dtype=np.float32)
            return z, z
        r = chip_track(bw, chrom, rs, re)
        c = r.copy() if is_diag else chip_track(bw, chrom, cs, ce)
        return r, c

    r_ctcf, c_ctcf = chip_pair("ctcf")
    r_hac, c_hac = chip_pair("hac")
    r_me1, c_me1 = chip_pair("h3k4me1")
    r_me3, c_me3 = chip_pair("h3k4me3")

    bulk = 0.25 * sum(phases[p] for p in PHASE_FILES)

    def t2(x):
        return torch.from_numpy(x).float().to(device).unsqueeze(0)

    return {
        "region": [region],
        "earlyG1": t2(phases["earlyG1"]),
        "midG1": t2(phases["midG1"]),
        "lateG1": t2(phases["lateG1"]),
        "anatelo": t2(phases["anatelo"]),
        "chip_seq_ctcf_row": t2(r_ctcf),
        "chip_seq_hac_row": t2(r_hac),
        "chip_seq_h3k4me1_row": t2(r_me1),
        "chip_seq_h3k4me3_row": t2(r_me3),
        "chip_seq_ctcf_col": t2(c_ctcf),
        "chip_seq_hac_col": t2(c_hac),
        "chip_seq_h3k4me1_col": t2(c_me1),
        "chip_seq_h3k4me3_col": t2(c_me3),
        "_bulk_np": bulk,
    }


def run_one_patch(
    inference: Inference,
    bt: dict,
    use_model: bool,
    enforce_sym: bool,
) -> np.ndarray:
    if not use_model:
        bulk_np = 0.25 * bt["_bulk_np"].astype(np.float32)
        t = torch.from_numpy(bulk_np).to(bt["earlyG1"].device)
        y = t.unsqueeze(0).expand(4, -1, -1)
        return y.cpu().numpy()

    bulk_map = 0.25 * (
        bt["earlyG1"] + bt["midG1"] + bt["lateG1"] + bt["anatelo"]
    ).unsqueeze(1)
    y = inference.sample(
        bulk_map,
        bt["chip_seq_ctcf_row"],
        bt["chip_seq_hac_row"],
        bt["chip_seq_h3k4me1_row"],
        bt["chip_seq_h3k4me3_row"],
        bt["chip_seq_ctcf_col"],
        bt["chip_seq_hac_col"],
        bt["chip_seq_h3k4me1_col"],
        bt["chip_seq_h3k4me3_col"],
        enforce_symmetry=enforce_sym,
    )
    return y[0].cpu().numpy().astype(np.float32)


def scatter(pred_4: np.ndarray, rs: int, cs: int, sums: list[np.ndarray], counts: list[np.ndarray], L: int):
    gi0, gj0 = rs // RESOLUTION_BP, cs // RESOLUTION_BP
    ii, jj = np.indices((N, N))
    gi, gj = gi0 + ii, gj0 + jj
    m = (gi >= 0) & (gi < L) & (gj >= 0) & (gj < L)
    gi_f, gj_f = gi[m], gj[m]
    for pi in range(4):
        sums[pi][gi_f, gj_f] += pred_4[pi][m]
        counts[pi][gi_f, gj_f] += 1


def midpoint_gap(rs: int, re: int, cs: int, ce: int) -> float:
    return abs(0.5 * (rs + re) - 0.5 * (cs + ce))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--chrom", default="2")
    p.add_argument("--data_dir", default=None, help="Folder with earlyG1.hic, … (default: raw_data/zhang_4dn)")
    p.add_argument("--chip_dir", default=None, help="Folder with bigWigs (default: same as data_dir)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--norm", default="KR")
    p.add_argument("--hic_unit", default="oe", choices=("oe", "observed"))
    p.add_argument("--no_log1p", action="store_true")
    p.add_argument("--near_band_bp", type=float, default=float(OFFDIAG_NEAR_BAND_BP))
    args = p.parse_args()

    data_dir = Path(args.data_dir or _REPO_ROOT / "raw_data" / "zhang_4dn")
    chip_dir = Path(args.chip_dir or data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_log = not args.no_log1p

    hic_paths = {ph: str(data_dir / f"{ph}.hic") for ph in PHASE_FILES}
    for ph, path in hic_paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {ph}: {path}")

    chip_files = {
        "ctcf": chip_dir / "GSE129997_CTCF_asyn_mm10.bw",
        "hac": chip_dir / "GSM1502751_534.mm10.bigWig",
        "h3k4me1": chip_dir / "h3k04me1.mm10.bigWig",
        "h3k4me3": chip_dir / "G1eH3k04me3.mm10.bigWig",
    }
    chip_bw: dict[str, object | None] = {}
    for k, path in chip_files.items():
        chip_bw[k] = pyBigWig.open(str(path)) if path.is_file() else None

    regions = regions_for_chrom(args.chrom)
    L = int(math.ceil(CHROMOSOME_SIZES[str(args.chrom)] / RESOLUTION_BP))
    sums = [np.zeros((L, L), dtype=np.float32) for _ in range(4)]
    counts = [np.zeros((L, L), dtype=np.int32) for _ in range(4)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(Path(args.checkpoint), device)
    inference = Inference(model, device, T=1000)

    try:
        for region in tqdm(regions, desc="patches"):
            _, rs, re, cs, ce = parse_region(region)
            use_model = midpoint_gap(rs, re, cs, ce) <= args.near_band_bp
            bt = load_patch_tensors(region, hic_paths, chip_bw, args.norm, args.hic_unit, use_log, device)
            enforce_sym = region_is_symmetric(bt["region"])
            pred = run_one_patch(inference, bt, use_model, enforce_sym)
            scatter(pred, rs, cs, sums, counts, L)
    finally:
        for bw in chip_bw.values():
            if bw is not None:
                bw.close()

    chrom = str(args.chrom)
    for pi, tag in enumerate(OUTPUT_SUFFIX):
        arr = sums[pi] / np.maximum(counts[pi].astype(np.float32), 1.0)
        path = out_dir / f"chr{chrom}_{tag}.hic"
        with open(path, "wb") as f:
            np.save(f, arr.astype(np.float32))
        print(f"Wrote {path} shape={arr.shape}")


if __name__ == "__main__":
    main()
