"""
Cache-backed dataloader for cell cycle Hi-C contact maps and ChIP-seq signals.

This loader does not generate or resample genomic regions. It simply enumerates
pre-stored `.npz` files under `processed_data` and serves them as training or
holdout samples based on the chromosome split.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple


class CellCycleDataLoader:
    """
    dataloader for cell cycle Hi-C contact maps.

    Each sample contains:
      region  : str, e.g. "1:10000000-10640000:10000000-10640000"
      earlyG1, lateG1, midG1, anatelo : numpy arrays of shape (N, N) = (64, 64)
      chip_seq_{mark}_row / _col       : numpy arrays of shape (N,) = (64,)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        resolution: int = 10000,
        region_size: int = 640000,
        normalization: str = "KR",
        chipseq_file: Optional[str] = None,
        hold_out_chromosome: Optional[str] = None,
        cluster3_loops_file: Optional[str] = None,
        save_normalization_stats: bool = False,
        normalization_stats_file: Optional[str] = None,
        hic_data_type: str = "oe",
        use_log_transform: bool = True,
        augment: Union[int, float] = 50,
        processed_data_dir: Optional[Union[str, Path]] = None,
        allow_live_fallback: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.region_size = region_size
        self.normalization = normalization
        self.image_size = region_size // resolution
        self.hold_out_chromosome = hold_out_chromosome
        self.hic_data_type = hic_data_type
        self.use_log_transform = use_log_transform
        self.augment = float(augment)
        self.allow_live_fallback = bool(allow_live_fallback)
        self.processed_data_dir = Path(processed_data_dir) if processed_data_dir else None

        self.save_normalization_stats = save_normalization_stats
        if normalization_stats_file is None:
            self.normalization_stats_file = self.data_dir / "normalization_stats.csv"
        else:
            self.normalization_stats_file = Path(normalization_stats_file)

        if self.save_normalization_stats:
            with open(self.normalization_stats_file, 'w') as f:
                f.write("region,phase,min,max\n")
            print(f"Saving normalization stats to: {self.normalization_stats_file}")

        self.phase_names = ('earlyG1', 'midG1', 'lateG1', 'anatelo')
        self.phase_paths: Dict[str, Path] = {}
        self._chipseq_paths: Dict[str, Optional[str]] = {}
        self.chipseq_files: Dict[str, Optional[object]] = {}

        default_processed_data_dir = Path(__file__).resolve().parent.parent / "processed_data"
        self.processed_data_dir = Path(processed_data_dir) if processed_data_dir else default_processed_data_dir
        if not self.processed_data_dir.exists() and not self.allow_live_fallback:
            raise ValueError(
                f"Processed data directory not found: {self.processed_data_dir}. "
                "Run preprocess/prestore_hic.py first (or set allow_live_fallback=True)."
            )

        self.region_to_path: Dict[str, Path] = {}
        if self.processed_data_dir.exists():
            self.regions, self.holdout_regions = self._generate_regions()
        else:
            self.regions, self.holdout_regions = [], []

        total_cached = len(self.regions) + len(self.holdout_regions)
        if total_cached == 0 and not self.allow_live_fallback:
            raise ValueError(
                f"No cached .npz files found under {self.processed_data_dir}. "
                "Run preprocess/prestore_hic.py first (or set allow_live_fallback=True)."
            )

        if self.allow_live_fallback:
            self._init_live_sources()

    # ------------------------------------------------------------------
    # ChIP-seq handle management
    # ------------------------------------------------------------------
    def _open_chipseq_handles(self):
        """Open all pyBigWig file handles (single-process inference)."""
        if not self.allow_live_fallback:
            return
        for key, path in self._chipseq_paths.items():
            if path is None:
                self.chipseq_files[key] = None
                continue
            try:
                import pyBigWig  # lazy import
                self.chipseq_files[key] = pyBigWig.open(path)
            except Exception:
                self.chipseq_files[key] = None

    # ------------------------------------------------------------------
    # Live sources (used only when region isn't cached)
    # ------------------------------------------------------------------
    def _init_live_sources(self) -> None:
        """Initialize `.hic` and bigWig paths for live fallback queries."""
        # Hi-C phase files
        for phase in self.phase_names:
            p = self.data_dir / f"{phase}.hic"
            if p.exists():
                self.phase_paths[phase] = p

        # bigWig files (optional)
        raw_data_dir = self.data_dir.parent.parent / "raw_data" / "zhang_4dn"
        chip_files = {
            'ctcf':    raw_data_dir / "GSE129997_CTCF_asyn_mm10.bw",
            'hac':     raw_data_dir / "GSM1502751_534.mm10.bigWig",
            'h3k4me1': raw_data_dir / "h3k04me1.mm10.bigWig",
            'h3k4me3': raw_data_dir / "G1eH3k04me3.mm10.bigWig",
        }
        for k, p in chip_files.items():
            self._chipseq_paths[k] = str(p) if p.exists() else None

        self._open_chipseq_handles()

    def _extract_region_matrix_live(self, hic_file: Path, region: str) -> np.ndarray:
        chrom, row_start, row_end, col_start, col_end = self._parse_region(region)
        is_diagonal = (row_start == col_start)
        try:
            import hicstraw as straw  # lazy import
            result = straw.straw(
                self.hic_data_type,
                self.normalization,
                str(hic_file),
                f"{chrom}:{row_start}:{row_end}",
                f"{chrom}:{col_start}:{col_end}",
                "BP",
                self.resolution,
            )
        except Exception as e:
            raise ValueError(f"Error reading region {region} from {hic_file}: {e}")

        matrix = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        for record in result:
            val = float(record.counts)
            # hicstraw always returns binX <= binY; try both coordinate mappings
            # so that off-diagonal regions with overlapping intervals get filled
            # on both sides of the submatrix.
            x_idx = int((record.binX - row_start) // self.resolution)
            y_idx = int((record.binY - col_start) // self.resolution)
            if 0 <= x_idx < self.image_size and 0 <= y_idx < self.image_size:
                matrix[x_idx, y_idx] = val

            x2 = int((record.binY - row_start) // self.resolution)
            y2 = int((record.binX - col_start) // self.resolution)
            if 0 <= x2 < self.image_size and 0 <= y2 < self.image_size and (x2, y2) != (x_idx, y_idx):
                matrix[x2, y2] = val
        return matrix

    def _extract_chipseq_signal_live(self, region_1d: str, bw) -> np.ndarray:
        if bw is None:
            return np.zeros(self.image_size, dtype=np.float32)
        parts = region_1d.split(':')
        chrom = parts[0]
        start, end = map(int, parts[1].split('-'))
        signal = np.zeros(self.image_size, dtype=np.float32)
        chrom_name = "chr" + chrom
        try:
            for i in range(self.image_size):
                bin_start = start + i * self.resolution
                bin_end = start + (i + 1) * self.resolution
                values = bw.stats(chrom_name, bin_start, bin_end, type="max")
                signal[i] = np.log1p(values[0] if values[0] is not None else 0.0)
        except Exception:
            pass
        return signal

    def _load_from_live(self, region: str, do_flip: bool) -> Dict[str, object]:
        if not self.phase_paths:
            raise ValueError(
                f"Live fallback requested for region {region}, but no .hic files were found under {self.data_dir}."
            )

        chrom, row_start, row_end, col_start, col_end = self._parse_region(region)
        is_diagonal = (row_start == col_start)
        row_1d = f"{chrom}:{row_start}-{row_end}"
        col_1d = f"{chrom}:{col_start}-{col_end}"

        sample: Dict[str, object] = {'region': region}

        for phase, hic_path in self.phase_paths.items():
            mat = self._extract_region_matrix_live(hic_path, region)
            if self.use_log_transform:
                mat = np.log1p(mat)

            threshold = np.percentile(mat, 99.9)
            mat = np.where(mat > threshold, threshold, mat)
            m_min, m_max = mat.min(), mat.max()
            self._save_normalization_stat(region, phase, float(m_min), float(m_max))

            normalized = (
                np.zeros_like(mat, dtype=np.float32)
                if m_max - m_min < 1e-10
                else ((mat - m_min) / (m_max - m_min) * 2.0 - 1.0).astype(np.float32)
            )
            if do_flip:
                normalized = np.flip(normalized, axis=(0, 1)).copy()
            sample[phase] = normalized

        for mark in ('ctcf', 'hac', 'h3k4me1', 'h3k4me3'):
            bw = self.chipseq_files.get(mark)
            row = self._extract_chipseq_signal_live(row_1d, bw)
            col = row.copy() if is_diagonal else self._extract_chipseq_signal_live(col_1d, bw)
            if do_flip:
                row = np.flip(row).copy()
                col = np.flip(col).copy()
            sample[f"chip_seq_{mark}_row"] = row.astype(np.float32)
            sample[f"chip_seq_{mark}_col"] = col.astype(np.float32)

        return sample

    # ------------------------------------------------------------------
    # Region parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_region(region: str) -> Tuple[str, int, int, int, int]:
        """
        Parse region string.

        Supported formats:
          "chrom:row_start-row_end:col_start-col_end"   (new, canonical)
          "chrom:start-end"                              (legacy diagonal)

        Returns (chrom, row_start, row_end, col_start, col_end).
        """
        parts = region.split(':')
        chrom = parts[0]
        row_start, row_end = map(int, parts[1].split('-'))
        if len(parts) == 3:
            col_start, col_end = map(int, parts[2].split('-'))
        else:
            col_start, col_end = row_start, row_end
        return chrom, row_start, row_end, col_start, col_end

    # ------------------------------------------------------------------
    # Region enumeration
    # ------------------------------------------------------------------
    def _generate_regions(self) -> Tuple[List[str], List[str]]:
        """
        Build training and holdout region lists directly from cached `.npz` files.

        Sampling policy lives in `preprocess/prestore_hic.py`; the dataloader only
        indexes what has already been written to disk.
        """
        training_regions, holdout_regions = self._enumerate_cached_regions()
        print(f"Indexed cached regions from: {self.processed_data_dir}")
        print(f"Training regions: {len(training_regions)}")
        if self.hold_out_chromosome is not None:
            print(f"Holdout chromosome '{self.hold_out_chromosome}': {len(holdout_regions)} regions")
        return training_regions, holdout_regions

    def _enumerate_cached_regions(self) -> Tuple[List[str], List[str]]:
        """
        Enumerate all cached `.npz` files under `processed_data_dir`.

        File naming convention:
            .../chr{chrom}/{rs}-{re},{cs}-{ce}.npz  →  "{chrom}:{rs}-{re}:{cs}-{ce}"
        """
        training_regions: List[str] = []
        holdout_regions: List[str] = []
        holdout = str(self.hold_out_chromosome) if self.hold_out_chromosome else None

        for npz_path in sorted(self.processed_data_dir.rglob("*.npz")):
            chrom_dir = npz_path.parent.name
            if not chrom_dir.startswith("chr"):
                continue

            chrom = chrom_dir[3:]
            row_part, col_part = npz_path.stem.split(",")
            region = f"{chrom}:{row_part}:{col_part}"
            if region in self.region_to_path and self.region_to_path[region] != npz_path:
                raise ValueError(
                    f"Duplicate cached region found for {region} under {self.processed_data_dir}. "
                    "Pass a more specific processed_data_dir."
                )
            self.region_to_path[region] = npz_path

            if holdout is not None and chrom == holdout:
                holdout_regions.append(region)
            else:
                training_regions.append(region)

        return training_regions, holdout_regions

    # ------------------------------------------------------------------
    # Normalization stats helpers
    # ------------------------------------------------------------------
    def _save_normalization_stat(self, region: str, phase: str, min_val: float, max_val: float):
        if not self.save_normalization_stats:
            return
        with open(self.normalization_stats_file, 'a') as f:
            f.write(f"{region},{phase},{min_val},{max_val}\n")

    @staticmethod
    def load_normalization_stats(stats_file: Union[str, Path]) -> Dict[Tuple[str, str], Tuple[float, float]]:
        stats_dict = {}
        stats_file = Path(stats_file)
        if not stats_file.exists():
            raise FileNotFoundError(f"Normalization stats file not found: {stats_file}")
        with open(stats_file, 'r') as f:
            next(f)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                region, phase, min_val, max_val = line.split(',')
                stats_dict[(region, phase)] = (float(min_val), float(max_val))
        return stats_dict

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _npz_path(self, region: str) -> Optional[Path]:
        """Return the cached .npz path for a region, if indexed."""
        return self.region_to_path.get(region)

    def _load_from_cache(self, region: str, do_flip: bool) -> Optional[Dict]:
        """
        Load a pre-stored .npz file and return a fully normalised sample dict,
        or None if the cache file does not exist.

        The .npz stores raw Hi-C counts (no log) and log1p chip signals, matching
        exactly what prestore_hic.py writes.
        """
        path = self._npz_path(region)
        if path is None or not path.exists():
            return None

        data   = np.load(path)
        sample = {'region': region}

        # ---- Hi-C phases (same normalisation as the live path) ----
        for phase in ('earlyG1', 'midG1', 'lateG1', 'anatelo'):
            mat = data[phase].copy()                   # (N, N) raw counts

            if self.use_log_transform:
                mat = np.log1p(mat)

            threshold = np.percentile(mat, 99.9)
            mat = np.where(mat > threshold, threshold, mat)

            m_min, m_max = mat.min(), mat.max()
            #self._save_normalization_stat(region, phase, float(m_min), float(m_max))

            normalized = (
                np.zeros_like(mat, dtype=np.float32)
                if m_max - m_min < 1e-10
                else ((mat - m_min) / (m_max - m_min) * 2.0 - 1.0).astype(np.float32)
            )
            if do_flip:
                normalized = np.flip(normalized, axis=(0, 1)).copy()
            sample[phase] = normalized

        # ---- ChIP-seq tracks (already log1p in the .npz) ----
        for mark in ('ctcf', 'hac', 'h3k4me1', 'h3k4me3'):
            row = data[f"chip_{mark}_row"].copy()
            col = data[f"chip_{mark}_col"].copy()
            if do_flip:
                row = np.flip(row).copy()
                col = np.flip(col).copy()
            sample[f"chip_seq_{mark}_row"] = row.astype(np.float32)
            sample[f"chip_seq_{mark}_col"] = col.astype(np.float32)

        return sample

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.regions)

    def get_holdout_regions(self) -> List[str]:
        return self.holdout_regions if hasattr(self, 'holdout_regions') else []

    def __getitem__(self, idx: Union[int, str]) -> Dict[str, object]:
        """
        Returns a sample dict with:
          region           : str
          earlyG1, midG1, lateG1, anatelo : float32 (N, N) matrices, normalised to [-1,1]
          chip_seq_{mark}_row / _col       : float32 (N,) log1p ChIP-seq signals
        """
        if isinstance(idx, str):
            region = idx
        else:
            region = self.regions[idx]

        do_flip = (self.augment > 0) and (np.random.rand() < (self.augment / 100.0))

        cached = self._load_from_cache(region, do_flip)
        if cached is not None:
            return cached

        if not self.allow_live_fallback:
            raise KeyError(f"Region not found in cached dataset: {region}")

        return self._load_from_live(region, do_flip)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_regions(self) -> List[str]:
        return self.regions.copy()

    def get_available_phases(self) -> List[str]:
        return list(self.phase_names)

    def close(self):
        if not self.allow_live_fallback:
            return
        for bw in self.chipseq_files.values():
            if bw is not None:
                try:
                    bw.close()
                except Exception:
                    pass

    def __del__(self):
        self.close()
