"""
dataloader for cell cycle Hi-C contact maps and chip seq signals.

Regions are now represented as 2D crops (not restricted to the diagonal):
  Format: "chrom:row_start-row_end:col_start-col_end"
  Diagonal crops (row == col range) are symmetric; off-diagonal crops are not.

Training set includes:
  1. Overlapping diagonal sliding-window crops (as before, 10-pixel step).
  2. An equal number of off-diagonal crops sampled from the upper-triangular
     part of the genome, with weights inversely proportional to their distance
     from the diagonal (i.e. nearby-diagonal crops are sampled more often).

Samples now return full (N, N) contact matrices (float32), not flattened
upper-triangular vectors.  ChIP-seq is returned as separate row and col tracks
(identical for diagonal crops).
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import hicstraw as straw
import pyBigWig


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

        self.save_normalization_stats = save_normalization_stats
        if normalization_stats_file is None:
            self.normalization_stats_file = self.data_dir / "normalization_stats.csv"
        else:
            self.normalization_stats_file = Path(normalization_stats_file)

        if self.save_normalization_stats:
            with open(self.normalization_stats_file, 'w') as f:
                f.write("region,phase,min,max\n")
            print(f"Saving normalization stats to: {self.normalization_stats_file}")

        self.step_pixels = 10
        self.step_bp = self.step_pixels * resolution

        raw_data_dir = self.data_dir.parent.parent / "raw_data" / "zhang_4dn"

        self.chipseq_files = {}

        def _load_chip(key: str, filename: str, label: str):
            chip_path = raw_data_dir / filename
            if chip_path.exists():
                try:
                    self.chipseq_files[key] = pyBigWig.open(str(chip_path))
                    print(f"Loaded {label} ChIP-seq from: {chip_path}")
                except Exception as e:
                    print(f"Warning: Failed to load {label} bigWig file {chip_path}: {e}")
                    self.chipseq_files[key] = None
            else:
                print(f"Warning: {label} bigWig file not found at expected path: {chip_path}")
                self.chipseq_files[key] = None

        _load_chip('ctcf',    "GSE129997_CTCF_asyn_mm10.bw",    "CTCF")
        _load_chip('hac',     "GSM1502751_534.mm10.bigWig",      "H3K27ac (HAC)")
        _load_chip('h3k4me1', "h3k04me1.mm10.bigWig",            "H3K4me1")
        _load_chip('h3k4me3', "G1eH3k04me3.mm10.bigWig",         "H3K4me3")
        self.chipseq_files['rad21'] = None

        self.phase_files = {
            'earlyG1': 'earlyG1.hic',
            'lateG1':  'lateG1.hic',
            'midG1':   'midG1.hic',
            'anatelo': 'anatelo.hic',
        }

        self.phase_paths = {}
        for phase, filename in self.phase_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                self.phase_paths[phase] = filepath

        if not self.phase_paths:
            raise ValueError(f"No .hic files found in {self.data_dir}")

        self.chromosome_sizes = {
            "1": 195471971, "2": 182113224, "3": 160039680, "4": 156508116,
            "5": 151834684, "6": 149736546, "7": 145441459, "8": 129401213,
            "9": 124595110, "10": 130694993, "11": 122082543, "12": 120129022,
            "13": 120421639, "14": 124902244, "15": 104043685, "16": 98207768,
            "17": 94987271, "18": 90702639, "19": 61431566,
            "X": 171031299, "Y": 91744698,
        }

        self.min_start_position = 3000000
        self.regions, self.holdout_regions = self._generate_regions()

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
    # Region generation
    # ------------------------------------------------------------------
    def _make_diag_region(self, chrom: str, start: int) -> str:
        end = start + self.region_size
        return f"{chrom}:{start}-{end}:{start}-{end}"

    def _sample_offdiag_regions(
        self,
        chrom: str,
        diag_positions: List[int],
        n_samples: int,
        rng: np.random.Generator,
    ) -> List[str]:
        """
        Sample n_samples off-diagonal upper-triangular crops.

        Distance is measured as the number of step_bp steps between the two
        window start positions.  Probability ∝ 1 / distance so near-diagonal
        crops are preferred.
        """
        pos_array = np.asarray(diag_positions, dtype=np.int64)
        n_pos = len(pos_array)
        if n_pos < 2:
            return []

        D_MAX = n_pos - 1
        d_vals = np.arange(1, D_MAX + 1, dtype=np.float64)
        d_probs = 1.0 / d_vals
        d_probs /= d_probs.sum()

        regions: List[str] = []
        attempts = 0
        max_attempts = n_samples * 15

        while len(regions) < n_samples and attempts < max_attempts:
            batch = min((n_samples - len(regions)) * 3, 4096)

            row_idx = rng.integers(0, n_pos - 1, size=batch)
            d_steps = rng.choice(D_MAX, size=batch, p=d_probs) + 1
            col_idx = row_idx + d_steps

            valid = col_idx < n_pos
            for k in range(batch):
                if not valid[k]:
                    continue
                rs = int(pos_array[row_idx[k]])
                cs = int(pos_array[col_idx[k]])
                regions.append(
                    f"{chrom}:{rs}-{rs + self.region_size}:{cs}-{cs + self.region_size}"
                )
                if len(regions) >= n_samples:
                    break
            attempts += batch

        return regions[:n_samples]

    def _generate_regions(self) -> Tuple[List[str], List[str]]:
        """
        Build training and holdout region lists.

        Training  = diagonal sliding windows + equal-count off-diagonal samples.
        Holdout   = diagonal sliding windows only (for clean eval).
        """
        training_regions: List[str] = []
        holdout_regions: List[str] = []
        rng = np.random.default_rng(42)

        for chrom, size in self.chromosome_sizes.items():
            if chrom == 'Y':
                continue

            is_holdout = (
                self.hold_out_chromosome is not None
                and str(chrom) == str(self.hold_out_chromosome)
            )

            diag_positions = list(range(
                self.min_start_position,
                size - self.region_size + 1,
                self.step_bp,
            ))

            diag_region_strs = [self._make_diag_region(chrom, s) for s in diag_positions]

            if is_holdout:
                holdout_regions.extend(diag_region_strs)
            else:
                training_regions.extend(diag_region_strs)
                # Off-diagonal: same count as diagonal, training only
                offdiag = self._sample_offdiag_regions(
                    chrom, diag_positions, len(diag_region_strs), rng
                )
                training_regions.extend(offdiag)

        if self.hold_out_chromosome:
            print(f"Holdout chromosome '{self.hold_out_chromosome}': {len(holdout_regions)} regions")
            print(f"Training regions: {len(training_regions)} "
                  f"(~50% diagonal, ~50% off-diagonal)")

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
    # Hi-C extraction
    # ------------------------------------------------------------------
    def _extract_region_matrix(self, hic_file: Path, region: str) -> np.ndarray:
        """
        Extract a (N, N) contact matrix for the given region.

        For diagonal crops (row == col) the matrix is symmetric.
        For off-diagonal crops the matrix is a raw rectangular slice of the
        full contact map (not symmetric).
        """
        chrom, row_start, row_end, col_start, col_end = self._parse_region(region)
        is_diagonal = (row_start == col_start)

        try:
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
            x_idx = int((record.binX - row_start) // self.resolution)
            y_idx = int((record.binY - col_start) // self.resolution)

            if 0 <= x_idx < self.image_size and 0 <= y_idx < self.image_size:
                matrix[x_idx, y_idx] = float(record.counts)
                if is_diagonal and x_idx != y_idx:
                    matrix[y_idx, x_idx] = float(record.counts)

        return matrix

    # ------------------------------------------------------------------
    # ChIP-seq extraction
    # ------------------------------------------------------------------
    def _extract_chipseq_signal(self, region_1d: str, chipseq_bw=None) -> np.ndarray:
        """
        Extract ChIP-seq signal for a 1-D genomic interval "chrom:start-end".
        Returns max-per-bin signal with log1p transformation.
        """
        if chipseq_bw is None:
            return np.zeros(self.image_size, dtype=np.float32)

        parts  = region_1d.split(':')
        chrom  = parts[0]
        start, end = map(int, parts[1].split('-'))
        signal = np.zeros(self.image_size, dtype=np.float32)
        chrom_name = "chr" + chrom
        try:
            for i in range(self.image_size):
                bin_start = start + i * self.resolution
                bin_end   = start + (i + 1) * self.resolution
                values    = chipseq_bw.stats(chrom_name, bin_start, bin_end, type="max")
                signal[i] = np.log1p(values[0] if values[0] is not None else 0.0)
            return signal
        except Exception as e:
            raise ValueError(f"Error extracting ChIP-seq signal for region {region_1d}: {e}")

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
            # Validate that size matches if not in known lists
            all_regions = self.regions + (self.holdout_regions if hasattr(self, 'holdout_regions') else [])
            if region not in all_regions:
                _, rs, re, cs, ce = self._parse_region(region)
                if (re - rs) != self.region_size or (ce - cs) != self.region_size:
                    raise KeyError(
                        f"Region {region} not in regions list and window size does not match "
                        f"region_size={self.region_size}"
                    )
        else:
            region = self.regions[idx]

        chrom, row_start, row_end, col_start, col_end = self._parse_region(region)
        is_diagonal = (row_start == col_start)

        # 1-D region strings for ChIP-seq
        row_1d = f"{chrom}:{row_start}-{row_end}"
        col_1d = f"{chrom}:{col_start}-{col_end}"

        sample: Dict[str, object] = {'region': region}
        do_flip = (self.augment > 0) and (np.random.rand() < (self.augment / 100.0))

        # ---- Hi-C phases ----
        for phase, filepath in self.phase_paths.items():
            mat = self._extract_region_matrix(filepath, region)  # (N, N)

            if self.use_log_transform:
                mat = np.log1p(mat)

            threshold = np.percentile(mat, 99.9)
            mat = np.where(mat > threshold, threshold, mat)

            m_min = mat.min()
            m_max = mat.max()
            self._save_normalization_stat(region, phase, float(m_min), float(m_max))

            if m_max - m_min < 1e-10:
                normalized = np.zeros_like(mat, dtype=np.float32)
            else:
                normalized = (mat - m_min) / (m_max - m_min) * 2.0 - 1.0

            if do_flip:
                normalized = np.flip(normalized, axis=(0, 1)).copy()

            sample[phase] = normalized.astype(np.float32)

        # ---- ChIP-seq tracks ----
        for key in ('ctcf', 'hac', 'h3k4me1', 'h3k4me3'):
            bw = self.chipseq_files.get(key)
            row_track = self._extract_chipseq_signal(row_1d, chipseq_bw=bw)
            col_track = row_track.copy() if is_diagonal else self._extract_chipseq_signal(col_1d, chipseq_bw=bw)

            if do_flip:
                row_track = np.flip(row_track).copy()
                col_track = np.flip(col_track).copy()

            tag = f"chip_seq_{key}"
            sample[f"{tag}_row"] = row_track.astype(np.float32)
            sample[f"{tag}_col"] = col_track.astype(np.float32)

        return sample

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_regions(self) -> List[str]:
        return self.regions.copy()

    def get_available_phases(self) -> List[str]:
        return list(self.phase_paths.keys())

    def close(self):
        for key, bw in self.chipseq_files.items():
            if bw is not None:
                try:
                    bw.close()
                except Exception:
                    pass

    def __del__(self):
        self.close()
