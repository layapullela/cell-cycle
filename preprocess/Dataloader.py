"""
dataloader for cell cycle Hi-C contact maps and chip seq signals.

Extracts 64x64 pixel (640kb if 10kb resolution) regions along the diagonal with sliding window.
we only need to store upper triangular portion of symmetric hic.
we should have chip seq tracks for ctcf, hac, h3k4me1, h3k4me3 corresponding to hic.
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
    - region: str, e.g., "1:10000000-10640000"
    earlyG1, lateG1, midG1, anatelo: each a numpy array of shape (2080,)
    note: for a 64x64 matrix, upper triangular has 64*65/2 = 2080 elements.
    - chip_seq: numpy array (shape: (64,)) - ChIP-seq signal for the region

    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        resolution: int = 10000,  # 10kb bins
        region_size: int = 640000,  # 640kb regions (64 pixels at 10kb resolution)
        normalization: str = "KR",  # or "NONE", "KR", etc. we use vanilla coverage. #TODO: VC or KR? epiphany paper uses KR, so does HiCDiff paper
        chipseq_file: Optional[str] = None,  # Path to ChIP-seq bigWig file
        hold_out_chromosome: Optional[str] = None,  # Chromosome to hold out for testing (e.g., "2")
        cluster3_loops_file: Optional[str] = None,  # Path to cluster_id == 3 loop coordinates file
        save_normalization_stats: bool = False,  # Whether to save min/max values to file
        normalization_stats_file: Optional[str] = None,  # Path to save normalization stats
        hic_data_type: str = "oe",  # experimenting with observed vs oe for predictions of ep loops.
        use_log_transform: bool = True,  # typically we want log transform on hic and chip seq signals
        augment: Union[int, float] = 50,  # data agumentation rotate hic 180 degrees and flip chip seq.
    ):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing .hic files (earlyG1.hic, lateG1.hic, midG1.hic, anatelo.hic)
            resolution: Bin size in base pairs (default: 10000 for 10kb)
            region_size: Size of square region to extract in base pairs (default: 640000 for 640kb = 64 pixels)
            normalization: Normalization method for Hi-C data (NONE, VC, VC_SQRT, KR)
            chipseq_file: Optional path to ChIP-seq bigWig file (if None, looks for default file)
            hold_out_chromosome: Optional chromosome to exclude from training (e.g., "2" for chr2)
        """
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.region_size = region_size
        self.normalization = normalization
        self.image_size = region_size // resolution 
        self.hold_out_chromosome = hold_out_chromosome  
        self.hic_data_type = hic_data_type 
        self.use_log_transform = use_log_transform
        self.augment = float(augment) 

        # setup normalization stats saving (this is incase we want to undo normalization to original observed values.)
        self.save_normalization_stats = save_normalization_stats
        if normalization_stats_file is None:
            self.normalization_stats_file = self.data_dir / "normalization_stats.csv"
        else:
            self.normalization_stats_file = Path(normalization_stats_file)

        # init stats file with header if saving is enabled
        if self.save_normalization_stats:
            with open(self.normalization_stats_file, 'w') as f:
                f.write("region,phase,min,max\n")
            print(f"Saving normalization stats to: {self.normalization_stats_file}")
        
        # step size: sample every 10 pixels along the diagonal
        self.step_pixels = 10
        self.step_bp = self.step_pixels * resolution  # 100kb step
        
        # load chip seq mm10 files (async)
        # data is in raw_data/zhang_4dn from zhang et al paper
        raw_data_dir = self.data_dir.parent.parent / "raw_data" / "zhang_4dn"

        # Dictionary to store multiple ChIP-seq files
        self.chipseq_files = {}

        # Small helper to DRY up bigWig loading + logging
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

        # CTCF, H3K27ac, H3K4me1, H3K4me3
        _load_chip('ctcf',    "GSE129997_CTCF_asyn_mm10.bw",    "CTCF")
        _load_chip('hac',     "GSM1502751_534.mm10.bigWig",     "H3K27ac (HAC)")
        _load_chip('h3k4me1', "h3k04me1.mm10.bigWig",           "H3K4me1")
        _load_chip('h3k4me3', "G1eH3k04me3.mm10.bigWig",        "H3K4me3")

        self.chipseq_files['rad21'] = None  # Dummy object - returns zeros when used
        
        # Phase files
        self.phase_files = {
            'earlyG1': 'earlyG1.hic',
            'lateG1': 'lateG1.hic',
            'midG1': 'midG1.hic',
            'anatelo': 'anatelo.hic',
        }
        
        # Load available phase files
        self.phase_paths = {}
        for phase, filename in self.phase_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                self.phase_paths[phase] = filepath
        
        if not self.phase_paths:
            raise ValueError(f"No .hic files found in {self.data_dir}")
        
        # Load mm10 chromosome sizes 
        # using chromosome sizes from mm10.chrom.sizes file
        self.chromosome_sizes = {
            "1": 195471971, "2": 182113224, "3": 160039680, "4": 156508116,
            "5": 151834684, "6": 149736546, "7": 145441459, "8": 129401213,
            "9": 124595110, "10": 130694993, "11": 122082543, "12": 120129022,
            "13": 120421639, "14": 124902244, "15": 104043685, "16": 98207768,
            "17": 94987271, "18": 90702639, "19": 61431566,
            "X": 171031299, "Y": 91744698,
        }

        # Load cluster_id == 3 loop coordinates for special handling from zhang et al paper
        self.cluster3_loop_regions = self._load_cluster3_loops(cluster3_loops_file)

        # theres no mappable data before 3MB of each chromosome.
        self.min_start_position = 3000000  # 3MB
        self.regions, self.holdout_regions = self._generate_regions()

    def _load_cluster3_loops(self, cluster3_loops_file: Optional[str] = None) -> List[Tuple[str, int, int, int, int]]:
        """
        Load cluster_id == 3 loop coordinates from file.

        Each line in the file should be a loop coordinate like:
        chr1:74920000-74940000_chr1:75090000-75110000

        Returns a list of tuples (chrom, anchor1_start, anchor1_end, anchor2_start, anchor2_end)
        representing the individual anchor coordinates for each loop.
        """
        cluster3_regions = []

        if cluster3_loops_file is None:
            # using excel file from zhang et al supp
            cluster3_loops_file = self.data_dir.parent.parent / "raw_data" / "zhang_4dn" / "cluster3_loop_coordinates.txt"
        else:
            cluster3_loops_file = Path(cluster3_loops_file)

        if not cluster3_loops_file.exists():
            print(f"Warning: cluster3 loops file not found at {cluster3_loops_file}")
            return cluster3_regions

        print(f"Loading cluster_id == 3 loop coordinates from: {cluster3_loops_file}")

        with open(cluster3_loops_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse loop coordinate: chr1:74920000-74940000_chr1:75090000-75110000
                parts = line.split('_')
                if len(parts) != 2:
                    continue

                # Parse both anchors
                anchor1 = parts[0].replace('chr', '')  # Remove 'chr' prefix
                anchor2 = parts[1].replace('chr', '')

                # Extract chromosome and coordinates for anchor1
                chrom1, coords1 = anchor1.split(':')
                start1, end1 = map(int, coords1.split('-'))

                # Extract chromosome and coordinates for anchor2
                chrom2, coords2 = anchor2.split(':')
                start2, end2 = map(int, coords2.split('-'))

                # Both anchors should be on the same chromosome
                if chrom1 != chrom2:
                    continue

                # Store individual anchor coordinates (ensure anchor1 is upstream)
                if start1 <= start2:
                    cluster3_regions.append((chrom1, start1, end1, start2, end2))
                else:
                    cluster3_regions.append((chrom1, start2, end2, start1, end1))

        print(f"Loaded {len(cluster3_regions)} cluster_id == 3 loops")
        return cluster3_regions

    def _generate_regions(self):
        """
        generate all 64x64 regions by sliding every 10 pixels along the diagonal.
        skip the beginning of chromosomes (no contacts here).
        exclude y chromosome.
        for now, holding out chr2 for testing.

        data augmentation: marks regions for flipping only if they contain both anchors of a cluster3 loop
        (so the loop is actually visible in the 64x64 Hi-C map).

        Returns:
            Tuple of (training_regions, holdout_regions):
            - training_regions: List of region strings for training
            - holdout_regions: List of region strings for holdout chromosome (empty if no holdout)
        """
        training_regions = []
        holdout_regions = []

        for chrom, size in self.chromosome_sizes.items():
            if chrom in ['Y']: # for now, skip Y chrom
                continue

            is_holdout = (self.hold_out_chromosome is not None and
                         str(chrom) == str(self.hold_out_chromosome))

            # Get cluster 3 anchors (for eval mostly, these are transient loops)
            chrom_cluster3_loops = [
                (a1_start, a1_end, a2_start, a2_end)
                for loop_chrom, a1_start, a1_end, a2_start, a2_end in self.cluster3_loop_regions
                if loop_chrom == chrom
            ]

            # start from 3MB
            start = self.min_start_position
            while start + self.region_size <= size:
                end = start + self.region_size
                region_str = f"{chrom}:{start}-{end}"

                if is_holdout:
                    holdout_regions.append(region_str)
                else:
                    training_regions.append(region_str)

                # Use 10 pixel step size
                start += self.step_bp

        if self.hold_out_chromosome:
            print(f"Holdout chromosome '{self.hold_out_chromosome}': {len(holdout_regions)} regions")
            print(f"Training regions: {len(training_regions)} regions")

        return training_regions, holdout_regions

    def _save_normalization_stat(self, region: str, phase: str, min_val: float, max_val: float):
        """
        saving min/max normalization statistics for a region and phase to file.

        Args:
            region: Region string (e.g., "1:10000000-10640000")
            phase: Phase name (e.g., "earlyG1", "midG1", etc.)
            min_val: min contact count in given region
            max_val: max count count in given region
        """
        if not self.save_normalization_stats:
            return

        # makes look up table for normalization stats
        with open(self.normalization_stats_file, 'a') as f:
            f.write(f"{region},{phase},{min_val},{max_val}\n")

    @staticmethod
    def load_normalization_stats(stats_file: Union[str, Path]) -> Dict[Tuple[str, str], Tuple[float, float]]:
        """
        when we normalize, we should save min and max contact counts so that we may reverse post eval.

        Args:
            stats_file: Path to normalization stats CSV file

        Returns:
            Dictionary mapping (region, phase) -> (min, max)
            Example: {("1:10000000-10640000", "earlyG1"): (0.0, 5.234), ...}
        """
        stats_dict = {}
        stats_file = Path(stats_file)

        if not stats_file.exists():
            raise FileNotFoundError(f"Normalization stats file not found: {stats_file}")

        with open(stats_file, 'r') as f:
            # Skip header
            next(f)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                region, phase, min_val, max_val = line.split(',')
                stats_dict[(region, phase)] = (float(min_val), float(max_val))

        return stats_dict

    def _reconstruct_matrix_from_upper_triangular(self, upper_tri_vec: np.ndarray) -> np.ndarray:
        """
        Reconstruct a full symmetric matrix from upper triangular vector. (this is for the flipping)

        Args:
            upper_tri_vec: 1D numpy array containing upper triangular elements (2080 for 64x64)

        Returns:
            2D symmetric numpy array (64x64)
        """
        n = self.image_size
        matrix = np.zeros((n, n), dtype=upper_tri_vec.dtype)

        idx = 0
        for i in range(n):
            for j in range(i, n):
                matrix[i, j] = upper_tri_vec[idx]
                matrix[j, i] = upper_tri_vec[idx]  # Symmetric
                idx += 1

        return matrix

    def _flip_hic_matrix(self, upper_tri_vec: np.ndarray) -> np.ndarray:
        """
        Flip HiC matrix (180-degree rotation) while maintaining symmetry and diagonal.

        The bottom-right corner becomes the top-left corner.

        Args:
            upper_tri_vec: 1D numpy array of upper triangular elements

        Returns:
            1D numpy array of upper triangular elements after flipping
        """
        # Reconstruct full matrix
        matrix = self._reconstruct_matrix_from_upper_triangular(upper_tri_vec)

        # Flip both horizontally and vertically (180-degree rotation)
        flipped_matrix = np.flip(matrix, axis=(0, 1))

        # Extract upper triangular again
        return self._extract_upper_triangular_vector(flipped_matrix)

    def _flip_chipseq_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Flip ChIP-seq signal (reverse the array).

        Args:
            signal: 1D numpy array of ChIP-seq values

        Returns:
            1D numpy array with reversed values
        """
        return np.flip(signal)

    def _extract_upper_triangular_vector(self, matrix: np.ndarray) -> np.ndarray:
        """
        Extract upper triangular portion of symmetric matrix as a flattened vector.
        Elements are stacked row by row (e.g., row 0, then row 1, etc.).
        
        Args:
            matrix: 2D numpy array (square, symmetric)
        
        Returns:
            1D numpy array containing upper triangular elements in row-major order
        """
        n = matrix.shape[0]
        upper_tri_vec = []
        
        # Stack rows: for row i, take elements from column i to n-1
        for i in range(n):
            upper_tri_vec.extend(matrix[i, i:].tolist())
        
        return np.array(upper_tri_vec, dtype=matrix.dtype)
    
    def _extract_region_matrix(self, hic_file: Path, region: str) -> np.ndarray:
        """
        Extract a square contact matrix for a given region and return upper triangular as vector.
        
        Args:
            hic_file: Path to .hic file
            region: Region string like "1:10000000-10500000" (0-based coordinates, no chr prefix)
        
        Returns:
            1D numpy array containing upper triangular elements flattened row by row
        """
        # Parse region
        chrom, coords = region.split(':')
        start, end = map(int, coords.split('-'))

        # Extract contact matrix using hicstraw
        # Use self.hic_data_type ("oe" for observed/expected or "observed" for raw counts)
        try:
            result = straw.straw(
                self.hic_data_type,
                self.normalization,
                str(hic_file),
                f"{chrom}:{start}:{end}",
                f"{chrom}:{start}:{end}",
                "BP",
                self.resolution
            )
        except Exception as e:
            raise ValueError(f"Error reading region {region} from {hic_file}: {e}")
        
        # Initialize matrix
        matrix = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Fill the matrix
        for record in result:
            x = record.binX
            y = record.binY
            count = record.counts
            
            # Convert genomic positions to bin indices (0-indexed)
            x_idx = int((x - start) // self.resolution)
            y_idx = int((y - start) // self.resolution)
            
            # Ensure indices are within bounds
            if 0 <= x_idx < self.image_size and 0 <= y_idx < self.image_size:
                matrix[x_idx, y_idx] = float(count)
                # matrix is symmetric, so also set the transpose
                if x_idx != y_idx:
                    matrix[y_idx, x_idx] = float(count)
        
        # return upper tri flattened vec (this is the bulk hic vec)
        return self._extract_upper_triangular_vector(matrix)

    def _extract_chipseq_signal(self, region: str, chipseq_bw=None) -> np.ndarray:
        """
        Extract ChIP-seq signal for a genomic region from bigWig data.
        Returns max-per-bin signal with log transformation applied.
        
        Args:
            region: example input region "1:10000000-10500000" means chr1 from start base pair 10000000 to end base pair 10500000
            chipseq_bw: Optional pyBigWig object to use (must be provided; if None, returns zeros)
        
        Returns:
            1d numpy array of shape (image_size,) with log-transformed max-per-bin ChIP-seq signal
        """
        if chipseq_bw is None:
            return np.zeros(self.image_size, dtype=np.float32)  # return zeros if chip-seq not available
        
        # Parse region
        chrom, coords = region.split(':')
        start, end = map(int, coords.split('-'))
        
        signal = np.zeros(self.image_size, dtype=np.float32)
        bin_size = self.resolution
    
        chrom_name = "chr" + chrom # see bedGraph chr{number} is the format
        try:
            for i in range(self.image_size):
                bin_start = start + i * bin_size
                bin_end = start + (i + 1) * bin_size
                # Use 'max' to get maximum value in bin 
                values = chipseq_bw.stats(chrom_name, bin_start, bin_end, type="max")
                max_val = values[0] if values[0] is not None else 0.0
                signal[i] = np.log1p(max_val) # oragami uses log1p(x).
            
            return signal
            
        except Exception as e: 
            raise ValueError(f"Error extracting ChIP-seq signal for region {region}: {e}")

    
    def __len__(self) -> int:
        """Return number of training regions (excludes holdout chromosome)."""
        return len(self.regions)
    
    def get_holdout_regions(self) -> List[str]:
        """
        Get regions from the holdout chromosome (for testing).
        
        Returns:
            List of region strings from holdout chromosome, empty list if no holdout specified
        """
        return self.holdout_regions if hasattr(self, 'holdout_regions') else []

    def __getitem__(self, idx: Union[int, str]) -> Dict[str, Union[str, np.ndarray]]:
        """
        Get a sample by index or region string.
        
        Args:
            idx: Integer index or region string like "1:10000000-10640000"
        
        Returns:
            Dictionary with keys: 'region', 'earlyG1', 'lateG1', 'midG1', 'anatelo', 
                                  'chip_seq_ctcf', 'chip_seq_rad21', 'chip_seq_hac'
            
            Hi-C phases (shape: (2080,) each):
                - Log-transformed: log1p(x) 
                - Normalized independently per sample (from each phase) to [-1, 1]:
                  (log1p(x) - phase_min) / (phase_max - phase_min) * 2 - 1
            
            ChIP-seq tracks (shape: (64,) each):
                - chip_seq_ctcf: CTCF ChIP-seq signal
                - chip_seq_rad21: RAD21 ChIP-seq signal (unused)
                - chip_seq_hac: H3k27ac ChIP-seq signal
                - chip_seq_h3k4me1: H3k4me1 ChIP-seq signal
                - chip_seq_h3k4me3: H3k4me3 ChIP-seq signal
                - Max signal per bin (10kb bins) with log1p transformation
                - No additional normalization (normalization handled by LayerNorm in model)
        """
        # Handle region string indexing
        if isinstance(idx, str):
            # Check both training and holdout regions, or allow user-specified region if size matches
            all_regions = self.regions
            if hasattr(self, 'holdout_regions'):
                all_regions = self.regions + self.holdout_regions
            if idx not in all_regions:
                # Allow user-specified region (e.g. --regions 2:18170000-18810000) if length matches
                try:
                    chrom, coords = idx.split(':')
                    start, end = map(int, coords.split('-'))
                    if end - start != self.region_size:
                        raise KeyError(
                            f"Region {idx} not in regions list and size {end - start} != region_size {self.region_size}"
                        )
                except ValueError:
                    raise KeyError(f"Region {idx} not found in regions list and could not parse as chrom:start-end")
            region = idx
        else:
            region = self.regions[idx]
        
        # build sample dict
        sample = {'region': region}
        do_flip = (self.augment > 0) and (np.random.rand() < (self.augment / 100.0))

        # chip seq helper + flip if this is on.
        def _chip_track(key: str) -> np.ndarray:
            track = self._extract_chipseq_signal(region, chipseq_bw=self.chipseq_files.get(key))
            if do_flip:
                track = self._flip_chipseq_signal(track)
            return track.astype(np.float32)

        # Load and normalize each phase INDEPENDENTLY using its own min/max
        for phase, filepath in self.phase_paths.items():
            vector = self._extract_region_matrix(filepath, region)
            
            if self.use_log_transform:
                vector = np.log1p(vector)

            # remove outliers using 99.9th percentile (from HiCDiff and DeepHiC papers)
            threshold = np.percentile(vector, 99.9)
            vector = np.where(vector > threshold, threshold, vector)
            
            # Use min/max normalization which is from HiCDiff and DeepHiC papers
            phase_min = vector.min()
            phase_max = vector.max()

            # save normalization if you want
            self._save_normalization_stat(region, phase, float(phase_min), float(phase_max))

            # Normalize to [-1, 1] using phase's own statistics
            if phase_max - phase_min < 1e-10:
                # All values are the same, map to 0
                normalized = np.zeros_like(vector, dtype=np.float32)
            else:
                # Linear transformation: (x - min) / (max - min) * 2 - 1
                normalized = (vector - phase_min) / (phase_max - phase_min) * 2.0 - 1.0

            if do_flip:
                normalized = self._flip_hic_matrix(normalized)
            sample[phase] = normalized.astype(np.float32)

        # Load ChIP-seq signals from multiple tracks (max per bin, log1p transformation)
        sample['chip_seq_ctcf']    = _chip_track('ctcf')
        sample['chip_seq_hac']     = _chip_track('hac')        # H3K27ac
        sample['chip_seq_h3k4me1'] = _chip_track('h3k4me1')
        sample['chip_seq_h3k4me3'] = _chip_track('h3k4me3')
        # for now no rad21.
        
        return sample
    
    def __iter__(self):
        """Make the dataloader iterable."""
        for i in range(len(self)):
            yield self[i]
    
    def get_regions(self) -> List[str]:
        """Get list of all available regions."""
        return self.regions.copy()
    
    def get_available_phases(self) -> List[str]:
        """Get list of available phase names."""
        return list(self.phase_paths.keys())
    
    def close(self):
        """Close all ChIP-seq bigWig files if open."""
        # Close all ChIP-seq files
        for key, bw in self.chipseq_files.items():
            if bw is not None:
                try:
                    bw.close()
                except:
                    pass
    
    def __del__(self):
        """Cleanup when object is deleted."""
        self.close()



# consider a similar previously used normalization for hic
# from the DeepHiC paper: 
# For Hi-C matrices in training, outliers 
# are set to the allowed maximum by setting
# the threshold be the 99.9-th percentile. 
# For example, 255 is about the average of
# 99.9-th percentiles for 10-kb Hi-C data, so 
# all values greater than 255 are set to 255 
# for 10-kb Hi-C data. Then all Hi-C matrices 
# are rescaled to values ranging from 0 to 1 by 
# min-max normalization [45] to ensure the training
# stability and efficiency. Besides, cutoff values
# for downsampled inputs of our model were 125,
# 100, 80, 50, and 25 for 1/10, 1/16, 1/25, 
# 1/50, and 1/100 downsampled ratios.

# procedure is used in DeepHiC and HicDiff papers
# https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007287
# HicDiff paper