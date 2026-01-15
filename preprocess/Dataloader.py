"""
Simple DataLoader for cell cycle Hi-C contact maps.

Extracts 64x64 pixel (640kb) regions along the diagonal with sliding window.
Only stores upper triangular portion of symmetric Hi-C matrices.
Includes ChIP-seq signal for each region.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional
import hicstraw as straw
import pyBigWig


class CellCycleDataLoader:
    """
    Simple DataLoader for cell cycle Hi-C contact maps.
    
    Each sample contains:
    - region: str, e.g., "1:10000000-10640000"
    - earlyG1: numpy array (flattened upper triangular vector, shape: (2080,))
    - lateG1: numpy array (flattened upper triangular vector, shape: (2080,))
    - midG1: numpy array (flattened upper triangular vector, shape: (2080,))
    - chip_seq: numpy array (shape: (64,)) - ChIP-seq signal for the region
    * dimention calculation: 64 x 65 x 0.5 = 2080
    
    For a 64x64 matrix, upper triangular has 64*65/2 = 2080 elements.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        resolution: int = 10000,  # 10kb bins
        region_size: int = 640000,  # 640kb regions (64 pixels at 10kb resolution)
        normalization: str = "VC",  # or "NONE", "KR", etc. we use vanilla coverage.
        chipseq_file: Optional[str] = None,  # Path to ChIP-seq bigWig file
        hold_out_chromosome: Optional[str] = None,  # Chromosome to hold out for testing (e.g., "2")
    ):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing .hic files (earlyG1.hic, lateG1.hic, midG1.hic)
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
        self.image_size = region_size // resolution  # 50 pixels for 500kb at 10kb
        self.hold_out_chromosome = hold_out_chromosome  # Chromosome to hold out for testing
        
        # Step size: sample every 10 pixels along the diagonal
        self.step_pixels = 10
        self.step_bp = self.step_pixels * resolution  # 100kb step
        
        # Load ChIP-seq file
        self.chipseq_bw = None
        self.chipseq_chrom_stats = {}  # Per-chromosome mean and std
        if chipseq_file is None:
            chipseq_file = self.data_dir / "GSM946535_mm9_wgEncodePsuHistoneG1eH3k04me1ME0S129InputSig.bigWig"
        if chipseq_file.exists() if isinstance(chipseq_file, Path) else Path(chipseq_file).exists():
                self.chipseq_bw = pyBigWig.open(str(chipseq_file))
        
        # Phase files
        self.phase_files = {
            'earlyG1': 'earlyG1.hic',
            'lateG1': 'lateG1.hic',
            'midG1': 'midG1.hic',
        }
        
        # Load available phase files
        self.phase_paths = {}
        for phase, filename in self.phase_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                self.phase_paths[phase] = filepath
        
        if not self.phase_paths:
            raise ValueError(f"No .hic files found in {self.data_dir}")
        
        # mm39 chromosome sizes (no "chr" prefix in .hic files)
        self.chromosome_sizes = {
            "1": 195154279, "2": 181755017, "3": 159745316, "4": 156860686,
            "5": 151758149, "6": 149588044, "7": 144995196, "8": 130127694,
            "9": 124359700, "10": 130530862, "11": 121973369, "12": 120092757,
            "13": 120883175, "14": 125139656, "15": 104073951, "16": 98008968,
            "17": 95294699, "18": 90720763, "19": 61420004,
            "X": 169476592, "Y": 91455967,
        }
        
        # Generate regions by sliding along diagonal every 10 pixels
        # Skip first 3MB of each chromosome (typically no mappable data)
        self.min_start_position = 3000000  # 3MB
        self.regions, self.holdout_regions = self._generate_regions()
        
        # Compute ChIP-seq statistics PER CHROMOSOME (after regions are generated)
        if self.chipseq_bw is not None:
            print(f"Loaded ChIP-seq from: {chipseq_file}")
            self._compute_chipseq_per_chromosome_stats()
    
    def _generate_regions(self):
        """
        Generate all 64x64 regions by sliding every 10 pixels along the diagonal.
        Skips the beginning of chromosomes (typically no mappable Hi-C data at start of chromosome).
        Excludes sex chromosomes (X and Y).
        Separates holdout chromosome regions if specified.

        Returns:
            Tuple of (training_regions, holdout_regions):
            - training_regions: List of region strings for training
            - holdout_regions: List of region strings for holdout chromosome (empty if no holdout)
        """
        training_regions = []
        holdout_regions = []

        for chrom, size in self.chromosome_sizes.items():
            # Skip sex chromosomes
            if chrom in ['X', 'Y']:
                continue
            
            # Check if this is the holdout chromosome
            is_holdout = (self.hold_out_chromosome is not None and 
                         str(chrom) == str(self.hold_out_chromosome))
            
            # Start from min_start_position (skip beginning of chromosome)
            start = self.min_start_position
            while start + self.region_size <= size:
                end = start + self.region_size
                region_str = f"{chrom}:{start}-{end}"
                
                if is_holdout:
                    holdout_regions.append(region_str)
                else:
                    training_regions.append(region_str)
                
                start += self.step_bp

        if self.hold_out_chromosome:
            print(f"Holdout chromosome '{self.hold_out_chromosome}': {len(holdout_regions)} regions")
            print(f"Training regions: {len(training_regions)} regions")
        
        return training_regions, holdout_regions
    
    def _compute_chipseq_per_chromosome_stats(self):
        """
        Compute per-chromosome mean and std for ChIP-seq signal.
        Normalization: log1p → z-score (per chromosome) → clip ±3σ → scale to [-1, 1]
        
        Samples regions for efficiency (~100 regions per chromosome).
        """
        if self.chipseq_bw is None:
            return
        
        print("Computing per-chromosome ChIP-seq statistics (z-score)...")
        
        # Group regions by chromosome
        chrom_regions = {}
        for region in self.regions:
            chrom = region.split(':')[0]
            if chrom not in chrom_regions:
                chrom_regions[chrom] = []
            chrom_regions[chrom].append(region)
        
        # Compute mean/std for each chromosome
        for chrom, regions in chrom_regions.items():
            # Sample ~100 regions per chromosome for efficiency
            sample_step = max(1, len(regions) // 100)
            sampled = regions[::sample_step]
            
            chrom_signals = []
            for region in sampled:
                _, coords = region.split(':')
                start, end = map(int, coords.split('-'))
                
                # Try with and without 'chr' prefix
                chrom_names = [chrom, f"chr{chrom}"]
                
                for chrom_name in chrom_names:
                    try:
                        signal = np.zeros(self.image_size, dtype=np.float32)
                        
                        for i in range(self.image_size):
                            bin_start = start + i * self.resolution
                            bin_end = start + (i + 1) * self.resolution
                            values = self.chipseq_bw.stats(chrom_name, bin_start, bin_end, type="mean")
                            signal[i] = values[0] if values[0] is not None else 0.0
                        
                        # Log transformation
                        signal = np.log1p(signal)
                        chrom_signals.extend(signal.tolist())
                        break
                    except:
                        continue
            
            # Compute and store per-chromosome statistics
            if chrom_signals:
                self.chipseq_chrom_stats[chrom] = {
                    'mean': float(np.mean(chrom_signals)),
                    'std': float(np.std(chrom_signals))
                }
                print(f"  Chr {chrom}: mean={self.chipseq_chrom_stats[chrom]['mean']:.4f}, "
                      f"std={self.chipseq_chrom_stats[chrom]['std']:.4f}")
            else:
                # Fallback
                self.chipseq_chrom_stats[chrom] = {'mean': 0.0, 'std': 1.0}
                print(f"  Chr {chrom}: Using default stats (no data)")
        
        print(f"Computed statistics for {len(self.chipseq_chrom_stats)} chromosomes")
    
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
        try:
            result = straw.straw(
                "observed",
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
                # Matrix is symmetric, so also set the transpose
                if x_idx != y_idx:
                    matrix[y_idx, x_idx] = float(count)
        
        # Return upper triangular portion as flattened vector
        return self._extract_upper_triangular_vector(matrix)
    
    def _extract_chipseq_signal(self, region: str) -> np.ndarray:
        """
        Extract ChIP-seq signal for a genomic region.
        Returns log-transformed signal (normalization happens in __getitem__ using per-chromosome stats).
        
        Args:
            region: Region string like "1:10000000-10500000"
        
        Returns:
            1D numpy array of shape (image_size,) with log-transformed ChIP-seq signal
        """
        if self.chipseq_bw is None:
            # Return zeros if ChIP-seq not available
            return np.zeros(self.image_size, dtype=np.float32)
        
        # Parse region
        chrom, coords = region.split(':')
        start, end = map(int, coords.split('-'))
        
        try:
            # Try with and without 'chr' prefix
            chrom_names = [chrom, f"chr{chrom}"]
            signal = None
            
            for chrom_name in chrom_names:
                try:
                    # Get average signal per bin
                    bin_size = self.resolution
                    signal = np.zeros(self.image_size, dtype=np.float32)
                    
                    for i in range(self.image_size):
                        bin_start = start + i * bin_size
                        bin_end = start + (i + 1) * bin_size
                        values = self.chipseq_bw.stats(chrom_name, bin_start, bin_end, type="mean")
                        signal[i] = values[0] if values[0] is not None else 0.0
                    
                    # Apply log transformation (z-score normalization in __getitem__)
                    signal = np.log1p(signal)
                    
                    return signal
                except:
                    continue
            
            # If all attempts failed, return zeros
            return np.zeros(self.image_size, dtype=np.float32)
            
        except Exception as e:
            # Return zeros on error
            return np.zeros(self.image_size, dtype=np.float32)
    
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
            Dictionary with keys: 'region', 'earlyG1', 'lateG1', 'midG1', 'chip_seq'
            
            Hi-C phases (shape: (2080,) each):
                - Log-transformed: log1p(x)
                - Normalized INDEPENDENTLY per phase to [-1, 1]:
                  (log1p(x) - phase_min) / (phase_max - phase_min) * 2 - 1
            
            ChIP-seq (shape: (64,)):
                - Log-transformed: log1p(x)
                - Per-chromosome z-score: (x - chrom_mean) / chrom_std
                - Clipped to ±3 sigma
                - Rescaled to [-1, 1]: x / 3.0
        """
        # Handle region string indexing
        if isinstance(idx, str):
            # Check both training and holdout regions
            all_regions = self.regions
            if hasattr(self, 'holdout_regions'):
                all_regions = self.regions + self.holdout_regions
            if idx not in all_regions:
                raise KeyError(f"Region {idx} not found in regions list")
            region = idx
        else:
            region = self.regions[idx]
        
        # Build sample dictionary
        sample = {'region': region}
        
        # Load and normalize each phase INDEPENDENTLY using its own min/max
        for phase, filepath in self.phase_paths.items():
            vector = self._extract_region_matrix(filepath, region)
            
            # Apply log transformation (standard for Hi-C data)
            # log1p = log(1 + x) handles zeros gracefully
            vector = np.log1p(vector)
            
            # Find min/max for THIS phase
            phase_min = vector.min()
            phase_max = vector.max()
            
            # Normalize to [-1, 1] using phase's own statistics
            if phase_max - phase_min < 1e-10:
                # All values are the same, map to 0
                normalized = np.zeros_like(vector, dtype=np.float32)
            else:
                # Linear transformation: (x - min) / (max - min) * 2 - 1
                normalized = (vector - phase_min) / (phase_max - phase_min) * 2.0 - 1.0
            
            sample[phase] = normalized.astype(np.float32)
        
        # Load ChIP-seq signal and normalize to [-1, 1]
        chip_seq = self._extract_chipseq_signal(region)  # Log-transformed
        
        # Get chromosome for per-chromosome z-score normalization
        chrom = region.split(':')[0]
        
        # Per-chromosome z-score normalization with clipping
        if chrom in self.chipseq_chrom_stats and self.chipseq_chrom_stats[chrom]['std'] > 0:
            # Step 1: z-score using per-chromosome mean/std
            mean = self.chipseq_chrom_stats[chrom]['mean']
            std = self.chipseq_chrom_stats[chrom]['std']
            chip_seq = (chip_seq - mean) / std
            
            # Step 2: Clip to ±3 sigma
            chip_seq = np.clip(chip_seq, -3.0, 3.0)
            
            # Step 3: Rescale to [-1, 1]
            chip_seq = chip_seq / 3.0  # Maps [-3, 3] → [-1, 1]
        else:
            # Fallback: map to zeros
            chip_seq = np.zeros_like(chip_seq, dtype=np.float32)
        
        sample['chip_seq'] = chip_seq.astype(np.float32)
        
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
        """Close ChIP-seq bigWig file if open."""
        if self.chipseq_bw is not None:
            self.chipseq_bw.close()
    
    def __del__(self):
        """Cleanup when object is deleted."""
        self.close()