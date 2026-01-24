"""
Simple DataLoader for cell cycle Hi-C contact maps.

Extracts 64x64 pixel (640kb) regions along the diagonal with sliding window.
Only stores upper triangular portion of symmetric Hi-C matrices.
Includes ChIP-seq signal for each region.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import hicstraw as straw


class CellCycleDataLoader:
    """
    Simple DataLoader for cell cycle Hi-C contact maps.
    
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
        normalization: str = "VC",  # or "NONE", "KR", etc. we use vanilla coverage. #TODO: VC or KR? epiphany paper uses KR
        chipseq_file: Optional[str] = None,  # Path to ChIP-seq bedGraph file
        hold_out_chromosome: Optional[str] = None,  # Chromosome to hold out for testing (e.g., "2")
    ):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing .hic files (earlyG1.hic, lateG1.hic, midG1.hic, anatelo.hic)
            resolution: Bin size in base pairs (default: 10000 for 10kb)
            region_size: Size of square region to extract in base pairs (default: 640000 for 640kb = 64 pixels)
            normalization: Normalization method for Hi-C data (NONE, VC, VC_SQRT, KR)
            chipseq_file: Optional path to ChIP-seq bedGraph file (if None, looks for default file)
            hold_out_chromosome: Optional chromosome to exclude from training (e.g., "2" for chr2)
        """
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.region_size = region_size
        self.normalization = normalization
        self.image_size = region_size // resolution  # 64 pixels for 640kb at 10kb
        self.hold_out_chromosome = hold_out_chromosome  # Chromosome to hold out for testing
        
        # Step size: sample every 10 pixels along the diagonal
        self.step_pixels = 10
        self.step_bp = self.step_pixels * resolution  # 100kb step
        
        # Load ChIP-seq bedGraph file
        # Store as dict: chrom -> list of (start, end, value) tuples
        self.chipseq_data: Dict[str, List[Tuple[int, int, float]]] = {}
        if chipseq_file is None:
            chipseq_file = self.data_dir / "GSE129997_CTCF_asyn.bedGraph"
        chipseq_path = Path(chipseq_file) if not isinstance(chipseq_file, Path) else chipseq_file
        if chipseq_path.exists():
            self._load_bedgraph(chipseq_path)
        
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
        
        # Load mm9 chromosome sizes (no "chr" prefix in .hic files)
        chrom_sizes_file = self.data_dir.parent / "mm9.chrom.sizes"
        if not chrom_sizes_file.exists():
            # Fallback: try in raw_data directory
            chrom_sizes_file = self.data_dir.parent.parent / "raw_data" / "mm9.chrom.sizes"
        
        # using chromosome sizes from mm9.chrom.sizes file
        self.chromosome_sizes = {
            "1": 197195432, "2": 181748087, "3": 159599783, "4": 155630120,
            "5": 152537259, "6": 149517037, "7": 152524553, "8": 131738871,
            "9": 124076172, "10": 129993255, "11": 121843856, "12": 121257530,
            "13": 120284312, "14": 125194864, "15": 103494974, "16": 98319150,
            "17": 95272651, "18": 90772031, "19": 61342430,
            "X": 166650296, "Y": 15902555,
        }
        
        # Generate regions by sliding along diagonal every 10 pixels
        # Skip first 3MB of each chromosome (typically no mappable data)
        self.min_start_position = 3000000  # 3MB
        self.regions, self.holdout_regions = self._generate_regions()
        
        # we normalize chip-seq signal by layer norm in the model.
    
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
            if chrom in ['X', 'Y']: # for now, skip sex chromosomes
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
    
    # _compute_chipseq_per_chromosome_stats() removed - no longer needed
    # new will normalize chip-seq signal by layer norm in the model.
    
    # convert bigwig to bedgraph using bigWigToBedGraph ucsc tool
    def _load_bedgraph(self, bedgraph_path: Path):
        """
        Load bedGraph file into memory.
        Format: chr1\tstart\tend\tvalue
        Stores as dict: chrom -> list of (start, end, value) tuples
        """
        print(f"Loading ChIP-seq bedGraph from {bedgraph_path}...")
        self.chipseq_data = {}
        
        with open(bedgraph_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) < 4:
                    continue
                
                chrom = parts[0]  # it will be like "chr1"
                start = int(parts[1])
                end = int(parts[2])
                value = float(parts[3])
                
                if chrom not in self.chipseq_data:
                    self.chipseq_data[chrom] = []
                self.chipseq_data[chrom].append((start, end, value))
        
        # Sort intervals by start position for each chromosome
        for chrom in self.chipseq_data:
            self.chipseq_data[chrom].sort(key=lambda x: x[0])
        
        total_intervals = sum(len(intervals) for intervals in self.chipseq_data.values())
        print(f"  Loaded {total_intervals:,} intervals across {len(self.chipseq_data)} chromosomes")
    
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
        Extract ChIP-seq signal for a genomic region from bedGraph data.
        Returns max-per-bin signal (no log transformation applied).
        
        Args:
            region: example input region "1:10000000-10500000" means chr1 from start base pair 10000000 to end base pair 10500000
        
        Returns:
            1d numpy array of shape (image_size,) with max-per-bin ChIP-seq signal
        """
        if not self.chipseq_data:
            return np.zeros(self.image_size, dtype=np.float32)  # return zeros if chip-seq not available
        
        # Parse region
        chrom, coords = region.split(':')
        start, end = map(int, coords.split('-'))
        
        # bedGraph files use chr{#} format
        chrom_name = f"chr{chrom}"
        
        signal = np.zeros(self.image_size, dtype=np.float32)
        bin_size = self.resolution
        
        # Get chromosome data
        if chrom_name not in self.chipseq_data:
            return signal  # No data for this chromosome
        
        intervals = self.chipseq_data[chrom_name]
        
        # For each bin, find max value from overlapping intervals
        # Since intervals are sorted by start, we can break early
        interval_idx = 0
        for i in range(self.image_size):
            bin_start = start + i * bin_size
            bin_end = start + (i + 1) * bin_size
            
            # Find all intervals that overlap with this bin
            max_val = 0.0
            # Start from where we left off (intervals are sorted)
            for j in range(interval_idx, len(intervals)):
                interval_start, interval_end, value = intervals[j]
                
                # If interval starts after bin ends, we're done with this bin
                if interval_start >= bin_end:
                    break
                
                # If interval ends before bin starts, skip it (but update start index)
                if interval_end <= bin_start:
                    interval_idx = j + 1
                    continue
                
                # Interval overlaps with bin
                if interval_end > bin_start and interval_start < bin_end:
                    max_val = max(max_val, value)
            
            signal[i] = max_val
        
        return signal
    
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
            Dictionary with keys: 'region', 'earlyG1', 'lateG1', 'midG1', 'anatelo', 'chip_seq'
            
            Hi-C phases (shape: (2080,) each):
                - Log-transformed: log1p(x)
                - Normalized INDEPENDENTLY per phase to [-1, 1]:
                  (log1p(x) - phase_min) / (phase_max - phase_min) * 2 - 1
            
            ChIP-seq (shape: (64,)):
                - Max signal per bin (10kb bins)
                - No normalization (normalization handled by LayerNorm in model)
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
        
        # Load ChIP-seq signal (max per bin, no additioanl normalization)
        chip_seq = self._extract_chipseq_signal(region)  # Max per bin, no log transformation
        
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
        """Cleanup ChIP-seq data (no-op for bedGraph, kept for compatibility)."""
        pass
    
    def __del__(self):
        """Cleanup when object is deleted."""
        self.close()