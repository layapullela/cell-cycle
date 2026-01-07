"""
Simple DataLoader for cell cycle Hi-C contact maps.

Extracts 50x50 pixel (500kb) regions along the diagonal with sliding window.
Only stores upper triangular portion of symmetric Hi-C matrices.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Union
import hicstraw as straw


class CellCycleDataLoader:
    """
    Simple DataLoader for cell cycle Hi-C contact maps.
    
    Each sample contains:
    - region: str, e.g., "1:10000000-10500000"
    - earlyG1: numpy array (flattened upper triangular vector, shape: (1275,))
    - lateG1: numpy array (flattened upper triangular vector, shape: (1275,))
    - midG1: numpy array (flattened upper triangular vector, shape: (1275,))
    
    For a 50x50 matrix, upper triangular has 50*51/2 = 1275 elements.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        resolution: int = 10000,  # 10kb bins
        region_size: int = 500000,  # 500kb regions (50 pixels at 10kb resolution)
        normalization: str = "VC",  # or "NONE", "KR", etc.
    ):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing .hic files (earlyG1.hic, lateG1.hic, midG1.hic)
            resolution: Bin size in base pairs (default: 10000 for 10kb)
            region_size: Size of square region to extract in base pairs (default: 500000 for 500kb = 50 pixels)
            normalization: Normalization method for Hi-C data (NONE, VC, VC_SQRT, KR)
        """
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.region_size = region_size
        self.normalization = normalization
        self.image_size = region_size // resolution  # 50 pixels for 500kb at 10kb
        
        # Step size: sample every 10 pixels along the diagonal
        self.step_pixels = 10
        self.step_bp = self.step_pixels * resolution  # 100kb step
        
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
        self.regions = self._generate_regions()
    
    def _generate_regions(self) -> List[str]:
        """
        Generate all 50x50 regions by sliding every 10 pixels along the diagonal.
        
        Returns:
            List of region strings like "1:10000000-10500000" (0-based coordinates, no chr prefix)
        """
        regions = []
        
        for chrom, size in self.chromosome_sizes.items():
            # Start from 0, step by step_bp (100kb = 10 pixels at 10kb resolution)
            start = 0
            while start + self.region_size <= size:
                end = start + self.region_size
                regions.append(f"{chrom}:{start}-{end}")
                start += self.step_bp
        
        return regions
    
    def _extract_upper_triangular_vector(self, matrix: np.ndarray) -> np.ndarray:
        """
        Extract upper triangular portion of symmetric matrix as a flattened vector.
        Elements are stacked row by row (e.g., row 0, then row 1, etc.).
        
        Example:
            [[1, 1, 0],
             [0, 2, 1],
             [0, 0, 0]]
            -> [1, 1, 0, 2, 1, 0]
        
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
    
    def __len__(self) -> int:
        """Return number of regions."""
        return len(self.regions)
    
    def __getitem__(self, idx: Union[int, str]) -> Dict[str, Union[str, np.ndarray]]:
        """
        Get a sample by index or region string.
        
        Args:
            idx: Integer index or region string like "1:10000000-10500000"
        
        Returns:
            Dictionary with keys: 'region', 'earlyG1', 'lateG1', 'midG1'
            Each phase contains flattened upper triangular vector (shape: (1275,))
        """
        # Handle region string indexing
        if isinstance(idx, str):
            if idx not in self.regions:
                raise KeyError(f"Region {idx} not found in regions list")
            region = idx
        else:
            region = self.regions[idx]
        
        # Build sample dictionary
        sample = {'region': region}
        
        # Load matrices for each available phase (upper triangular as vector)
        for phase, filepath in self.phase_paths.items():
            vector = self._extract_region_matrix(filepath, region)
            sample[phase] = vector
        
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


def upper_tri_vec_to_matrix(vec: np.ndarray, n: int) -> np.ndarray:
    """
    Convert flattened upper triangular vector back to square matrix.
    
    Args:
        vec: 1D numpy array containing upper triangular elements (length: n*(n+1)/2)
        n: Size of the square matrix (e.g., 50 for 50x50)
    
    Returns:
        2D numpy array of shape (n, n) with upper triangular filled and lower triangular zeros
    
    Example:
        vec = [1, 1, 0, 2, 1, 0], n = 3
        -> [[1, 1, 0],
            [0, 2, 1],
            [0, 0, 0]]
    """
    matrix = np.zeros((n, n), dtype=vec.dtype)
    
    idx = 0
    for i in range(n):
        # For row i, fill columns from i to n-1
        num_elements = n - i
        matrix[i, i:] = vec[idx:idx + num_elements]
        idx += num_elements
    
    return matrix
