"""
DataLoader for cell cycle Hi-C contact maps.

This module provides a DataLoader class that:
- Reads .hic files binned at 10kb resolution
- Extracts 5Mb square regions along the main diagonal
- Organizes samples with keys: region, earlyG1, lateG1, midG1, prometa, anaTelo
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
import hicstraw as straw
import matplotlib.pyplot as plt


class CellCycleDataLoader:
    """
    DataLoader for cell cycle Hi-C contact maps.
    
    Each sample contains:
    - region: str, e.g., "chr1:10000000-15000000"
    - earlyG1: numpy array (500x500 contact matrix)
    - lateG1: numpy array (500x500 contact matrix)
    - midG1: numpy array (500x500 contact matrix)
    - prometa: numpy array (optional, 500x500 contact matrix)
    - anaTelo: numpy array (optional, 500x500 contact matrix)
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        resolution: int = 10000,  # 10kb bins
        region_size: int = 5000000,  # 5Mb regions
        normalization: str = "NONE",  # or "VC", "VC_SQRT", "KR"
        chromosome_sizes: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing .hic files (earlyG1.hic, lateG1.hic, etc.)
            resolution: Bin size in base pairs (default: 10000 for 10kb)
            region_size: Size of square region to extract in base pairs (default: 5000000 for 5Mb)
            normalization: Normalization method for Hi-C data (NONE, VC, VC_SQRT, KR)
            chromosome_sizes: Optional dict mapping chromosome names to sizes.
                             If None, will try to infer from .hic files.
        """
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.region_size = region_size
        self.normalization = normalization
        
        # Number of bins per region
        self.bins_per_region = region_size // resolution  # 500 bins for 5Mb at 10kb
        
        # Expected phase files
        self.phase_files = {
            'earlyG1': 'earlyG1.hic',
            'lateG1': 'lateG1.hic',
            'midG1': 'midG1.hic',
            'prometa': 'prometa.hic',  # optional
            'anaTelo': 'anaTelo.hic',  # optional
        }
        
        # Load available phase files
        self.phase_paths = {}
        for phase, filename in self.phase_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                self.phase_paths[phase] = filepath
            else:
                warnings.warn(f"Phase file not found: {filepath}. Phase '{phase}' will be skipped.")
        
        if not self.phase_paths:
            raise ValueError(f"No .hic files found in {self.data_dir}")
        
        # Get chromosome information from first available file
        self.chromosomes = self._get_chromosomes()
        
        # Generate all possible regions
        self.regions = self._generate_regions(chromosome_sizes)
        
        # Cache for loaded matrices
        self._matrix_cache: Dict[Tuple[str, str], np.ndarray] = {}
    
    def _get_chromosomes(self) -> Dict[str, int]:
        """Extract chromosome information from .hic file header."""
        # Use first available phase file to get chromosome info
        first_file = next(iter(self.phase_paths.values()))
        
        try:
            # Try to get chromosome sizes from the .hic file
            # Note: straw doesn't directly provide this, so we'll need to extract regions dynamically
            # For now, we'll use a common approach: try common chromosomes
            chromosomes = {}
            
            # Mouse chromosomes in hic file: '1', '2', '3', ..., '19', 'X', 'Y' (no "chr" prefix)
            common_chroms = [str(i) for i in range(1, 20)] + ["X", "Y"]
            
            # Try to determine chromosome sizes by querying the file
            # We'll do a test query to see which chromosomes exist
            for chrom in common_chroms:
                try:
                    # Try to get a small region to check if chromosome exists
                    # Usage: straw [observed/oe/expected] <NONE/VC/VC_SQRT/KR> <hicFile> <chr1>[:x1:x2] <chr2>[:y1:y2] <BP/FRAG> <binsize>
                    # Coordinates are 0-based
                    result = straw.straw(
                        "observed",                  # data type
                        self.normalization,          # normalization
                        str(first_file),             # file
                        f"{chrom}:1000000:1500000",  # chr:start:end (0-based)
                        f"{chrom}:1000000:1500000",  # chr:start:end (0-based)
                        "BP",                        # unit
                        self.resolution              # binsize
                    )
                    if result and len(result) > 0:
                        # Estimate size - we'll refine this when generating regions
                        chromosomes[chrom] = None  # Placeholder, will determine dynamically
                except:
                    continue
            
            return chromosomes if chromosomes else {"1": None}  # Default fallback
            
        except Exception as e:
            warnings.warn(f"Could not determine chromosomes from .hic file: {e}. Using default.")
            return {"1": None}
    
    def _generate_regions(
        self, 
        chromosome_sizes: Optional[Dict[str, int]] = None
    ) -> List[str]:
        """
        Generate all possible 5Mb regions along the diagonal.
        
        Returns:
            List of region strings like "1:10000000-15000000" (0-based coordinates, no chr prefix)
        """
        regions = []
        
        # If chromosome sizes provided, use them
        if chromosome_sizes:
            chrom_sizes = chromosome_sizes
        else:
            # Try to determine chromosome sizes dynamically
            chrom_sizes = self._estimate_chromosome_sizes()
        
        # Generate regions (non-overlapping by default, can be modified for overlap)
        # Use step_size = region_size for non-overlapping regions
        # Use step_size = region_size // 2 for 50% overlap
        step_size = self.region_size  # Non-overlapping regions
        
        for chrom, size in chrom_sizes.items():
            if size is None:
                continue
            
            # Start from 0 (0-based coordinates), extract regions until we run out of chromosome
            start = 0
            while start + self.region_size <= size:
                end = start + self.region_size
                # Region format: "chrom:start-end" (0-based, inclusive start, exclusive end for straw)
                region_str = f"{chrom}:{start}-{end}"
                regions.append(region_str)
                start += step_size
        
        return regions
    
    def _estimate_chromosome_sizes(self) -> Dict[str, int]:
        """
        Estimate chromosome sizes by querying .hic files.
        This is a heuristic approach - ideally chromosome sizes should be provided.
        """
        chrom_sizes = {}
        first_file = next(iter(self.phase_paths.values()))
        
        # mm39 chromosome sizes (matching the actual hic file format: '1', '2', ..., 'X', 'Y')
        # Based on mm39.chrom.sizes and user's chromosome list
        default_sizes = {
            "1": 195154279,
            "2": 181755017,
            "3": 159745316,
            "4": 156860686,
            "5": 151758149,
            "6": 149588044,
            "7": 144995196,
            "8": 130127694,
            "9": 124359700,
            "10": 130530862,
            "11": 121973369,
            "12": 120092757,
            "13": 120883175,
            "14": 125139656,
            "15": 104073951,
            "16": 98008968,
            "17": 95294699,
            "18": 90720763,
            "19": 61420004,
            "X": 169476592,
            "Y": 91455967,
        }
        
        # Try to verify which chromosomes exist by testing queries
        for chrom, default_size in default_sizes.items():
            try:
                # Test query - coordinates are 0-based
                # Usage: straw [observed/oe/expected] <NONE/VC/VC_SQRT/KR> <hicFile> <chr1>[:x1:x2] <chr2>[:y1:y2] <BP/FRAG> <binsize>
                result = straw.straw(
                    "observed",                                    # data type
                    self.normalization,                            # normalization
                    str(first_file),                               # file
                    f"{chrom}:0:{self.resolution * 10}",           # chr:start:end (0-based)
                    f"{chrom}:0:{self.resolution * 10}",           # chr:start:end (0-based)
                    "BP",                                          # unit
                    self.resolution                                # binsize
                )
                if result and len(result) > 0:
                    chrom_sizes[chrom] = default_size
            except:
                continue
        
        return chrom_sizes if chrom_sizes else {"1": 195154279}  # Default fallback
    
    def _extract_region_matrix(
        self,
        hic_file: Path,
        region: str,
    ) -> np.ndarray:
        """
        Extract a square contact matrix for a given region.
        
        Args:
            hic_file: Path to .hic file
            region: Region string like "1:10000000-15000000" (0-based coordinates, no chr prefix)
        
        Returns:
            2D numpy array of shape (bins_per_region, bins_per_region)
        """
        # Parse region
        chrom, coords = region.split(':')
        start, end = map(int, coords.split('-'))
        
        # Extract contact matrix using hicstraw
        # Usage: straw [observed/oe/expected] <NONE/VC/VC_SQRT/KR> <hicFile> <chr1>[:x1:x2] <chr2>[:y1:y2] <BP/FRAG> <binsize>
        # So the order is: data_type, normalization, file, chr1, chr2, unit, binsize
        try:
            result = straw.straw(
                "observed",                   # arg0: str (data type: observed/oe/expected)
                self.normalization,           # arg1: str (normalization: NONE/VC/VC_SQRT/KR)
                str(hic_file),                # arg2: str (file path)
                f"{chrom}:{start}:{end}",     # arg3: str (chr1:start:end)
                f"{chrom}:{start}:{end}",     # arg4: str (chr2:start:end)
                "BP",                         # arg5: str (unit: BP or FRAG)
                self.resolution               # arg6: int (binsize)
            )
        except Exception as e:
            raise ValueError(f"Error reading region {region} from {hic_file}: {e}")
        
        # Convert to dense matrix
        bins = self.bins_per_region
        matrix = np.zeros((bins, bins), dtype=np.float32)
        
        # hicstraw.straw returns a list of contactRecord objects
        # Each contactRecord has: binX, binY, counts attributes
        if not isinstance(result, list):
            raise ValueError(f"Unexpected return format from straw: {type(result)}, expected list")
        
        # Fill the matrix
        for record in result:
            x = record.binX
            y = record.binY
            count = record.counts
            # Convert genomic positions to bin indices (0-indexed)
            x_idx = int((x - start) // self.resolution)
            y_idx = int((y - start) // self.resolution)
            
            # Ensure indices are within bounds
            if 0 <= x_idx < bins and 0 <= y_idx < bins:
                matrix[x_idx, y_idx] = float(count)
                # Matrix is symmetric
                if x_idx != y_idx:
                    matrix[y_idx, x_idx] = float(count)
        
        return matrix
    
    def __len__(self) -> int:
        """Return number of regions."""
        return len(self.regions)
    
    def __getitem__(self, idx: Union[int, str]) -> Dict[str, Union[str, np.ndarray]]:
        """
        Get a sample by index or region string.
        
        Args:
            idx: Integer index or region string like "1:10000000-15000000" (0-based, no chr prefix)
        
        Returns:
            Dictionary with keys: 'region', 'earlyG1', 'lateG1', 'midG1', etc.
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
        
        # Load matrices for each available phase
        for phase, filepath in self.phase_paths.items():
            cache_key = (phase, region)
            
            # Check cache
            if cache_key in self._matrix_cache:
                matrix = self._matrix_cache[cache_key]
            else:
                matrix = self._extract_region_matrix(filepath, region)
                self._matrix_cache[cache_key] = matrix
            
            sample[phase] = matrix
        
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
    
    def clear_cache(self):
        """Clear the matrix cache."""
        self._matrix_cache.clear()


def example_usage():
    """Example of how to use the DataLoader."""
    # Initialize dataloader
    data_dir = "/nfs/turbo/umms-minjilab/lpullela/cell-cycle/raw_data/zhang_4dn"
    
    # Optional: provide chromosome sizes for more accurate region generation
    # chromosome_sizes = {"1": 195154279, "2": 181755017, ..., "X": 169476592, "Y": 91455967}
    
    loader = CellCycleDataLoader(
        data_dir=data_dir,
        resolution=10000,  # 10kb bins
        region_size=1000000,  # # for now we will do 1 Mb. 5Mb regions
        normalization="NONE",  # or "VC", "KR", etc.
        # chromosome_sizes=chromosome_sizes,  # optional
    )
    
    print(f"Number of regions: {len(loader)}")
    print(f"Available phases: {loader.get_available_phases()}")
    print(f"First 5 regions: {loader.get_regions()[:5]}")
    
    # Get a sample by index
    sample = loader[0]
    print(f"\nSample region: {sample['region']}")
    print(f"EarlyG1 matrix shape: {sample['earlyG1'].shape}")
    print(f"LateG1 matrix shape: {sample['lateG1'].shape}")
    print(f"MidG1 matrix shape: {sample['midG1'].shape}")
    
    # Get a sample by region string (0-based coordinates, no chr prefix)
    region = "1:10000000-15000000"
    if region in loader.get_regions():
        sample = loader[region]
        print(f"\nSample from region {region}:")
        print(f"Matrix shapes: {[sample[k].shape for k in loader.get_available_phases()]}")
    
    # Iterate through samples
    print("\nIterating through first 3 samples:")
    for i, sample in enumerate(loader):
        if i >= 3:
            break
        print(f"  Sample {i}: {sample['region']}")
    
    # Plot contact matrices for one sample
    print("\nPlotting contact matrices for first sample...")
    sample = loader[1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    phases = ['earlyG1', 'midG1', 'lateG1']
    for idx, phase in enumerate(phases):
        if phase in sample:
            matrix = sample[phase]
            # Use log scale for better visualization
            im = axes[idx].imshow(np.log1p(matrix), cmap='YlOrRd', aspect='auto')
            axes[idx].set_title(f'{phase}\n{sample["region"]}')
            axes[idx].set_xlabel('Bin')
            axes[idx].set_ylabel('Bin')
            plt.colorbar(im, ax=axes[idx], label='log(contacts + 1)')
    
    plt.tight_layout()
    plt.savefig('contact_matrices_example.png', dpi=150, bbox_inches='tight')
    print("  Saved plot to 'contact_matrices_example.png'")


if __name__ == "__main__":
    example_usage()
