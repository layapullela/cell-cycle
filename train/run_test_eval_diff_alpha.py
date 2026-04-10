"""
Run test set evaluation on a trained SR3 model, focusing on cluster 3 loops in chromosome 2.

Evaluates one 64x64 Hi-C map per cluster 3 loop, showing only maps that contain
both anchors of the loop. Uses cross-phase quantile normalization with fixed
color scale [0, 40] for visualization, allowing direct comparison across all maps.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import Tuple

# Add preprocess dir to path
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocess"))
from Dataloader import CellCycleDataLoader

# Import model and utilities from alpha training script
from train_diffusion_alpha import (
    SR3UNet,
    NoiseEmbedding,
    T, N,
    RESOLUTION_BP,
    REGION_SIZE_BP,
)
from inference import run_inference_and_visualize, Inference


def load_checkpoint(checkpoint_path, device):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: torch device
    
    Returns:
        model: Loaded SR3UNet model in eval mode
        checkpoint: Full checkpoint dict with metadata
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Recreate model architecture (must match training!)
    d_t = 256  # Time embedding dimension - must match training
    noise_embed_module = NoiseEmbedding(d_t, max_value=1000)
    
    model = SR3UNet(
        n=N,
        noise_embed_module=noise_embed_module,
        base_ch=64            # Base channels for U-Net (64 -> 128 -> 256 -> 512)
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch'] + 1}")
    print(f"  Loss: {checkpoint['loss']:.6f}")
    print(f"  Global step: {checkpoint['global_step']}")
    
    return model, checkpoint


def _parse_interval(interval_str: str) -> Tuple[int, int]:
    """
    Parse a genomic interval like "44700000-45300000".
    """
    start, end = map(int, interval_str.split('-'))
    if end <= start:
        raise ValueError(
            f"Invalid interval '{interval_str}': end must be greater than start."
        )
    return start, end


def normalize_region(region_str: str) -> str:
    """
    Normalize region strings to the loader's canonical 2D format.

    Supported inputs:
      "chrom:start-end"                           -> diagonal crop
      "chrom:row_start-row_end:col_start-col_end" -> canonical 2D crop
      "chrom:row_start-row_end,col_start-col_end" -> shorthand 2D crop
    """
    parts = region_str.split(':')
    if len(parts) == 2:
        chrom, coords = parts
        if ',' in coords:
            row_coords, col_coords = coords.split(',', 1)
        else:
            row_coords = coords
            col_coords = coords
    elif len(parts) == 3:
        chrom, row_coords, col_coords = parts
    else:
        raise ValueError(
            "Region must be one of: 'chrom:start-end', "
            "'chrom:row_start-row_end:col_start-col_end', or "
            "'chrom:row_start-row_end,col_start-col_end'. "
            f"Received: '{region_str}'"
        )

    row_start, row_end = _parse_interval(row_coords)
    col_start, col_end = _parse_interval(col_coords)
    return f"{chrom}:{row_start}-{row_end}:{col_start}-{col_end}"


def parse_region(region_str: str) -> Tuple[str, int, int, int, int]:
    """
    Parse a region string into row/column genomic ranges.

    Returns:
        Tuple of (chrom, row_start, row_end, col_start, col_end)
    """
    chrom, row_coords, col_coords = normalize_region(region_str).split(':')
    row_start, row_end = _parse_interval(row_coords)
    col_start, col_end = _parse_interval(col_coords)
    return chrom, row_start, row_end, col_start, col_end


def region_in_range(region_str, target_start, target_end):
    """
    Check if a region overlaps with target range.
    
    Args:
        region_str: Region string like "2:44700000-45300000"
        target_start: Start of target range (bp)
        target_end: End of target range (bp)
    
    Returns:
        True if region overlaps with target range
    """
    _, row_start, row_end, col_start, col_end = parse_region(region_str)
    region_start = min(row_start, col_start)
    region_end = max(row_end, col_end)
    return region_start < target_end and region_end > target_start


def get_cluster3_regions_chr2(all_chr2_regions, data_dir):
    """
    Get one region per cluster 3 loop on chromosome 2.

    Args:
        all_chr2_regions: List of all chromosome 2 region strings
        data_dir: Path to data directory

    Returns:
        List of tuples (region_string, loop_info) - one per cluster 3 loop
    """
    # Load cluster 3 loops from file
    #cluster3_file = Path(data_dir).parent / "cluster3_loop_coordinates.txt"

    cluster3_file = "/nfs/turbo/umms-minjilab/lpullela/cell-cycle/raw_data/zhang_4dn/cluster3_loop_coordinates.txt"
    cluster3_file = Path(cluster3_file)

    if not cluster3_file.exists():
        print(f"Warning: cluster3 loops file not found at {cluster3_file}")
        return []

    print(f"Loading cluster 3 loops from: {cluster3_file}")

    cluster3_loops_chr2 = []
    with open(cluster3_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse: chr2:44920000-44940000_chr2:45090000-45110000
            parts = line.split('_')
            if len(parts) != 2:
                continue

            anchor1 = parts[0].replace('chr', '')
            anchor2 = parts[1].replace('chr', '')

            chrom1, coords1 = anchor1.split(':')
            start1, end1 = map(int, coords1.split('-'))

            chrom2, coords2 = anchor2.split(':')
            start2, end2 = map(int, coords2.split('-'))

            # Only chromosome 2 loops
            if chrom1 != '2' or chrom2 != '2':
                continue

            # Ensure anchor1 is upstream
            if start1 <= start2:
                cluster3_loops_chr2.append((start1, end1, start2, end2, line))
            else:
                cluster3_loops_chr2.append((start2, end2, start1, end1, line))

    print(f"Found {len(cluster3_loops_chr2)} cluster 3 loops on chromosome 2")

    # For each loop, find the BEST CENTERED region that contains both anchors
    target_regions = []
    for a1_start, a1_end, a2_start, a2_end, loop_coord in cluster3_loops_chr2:
        # Calculate loop midpoint (center between the two anchors)
        loop_midpoint = (a1_start + a2_end) / 2

        # Find all regions that contain both anchors
        candidate_regions = []
        for region_str in all_chr2_regions:
            chrom, row_start, row_end, col_start, col_end = parse_region(region_str)

            anchor1_in_row = (row_start <= a1_start and a1_end <= row_end)
            anchor1_in_col = (col_start <= a1_start and a1_end <= col_end)
            anchor2_in_row = (row_start <= a2_start and a2_end <= row_end)
            anchor2_in_col = (col_start <= a2_start and a2_end <= col_end)

            # For off-diagonal crops, one anchor must lie on each axis.
            contains_loop = (
                (anchor1_in_row and anchor2_in_col) or
                (anchor1_in_col and anchor2_in_row)
            )

            if contains_loop:
                region_midpoint = (
                    min(row_start, col_start) + max(row_end, col_end)
                ) / 2
                # Calculate distance from loop midpoint
                distance = abs(region_midpoint - loop_midpoint)
                candidate_regions.append((region_str, distance))

        # Pick the region with minimum distance (best centered on loop)
        if candidate_regions:
            best_region = min(candidate_regions, key=lambda x: x[1])[0]
            target_regions.append((best_region, loop_coord))
        else:
            print(f"  Warning: No region found containing loop {loop_coord}")

    return target_regions


def run_test_evaluation_chromosome2(
    checkpoint_path,
    phase_name='earlyG1',
    data_dir=None,
    output_dir="./test_inference_visualizations_chr2",
    target_regions=None,
):
    """
    Run inference on cluster 3 loops on chromosome 2 from a trained checkpoint.

    Evaluates one 64x64 region per cluster 3 loop, showing only maps that
    contain both anchors of the loop. Uses cross-phase quantile normalization
    with fixed [0, 40] color scale for all visualizations.

    Args:
        checkpoint_path: Path to model checkpoint
        phase_name: Which phase to evaluate ('earlyG1', 'midG1', 'lateG1', 'anatelo')
        data_dir: Path to data directory (if None, uses default)
        output_dir: Where to save visualizations
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("="*80)
    
    # Load trained model
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Load data with chromosome 2 held out (same as training)
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "raw_data" / "zhang_4dn"
    
    print(f"\nLoading data from: {data_dir}")
    print(f"Resolution: {RESOLUTION_BP} bp ({RESOLUTION_BP // 1000}kb), region size: {REGION_SIZE_BP} bp (64 bins)")
    # Determine data type and log transform from checkpoint name
    # If checkpoint contains "observed_expected" or "oe", use "oe" data type
    # If checkpoint contains "observed_only" or "observed", use "observed" data type
    use_oe = "observed_expected" in checkpoint_path.lower() or "oe" in checkpoint_path.lower()
    hic_data_type = "oe" if use_oe else "observed"
    
    # IMPORTANT: Model was trained on "oe" data WITH log1p transformation
    # So we must use log1p for "oe" data during inference to match training
    # For "observed" data, also use log1p (standard)
    use_log_transform = True  # Always use log1p to match training
    
    print(f"Using hic_data_type='{hic_data_type}', use_log_transform={use_log_transform}")
    print(f"  (Model trained on '{hic_data_type}' data with log1p transformation)")
    
    # Default cache location (matches training)
    processed_data_dir = Path(__file__).parent.parent / "processed_data" / "zhang" / "oe_kr"

    cell_cycle_loader = CellCycleDataLoader(
        data_dir=data_dir,
        resolution=RESOLUTION_BP,
        region_size=REGION_SIZE_BP,
        normalization="KR",  # IMPORTANT: Use the same normalization as training!
        hold_out_chromosome="2",  # Match training setup
        hic_data_type=hic_data_type,  # Match training data type
        use_log_transform=use_log_transform,  # Match training preprocessing
        processed_data_dir=processed_data_dir,
        augment=0,
        allow_live_fallback=True,
    )

    # Decide which regions to evaluate
    if target_regions is None:
        # Default: use one region per cluster 3 loop on chromosome 2
        all_chr2_regions = cell_cycle_loader.get_holdout_regions()
        print(f"Total chromosome 2 regions: {len(all_chr2_regions)}")
        
        target_regions_with_loops = get_cluster3_regions_chr2(all_chr2_regions, data_dir)
        
        print(f"\nCluster 3 loops found on chromosome 2: {len(target_regions_with_loops)}")
        print("\nRegions to evaluate (one per cluster 3 loop):")
        for i, (reg, loop_coord) in enumerate(target_regions_with_loops, 1):
            print(f"  {i}. {reg}")
            print(f"     Loop: {loop_coord}")
        print()
        
        if len(target_regions_with_loops) == 0:
            print("ERROR: No regions found containing cluster 3 loops on chromosome 2!")
            print("Please check that the cluster3_loop_coordinates.txt file exists.")
            cell_cycle_loader.close()
            return
        
        # Extract just the region strings for the dataset
        target_regions = [reg for reg, _ in target_regions_with_loops]
    else:
        # Use user-specified explicit regions (e.g. 2:18563263-19203263)
        target_regions = [normalize_region(region) for region in target_regions]
        print("\nUsing user-specified target regions:")
        for i, reg in enumerate(target_regions, 1):
            print(f"  {i}. {reg}")
        print()

    # Create custom dataset for target regions
    class TargetRegionDataset(Dataset):
        """Dataset for specific target regions."""
        def __init__(self, loader, target_regions):
            self.loader = loader
            self.target_regions = target_regions

        def __len__(self):
            return len(self.target_regions)

        def __getitem__(self, idx):
            region_str = self.target_regions[idx]
            return self.loader[region_str]

    # Create dataset for cluster 3 loop regions
    test_dataset = TargetRegionDataset(cell_cycle_loader, target_regions)

    # Create test dataloader with batch_size=1
    test_dataloader = TorchDataLoader(
        test_dataset,
        batch_size=1,  # Process one sample at a time
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Generate visualizations for cluster 3 loops with fixed [0, 40] color scale
    print(f"\n{'='*80}")
    print(f"Generating visualizations for cluster 3 loops")
    print(f"Using cross-phase quantile normalization with fixed color scale [0, 40]")
    print(f"Processing {len(target_regions)} regions (one per cluster 3 loop)")
    print("="*80 + "\n")

    with torch.no_grad():
        for sample_idx, batch in enumerate(test_dataloader):
            # Get region info for filename
            region = batch['region'][0] if 'region' in batch else f"sample_{sample_idx}"
            region_clean = region.replace(':', '_').replace('-', '_').replace(',', '_')

            # Parse region for better filename
            chrom, row_start, row_end, col_start, col_end = parse_region(region)
            row_start_mb = row_start / 1e6
            row_end_mb = row_end / 1e6
            col_start_mb = col_start / 1e6
            col_end_mb = col_end / 1e6

            # Run inference and visualize (all four channels)
            save_path = run_inference_and_visualize(
                model=model,
                batch=batch,
                device=device,
                step=(
                    f"chr{chrom}_rows_{row_start_mb:.2f}Mb-{row_end_mb:.2f}Mb_"
                    f"cols_{col_start_mb:.2f}Mb-{col_end_mb:.2f}Mb_{region_clean}"
                ),
                output_dir=output_dir
            )

            print(f"✓ [{sample_idx+1}/{len(target_regions)}] {region} -> {save_path.name}")

    print(f"\n{'='*80}")
    print(f"✓ Completed test evaluation on {len(target_regions)} cluster 3 loops")
    print(f"✓ All visualizations use cross-phase quantile normalization with fixed color scale [0, 40]")
    print(f"✓ Visualizations saved to: {output_dir}")
    print("="*80)
    
    # Cleanup
    cell_cycle_loader.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run test evaluation on cluster 3 loops in chromosome 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Evaluates one 64x64 Hi-C map per cluster 3 loop on chromosome 2.
Only regions that contain both anchors of a loop are evaluated.

Visualization uses cross-phase quantile normalization with fixed
color scale [0, 40], allowing direct comparison across all maps.
        """
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="earlyG1",
        choices=["earlyG1", "midG1", "lateG1", "anatelo"],
        help="Which phase to evaluate (default: earlyG1)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to data directory (default: ../raw_data/zhang_4dn)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_inference_visualizations_chr2",
        help="Output directory for visualizations (default: ./test_inference_visualizations_chr2)"
    )
    parser.add_argument(
        "--regions",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional explicit list of regions like '2:18563263-19203263', "
            "'2:18400000-19040000,18650000-19290000', or "
            "'2:18400000-19040000:18650000-19290000'. "
            "If provided, evaluation is run ONLY on these regions instead of cluster 3 loops."
        ),
    )
    
    args = parser.parse_args()
    
    run_test_evaluation_chromosome2(
        checkpoint_path=args.checkpoint,
        phase_name=args.phase,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_regions=args.regions,
    )
