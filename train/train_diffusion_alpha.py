"""
Cell-Cycle Hi-C Phase Decomposition via SR3-Style Iterative Refinement

Model inputs/outputs are full 2-D contact matrices (B, 4, N, N) – no upper-tri
vectors.  Training samples now include both diagonal and off-diagonal crops.

NOTATION (γ = signal fraction, NOT noise variance):
    γ_t: Signal fraction at timestep t  (γ≈1 → clean, γ≈0 → pure noise)
    α_t: Step ratio = γ_t / γ_{t-1}

FORWARD PROCESS:
    y_γ = √γ · y_0 + √(1-γ) · ϵ,  ϵ ~ N(0, I)

TRAINING (SR3 Algorithm 1):
    - Sample γ ~ Uniform(γ_min, γ_max) continuously
    - Create noisy: y_γ = √γ · y_0 + √(1-γ) · ϵ
    - Train: loss = MSE(model(y_γ, γ, conditioning), ϵ)

SAMPLING:
    - Start from pure noise y_{T-1} ~ N(0, I)
    - For t = T-1, T-2, ..., 1:
        y_{t-1}  = (1/√α_t)(y_t - (1-α_t)/√(1-γ_t) · ε_θ) + √(1-α_t) · z
"""

import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocess"))
from Dataloader import CellCycleDataLoader

from schedule import T, gammas, alphas, GAMMA_MIN, GAMMA_MAX
from model import SR3UNet, NoiseEmbedding

torch.manual_seed(42)


############################################
# 0) PYTORCH DATASET WRAPPER
############################################
class CellCycleDataset(Dataset):
    """PyTorch Dataset wrapper for CellCycleDataLoader to enable batching."""

    def __init__(self, cell_cycle_loader):
        self.loader = cell_cycle_loader
        self.length = len(cell_cycle_loader)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.loader[idx]


############################################
# 1) CONFIG
############################################
# Four-channel decomposition: bulk = average(earlyG1, midG1, lateG1, anatelo)
# Model outputs channel 0=earlyG1, 1=midG1, 2=lateG1, 3=anatelo.

N = 64                           # contact map size (64 x 64)

# Genomic resolution and region size (in base pairs)
RESOLUTION_BP  = 10000           # bin size in base pairs (10kb)
REGION_SIZE_BP = RESOLUTION_BP * N

L          = 2                   # (kept for reference; bottleneck depth in U-Net)
HIDDEN_DIM = 128                 # base channel dimension for U-Net
d_t        = 256                 # time embedding dimension

BATCH_SIZE  = 32
LR          = 1e-4
NUM_EPOCHS  = 40
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

RESUME_CHECKPOINT = None

# Chip aux target: per phase, DoG(phase) − DoG(bulk).
# DoG(z) = blur_small(z) − blur_large(z), which tends to emphasize outlines/corners more
# than a single Gaussian blur.
# CHIP_DOG_KERNEL       = 15  # odd
# CHIP_DOG_SIGMA_SMALL  = 5
# CHIP_DOG_SIGMA_LARGE  = 11


############################################
# 2) CHECKPOINT LOADING
############################################
def load_checkpoint_for_training(checkpoint_path, model, optimizer, device):
    if checkpoint_path is None:
        return 0, 0, float('inf')

    path = Path(checkpoint_path)
    if not path.is_absolute():
        if checkpoint_path.startswith("checkpoints/"):
            path = CHECKPOINT_DIR / checkpoint_path.replace("checkpoints/", "")
        else:
            path = CHECKPOINT_DIR / checkpoint_path

    if not path.exists():
        print(f"WARNING: Checkpoint not found: {path}")
        return 0, 0, float('inf')

    print(f"\n{'='*80}")
    print(f"Loading checkpoint: {path}")
    print("="*80)

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch  = checkpoint['epoch'] + 1
    global_step  = checkpoint.get('global_step', 0)
    best_loss    = checkpoint.get('loss', float('inf'))

    print(f"✓ Resuming from epoch {checkpoint['epoch'] + 1}")
    print(f"  Loss: {checkpoint['loss']:.6f}, Global step: {global_step}")
    print("="*80 + "\n")

    return start_epoch, global_step, best_loss


############################################
# 3) VALIDATION SET (chr2, excluding test-eval regions)
############################################
TEST_EVAL_TARGET_RANGES_CHR2 = [
    (44700000, 45100000),
    (18400000, 19400000),
]


def _parse_region(region_str: str):
    """
    Parse region string to (chrom, start, end).

    Handles:
      "chrom:start-end"                           (legacy diagonal)
      "chrom:row_start-row_end:col_start-col_end" (new format)

    For the new format, start = row_start and end = col_end (full genomic span).
    """
    parts = region_str.split(":")
    chrom = parts[0]
    row_start, row_end = map(int, parts[1].split("-"))
    if len(parts) == 3:
        _, col_end = map(int, parts[2].split("-"))
        return chrom, row_start, col_end   # full genomic span
    return chrom, row_start, row_end


def _region_overlaps_any(region_str, ranges):
    """True if region overlaps any (t_start, t_end) in ranges."""
    _, start, end = _parse_region(region_str)
    for t_start, t_end in ranges:
        if start < t_end and end > t_start:
            return True
    return False


def get_validation_regions_chr2(holdout_regions, n=10, seed=42):
    """
    From chr2 holdout regions (diagonal only), exclude test-eval targets,
    then return n regions for validation.
    """
    rng = np.random.default_rng(seed)
    valid = [r for r in holdout_regions
             if not _region_overlaps_any(r, TEST_EVAL_TARGET_RANGES_CHR2)]
    if len(valid) <= n:
        return valid
    indices = rng.choice(len(valid), size=n, replace=False)
    return [valid[i] for i in indices]


############################################
# 4) TRAINING LOOP
############################################
def _build_targets(batch, device):
    """
    Construct four-channel target matrices and bulk conditioning.

    Returns:
        x0_current : (B, 4, N, N)  earlyG1 / midG1 / lateG1 / anatelo matrices
        bulk_map   : (B, 1, N, N)  average of all four phases
        chip_*_row : (B, N)
        chip_*_col : (B, N)
    """
    x0_early   = batch["earlyG1"].float().to(device)   # (B, N, N)
    x0_mid     = batch["midG1"].float().to(device)
    x0_late    = batch["lateG1"].float().to(device)
    x0_anatelo = batch["anatelo"].float().to(device)

    x0_current = torch.stack([x0_early, x0_mid, x0_late, x0_anatelo], dim=1)  # (B, 4, N, N)
    bulk_map   = (x0_early + x0_mid + x0_late + x0_anatelo).mul(0.25).unsqueeze(1)  # (B, 1, N, N)

    chip_ctcf_row = batch["chip_seq_ctcf_row"].float().to(device)
    chip_hac_row  = batch["chip_seq_hac_row"].float().to(device)
    chip_me1_row  = batch["chip_seq_h3k4me1_row"].float().to(device)
    chip_me3_row  = batch["chip_seq_h3k4me3_row"].float().to(device)

    chip_ctcf_col = batch["chip_seq_ctcf_col"].float().to(device)
    chip_hac_col  = batch["chip_seq_hac_col"].float().to(device)
    chip_me1_col  = batch["chip_seq_h3k4me1_col"].float().to(device)
    chip_me3_col  = batch["chip_seq_h3k4me3_col"].float().to(device)

    return (x0_current, bulk_map,
            chip_ctcf_row, chip_hac_row, chip_me1_row, chip_me3_row,
            chip_ctcf_col, chip_hac_col, chip_me1_col, chip_me3_col)


# def _gaussian_blur_depthwise(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
#     """Depthwise isotropic Gaussian blur. x: (B, C, H, W)."""
#     _ks = kernel_size
#     if _ks % 2 != 1 or _ks < 1:
#         raise ValueError("kernel_size must be a positive odd integer")
#     B, C, H, W = x.shape
#     device, dtype = x.device, x.dtype
#     coords = torch.arange(_ks, device=device, dtype=dtype) - (_ks - 1) / 2.0
#     g1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
#     g1d = g1d / g1d.sum()
#     k2d = torch.outer(g1d, g1d)
#     k2d = k2d / k2d.sum()
#     weight = k2d.view(1, 1, _ks, _ks).expand(C, 1, _ks, _ks).contiguous()
#     pad = _ks // 2
#     return F.conv2d(x, weight, padding=pad, groups=C)


# def high_pass_x0_maps(x0: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
#     """High-pass each channel: x0 - Gaussian_blur(x0). No bulk subtraction."""
#     low = _gaussian_blur_depthwise(x0, kernel_size, sigma)
#     return x0 - low


# def gaussian_blur_residual_vs_bulk(
#     x0: torch.Tensor,
#     bulk_map: torch.Tensor,
#     kernel_size: int,
#     sigma: float,
# ) -> torch.Tensor:
#     """
#     Per phase c: Gaussian_blur(x0_c) − Gaussian_blur(bulk).

#     x0:       (B, 4, H, W)
#     bulk_map: (B, 1, H, W)
#     """
#     low_x0 = _gaussian_blur_depthwise(x0, kernel_size, sigma)
#     low_bulk = _gaussian_blur_depthwise(bulk_map, kernel_size, sigma)
#     return low_x0 - low_bulk


# def dog_residual_vs_bulk(
#     x0: torch.Tensor,
#     bulk_map: torch.Tensor,
#     kernel_size: int,
#     sigma_small: float,
#     sigma_large: float,
# ) -> torch.Tensor:
#     """
#     Per phase c: DoG(x0_c) − DoG(bulk), where DoG(z)=blur_small(z)−blur_large(z).

#     x0:       (B, 4, H, W)
#     bulk_map: (B, 1, H, W)
#     """
#     low_x0_small = _gaussian_blur_depthwise(x0, kernel_size, sigma_small)
#     low_x0_large = _gaussian_blur_depthwise(x0, kernel_size, sigma_large)

#     #low_bulk_small = _gaussian_blur_depthwise(bulk_map, kernel_size, sigma_small)
#     #low_bulk_large = _gaussian_blur_depthwise(bulk_map, kernel_size, sigma_large)

#     dog_x0 = low_x0_small - low_x0_large
#     #dog_bulk = low_bulk_small - low_bulk_large
#     return dog_x0 #- dog_bulk


def eval_batch_loss(model, batch, device, generator: torch.Generator | None = None):
    """Compute SR3 MSE loss for one batch (no backward)."""
    (x0_current, bulk_map,
     chip_ctcf_row, chip_hac_row, chip_me1_row, chip_me3_row,
     chip_ctcf_col, chip_hac_col, chip_me1_col, chip_me3_col) = _build_targets(batch, device)

    batch_size = x0_current.shape[0]

    if generator is not None:
        gamma_t  = torch.rand(batch_size, device=device, generator=generator) * (GAMMA_MAX - GAMMA_MIN) + GAMMA_MIN
        eps_true = torch.randn(x0_current.shape, device=device, generator=generator)
    else:
        gamma_t  = torch.rand(batch_size, device=device) * (GAMMA_MAX - GAMMA_MIN) + GAMMA_MIN
        eps_true = torch.randn_like(x0_current)

    gamma_4d = gamma_t[:, None, None, None]   # (B, 1, 1, 1) broadcasts with (B, 4, N, N)
    y_gamma  = torch.sqrt(gamma_4d) * x0_current + torch.sqrt(1.0 - gamma_4d) * eps_true

    eps_pred, _ = model(
        y_gamma, gamma_t,
        chip_ctcf_row, chip_hac_row, chip_me1_row, chip_me3_row,
        chip_ctcf_col, chip_hac_col, chip_me1_col, chip_me3_col,
        bulk_map,
    )
    return F.mse_loss(eps_pred, eps_true).item()


def compute_validation_loss(model, val_dataloader, device):
    """Average loss over validation set (model in eval mode, no grad)."""
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(12345)
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for batch in val_dataloader:
            total_loss += eval_batch_loss(model, batch, device, generator=gen)
            n_batches  += 1
    model.train()
    return total_loss / n_batches if n_batches else 0.0


def train_step(model, raw_model, optimizer, batch, device):
    """
    Single training step for SR3-style iterative refinement.

    Args:
        model:     nn.DataParallel-wrapped (or plain) SR3UNet — used for forward pass
        raw_model: Underlying SR3UNet (model.module when DataParallel, else model itself).
                   Used directly for chip_aux_pred to avoid DP re-scattering a small tensor.
    Returns:
        (total_loss, mse_loss, chip_aux_loss) as floats
    """
    (x0_current, bulk_map,
     chip_ctcf_row, chip_hac_row, chip_me1_row, chip_me3_row,
     chip_ctcf_col, chip_hac_col, chip_me1_col, chip_me3_col) = _build_targets(batch, device)

    batch_size = x0_current.shape[0]

    # SR3: sample γ ~ Uniform(γ_min, γ_max) continuously
    gamma_t  = torch.rand(batch_size, device=device) * (GAMMA_MAX - GAMMA_MIN) + GAMMA_MIN
    gamma_4d = gamma_t[:, None, None, None]  # (B, 1, 1, 1) broadcasts with (B, 4, N, N)

    eps_true = torch.randn_like(x0_current)
    y_gamma  = torch.sqrt(gamma_4d) * x0_current + torch.sqrt(1.0 - gamma_4d) * eps_true

    # DataParallel splits along dim=0; h_chip is gathered back to GPU 0 automatically
    eps_pred, h_chip = model(
        y_gamma, gamma_t,
        chip_ctcf_row, chip_hac_row, chip_me1_row, chip_me3_row,
        chip_ctcf_col, chip_hac_col, chip_me1_col, chip_me3_col,
        bulk_map,
    )

    channel_weights  = torch.tensor([0.20, 0.20, 0.20, 0.40], device=device)
    mse_per_channel  = ((eps_pred - eps_true) ** 2).mean(dim=(0, 2, 3))  # (4,)
    mse_loss         = (channel_weights * mse_per_channel).sum()

    # ---- Chip aux: 50/50 CTCF + H3K27ac (HAC) outer-product weighting ----
    # CTCF anchors anaphase / cohesin loops; H3K27ac marks G1 active loops.
    # Mixing them lets the chip features learn both anchor types in the same head.
    chip_pred         = raw_model.chip_aux_pred(h_chip)
    chip_ctcf_weight  = chip_ctcf_row[:, :, None] * chip_ctcf_col[:, None, :]   # (B, N, N)
    chip_hac_weight   = chip_hac_row[:, :, None]  * chip_hac_col[:, None, :]    # (B, N, N)
    chip_combo_weight = 0.5 * chip_ctcf_weight + 0.5 * chip_hac_weight           # (B, N, N)
    chip_aux_target   = x0_current * chip_combo_weight.unsqueeze(1)              # (B, 4, N, N)
    chip_aux_loss     = F.mse_loss(chip_pred, chip_aux_target)

    loss = mse_loss + chip_aux_loss / 20

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), mse_loss.item(), chip_aux_loss.item()


############################################
# 5) MAIN TRAINING
############################################
def main():
    parser = argparse.ArgumentParser(description='Train diffusion model for Hi-C phase decomposition')
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    args = parser.parse_args()

    resume_checkpoint = args.resume_checkpoint if args.resume_checkpoint is not None else RESUME_CHECKPOINT
    num_epochs        = args.num_epochs if args.num_epochs is not None else NUM_EPOCHS

    print("="*80)
    print("TRAINING: all four phases (matrix I/O, diagonal + off-diagonal crops)")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Matrix size: {N}×{N}")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {num_epochs}")
    if resume_checkpoint:
        print(f"Resume checkpoint: {resume_checkpoint}")

    noise_embed_module = NoiseEmbedding(d_t, max_value=1000)

    raw_model = SR3UNet(
        n=N,
        noise_embed_module=noise_embed_module,
        base_ch=64,
    ).to(DEVICE)

    num_params = sum(p.numel() for p in raw_model.parameters())
    print(f"Parameters: {num_params:,}")
    print(f"Estimated memory: ~{num_params * 4 / 1e9:.2f} GB (fp32)")

    optimizer = torch.optim.Adam(raw_model.parameters(), lr=LR)

    # Load checkpoint into raw_model BEFORE wrapping with DataParallel so that
    # state-dict keys never have the "module." prefix.
    start_epoch, global_step, best_loss = load_checkpoint_for_training(
        resume_checkpoint, raw_model, optimizer, DEVICE
    )

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs with DataParallel (batch split: {BATCH_SIZE} → {BATCH_SIZE // n_gpus} per GPU)")
        model = torch.nn.DataParallel(raw_model)
    else:
        print(f"Using {'GPU' if n_gpus == 1 else 'CPU'}")
        model = raw_model

    data_dir = Path(__file__).parent.parent / "raw_data" / "zhang_4dn"
    print(f"Loading data from: {data_dir}")

    HOLD_OUT_CHROMOSOME = "2"

    processed_data_dir = Path(__file__).parent.parent / "processed_data" / "zhang" / "obs"
    if not processed_data_dir.exists():
        raise ValueError(
            f"Cache directory not found at {processed_data_dir}. "
            "Training is cache-only; run preprocess/prestore_hic.py first."
        )
    print(f"Using pre-stored cache (cache-only training): {processed_data_dir}")

    base_loader_kwargs = dict(
        data_dir=data_dir,
        resolution=RESOLUTION_BP,
        region_size=REGION_SIZE_BP,
        normalization="KR",
        hold_out_chromosome=HOLD_OUT_CHROMOSOME,
        hic_data_type="observed",
        use_log_transform=True,
        normalization_stats_file=data_dir / "normalization_stats.csv",
        processed_data_dir=processed_data_dir,
        allow_live_fallback=False,
    )

    cell_cycle_loader_train = CellCycleDataLoader(
        save_normalization_stats=False,  # only needed once; disable to reduce I/O overhead
        augment=50,
        **base_loader_kwargs,
    )
    cell_cycle_loader_eval = CellCycleDataLoader(
        save_normalization_stats=False,
        augment=0,
        **base_loader_kwargs,
    )

    print(f"Training regions: {len(cell_cycle_loader_train)}")
    print(f"Holdout regions (chr{HOLD_OUT_CHROMOSOME}): "
          f"{len(cell_cycle_loader_eval.get_holdout_regions())}")
    print(f"Available phases: {cell_cycle_loader_train.get_available_phases()}")

    train_dataset = CellCycleDataset(cell_cycle_loader_train)

    holdout_regions = cell_cycle_loader_eval.get_holdout_regions()
    if not holdout_regions:
        raise ValueError(f"No regions found for holdout chromosome '{HOLD_OUT_CHROMOSOME}'")

    class HoldoutDataset(Dataset):
        def __init__(self, loader, holdout_regions):
            self.loader          = loader
            self.holdout_regions = holdout_regions

        def __len__(self):
            return len(self.holdout_regions)

        def __getitem__(self, idx):
            return self.loader[self.holdout_regions[idx]]

    test_dataset = HoldoutDataset(cell_cycle_loader_eval, holdout_regions)

    NUM_VAL_SAMPLES     = 30
    validation_regions  = get_validation_regions_chr2(holdout_regions, n=NUM_VAL_SAMPLES)
    if not validation_regions:
        raise ValueError("No chr2 regions left for validation after excluding test-eval targets")

    val_dataset    = HoldoutDataset(cell_cycle_loader_eval, validation_regions)
    val_dataloader = TorchDataLoader(
        val_dataset,
        batch_size=min(5, len(validation_regions)),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Validation regions (chr2, excluding test-eval): "
          f"{validation_regions[:3]}{'...' if len(validation_regions) > 3 else ''} "
          f"(n={len(validation_regions)})")
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}, Val: {len(val_dataset)}")

    NUM_WORKERS = 4  # each worker pre-fetches independently, overlapping NFS I/O with GPU
    train_dataloader = TorchDataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,  # keep workers alive between epochs to avoid re-fork cost
    )

    print(f"Batches per epoch: {len(train_dataloader)}")
    print("="*80)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_losses, epoch_mse, epoch_chip = [], [], []
        model.train()

        total_epochs = start_epoch + num_epochs
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [4-phase]")
        for batch in pbar:
            loss, mse, chip = train_step(model, raw_model, optimizer, batch, DEVICE)
            epoch_losses.append(loss)
            epoch_mse.append(mse)
            epoch_chip.append(chip)
            global_step += 1

            if global_step % 100 == 0:
                val_loss = compute_validation_loss(model, val_dataloader, DEVICE)
                print(f"  [step {global_step}] val_loss = {val_loss:.6f}")
            # Only update the progress bar postfix every 20 iterations to avoid printing every single iteration
            if global_step % 20 == 0:
                pbar.set_postfix({'total': f"{loss:.4f}", 'mse': f"{mse:.4f}", 'chip': f"{chip:.4f}"})
     

        avg_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch+1}/{total_epochs} - "
              f"total={avg_loss:.6f}  mse={np.mean(epoch_mse):.6f}  chip={np.mean(epoch_chip):.6f}")

        # Save only selected epochs to reduce checkpoint churn.
        if (epoch + 1) in (10, 20, 30, 40):
            data_type_str = cell_cycle_loader_train.hic_data_type
            log_str       = "log" if cell_cycle_loader_train.use_log_transform else "nolog"
            checkpoint_path = (CHECKPOINT_DIR /
                               f"{data_type_str}_{log_str}_4phase_epoch{epoch+1}_5_4_observed.pth")
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     raw_model.state_dict(),  # never has "module." prefix
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':                 avg_loss,
                'global_step':          global_step,
            }, checkpoint_path)
            print(f"✓ Saved epoch checkpoint: {checkpoint_path}")

    print("\n" + "="*80)
    print("Training complete for all four phases!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    print("="*80)

    cell_cycle_loader_train.close()
    cell_cycle_loader_eval.close()


if __name__ == "__main__":
    main()
