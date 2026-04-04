"""
Cell-Cycle Hi-C Phase Decomposition via SR3-Style Iterative Refinement

We train a single model to steer a diffusion process towards the true Hi-C contact map of a given phase.

NOTATION (γ = signal fraction, NOT noise variance):
    γ_t: Signal fraction at timestep t  (γ≈1 → clean, γ≈0 → pure noise)
    α_t: Step ratio = γ_t / γ_{t-1}    (ratio of adjacent signal levels)

FORWARD PROCESS:
    y_γ = √γ · y_0 + √(1-γ) · ϵ,  where ϵ ~ N(0, I)
    γ→1: y_γ ≈ y_0 (clean);  γ→0: y_γ ≈ ϵ (pure noise)

NOISE SCHEDULE:
    Uniform schedule (γ_min = 1e-4, γ_max = 1.0)

TRAINING (SR3 Algorithm 1):
    - Sample γ ~ Uniform(γ_min, γ_max) continuously
    - Sample noise ϵ ~ N(0, I)
    - Create noisy: y_γ = √γ · y_0 + √(1-γ) · ϵ
    - Train: loss = MSE(model(y_γ, γ, conditioning), ϵ)  ← Model predicts ε directly

SAMPLING:
    - Start from pure noise y_{T-1} ~ N(0, I)
    - For t = T-1, T-2, ..., 1:
        x̂_0      = model(y_t, γ_t, conditioning)          ← Predict clean image
        ε_implied = (y_t - √γ_t · x̂_0) / √(1-γ_t)        ← Recover implied noise
        y_{t-1}  = (1/√α_t)(y_t - (1-α_t)/√(1-γ_t) · ε_implied) + √(1-α_t) · z
    - Result: y_0 (phase-specific Hi-C)

MEMORY OPTIMIZATION: Train one phase at a time to reduce GPU memory usage

Architecture: SR3-Style U-Net with BigGAN Residual Blocks
    - Converts upper triangular Hi-C vector to 2D symmetric matrix
    - Input: Concatenate noisy image + bulk Hi-C conditioning → (B, 2, 64, 64)
    - Four downsampling stages: 64 → 32 → 16 → 8 (bottleneck)
    - BigGAN residual blocks at each resolution with time conditioning
    - Standard U-Net skip connections via concatenation
    - Output: Predicted clean image x_0

Conditioning:
    (1) Noise level γ / time t  -> Sinusoidal embeddings + adaptive group norm
    (2) Bulk Hi-C              -> Concatenated with noisy input at start
"""

import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

# Add preprocess dir to path
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocess"))
from Dataloader import CellCycleDataLoader
from utils import upper_tri_vec_to_matrix  # re-exported for inference.py

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
# Two-channel decomposition: bulk = average(anaphase, G1)
#   anaphase = anatelo phase
#   G1       = mean(earlyG1, midG1, lateG1)
# Model outputs channel 0 = anaphase, channel 1 = G1.

# T is imported from schedule.py
N = 64                 # contact map size (64 x 64)
VEC_DIM = 2080         # upper triangular vector dimension (64*65/2)

# Genomic resolution and region size (in base pairs)
RESOLUTION_BP = 10000           # bin size in base pairs (10kb)
REGION_SIZE_BP = RESOLUTION_BP * N  # total region size in bp (64 bins)
L = 2                 # number of bottleneck blocks in U-Net
HIDDEN_DIM = 128      # base channel dimension for U-Net
d_t = 256             # time embedding dimension

BATCH_SIZE = 32
LR = 1e-4
NUM_EPOCHS = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Resume training from checkpoint (set to None to start from scratch)
# Can be overridden via --resume_checkpoint CLI argument
RESUME_CHECKPOINT = None


############################################
# 2) CHECKPOINT LOADING
############################################
def load_checkpoint_for_training(checkpoint_path, model, optimizer, device):
    """
    Load checkpoint for resuming training.

    Args:
        checkpoint_path: Path to checkpoint file (relative to CHECKPOINT_DIR or absolute)
        model: Model to load weights into
        optimizer: Optimizer to load state into
        device: Device to load checkpoint on

    Returns:
        Tuple of (start_epoch, global_step, best_loss)
        Returns (0, 0, float('inf')) if checkpoint not found
    """
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

    start_epoch = checkpoint['epoch'] + 1
    global_step = checkpoint.get('global_step', 0)
    best_loss = checkpoint.get('loss', float('inf'))

    print(f"✓ Resuming from epoch {checkpoint['epoch'] + 1}")
    print(f"  Loss: {checkpoint['loss']:.6f}, Global step: {global_step}")
    print("="*80 + "\n")

    return start_epoch, global_step, best_loss


############################################
# 3) VALIDATION SET (chr2, excluding test-eval regions)
############################################
# Ranges used by run_test_evaluation_chromosome2.sh – excluded so validation ≠ test eval
TEST_EVAL_TARGET_RANGES_CHR2 = [
    (44700000, 45100000),   # chr2:44.7Mb-45.1Mb
    (18400000, 19400000),   # chr2:18.4Mb-19.4Mb
]


def _parse_region(region_str):
    """Parse 'chrom:start-end' -> (chrom, start, end)."""
    chrom, coords = region_str.split(":")
    start, end = coords.split("-")
    return chrom, int(start), int(end)


def _region_overlaps_any(region_str, ranges):
    """True if region overlaps any (t_start, t_end) in ranges."""
    _, start, end = _parse_region(region_str)
    for t_start, t_end in ranges:
        if start < t_end and end > t_start:
            return True
    return False


def get_validation_regions_chr2(holdout_regions, n=10, seed=42):
    """
    From chr2 holdout regions, exclude those used in run_test_evaluation_chromosome2,
    then return n regions for validation.
    """
    rng = np.random.default_rng(seed)
    valid = [r for r in holdout_regions if not _region_overlaps_any(r, TEST_EVAL_TARGET_RANGES_CHR2)]
    if len(valid) <= n:
        return valid
    indices = rng.choice(len(valid), size=n, replace=False)
    return [valid[i] for i in indices]


############################################
# 4) TRAINING LOOP
############################################
def _build_targets(batch, device):
    """
    Construct two-channel target and bulk conditioning.

    Returns:
        x0_current: (B, 2, vec_dim)  channel 0=anaphase, channel 1=G1
        x0_bulk:    (B, vec_dim)     average of anaphase and G1
        chip_*:     (B, N) chip-seq tracks
    """
    x0_early   = batch["earlyG1"].float().to(device)
    x0_mid     = batch["midG1"].float().to(device)
    x0_late    = batch["lateG1"].float().to(device)
    x0_anatelo = batch["anatelo"].float().to(device)

    x0_anaphase = x0_anatelo                                  # (B, vec_dim)
    x0_G1       = (x0_early + x0_mid + x0_late) / 3.0        # (B, vec_dim)
    x0_bulk     = (x0_early + x0_mid + x0_late + x0_anatelo) * 0.25              # (B, vec_dim)
    x0_current  = torch.stack([x0_anaphase, x0_G1], dim=1)   # (B, 2, vec_dim)

    chip_ctcf = batch["chip_seq_ctcf"].float().to(device)
    chip_hac  = batch["chip_seq_hac"].float().to(device)
    chip_me1  = batch["chip_seq_h3k4me1"].float().to(device)
    chip_me3  = batch["chip_seq_h3k4me3"].float().to(device)

    return x0_current, x0_bulk, chip_ctcf, chip_hac, chip_me1, chip_me3


def eval_batch_loss(model, batch, device, generator: torch.Generator | None = None):
    """Compute SR3 MSE loss for one batch (no backward)."""
    x0_current, x0_bulk, chip_ctcf, chip_hac, chip_me1, chip_me3 = _build_targets(batch, device)
    batch_size = x0_current.shape[0]

    # SR3: sample γ ~ Uniform(γ_min, γ_max) continuously; deterministic if generator given
    if generator is not None:
        gamma_t = torch.rand(batch_size, device=device, generator=generator) * (GAMMA_MAX - GAMMA_MIN) + GAMMA_MIN
        eps_true = torch.randn(x0_current.shape, device=device, generator=generator)
    else:
        gamma_t = torch.rand(batch_size, device=device) * (GAMMA_MAX - GAMMA_MIN) + GAMMA_MIN
        eps_true = torch.randn_like(x0_current)

    # gamma broadcast: (B,) -> (B, 1, 1) for (B, 2, vec_dim)
    gamma_3d = gamma_t[:, None, None]
    y_gamma = torch.sqrt(gamma_3d) * x0_current + torch.sqrt(1.0 - gamma_3d) * eps_true

    eps_pred, _ = model(y_gamma, gamma_t, chip_ctcf, chip_hac, chip_me1, chip_me3, x0_bulk)
    return F.mse_loss(eps_pred, eps_true).item()


def compute_validation_loss(model, val_dataloader, device):
    """Average loss over validation set (model in eval mode, no grad)."""
    model.eval()
    # Fixed seed so validation loss is comparable across epochs
    gen = torch.Generator(device=device)
    gen.manual_seed(12345)
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_dataloader:
            total_loss += eval_batch_loss(model, batch, device, generator=gen)
            n_batches += 1
    model.train()
    return total_loss / n_batches if n_batches else 0.0


def train_step(model, optimizer, batch, device, global_step=0):
    """
    Single training step for SR3-style iterative refinement.

    Model predicts ε for both channels simultaneously:
        channel 0 = anaphase (anatelo)
        channel 1 = G1 (mean of earlyG1, midG1, lateG1)
    Bulk conditioning = average(anaphase, G1).

    Returns:
        (total_loss, mse_loss, chip_aux_loss) as floats
    """
    x0_current, x0_bulk, chip_ctcf, chip_hac, chip_me1, chip_me3 = _build_targets(batch, device)
    batch_size = x0_current.shape[0]

    # SR3: sample γ ~ Uniform(γ_min, γ_max) continuously
    gamma_t = torch.rand(batch_size, device=device) * (GAMMA_MAX - GAMMA_MIN) + GAMMA_MIN  # (B,)
    gamma_3d = gamma_t[:, None, None]  # (B, 1, 1) for broadcast with (B, 2, vec_dim)

    eps_true = torch.randn_like(x0_current)  # (B, 2, vec_dim)
    y_gamma = torch.sqrt(gamma_3d) * x0_current + torch.sqrt(1.0 - gamma_3d) * eps_true

    # SR3 Algorithm 1: model predicts ε directly
    eps_pred, h_chip = model(y_gamma, gamma_t, chip_ctcf, chip_hac, chip_me1, chip_me3, x0_bulk)

    mse_loss = F.mse_loss(eps_pred, eps_true)

    # ---- Commented-out deviations from SR3 ----
    # Min-SNR weighted MSE (Hang et al. 2023):
    # snr = gamma_t / (1.0 - gamma_t)
    # weight = torch.clamp(snr, max=5.0)
    # mse_loss = (weight * (eps_pred - eps_true) ** 2).mean()

    # Aux loss: predict both channels from chip features alone
    chip_pred = model.chip_aux_pred(h_chip)                   # (B, 2, vec_dim)
    chip_aux_loss = 0.20 * F.mse_loss(chip_pred, x0_current)
    loss = mse_loss + chip_aux_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), mse_loss.item(), chip_aux_loss.item()


############################################
# 5) MAIN TRAINING
############################################
def main():
    parser = argparse.ArgumentParser(description='Train diffusion model for Hi-C phase decomposition')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from (relative to checkpoints/ or absolute).')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of epochs to train (overrides NUM_EPOCHS constant).')
    args = parser.parse_args()

    resume_checkpoint = args.resume_checkpoint if args.resume_checkpoint is not None else RESUME_CHECKPOINT
    num_epochs = args.num_epochs if args.num_epochs is not None else NUM_EPOCHS

    print("="*80)
    print("TRAINING: anaphase + G1 (two-channel decomposition)")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Vector dimension: {VEC_DIM}, Matrix size: {N}x{N}")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {num_epochs}")
    if resume_checkpoint:
        print(f"Resume checkpoint: {resume_checkpoint}")
    else:
        print("Starting from scratch (no checkpoint)")

    noise_embed_module = NoiseEmbedding(d_t, max_value=1000)

    model = SR3UNet(
        vec_dim=VEC_DIM,
        n=N,
        noise_embed_module=noise_embed_module,
        base_ch=64
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print(f"Estimated memory: ~{num_params * 4 / 1e9:.2f} GB (fp32)")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch, global_step, best_loss = load_checkpoint_for_training(
        resume_checkpoint, model, optimizer, DEVICE
    )

    data_dir = Path(__file__).parent.parent / "raw_data" / "zhang_4dn"
    print(f"Loading data from: {data_dir}")

    HOLD_OUT_CHROMOSOME = "2"

    base_loader_kwargs = dict(
        data_dir=data_dir,
        resolution=RESOLUTION_BP,
        region_size=REGION_SIZE_BP,
        normalization="KR",
        hold_out_chromosome=HOLD_OUT_CHROMOSOME,
        hic_data_type="oe",
        use_log_transform=True,
        normalization_stats_file=data_dir / "normalization_stats.csv",
    )

    cell_cycle_loader_train = CellCycleDataLoader(
        save_normalization_stats=True,
        augment=50,
        **base_loader_kwargs,
    )
    cell_cycle_loader_eval = CellCycleDataLoader(
        save_normalization_stats=False,
        augment=0,
        **base_loader_kwargs,
    )

    print(f"Training regions: {len(cell_cycle_loader_train)}")
    print(f"Holdout regions (chr{HOLD_OUT_CHROMOSOME}): {len(cell_cycle_loader_eval.get_holdout_regions())}")
    print(f"Available phases: {cell_cycle_loader_train.get_available_phases()}")

    train_dataset = CellCycleDataset(cell_cycle_loader_train)

    holdout_regions = cell_cycle_loader_eval.get_holdout_regions()
    if len(holdout_regions) == 0:
        raise ValueError(f"No regions found for holdout chromosome '{HOLD_OUT_CHROMOSOME}'")

    class HoldoutDataset(Dataset):
        """Dataset for holdout chromosome regions."""
        def __init__(self, loader, holdout_regions):
            self.loader = loader
            self.holdout_regions = holdout_regions

        def __len__(self):
            return len(self.holdout_regions)

        def __getitem__(self, idx):
            return self.loader[self.holdout_regions[idx]]

    test_dataset = HoldoutDataset(cell_cycle_loader_eval, holdout_regions)

    NUM_VAL_SAMPLES = 30
    validation_regions = get_validation_regions_chr2(holdout_regions, n=NUM_VAL_SAMPLES)
    if len(validation_regions) == 0:
        raise ValueError("No chr2 regions left for validation after excluding test-eval targets")
    val_dataset = HoldoutDataset(cell_cycle_loader_eval, validation_regions)
    val_dataloader = TorchDataLoader(
        val_dataset,
        batch_size=min(5, len(validation_regions)),
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    print(f"Validation regions (chr2, excluding test-eval): {validation_regions[:3]}{'...' if len(validation_regions) > 3 else ''} (n={len(validation_regions)})")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}, Val samples: {len(val_dataset)}")

    train_dataloader = TorchDataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Number of batches per epoch: {len(train_dataloader)}")
    print("="*80)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_losses, epoch_mse, epoch_chip = [], [], []
        model.train()

        total_epochs = start_epoch + num_epochs
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [anaphase+G1]")
        for batch in pbar:
            loss, mse, chip = train_step(model, optimizer, batch, DEVICE, global_step)
            epoch_losses.append(loss)
            epoch_mse.append(mse)
            epoch_chip.append(chip)
            global_step += 1

            if global_step % 100 == 0:
                val_loss = compute_validation_loss(model, val_dataloader, DEVICE)
                print(f"  [step {global_step}] val_loss = {val_loss:.6f}")

            pbar.set_postfix({'total': f"{loss:.4f}", 'mse': f"{mse:.4f}", 'chip': f"{chip:.4f}"})

        avg_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch+1}/{total_epochs} - total={avg_loss:.6f}  mse={np.mean(epoch_mse):.6f}  chip={np.mean(epoch_chip):.6f}")

        data_type_str = cell_cycle_loader_train.hic_data_type
        log_str = "log" if cell_cycle_loader_train.use_log_transform else "nolog"
        checkpoint_path = CHECKPOINT_DIR / f"{data_type_str}_{log_str}_anaphase_G1_epoch{epoch+1}_4-3.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'global_step': global_step,
        }, checkpoint_path)
        print(f"✓ Saved epoch checkpoint: {checkpoint_path}")

    print("\n" + "="*80)
    print("Training complete for anaphase + G1 two-channel model!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    print("="*80)

    cell_cycle_loader_train.close()
    cell_cycle_loader_eval.close()


if __name__ == "__main__":
    main()
