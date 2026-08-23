"""
Cell-Cycle Hi-C Phase Decomposition via SR3-Style Iterative Refinement

Model inputs/outputs are full 2-D contact matrices (B, 3, N, N) – no upper-tri
vectors.  Training samples now include both diagonal and off-diagonal crops.
Pelham-Webb dataset: three phases (mitosis / earlyG1 / lateG1); bulk = async.

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

# import re  # loop label parsing (disabled)
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
# import pandas as pd  # loop label Excel I/O (disabled)
import pytorch_msssim
from iw_ssim import InformationWeightedSSIMLoss
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
# Three-channel decomposition (Pelham-Webb): bulk = async Hi-C measurement.
# Model outputs channel 0=mitosis, 1=earlyG1, 2=lateG1.

N = 64                           # contact map size (64 x 64)

# Genomic resolution and region size (in base pairs)
RESOLUTION_BP  = 10000           # bin size in base pairs (10kb)
REGION_SIZE_BP = RESOLUTION_BP * N

L          = 2                   # (kept for reference; bottleneck depth in U-Net)
HIDDEN_DIM = 128                 # base channel dimension for U-Net
d_t        = 256                 # time embedding dimension

BATCH_SIZE  = 32
LR          = 1e-5
NUM_EPOCHS  = 40
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "final" # TODO: remove later
CHECKPOINT_DIR.mkdir(exist_ok=True)

RESUME_CHECKPOINT = None


############################################
# 3) CHECKPOINT LOADING  (was §2)
############################################
def load_checkpoint_for_training(checkpoint_path, model, optimizer, device, scheduler=None):
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

    load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if load_result.missing_keys:
        print(f"  New params (random init): {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"  Ignored keys: {load_result.unexpected_keys}")
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"  Scheduler state restored (last_epoch={scheduler.last_epoch})")

    start_epoch  = checkpoint['epoch'] + 1
    global_step  = checkpoint.get('global_step', 0)
    best_loss    = checkpoint.get('loss', float('inf'))

    current_lr = optimizer.param_groups[0]['lr']
    print(f"✓ Resuming from epoch {checkpoint['epoch'] + 1}")
    print(f"  Loss: {checkpoint['loss']:.6f}, Global step: {global_step}")
    print(f"  Learning rate: {current_lr:.2e}")
    print("="*80 + "\n")

    return start_epoch, global_step, best_loss


############################################
# 3) VALIDATION SET (random holdout sample)
############################################
VAL_SPLIT_SEED = 42


def get_validation_regions(holdout_regions, n=10, seed=VAL_SPLIT_SEED):
    """Sample ``n`` validation regions from holdout tiles (reproducible via ``seed``)."""
    if not holdout_regions:
        return []
    rng = np.random.default_rng(seed)
    n_val = min(n, len(holdout_regions))
    indices = rng.choice(len(holdout_regions), size=n_val, replace=False)
    return [holdout_regions[i] for i in indices]


############################################
# 4) TRAINING LOOP
############################################
def _build_targets(batch, device):
    """
    Construct three-channel target matrices and bulk conditioning (Pelham-Webb).

    Returns:
        x0_current : (B, 3, N, N)  mitosis / earlyG1 / lateG1 matrices
        bulk_map   : (B, 1, N, N)  async Hi-C (loaded directly as bulk)
        chip_*_row : (B, N)
        chip_*_col : (B, N)
    """
    x0_mitosis = batch["mitosis"].float().to(device)
    x0_early   = batch["earlyG1"].float().to(device)
    x0_late    = batch["lateG1"].float().to(device)
    x0_current = torch.stack([x0_mitosis, x0_early, x0_late], dim=1)  # (B, 3, N, N)

    bulk_map   = batch["bulk"].float().to(device).unsqueeze(1)  # (B, 1, N, N)  — sum-then-normalize

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


_SSIM_DATA_RANGE = 4.0


def ssim_1_minus_mean(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    win_size: int = 11,
    win_sigma: float = 1.5,
) -> torch.Tensor:
    """
    Differentiable 1 − SSIM via pytorch_msssim. pred and target: (B, C, H, W).

    Fixed data_range for [-1, 1] maps (see _SSIM_DATA_RANGE).
    """
    if pred.shape != target.shape:
        raise ValueError(f"ssim: shape mismatch {pred.shape} vs {target.shape}")
    _, _C, H, W = pred.shape
    if win_size % 2 != 1 or win_size < 3:
        raise ValueError("win_size must be an odd integer >= 3")
    if H < win_size or W < win_size:
        raise ValueError(f"map spatial size ({H},{W}) must be >= win_size ({win_size})")

    ssim_val = pytorch_msssim.ssim(
        pred,
        target,
        data_range=_SSIM_DATA_RANGE,
        size_average=True,
        win_size=win_size,
        win_sigma=win_sigma,
    )
    return 1.0 - ssim_val


############################################
# IW-SSIM  (Information-Weighted SSIM)
############################################
# Log-normalised Hi-C maps span roughly [-2, 2]; data_range = max - min = 4.
# PIQ default uses 5 pyramid levels (min 161×161); N=64 maps support at most 3.
_iw_ssim_loss = InformationWeightedSSIMLoss(
    data_range=4,
    scale_weights=torch.tensor([0.0448, 0.2856, 0.3001]),
)


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

    gamma_4d = gamma_t[:, None, None, None]   # (B, 1, 1, 1) broadcasts with (B, 3, N, N)
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
        model:     nn.DataParallel-wrapped (or plain) SR3UNet — used for forward pass.
        raw_model: Underlying SR3UNet; used for chip_aux_pred without DataParallel
                   re-scattering small tensors.
    Returns:
        (total_loss, mse_loss, chip_aux_loss) as floats
        mse_loss:     channel-weighted MSE on noise residuals (main diffusion objective).
        chip_aux_loss: IW-SSIM loss between chip_aux_pred(h_chip) and x0_current.
    """
    (x0_current, bulk_map,
     chip_ctcf_row, chip_hac_row, chip_me1_row, chip_me3_row,
     chip_ctcf_col, chip_hac_col, chip_me1_col, chip_me3_col) = _build_targets(batch, device)

    batch_size = x0_current.shape[0]

    # SR3: sample γ ~ Uniform(γ_min, γ_max) continuously
    gamma_t  = torch.rand(batch_size, device=device) * (GAMMA_MAX - GAMMA_MIN) + GAMMA_MIN
    gamma_4d = gamma_t[:, None, None, None]  # (B, 1, 1, 1) broadcasts with (B, 3, N, N)

    eps_true = torch.randn_like(x0_current)
    y_gamma  = torch.sqrt(gamma_4d) * x0_current + torch.sqrt(1.0 - gamma_4d) * eps_true

    # DataParallel splits along dim=0; h_chip is gathered back to GPU 0 automatically
    eps_pred, h_chip = model(
        y_gamma, gamma_t,
        chip_ctcf_row, chip_hac_row, chip_me1_row, chip_me3_row,
        chip_ctcf_col, chip_hac_col, chip_me1_col, chip_me3_col,
        bulk_map,
    )

    channel_weights  = torch.tensor([1/3, 1/3, 1/3], device=device)
    mse_per_channel  = ((eps_pred - eps_true) ** 2).mean(dim=(0, 2, 3))  # (3,)
    mse_loss         = (channel_weights * mse_per_channel).sum()

    # ---- ChIP aux head: predict phase maps, supervise with IW-SSIM ----
    # chip_pred outputs (B, 3, N, N) phase-map predictions from ChIP pair features.
    # Compared directly against x0_current (log-normalised Hi-C targets).
    chip_pred     = raw_model.chip_aux_pred(h_chip)          # (B, 3, N, N)
    # PIQ IW-SSIM expects inputs in [0, data_range]; log-normalised Hi-C maps
    # span roughly [-2, 2], so shift into [0, 4] to satisfy the range check.
    chip_aux_loss = _iw_ssim_loss(
        (chip_pred + 2).clamp(0, 4),
        (x0_current.detach() + 2).clamp(0, 4),
    )

    loss = mse_loss + chip_aux_loss / 3

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), mse_loss.item(), chip_aux_loss.item() / 3


############################################
# 5) MAIN TRAINING
############################################
def main():
    parser = argparse.ArgumentParser(description='Train diffusion model for Hi-C phase decomposition')
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--hold_out_chromosome', type=str, default='19')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory for saved checkpoints (default: train/checkpoints)')
    parser.add_argument('--save_epochs', type=str, default='20,40,60,80',
                        help='Comma-separated epoch numbers at which to save checkpoints')
    parser.add_argument('--checkpoint_basename', type=str, default=None,
                        help='If set, save as {checkpoint_dir}/{basename}.pth instead of the default name')
    args = parser.parse_args()

    resume_checkpoint = args.resume_checkpoint if args.resume_checkpoint is not None else RESUME_CHECKPOINT
    num_epochs        = args.num_epochs if args.num_epochs is not None else NUM_EPOCHS
    hold_out_chromosome = args.hold_out_chromosome
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_epochs = tuple(int(e.strip()) for e in args.save_epochs.split(',') if e.strip())

    print("="*80)
    print("TRAINING: Pelham-Webb three phases (mitosis / earlyG1 / lateG1, matrix I/O)")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Matrix size: {N}×{N}")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {num_epochs}")
    if resume_checkpoint:
        print(f"Resume checkpoint: {resume_checkpoint}")

    noise_embed_module = NoiseEmbedding(d_t, max_value=1000)

    raw_model = SR3UNet(n=N, noise_embed_module=noise_embed_module, base_ch=64, n_phases=3).to(DEVICE)


    num_params = sum(p.numel() for p in raw_model.parameters())
    print(f"Parameters: {num_params:,}")
    print(f"Estimated memory: ~{num_params * 4 / 1e9:.2f} GB (fp32)")

    optimizer = torch.optim.Adam(raw_model.parameters(), lr=LR)

    # Cosine annealing over the total planned epochs (T_max).
    # On resume the saved state_dict restores the exact position in the schedule.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=LR / 100,
    )

    # Load checkpoint into raw_model BEFORE wrapping with DataParallel so that
    # state-dict keys never have the "module." prefix.
    start_epoch, global_step, best_loss = load_checkpoint_for_training(
        resume_checkpoint, raw_model, optimizer, DEVICE, scheduler=scheduler,
    )

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs with DataParallel (batch split: {BATCH_SIZE} → {BATCH_SIZE // n_gpus} per GPU)")
        model = torch.nn.DataParallel(raw_model)
    else:
        print(f"Using {'GPU' if n_gpus == 1 else 'CPU'}")
        model = raw_model

    data_dir = Path(__file__).parent.parent / "raw_data" / "pelham-webb"
    print(f"Loading data from: {data_dir}")

    processed_data_dir = [
        Path(__file__).parent.parent / "processed_data" / "pelham-webb" / "rep1",
        Path(__file__).parent.parent / "processed_data" / "pelham-webb" / "rep2",
    ]
    for _d in processed_data_dir:
        if not _d.exists():
            raise ValueError(
                f"Cache directory not found at {_d}. "
                "Run preprocess/prestore_hic.py or preprocess/kang/prestore_kang.py first."
            )
    print(f"Using pre-stored caches (cache-only training): {[str(d) for d in processed_data_dir]}")

    base_loader_kwargs = dict(
        data_dir=data_dir,
        resolution=RESOLUTION_BP,
        region_size=REGION_SIZE_BP,
        normalization="KR",
        hold_out_chromosome=hold_out_chromosome,
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
    print(f"Holdout regions (chr{hold_out_chromosome}): "
          f"{len(cell_cycle_loader_eval.get_holdout_regions())}")
    print(f"Available phases: {cell_cycle_loader_train.get_available_phases()}")

    train_dataset = CellCycleDataset(cell_cycle_loader_train)

    holdout_regions = cell_cycle_loader_eval.get_holdout_regions()
    if not holdout_regions:
        raise ValueError(f"No regions found for holdout chromosome '{hold_out_chromosome}'")

    class HoldoutDataset(Dataset):
        def __init__(self, loader, holdout_regions):
            self.loader          = loader
            self.holdout_regions = holdout_regions

        def __len__(self):
            return len(self.holdout_regions)

        def __getitem__(self, idx):
            return self.loader[self.holdout_regions[idx]]

    test_dataset = HoldoutDataset(cell_cycle_loader_eval, holdout_regions)

    NUM_VAL_SAMPLES = 30
    validation_regions = get_validation_regions(holdout_regions, n=NUM_VAL_SAMPLES)
    if not validation_regions:
        raise ValueError(f"No holdout regions on chr{hold_out_chromosome} for validation")

    val_dataset    = HoldoutDataset(cell_cycle_loader_eval, validation_regions)
    val_dataloader = TorchDataLoader(
        val_dataset,
        batch_size=min(5, len(validation_regions)),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Validation regions (chr{hold_out_chromosome}, seed={VAL_SPLIT_SEED}): "
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
    print(f"LR schedule: cosine annealing {LR:.1e} → {LR / 100:.1e} over {num_epochs} epochs")
    print("="*80)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_losses, epoch_mse, epoch_chip_aux = [], [], []
        model.train()

        total_epochs = start_epoch + num_epochs
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [3-phase/pelham-webb]")
        for batch in pbar:
            loss, mse, chip_aux = train_step(
                model, raw_model, optimizer, batch, DEVICE,
            )
            epoch_losses.append(loss)
            epoch_mse.append(mse)
            epoch_chip_aux.append(chip_aux)
            global_step += 1

            if global_step % 100 == 0:
                val_loss = compute_validation_loss(model, val_dataloader, DEVICE)
                cur_lr   = scheduler.get_last_lr()[0]
                print(f"  [step {global_step}] val_loss = {val_loss:.6f}  lr = {cur_lr:.2e}")
            if global_step % 20 == 0:
                pbar.set_postfix({
                    'total':    f"{loss:.4f}",
                    'mse':      f"{mse:.4f}",
                    'chip_aux': f"{chip_aux:.4f}",
                    'lr':       f"{scheduler.get_last_lr()[0]:.2e}",
                })

        scheduler.step()

        avg_loss = np.mean(epoch_losses)
        cur_lr   = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch+1}/{total_epochs} - "
              f"total={avg_loss:.6f}  mse={np.mean(epoch_mse):.6f}  "
              f"chip_aux={np.mean(epoch_chip_aux):.6f}  "
              f"lr={cur_lr:.2e}")

        # Save only selected epochs to reduce checkpoint churn.
        if (epoch + 1) in save_epochs:
            if args.checkpoint_basename:
                checkpoint_path = checkpoint_dir / f"{args.checkpoint_basename}.pth"
            else:
                data_type_str = cell_cycle_loader_train.hic_data_type
                log_str       = "log" if cell_cycle_loader_train.use_log_transform else "nolog"
                checkpoint_path = (checkpoint_dir /
                                   f"{data_type_str}_{log_str}_3phase_pelhamwebb_epoch{epoch+1}_"
                                   f"holdout{hold_out_chromosome}.pth")
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     raw_model.state_dict(),  # never has "module." prefix
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss':                 avg_loss,
                'global_step':          global_step,
            }, checkpoint_path)
            print(f"✓ Saved epoch checkpoint: {checkpoint_path}")

    print("\n" + "="*80)
    print("Training complete for all three phases (Pelham-Webb)!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("="*80)

    cell_cycle_loader_train.close()
    cell_cycle_loader_eval.close()


if __name__ == "__main__":
    main()