"""
Cell-Cycle Hi-C Phase Decomposition via Conditional DDPM

Train 3 separate conditional denoisers (one per phase: earlyG1/midG1/lateG1)
Each denoiser predicts injected Gaussian noise ε from a noisy phase map x_t

MEMORY OPTIMIZATION: Train one phase at a time to reduce GPU memory usage

Architecture: ControlNet-Inspired U-Net with Parallel Conditioning Encoder
    - Converts upper triangular Hi-C vector to 2D symmetric matrix
    - Parallel control encoder processes bulk Hi-C at multiple resolutions
    - Main U-Net processes noisy input
    - Conditioning features ADDED at each resolution level
    - Time FiLM conditioning (scale & shift) globally injected at every ResBlock

Conditioning signals:
    (1) Diffusion time t         -> Sinusoidal embeddings + FiLM at every layer
    (2) Bulk Hi-C               -> Parallel encoder → multi-resolution features
                                    → Added to main U-Net at each level

Loss: MSE between predicted noise and true noise (standard DDPM)
    loss = MSE(ε_pred, ε_true)  [per phase]
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

# Add preprocess dir to path
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocess"))
from Dataloader import CellCycleDataLoader, upper_tri_vec_to_matrix

torch.manual_seed(42)

############################################
# HELPER FUNCTIONS
############################################
def matrix_to_upper_tri_vec(matrix):
    """
    Convert batch of symmetric matrices to upper triangular vectors.
    
    Args:
        matrix: (batch, n, n) symmetric matrices
    
    Returns:
        vec: (batch, n*(n+1)/2) upper triangular vectors
    """
    batch_size, n, _ = matrix.shape
    indices = torch.triu_indices(n, n, device=matrix.device)
    vec = matrix[:, indices[0], indices[1]]  # (batch, n*(n+1)/2)
    return vec


def upper_tri_vec_to_matrix(vec, n):
    """
    Convert batch of upper triangular vectors to symmetric matrices.
    
    Args:
        vec: (batch, n*(n+1)/2) upper triangular vectors
        n: matrix size
    
    Returns:
        matrix: (batch, n, n) symmetric matrices
    """
    batch_size = vec.shape[0]
    device = vec.device
    matrix = torch.zeros(batch_size, n, n, device=device)
    
    indices = torch.triu_indices(n, n, device=device)
    matrix[:, indices[0], indices[1]] = vec
    matrix[:, indices[1], indices[0]] = vec  # Make symmetric
    
    return matrix


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
        """
        Returns a sample as a dict with numpy arrays.
        PyTorch DataLoader will automatically convert to tensors and batch.
        """
        return self.loader[idx]


############################################
# 1) CONFIG
############################################
PHASES = ["earlyG1", "midG1", "lateG1"]

# MEMORY OPTIMIZATION: Train one phase at a time
# Change this to 'earlyG1', 'midG1', or 'lateG1' to train different phases
CURRENT_PHASE = 'earlyG1'  # <-- CHANGE THIS to train different phases

T = 1000              # diffusion steps
N = 64                # contact map size (64 x 64)
VEC_DIM = 2080        # upper triangular vector dimension (64*65/2)
L = 2                 # number of bottleneck blocks in U-Net
HIDDEN_DIM = 128      # base channel dimension for U-Net (reduced from 256)
d_t = 256             # time embedding dimension

BATCH_SIZE = 32       # Increased from 8 since we have more memory with single model
LR = 1e-4
NUM_EPOCHS = 2        # More epochs since we're training one at a time
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model checkpoints directory
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)




############################################
# 1) DIFFUSION SCHEDULE (DDPM)
############################################
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02): # same as ddpm paper
    """Linear schedule from Ho et al. (DDPM paper)"""
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


# Initialize schedule
betas, alphas, alphas_cumprod = linear_beta_schedule(T)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)


############################################
# 2) TIME EMBEDDING (Sinusoidal)
############################################
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        time: (batch,) tensor of timestep indices
        returns: (batch, dim) embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    def __init__(self, dim, max_timesteps=1000):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbeddings(dim)
        self.max_timesteps = max_timesteps
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, t):
        """
        t: (batch,) integer timestep in [0, max_timesteps-1]
        returns: (batch, dim)
        """
        emb = self.sinusoidal(t)
        return self.mlp(emb)


############################################
# 3) CONVOLUTIONAL RESIDUAL BLOCK (with Time FiLM)
############################################
class ConvResBlock(nn.Module):
    """
    Convolutional residual block with group normalization and TIME conditioning only.
    
    Inputs:
        x: (batch, channels, H, W) feature map
        time_emb: (batch, time_dim) time embedding for FiLM conditioning
    """
    def __init__(self, channels, time_dim):
        super().__init__()
        self.channels = channels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        # Group normalization (use 8 groups or channels if fewer)
        num_groups = min(8, channels)
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.gn2 = nn.GroupNorm(num_groups, channels)
        
        # Time conditioning projection (FiLM: scale and shift)
        self.time_proj = nn.Linear(time_dim, channels * 2)
        
        self.act = nn.SiLU()
    
    def forward(self, x, time_emb):
        """
        x: (batch, channels, H, W)
        time_emb: (batch, time_dim) - time embedding only
        """
        residual = x
        
        # Project time embedding to scale and shift
        time_params = self.time_proj(time_emb)  # (batch, channels * 2)
        scale, shift = time_params.chunk(2, dim=1)  # Each (batch, channels)
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # (batch, channels, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)  # (batch, channels, 1, 1)
        
        # First conv + FiLM conditioning (time only)
        h = self.conv1(x)
        h = self.gn1(h)
        h = h * (1 + scale) + shift  # FiLM: scale and shift from time
        h = self.act(h)
        
        # Second conv
        h = self.conv2(h)
        h = self.gn2(h)
        
        return self.act(h + residual)


############################################
# 4) CONTROL ENCODER (Parallel Conditioning Pathway)
############################################
class ControlEncoder(nn.Module):
    """
    Parallel encoder that processes bulk Hi-C conditioning image.
    Produces multi-resolution features to be added to main U-Net at each level.
    
    Architecture matches the main U-Net structure to ensure resolution compatibility.
    """
    def __init__(self, base_ch: int = 64, time_dim: int = 256):
        super().__init__()
        
        # Input: (B, 1, 64, 64) bulk Hi-C map
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.SiLU(),
        )
        
        # Level 1: 64x64 @ base_ch
        self.enc1 = ConvResBlock(base_ch, time_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1)
        
        # Level 2: 32x32 @ base_ch*2
        self.enc2 = ConvResBlock(base_ch * 2, time_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1)
        
        # Level 3: 16x16 @ base_ch*4 (bottleneck)
        self.enc3 = ConvResBlock(base_ch * 4, time_dim)
        
        # Zero-initialize output to start with no conditioning
        for module in [self.enc1, self.enc2, self.enc3]:
            if hasattr(module, 'conv2'):
                nn.init.zeros_(module.conv2.weight)
                nn.init.zeros_(module.conv2.bias)
    
    def forward(self, cond_img: torch.Tensor, t_emb: torch.Tensor):
        """
        Args:
            cond_img: (B, 1, 64, 64) bulk Hi-C conditioning image
            t_emb: (B, time_dim) time embedding
        
        Returns:
            Dictionary of conditioning features at each resolution:
            {
                'feat_64': (B, base_ch, 64, 64),
                'feat_32': (B, base_ch*2, 32, 32),
                'feat_16': (B, base_ch*4, 16, 16)
            }
        """
        # Input processing
        h = self.input_conv(cond_img)  # (B, base_ch, 64, 64)
        
        # Level 1: 64x64
        feat_64 = self.enc1(h, t_emb)  # (B, base_ch, 64, 64)
        h = self.down1(feat_64)        # (B, base_ch*2, 32, 32)
        
        # Level 2: 32x32
        feat_32 = self.enc2(h, t_emb)  # (B, base_ch*2, 32, 32)
        h = self.down2(feat_32)        # (B, base_ch*4, 16, 16)
        
        # Level 3: 16x16 (bottleneck)
        feat_16 = self.enc3(h, t_emb)  # (B, base_ch*4, 16, 16)
        
        return {
            'feat_64': feat_64,
            'feat_32': feat_32,
            'feat_16': feat_16
        }


############################################
# 5) MAIN U-NET (EpsilonNet with ControlNet Architecture)
############################################
class EpsilonNet(nn.Module):
    """
    Main denoising U-Net with parallel control encoder.
    
    Architecture:
        - Control encoder processes bulk Hi-C → multi-resolution features
        - Main U-Net processes noisy input
        - Conditioning features ADDED at each level
        - Time FiLM at every ResBlock
    """
    def __init__(self, vec_dim, n, time_embed_module, base_ch: int = 64):
        super().__init__()
        self.vec_dim = vec_dim
        self.n = n
        self.time_embed = time_embed_module  # TimeEmbedding module
        
        time_dim = self.time_embed.mlp[-1].out_features
        
        # ---- Control Encoder (processes bulk Hi-C) ----
        self.control_encoder = ControlEncoder(base_ch=base_ch, time_dim=time_dim)
        
        # ---- Main U-Net (processes noisy input) ----
        # Input: (B, 1, 64, 64) noisy Hi-C map
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.SiLU(),
        )
        
        # Encoder Level 1: 64x64 @ base_ch
        self.enc1 = ConvResBlock(base_ch, time_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1)
        
        # Encoder Level 2: 32x32 @ base_ch*2
        self.enc2 = ConvResBlock(base_ch * 2, time_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1)
        
        # Bottleneck: 16x16 @ base_ch*4
        self.bottleneck = nn.ModuleList([
            ConvResBlock(base_ch * 4, time_dim),
            ConvResBlock(base_ch * 4, time_dim),
        ])
        
        # Decoder Level 2: 16x16 -> 32x32 @ base_ch*2
        self.up2_conv = nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1)
        self.dec2_reduce = nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=1)  # Reduce after concat
        self.dec2 = ConvResBlock(base_ch * 2, time_dim)
        
        # Decoder Level 1: 32x32 -> 64x64 @ base_ch
        self.up1_conv = nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1)
        self.dec1_reduce = nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)  # Reduce after concat
        self.dec1 = ConvResBlock(base_ch, time_dim)
        
        # Output: (B, 1, 64, 64) predicted noise
        self.output_conv = nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
    
    def forward(self, x_t_vec, t_idx, chip_1d, bulk_vec):
        """
        Args:
            x_t_vec:  (B, vec_dim) noisy Hi-C vector
            t_idx:    (B,) integer timestep in [0..T-1]
            chip_1d:  (B, N) ChIP-seq signal (IGNORED for now)
            bulk_vec: (B, vec_dim) bulk Hi-C vector (conditioning)
        
        Returns:
            eps_vec: (B, vec_dim) predicted noise vector
        """
        B = x_t_vec.shape[0]
        N = self.n
        
        # --- Build time embedding ---
        t_emb = self.time_embed(t_idx)  # (B, time_dim)
        
        # --- Convert vectors to maps ---
        x_t_map = upper_tri_vec_to_matrix(x_t_vec, N).unsqueeze(1)     # (B, 1, 64, 64)
        bulk_map = upper_tri_vec_to_matrix(bulk_vec, N).unsqueeze(1)   # (B, 1, 64, 64)
        
        # ========== CONTROL ENCODER: Process bulk Hi-C ==========
        cond_feats = self.control_encoder(bulk_map, t_emb)
        # cond_feats = {
        #     'feat_64': (B, base_ch, 64, 64),
        #     'feat_32': (B, base_ch*2, 32, 32),
        #     'feat_16': (B, base_ch*4, 16, 16)
        # }
        
        # ========== MAIN U-NET: Process noisy input ==========
        # Input stem
        h = self.input_conv(x_t_map)  # (B, base_ch, 64, 64)
        
        # ENCODER LEVEL 1: 64x64
        h = h + cond_feats['feat_64']  # ← ADD control features
        enc1 = self.enc1(h, t_emb)     # (B, base_ch, 64, 64)
        h = self.down1(enc1)           # (B, base_ch*2, 32, 32)
        
        # ENCODER LEVEL 2: 32x32
        h = h + cond_feats['feat_32']  # ← ADD control features
        enc2 = self.enc2(h, t_emb)     # (B, base_ch*2, 32, 32)
        h = self.down2(enc2)           # (B, base_ch*4, 16, 16)
        
        # BOTTLENECK: 16x16
        h = h + cond_feats['feat_16']  # ← ADD control features
        for blk in self.bottleneck:
            h = blk(h, t_emb)          # (B, base_ch*4, 16, 16)
        
        # DECODER LEVEL 2: 16x16 -> 32x32
        h = F.interpolate(h, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        h = self.up2_conv(h)                    # (B, base_ch*2, 32, 32)
        h = h + cond_feats['feat_32']          # ← ADD control features again
        h = torch.cat([h, enc2], dim=1)        # (B, base_ch*4, 32, 32)
        h = self.dec2_reduce(h)                # (B, base_ch*2, 32, 32) - reduce channels
        h = self.dec2(h, t_emb)                # (B, base_ch*2, 32, 32)
        
        # DECODER LEVEL 1: 32x32 -> 64x64
        h = F.interpolate(h, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        h = self.up1_conv(h)                    # (B, base_ch, 64, 64)
        h = h + cond_feats['feat_64']          # ← ADD control features again
        h = torch.cat([h, enc1], dim=1)        # (B, base_ch*2, 64, 64)
        h = self.dec1_reduce(h)                # (B, base_ch, 64, 64) - reduce channels
        h = self.dec1(h, t_emb)                # (B, base_ch, 64, 64)
        
        # Output
        eps_map = self.output_conv(h).squeeze(1)  # (B, 64, 64)
        eps_vec = matrix_to_upper_tri_vec(eps_map)  # (B, vec_dim)
        
        return eps_vec


############################################
# 6) FORWARD NOISING (q_sample)
############################################
def q_sample(x0, t, noise):
    """
    Sample from q(x_t | x_0)
    
    x0: (batch, vec_dim) clean data
    t: (batch,) timesteps
    noise: (batch, vec_dim) Gaussian noise
    
    returns: x_t (batch, vec_dim)
    """
    device = x0.device
    # Move schedule tensors to the same device as data
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod.to(device)[t][:, None]  # (batch, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod.to(device)[t][:, None]
    
    x_t = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
    return x_t


############################################
# 6) TRAINING LOOP
############################################
def train_step(model, optimizer, batch, device, phase_name):
    """
    Single training step for ONE phase denoiser.
    
    Args:
        model: EpsilonNet for the current phase
        optimizer: optimizer for this model
        batch: dict with keys 'region', 'earlyG1', 'midG1', 'lateG1', 'chip_seq'
        device: torch device
        phase_name: 'earlyG1', 'midG1', or 'lateG1'
    
    Returns:
        float: loss for this phase
    """
    # Extract clean x0 for each phase (needed for bulk conditioning)
    x0_early = batch['earlyG1'].float().to(device)  # (batch_size, vec_dim)
    x0_mid = batch['midG1'].float().to(device)
    x0_late = batch['lateG1'].float().to(device)
    
    # Select the current phase's ground truth
    phase_data = {
        'earlyG1': x0_early,
        'midG1': x0_mid,
        'lateG1': x0_late
    }
    x0_current = phase_data[phase_name]  # (batch_size, vec_dim)
    
    # Compute bulk Hi-C (average of three phases) for conditioning
    x0_bulk_normalized = (x0_early + x0_mid + x0_late) / 3  # (batch_size, vec_dim)
    batch_size = x0_bulk_normalized.shape[0]
    
    # Get ChIP-seq conditioning (already batched)
    chip_1d = batch['chip_seq'].float().to(device)  # (batch_size, N)
    
    # Sample random timestep for each sample in batch
    t = torch.randint(0, T, (batch_size,), device=device).long()  # (batch_size,)
    
    # Sample random noise (epsilon)
    eps_true = torch.randn_like(x0_current)  # (batch_size, vec_dim)
    
    # Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    x_t = q_sample(x0_current, t, eps_true)
    
    # Predict the noise that was added (with bulk conditioning)
    eps_hat = model(x_t, t, chip_1d, x0_bulk_normalized)
    
    # DDPM loss: MSE between predicted noise and true noise
    loss = F.mse_loss(eps_hat, eps_true)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def test_training(model, batch, device, step, phase_name):
    """
    Test the model by reconstruction and visualizing predictions.
    Called every ~100 training steps to monitor progress.
    
    Args:
        model: EpsilonNet for the current phase
        batch: training batch
        device: torch device
        step: current training step number
        phase_name: 'earlyG1', 'midG1', or 'lateG1'
    
    Returns:
        Path to saved visualization
    """
    from visualize_diffusion_training import visualize_training_step_single_phase
    
    print(f"\n{'='*60}")
    print(f"Testing {phase_name} at step {step}...")
    
    save_path = visualize_training_step_single_phase(
        model=model,
        phase_name=phase_name,
        batch=batch,
        device=device,
        step=step,
        T=T,
        output_dir="./training_visualizations"
    )
    
    print(f"{'='*60}\n")
    
    return save_path


############################################
# 7) MAIN TRAINING
############################################
def main():
    print("="*80)
    print(f"TRAINING PHASE: {CURRENT_PHASE}")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Vector dimension: {VEC_DIM}, Matrix size: {N}x{N}")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}")
    
    # Create time embedding module
    time_embed_module = TimeEmbedding(d_t, max_timesteps=T)
    
    # Initialize ONE model for the current phase with ControlNet architecture
    model = EpsilonNet(
        vec_dim=VEC_DIM,
        n=N,
        time_embed_module=time_embed_module,
        base_ch=64            # Base channels for U-Net and Control Encoder
    ).to(DEVICE)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print(f"Estimated memory: ~{num_params * 4 / 1e9:.2f} GB (fp32)")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Load data
    data_dir = Path(__file__).parent.parent / "raw_data" / "zhang_4dn"
    print(f"Loading data from: {data_dir}")
    
    cell_cycle_loader = CellCycleDataLoader(
        data_dir=data_dir,
        resolution=10000,
        region_size=640000,
        normalization="VC"
    )
    
    print(f"Number of regions: {len(cell_cycle_loader)}")
    print(f"Available phases: {cell_cycle_loader.get_available_phases()}")
    
    # Wrap in PyTorch Dataset and DataLoader for batching
    dataset = CellCycleDataset(cell_cycle_loader)
    dataloader = TorchDataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with hicstraw
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Number of batches per epoch: {len(dataloader)}")
    print("="*80)
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        model.train()
        
        # Iterate through batches
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [{CURRENT_PHASE}]")
        for batch_idx, batch in enumerate(pbar):
            loss = train_step(model, optimizer, batch, DEVICE, CURRENT_PHASE)
            epoch_losses.append(loss)
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss:.4f}"})
            
            # Visualize every 100 steps
            if global_step % 100 == 0:
                test_training(model, batch, DEVICE, global_step, CURRENT_PHASE)
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = CHECKPOINT_DIR / f"{CURRENT_PHASE}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'global_step': global_step,
            }, checkpoint_path)
            print(f"✓ Saved best checkpoint: {checkpoint_path}")
        
        # Save epoch checkpoint
        checkpoint_path = CHECKPOINT_DIR / f"{CURRENT_PHASE}_epoch{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'global_step': global_step,
        }, checkpoint_path)
    
    print("\n" + "="*80)
    print(f"Training complete for {CURRENT_PHASE}!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    print("="*80)
    
    # Cleanup
    cell_cycle_loader.close()


if __name__ == "__main__":
    main()
