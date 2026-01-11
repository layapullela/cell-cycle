"""
Cell-Cycle Hi-C Phase Decomposition via SR3-Style Iterative Refinement

Train 3 separate conditional denoisers (one per phase: earlyG1/midG1/lateG1)
Each denoiser iteratively refines noisy bulk Hi-C toward phase-specific data

SR3 NOTATION (from "Image Super-Resolution via Iterative Refinement"):
    γ_t: Noise level at timestep t (linearly spaced: 1e-4 → 0.02)
    α_t: Step size = γ_{t-1} / γ_t (relates consecutive noise levels)
    ᾱ_t: Cumulative product = ∏_{s=1}^t α_s

FORWARD PROCESS:
    y_t = √(1-γ_t) · y_0 + √γ_t · ϵ,  where ϵ ~ N(0, I)
    As t increases, γ_t increases (more noise)

TRAINING (Algorithm 1):
    - Sample noise level γ ~ Uniform(γ_min, γ_max)
    - Sample noise ϵ ~ N(0, I)
    - Create noisy: y_γ = √γ · y_0 + √(1-γ) · ϵ
    - Train: loss = MSE(model(y_γ, γ), ϵ)  ← Model predicts NOISE!
    - Model learns to predict the noise that was added

SAMPLING (Algorithm 2):
    - Start with pure noise y_T ~ N(0, I)
    - For t = T-1, T-2, ..., 0:
        ε_pred = model(y_t, t, conditioning)  ← Predict noise
        y_{t-1} = 1/√α_t * (y_t - (1-α_t)/√(1-γ_t) * ε_pred)  ← Remove noise
    - Result: y_0 (phase-specific Hi-C)

MEMORY OPTIMIZATION: Train one phase at a time to reduce GPU memory usage

Architecture: SR3-Style U-Net with BigGAN Residual Blocks
    - Converts upper triangular Hi-C vector to 2D symmetric matrix
    - Input: Concatenate noisy image + bulk Hi-C conditioning → (B, 2, 64, 64)
    - Four downsampling stages: 64 → 32 → 16 → 8 (bottleneck)
    - BigGAN residual blocks at each resolution with time conditioning
    - Standard U-Net skip connections via concatenation
    - Output: Predicted noise ε (what was added to create noisy image)

Conditioning:
    (1) Noise level γ / time t  -> Sinusoidal embeddings + adaptive group norm
    (2) Bulk Hi-C              -> Concatenated with noisy input at start
"""

import os
import sys
import gc
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
NUM_EPOCHS = 5        # More epochs since we're training one at a time
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model checkpoints directory
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)




############################################
# 1) SR3 NOISE SCHEDULE
############################################
def sr3_noise_schedule(timesteps, gamma_min=1e-4, gamma_max=0.02):
    """
    SR3 noise schedule following Algorithm 1 & 2 from SR3 paper.
    
    γ_t: Linearly spaced noise levels from γ_min to γ_max
    α_t: Step-wise alpha = γ_{t-1} / γ_t
    ᾱ_t: Cumulative product of alphas
    
    Args:
        timesteps: Number of diffusion steps T
        gamma_min: Minimum noise level (start, less noisy)
        gamma_max: Maximum noise level (end, more noisy)
    
    Returns:
        gammas: (T,) tensor of noise levels
        alphas: (T,) tensor of step-wise alphas
        alphas_cumprod: (T,) tensor of cumulative product alphas
    """
    # γ_t linearly spaced from γ_min to γ_max
    gammas = torch.linspace(gamma_min, gamma_max, timesteps)
    
    # Compute α_t = γ_{t-1} / γ_t
    # For t=0, we need γ_{-1}. Convention: γ_0 = γ_min (or could be 0)
    # We'll use γ_{-1} = 0 so α_0 ≈ 0/γ_0 → 0
    # Actually, let's set γ_{-1} = 0 for mathematical convenience
    gammas_prev = torch.cat([torch.tensor([0.0]), gammas[:-1]])
    alphas = gammas_prev / (gammas + 1e-10)  # Add small epsilon to avoid division by zero
    
    # For t=0, α_0 should be close to 0 (starting from clean)
    alphas[0] = 0.0
    
    # Cumulative product: ᾱ_t = ∏_{s=1}^t α_s
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return gammas, alphas, alphas_cumprod


# Initialize SR3 schedule
gammas, alphas, alphas_cumprod = sr3_noise_schedule(T, gamma_min=1e-4, gamma_max=0.02)
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
# 3) BIGGAN RESIDUAL BLOCK (SR3-style)
############################################
class BigGANResBlock(nn.Module):
    """
    BigGAN-style residual block with time conditioning via adaptive group norm.
    Used in SR3 paper for super-resolution diffusion.
    
    Inputs:
        x: (batch, in_channels, H, W) feature map
        time_emb: (batch, time_dim) time embedding
    """
    def __init__(self, in_channels, out_channels, time_dim, up=False, down=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        
        # Group normalization
        num_groups = min(8, in_channels)
        self.gn1 = nn.GroupNorm(num_groups, in_channels)
        self.gn2 = nn.GroupNorm(min(8, out_channels), out_channels)
        
        # Time conditioning (adaptive group norm - scale and shift)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2)
        )
        
        # Main pathway
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Residual/skip connection
        if in_channels != out_channels or up or down:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
        
        self.act = nn.SiLU()
    
    def forward(self, x, time_emb):
        """
        x: (batch, in_channels, H, W)
        time_emb: (batch, time_dim)
        """
        # Residual path
        residual = x
        
        # Upsample/downsample residual if needed
        if self.up:
            residual = F.interpolate(residual, scale_factor=2, mode='nearest')
        elif self.down:
            residual = F.avg_pool2d(residual, kernel_size=2, stride=2)
        
        residual = self.residual_conv(residual)
        
        # Main path
        h = self.gn1(x)
        h = self.act(h)
        
        # Upsample/downsample main path
        if self.up:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
        elif self.down:
            h = F.avg_pool2d(h, kernel_size=2, stride=2)
        
        h = self.conv1(h)
        
        # Apply time conditioning (adaptive group norm)
        time_params = self.time_proj(time_emb)  # (batch, out_channels * 2)
        scale, shift = time_params.chunk(2, dim=1)  # Each (batch, out_channels)
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # (batch, out_channels, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)  # (batch, out_channels, 1, 1)
        
        h = self.gn2(h)
        h = h * (1 + scale) + shift  # Adaptive group norm
        h = self.act(h)
        h = self.conv2(h)
        
        return h + residual


############################################
# 4) SR3-STYLE U-NET (Denoised Image Predictor)
############################################
class SR3UNet(nn.Module):
    """
    SR3-style U-Net that predicts noise ε (following Algorithm 1).
    
    Architecture:
        - Input: Concatenate noisy image + bulk Hi-C conditioning → (B, 2, 64, 64)
        - Four downsampling stages: 64 → 32 → 16 → 8
        - BigGAN residual blocks at each resolution
        - Standard U-Net skip connections (concatenation)
        - Output: Predicted noise ε
    """
    def __init__(self, vec_dim, n, time_embed_module, base_ch: int = 64):
        super().__init__()
        self.vec_dim = vec_dim
        self.n = n
        self.time_embed = time_embed_module
        
        time_dim = self.time_embed.mlp[-1].out_features
        
        # ---- Input: Concatenate noisy + conditioning → (B, 2, 64, 64) ----
        self.input_conv = nn.Conv2d(2, base_ch, kernel_size=3, padding=1)
        
        # ---- ENCODER ----
        # Level 1: 64x64 @ base_ch
        self.enc1 = BigGANResBlock(base_ch, base_ch, time_dim)
        self.enc1_down = BigGANResBlock(base_ch, base_ch * 2, time_dim, down=True)
        
        # Level 2: 32x32 @ base_ch*2
        self.enc2 = BigGANResBlock(base_ch * 2, base_ch * 2, time_dim)
        self.enc2_down = BigGANResBlock(base_ch * 2, base_ch * 4, time_dim, down=True)
        
        # Level 3: 16x16 @ base_ch*4
        self.enc3 = BigGANResBlock(base_ch * 4, base_ch * 4, time_dim)
        self.enc3_down = BigGANResBlock(base_ch * 4, base_ch * 8, time_dim, down=True)
        
        # ---- BOTTLENECK: 8x8 @ base_ch*8 ----
        self.bottleneck = nn.ModuleList([
            BigGANResBlock(base_ch * 8, base_ch * 8, time_dim),
            BigGANResBlock(base_ch * 8, base_ch * 8, time_dim),
        ])
        
        # ---- DECODER ----
        # Level 3: 8x8 -> 16x16 @ base_ch*4
        self.dec3_up = BigGANResBlock(base_ch * 8, base_ch * 4, time_dim, up=True)
        self.dec3_reduce = nn.Conv2d(base_ch * 8, base_ch * 4, kernel_size=1)  # After concat
        self.dec3 = BigGANResBlock(base_ch * 4, base_ch * 4, time_dim)
        
        # Level 2: 16x16 -> 32x32 @ base_ch*2
        self.dec2_up = BigGANResBlock(base_ch * 4, base_ch * 2, time_dim, up=True)
        self.dec2_reduce = nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=1)  # After concat
        self.dec2 = BigGANResBlock(base_ch * 2, base_ch * 2, time_dim)
        
        # Level 1: 32x32 -> 64x64 @ base_ch
        self.dec1_up = BigGANResBlock(base_ch * 2, base_ch, time_dim, up=True)
        self.dec1_reduce = nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)  # After concat
        self.dec1 = BigGANResBlock(base_ch, base_ch, time_dim)
        
        # ---- OUTPUT: Predict denoised image ----
        self.output_block = nn.Sequential(
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)
        )
        
        # Zero-initialize output for stability
        nn.init.zeros_(self.output_block[-1].weight)
        nn.init.zeros_(self.output_block[-1].bias)
    
    def forward(self, x_t_vec, t_idx, chip_1d, bulk_vec):
        """
        SR3 forward pass: Predict x_{t-1} from x_t (one iterative refinement step).
        
        Args:
            x_t_vec:  (B, vec_dim) noisy Hi-C vector at timestep t
            t_idx:    (B,) integer timestep in [0..T-1]
            chip_1d:  (B, N) ChIP-seq signal (UNUSED - kept for compatibility)
            bulk_vec: (B, vec_dim) bulk Hi-C vector (conditioning)
        
        Returns:
            x_prev_vec: (B, vec_dim) predicted x_{t-1} (less noisy image at previous timestep)
        """
        B = x_t_vec.shape[0]
        N = self.n
        
        # --- Time embedding ---
        t_emb = self.time_embed(t_idx)  # (B, time_dim)
        
        # --- Convert vectors to 2D matrices ---
        x_t_map = upper_tri_vec_to_matrix(x_t_vec, N).unsqueeze(1)     # (B, 1, 64, 64)
        bulk_map = upper_tri_vec_to_matrix(bulk_vec, N).unsqueeze(1)   # (B, 1, 64, 64)
        
        # --- Concatenate noisy input + bulk conditioning ---
        x_in = torch.cat([x_t_map, bulk_map], dim=1)  # (B, 2, 64, 64)
        
        # --- Input convolution ---
        h = self.input_conv(x_in)  # (B, base_ch, 64, 64)
        
        # ========== ENCODER ==========
        # Level 1: 64x64
        h = self.enc1(h, t_emb)           # (B, base_ch, 64, 64)
        skip1 = h                         # Save for skip connection
        h = self.enc1_down(h, t_emb)      # (B, base_ch*2, 32, 32)
        
        # Level 2: 32x32
        h = self.enc2(h, t_emb)           # (B, base_ch*2, 32, 32)
        skip2 = h                         # Save for skip connection
        h = self.enc2_down(h, t_emb)      # (B, base_ch*4, 16, 16)
        
        # Level 3: 16x16
        h = self.enc3(h, t_emb)           # (B, base_ch*4, 16, 16)
        skip3 = h                         # Save for skip connection
        h = self.enc3_down(h, t_emb)      # (B, base_ch*8, 8, 8)
        
        # ========== BOTTLENECK: 8x8 ==========
        for block in self.bottleneck:
            h = block(h, t_emb)           # (B, base_ch*8, 8, 8)
        
        # ========== DECODER ==========
        # Level 3: 8x8 -> 16x16
        h = self.dec3_up(h, t_emb)                    # (B, base_ch*4, 16, 16)
        h = torch.cat([h, skip3], dim=1)              # (B, base_ch*8, 16, 16)
        h = self.dec3_reduce(h)                       # (B, base_ch*4, 16, 16)
        h = self.dec3(h, t_emb)                       # (B, base_ch*4, 16, 16)
        
        # Level 2: 16x16 -> 32x32
        h = self.dec2_up(h, t_emb)                    # (B, base_ch*2, 32, 32)
        h = torch.cat([h, skip2], dim=1)              # (B, base_ch*4, 32, 32)
        h = self.dec2_reduce(h)                       # (B, base_ch*2, 32, 32)
        h = self.dec2(h, t_emb)                       # (B, base_ch*2, 32, 32)
        
        # Level 1: 32x32 -> 64x64
        h = self.dec1_up(h, t_emb)                    # (B, base_ch, 64, 64)
        h = torch.cat([h, skip1], dim=1)              # (B, base_ch*2, 64, 64)
        h = self.dec1_reduce(h)                       # (B, base_ch, 64, 64)
        h = self.dec1(h, t_emb)                       # (B, base_ch, 64, 64)
        
        # ========== OUTPUT: Predict x_{t-1} (one refinement step) ==========
        x_prev_map = self.output_block(h).squeeze(1)  # (B, 64, 64)
        x_prev_vec = matrix_to_upper_tri_vec(x_prev_map)  # (B, vec_dim)
        
        return x_prev_vec


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
    Single training step for SR3-style iterative refinement.
    
    Model learns to predict x_{t-1} (less noisy image) from x_t (current noisy image).
    This is the core of SR3: iteratively refining from bulk Hi-C to phase-specific data.
    
    Args:
        model: SR3UNet for the current phase
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
    
    # Get ChIP-seq conditioning (already batched - kept for compatibility but unused)
    chip_1d = batch['chip_seq'].float().to(device)  # (batch_size, N)
    
    # SR3 TRAINING (Algorithm 1): Sample random noise level γ ~ p(γ)
    gamma_min, gamma_max = 1e-4, 0.02
    gamma_t = torch.rand(batch_size, 1, device=device) * (gamma_max - gamma_min) + gamma_min  # (batch_size, 1)
    
    # Sample random noise ϵ ~ N(0, I) - THIS IS THE TARGET!
    eps_true = torch.randn_like(x0_current)  # (batch_size, vec_dim)
    
    # SR3 forward process: y_γ = √γ · y_0 + √(1-γ) · ϵ
    # Following SR3 Algorithm 1 line 5
    sqrt_gamma_t = torch.sqrt(gamma_t)
    sqrt_one_minus_gamma_t = torch.sqrt(1.0 - gamma_t)
    y_gamma = sqrt_gamma_t * x0_current + sqrt_one_minus_gamma_t * eps_true
    
    # Map γ to timestep t for model conditioning
    t_continuous = ((gamma_t.squeeze() - gamma_min) / (gamma_max - gamma_min)) * (T - 1)
    t = t_continuous.long().clamp(0, T-1)  # (batch_size,)
    
    # SR3 MODEL: Predicts noise ε (Algorithm 1)
    eps_pred = model(y_gamma, t, chip_1d, x0_bulk_normalized)
    
    # SR3 LOSS: MSE between predicted noise and true noise
    # Algorithm 1 line 5: minimize ||f_θ(x, √γ y_0 + √(1-γ)ε, γ) - ε||²
    loss = F.mse_loss(eps_pred, eps_true)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def test_training(model, batch, device, step, phase_name):
    """
    Run inference and visualize results during training.
    
    Args:
        model: SR3UNet for the current phase
        batch: training batch
        device: torch device
        step: current training step number
        phase_name: 'earlyG1', 'midG1', or 'lateG1'
    
    Returns:
        Path to saved visualization
    """
    from inference import run_inference_and_visualize
    
    print(f"\n{'='*60}")
    print(f"Running inference at step {step}...")
    
    save_path = run_inference_and_visualize(
        model=model,
        batch=batch,
        phase_name=phase_name,
        device=device,
        step=step,
        output_dir="./inference_visualizations"
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
    
    # Initialize SR3-style U-Net model
    model = SR3UNet(
        vec_dim=VEC_DIM,
        n=N,
        time_embed_module=time_embed_module,
        base_ch=64            # Base channels for U-Net (64 -> 128 -> 256 -> 512)
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
            
            # Visualize every 200 steps (reduced frequency to save memory)
            if global_step % 200 == 0:
                test_training(model, batch, DEVICE, global_step, CURRENT_PHASE)
                
                # Clear CUDA memory cache to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()  # Force garbage collection
        
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
