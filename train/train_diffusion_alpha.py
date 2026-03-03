"""
Cell-Cycle Hi-C Phase Decomposition via SR3-Style Iterative Refinement

Train 4 separate conditional denoisers (one per phase: earlyG1/midG1/lateG1/anatelo)
Each denoiser iteratively refines noisy bulk Hi-C toward phase-specific data

SR3 NOTATION (from "Image Super-Resolution via Iterative Refinement"):
    γ_t: Noise level at timestep t (linearly spaced: 1e-4 → 1.0)
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
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, random_split

# Add preprocess dir to path
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocess"))
from Dataloader import CellCycleDataLoader

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
PHASES = ["earlyG1", "midG1", "lateG1", "anatelo"]

# MEMORY OPTIMIZATION: Train one phase at a time
# Change this to 'earlyG1', 'midG1', 'lateG1', or 'anatelo' to train different phases
CURRENT_PHASE = 'anatelo'  # <-- CHANGE THIS to train different phases

T = 1000              # diffusion steps
N = 64                # contact map size (64 x 64)
VEC_DIM = 2080        # upper triangular vector dimension (64*65/2)

# Genomic resolution and region size (in base pairs)
# Keeping N fixed at 64 ensures VEC_DIM stays 2080 independent of resolution.
RESOLUTION_BP = 10000          # bin size in base pairs (25kb; must match .hic expected vectors)
REGION_SIZE_BP = RESOLUTION_BP * N  # total region size in bp (64 bins)
L = 2                 # number of bottleneck blocks in U-Net
HIDDEN_DIM = 128      # base channel dimension for U-Net (reduced from 256)
d_t = 256             # time embedding dimension

BATCH_SIZE = 32       # Increased from 8 since we have more memory with single model
LR = 1e-4
NUM_EPOCHS = 1        # More epochs since we're training one at a time
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model checkpoints directory
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Resume training from checkpoint (set to None to start from scratch)
# Can be overridden via command-line argument --resume_checkpoint
# Example: RESUME_CHECKPOINT = "anatelo_epoch2_1-21.pth"
#RESUME_CHECKPOINT = "anatelo_epoch1_final_feb_9_2026_1.pth"
#RESUME_CHECKPOINT = "checkpoint_anatelo_epoch1_20260212_115923.pth"
RESUME_CHECKPOINT = None  # Default: start from scratch (can be set via CLI)





############################################
# 1) SR3 NOISE SCHEDULE
############################################
def sr3_noise_schedule(timesteps, gamma_min=1e-4, gamma_max=1.0):
    """
    SR3 inference schedule: γ decreases from 1.0 (pure noise) to ~0 (clean).
    
    During inference, we start at t=T (γ=1.0, pure noise) and denoise
    down to t=0 (γ≈0, clean image).
    
    Args:
        timesteps: Number of diffusion steps T
        gamma_min: Minimum noise level (end, almost clean) - default 1e-4
        gamma_max: Maximum noise level (start, pure noise) - default 1.0
    
    Returns:
        gammas: (T,) tensor of noise levels, decreasing from gamma_max to gamma_min
        alphas: (T,) tensor of step-wise alphas α_t = γ_{t-1} / γ_t
        alphas_cumprod: (T,) tensor (kept for compatibility, not used in SR3)
    """
    # at inference, we start with gamma close to 0 at T - 1 and end with gamma close to 1 at time step 0
    gammas = torch.linspace(gamma_max, gamma_min, T)

    alphas = torch.ones(T)
    
    # Compute α_t = γ_{t-1} / γ_t for reverse process
    alphas[1:] = gammas[1:] / (gammas[:-1] + 1e-10)
    
    # Cumulative product (not used in SR3, kept for compatibility)
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return gammas, alphas, alphas_cumprod


# Initialize SR3 inference schedule (only used during inference, not training)
gammas, alphas, alphas_cumprod = sr3_noise_schedule(T, gamma_min=1e-4, gamma_max=1.0)
# Note: Training does NOT use this schedule - it samples γ ~ Uniform(0,1) directly


############################################
# 2) NOISE LEVEL EMBEDDING (Gamma)
############################################
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, value):
        """
        value: (batch,) tensor of values to embed (e.g., scaled gamma)
        returns: (batch, dim) sinusoidal embeddings
        """
        device = value.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = value[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class NoiseEmbedding(nn.Module):
    """
    Embeds noise level γ ∈ [0, 1] using sinusoidal embeddings.
    
    In SR3, γ represents the noise variance:
    - γ = 0: clean image
    - γ = 1: pure noise
    """
    def __init__(self, dim, max_value=1000):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbeddings(dim)
        self.max_value = max_value
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, gamma):
        """
        gamma: (batch,) noise level γ (pre-scaled to ~[0, 1000])
        returns: (batch, dim) noise level embedding
        """
        emb = self.sinusoidal(gamma)
        return self.mlp(emb)


############################################
# 3) BIGGAN RESIDUAL BLOCK (SR3-style)
############################################
class BigGANResBlock(nn.Module):
    """
    BigGAN-style residual block with noise level conditioning via adaptive group norm.
    Used in SR3 paper for super-resolution diffusion.
    
    Inputs:
        x: (batch, in_channels, H, W) feature map
        noise_emb: (batch, noise_dim) noise level embedding (γ)
    """
    def __init__(self, in_channels, out_channels, noise_dim, up=False, down=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        
        # Group normalization
        num_groups = min(8, in_channels)
        self.gn1 = nn.GroupNorm(num_groups, in_channels)
        self.gn2 = nn.GroupNorm(min(8, out_channels), out_channels)
        
        # Noise level conditioning (adaptive group norm - scale and shift)
        self.noise_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(noise_dim, out_channels * 2)
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
    
    def forward(self, x, noise_emb):
        """
        x: (batch, in_channels, H, W)
        noise_emb: (batch, noise_dim) - embedding of noise level γ
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
        
        # Apply noise level conditioning (adaptive group norm - FiLM)
        noise_params = self.noise_proj(noise_emb)  # (batch, out_channels * 2)
        scale, shift = noise_params.chunk(2, dim=1)  # Each (batch, out_channels)
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # (batch, out_channels, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)  # (batch, out_channels, 1, 1)
        
        h = self.gn2(h)
        h = h * (1 + scale) + shift  # Adaptive group norm
        h = self.act(h)
        h = self.conv2(h)
        
        return h + residual
############################################
# 3.6) SELF-ATTENTION BLOCK
############################################
class SelfAttentionBlock(nn.Module):
    """
    Self-attention block for exchanging information between spatial positions.
    Similar to C.Origami's transformer, but in 2D.
    """
    def __init__(self, channels, n_heads=8):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.d_head = channels // n_heads

        assert channels % n_heads == 0

        # Layer norm
        self.norm = nn.GroupNorm(min(32, channels), channels)

        # Q, K, V projections (all from same input - that's what makes it self-attention!)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)

        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # Small initialization for stability
        nn.init.normal_(self.out_proj.weight, std=0.01)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map

        Returns:
            x_out: (B, C, H, W) attended feature map
        """
        B, C, H, W = x.shape

        # Layer norm
        h = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(h)  # (B, C*3, H, W)
        q, k, v = qkv.chunk(3, dim=1)  # Each (B, C, H, W)

        # Reshape for multi-head attention
        # (B, C, H, W) → (B, n_heads, H*W, d_head)
        q = q.view(B, self.n_heads, self.d_head, H * W).transpose(2, 3)  # (B, n_heads, H*W, d_head)
        k = k.view(B, self.n_heads, self.d_head, H * W).transpose(2, 3)
        v = v.view(B, self.n_heads, self.d_head, H * W).transpose(2, 3)

        # Attention: each spatial position attends to all others
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B, n_heads, H*W, H*W)
        attn = F.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, n_heads, H*W, d_head)

        # Merge heads and reshape back to spatial
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)

        # Output projection
        out = self.out_proj(out)

        # Residual connection
        return x + out


############################################
# 3.7) ChIP-SEQ PAIR ENCODER (AlphaFold-style: 4 tracks → outer product 4×4 per (i,j))
############################################
class ChipPairEncoderAlpha(nn.Module):
    """
    AlphaFold-style encoder for 4 ChIP tracks.

    1) Stack 4 ChIP tracks into an MSA-style tensor (B, s=4, r=64, c_msa).
       - Start from scalars (no channel dim) and embed to c_msa via a linear layer.
    2) Apply row-wise self-attention over residues (axis r) independently for each track.
    3) Apply column-wise self-attention over tracks (axis s) independently for each bin.
       Result: updated MSA representation of shape (B, 4, 64, c_msa).
    4) Compute outer-product mean over the MSA (as in AlphaFold):
           pair[i, j] = mean_s( msa[s, i] ⊗ msa[s, j] )  ∈ ℝ^{c_msa×c_msa}
       Flatten c_msa×c_msa and project to c_pair, then return (B, c_pair, 64, 64).
    """

    def __init__(self, n_bins: int = 64, c_msa: int = 32, c_pair: int = 16, n_heads: int = 4):
        super().__init__()
        self.n_bins = n_bins
        self.c_msa = c_msa
        self.c_pair = c_pair

        # Embed scalar ChIP signal at each (track, bin) into c_msa channels
        self.msa_embed = nn.Linear(1, c_msa)
        # Learnable positional encoding over residues (columns 0..n_bins-1), shared across tracks
        # Shape: (1, 1, n_bins, c_msa) → broadcast over batch and 4 tracks
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, n_bins, c_msa))

        # Row-wise self-attention over residues (length 64) for each track
        self.row_attn = nn.MultiheadAttention(embed_dim=c_msa, num_heads=n_heads, batch_first=True)
        self.row_norm = nn.LayerNorm(c_msa)

        # Column-wise self-attention over tracks (4) for each residue
        self.col_attn = nn.MultiheadAttention(embed_dim=c_msa, num_heads=min(n_heads, 4), batch_first=True)
        self.col_norm = nn.LayerNorm(c_msa)

        #  (c_msa × c_msa) flattened → project to c_pair
        self.pair_proj = nn.Sequential(
            nn.Linear(c_msa * c_msa, c_pair),
            nn.LayerNorm(c_pair),
            nn.SiLU(),
        )

    def forward(self, chip_ctcf, chip_hac, chip_me1, chip_me3):
        """
        Args:
            chip_ctcf: (B, 64) CTCF
            chip_hac:  (B, 64) histone ac (e.g. H3K27ac)
            chip_me1:  (B, 64) histone me1 (e.g. H3K4me1)
            chip_me3:  (B, 64) histone me3 (e.g. H3K4me3)
        Returns:
            pair_map: (B, c_pair, 64, 64) pairwise features to concatenate with h
        """
        B = chip_ctcf.shape[0]

        # Stack 4 tracks → (B, s=4, r=64) "MSA" (no channel dim yet)
        signals = torch.stack(
            [chip_ctcf.float(), chip_hac.float(), chip_me1.float(), chip_me3.float()],
            dim=1,
        )  # (B, 4, 64)

        # Add channel dim and embed scalars to c_msa features
        msa = signals.unsqueeze(-1)  # (B, 4, 64, 1)
        msa = self.msa_embed(msa)    # (B, 4, 64, c_msa)
        # Add residue positional encoding along columns (bins)
        msa = msa + self.pos_embed   # broadcast over batch and tracks

        # ----- Row-wise attention over residues (axis r) -----
        B, S, L, C = msa.shape  # S=4, L=n_bins, C=c_msa
        x = msa.view(B * S, L, C)                     # (B*S, L, C)
        row_out, _ = self.row_attn(x, x, x)           # self-attention over L
        x = self.row_norm(x + row_out)                # (B*S, L, C)
        msa = x.view(B, S, L, C)                      # (B, 4, 64, C)

        # ----- Column-wise attention over tracks (axis s) -----
        # Treat each residue independently, attend over 4 tracks
        x = msa.permute(0, 2, 1, 3).reshape(B * L, S, C)  # (B*L, 4, C)
        col_out, _ = self.col_attn(x, x, x)
        x = self.col_norm(x + col_out)                    # (B*L, 4, C)
        msa = x.view(B, L, S, C).permute(0, 2, 1, 3)      # (B, 4, 64, C)

        # ----- Outer-product mean over MSA (AlphaFold-style) -----
        # Mean over tracks (sequences) → (B, 64, C)
        msa_mean = msa.mean(dim=1)  # (B, L, C)

        # For each (i,j): outer product of feature vectors at bins i and j
        pair_2d = torch.einsum("bic,bjd->bijcd", msa_mean, msa_mean)  # (B, L, L, C, C)
        pair_flat = pair_2d.reshape(B, self.n_bins, self.n_bins, C * C)  # (B, 64, 64, C*C)

        # Normalize over the flattened channel dimension
        pair_flat = F.normalize(pair_flat, dim=-1, eps=1e-6)

        # Project to c_pair
        pair_feat = self.pair_proj(pair_flat)               # (B, 64, 64, c_pair)
        pair_map = pair_feat.permute(0, 3, 1, 2)            # (B, c_pair, 64, 64)
        return pair_map


############################################
# 4) SR3-STYLE U-NET (Denoised Image Predictor)
############################################
class SR3UNet(nn.Module):
    """
    SR3-style U-Net that predicts noise ε (following Algorithm 1).

    Architecture:
        - Input: noisy (1) + bulk Hi-C (1) only → (B, 2, 64, 64); ChIP is supplemental
        - ChIP-seq: 4 tracks (CTCF, H3K27ac, H3K4me1, H3K4me3) → AlphaFold-style outer product
          per (i,j) → 4×4 flattened, normalized, projected; concat with h and project at 64×64
        - Four downsampling stages: 64 → 32 → 16 → 8
        - BigGAN residual blocks + self-attention at 16×16 (encoder/decoder) and 8×8 (bottleneck)
        - Standard U-Net skip connections (concatenation)
        - Output: Predicted noise ε
    """
    def __init__(self, vec_dim, n, noise_embed_module, base_ch: int = 64, c_pair: int = 16):
        super().__init__()
        self.vec_dim = vec_dim
        self.n = n
        self.noise_embed = noise_embed_module
        self.base_ch = base_ch
        self.c_pair = c_pair

        noise_dim = self.noise_embed.mlp[-1].out_features

        # Input: noisy + bulk ONLY (foundation)
        self.input_conv = nn.Conv2d(2, base_ch, kernel_size=3, padding=1)

        # ChIP: AlphaFold-style 4 tracks → outer product 4×4 per (i,j) → (B, c_pair, 64, 64)
        self.chip_pair_encoder = ChipPairEncoderAlpha(n_bins=64, c_pair=c_pair)
        # Concat h (base_ch) + pair_map (c_pair) then project back to base_ch
        self.chip_combine_64 = nn.Sequential(
            nn.Conv2d(base_ch + c_pair, base_ch, kernel_size=1),
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.SiLU(),
        )

        # ---- ENCODER ----
        self.enc1 = BigGANResBlock(base_ch, base_ch, noise_dim)
        self.enc1_down = BigGANResBlock(base_ch, base_ch * 2, noise_dim, down=True)

        self.enc2 = BigGANResBlock(base_ch * 2, base_ch * 2, noise_dim)
        self.enc2_down = BigGANResBlock(base_ch * 2, base_ch * 4, noise_dim, down=True)

        self.enc3 = BigGANResBlock(base_ch * 4, base_ch * 4, noise_dim)
        self.enc3_self_attn = SelfAttentionBlock(base_ch * 4)
        self.enc3_down = BigGANResBlock(base_ch * 4, base_ch * 8, noise_dim, down=True)

        # ---- BOTTLENECK: 8x8 @ base_ch*8 with self-attention ----
        self.bottleneck = nn.ModuleList([
            BigGANResBlock(base_ch * 8, base_ch * 8, noise_dim),
            SelfAttentionBlock(base_ch * 8),
            BigGANResBlock(base_ch * 8, base_ch * 8, noise_dim),
        ])

        # ---- DECODER ----
        self.dec3_up = BigGANResBlock(base_ch * 8, base_ch * 4, noise_dim, up=True)
        self.dec3_reduce = nn.Conv2d(base_ch * 8, base_ch * 4, kernel_size=1)
        self.dec3 = BigGANResBlock(base_ch * 4, base_ch * 4, noise_dim)
        self.dec3_self_attn = SelfAttentionBlock(base_ch * 4)

        self.dec2_up = BigGANResBlock(base_ch * 4, base_ch * 2, noise_dim, up=True)
        self.dec2_reduce = nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=1)
        self.dec2 = BigGANResBlock(base_ch * 2, base_ch * 2, noise_dim)

        self.dec1_up = BigGANResBlock(base_ch * 2, base_ch, noise_dim, up=True)
        self.dec1_reduce = nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)
        self.dec1 = BigGANResBlock(base_ch, base_ch, noise_dim)

        # ---- OUTPUT ----
        self.output_block = nn.Sequential(
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)
        )
        nn.init.zeros_(self.output_block[-1].weight)
        nn.init.zeros_(self.output_block[-1].bias)

    def forward(self, x_t_vec, gamma, chip_ctcf, chip_hac, chip_me1, chip_me3, bulk_vec):
        """
        SR3 forward pass: Predict noise ε given noisy input y_γ.

        ChIP-seq: 4 tracks (CTCF, H3K27ac, H3K4me1, H3K4me3) → AlphaFold-style outer product
        per (i,j) → 4×4 flattened, normalized, projected; then concatenated with h and projected.

        Args:
            x_t_vec:   (B, vec_dim) noisy Hi-C vector y_γ
            gamma:    (B,) or (B, 1) noise level γ ∈ [0, 1]
            chip_ctcf: (B, 64) CTCF ChIP-seq
            chip_hac:  (B, 64) histone ac (e.g. H3K27ac)
            chip_me1:  (B, 64) histone me1 (e.g. H3K4me1)
            chip_me3:  (B, 64) histone me3 (e.g. H3K4me3)
            bulk_vec:  (B, vec_dim) bulk Hi-C vector (conditioning)

        Returns:
            eps_vec: (B, vec_dim) predicted noise ε
        """
        B = x_t_vec.shape[0]
        N = self.n

        if gamma.dim() == 2:
            gamma = gamma.squeeze(-1)
        gamma_scaled = gamma * 999.0
        noise_emb = self.noise_embed(gamma_scaled)

        x_t_map = upper_tri_vec_to_matrix(x_t_vec, N).unsqueeze(1)     # (B, 1, 64, 64)
        bulk_map = upper_tri_vec_to_matrix(bulk_vec, N).unsqueeze(1)   # (B, 1, 64, 64)

        # Foundation: noisy + bulk only
        x_in = torch.cat([x_t_map, bulk_map], dim=1)                    # (B, 2, 64, 64)
        h = self.input_conv(x_in)                                       # (B, base_ch, 64, 64)

        # ChIP: 4 tracks → outer product 4×4 per (i,j) → (B, c_pair, 64, 64); concat with h and project
        chip_pair_64 = self.chip_pair_encoder(chip_ctcf, chip_hac, chip_me1, chip_me3)
        h = self.chip_combine_64(torch.cat([h, chip_pair_64], dim=1))

        # ========== ENCODER ==========
        # 64×64, C = base_ch
        h = self.enc1(h, noise_emb)
        skip1 = h
        h = self.enc1_down(h, noise_emb)                                # (B, base_ch*2, 32, 32)

        # 32×32, C = base_ch*2
        h = self.enc2(h, noise_emb)
        skip2 = h
        h = self.enc2_down(h, noise_emb)                                # (B, base_ch*4, 16, 16)

        # 16×16, C = base_ch*4
        h = self.enc3(h, noise_emb)
        h = self.enc3_self_attn(h)
        skip3 = h
        h = self.enc3_down(h, noise_emb)                                # (B, base_ch*8, 8, 8)

        # ========== BOTTLENECK: 8x8 ==========
        for block in self.bottleneck:
            h = block(h, noise_emb) if isinstance(block, BigGANResBlock) else block(h)

        # ========== DECODER ==========
        # 16×16, C = base_ch*4
        h = self.dec3_up(h, noise_emb)                                  # (B, base_ch*4, 16, 16)
        h = torch.cat([h, skip3], dim=1)                                # (B, base_ch*8, 16, 16)
        h = self.dec3_reduce(h)
        h = self.dec3(h, noise_emb)
        #h = self.dec3_self_attn(h)

        # 32×32, C = base_ch*2
        h = self.dec2_up(h, noise_emb)                                  # (B, base_ch*2, 32, 32)
        h = torch.cat([h, skip2], dim=1)
        h = self.dec2_reduce(h)
        h = self.dec2(h, noise_emb)

        # 64×64, C = base_ch (ChIP pair concat only here)
        h = self.dec1_up(h, noise_emb)                                  # (B, base_ch, 64, 64)
        h = torch.cat([h, skip1], dim=1)
        h = self.dec1_reduce(h)
        h = self.chip_combine_64(torch.cat([h, chip_pair_64], dim=1))
        h = self.dec1(h, noise_emb)

        eps_map = self.output_block(h).squeeze(1)
        eps_vec = matrix_to_upper_tri_vec(eps_map)
        return eps_vec


############################################
# 6) FORWARD NOISING - NOT USED IN SR3
############################################
# SR3 training uses direct gamma sampling instead of timestep-based noising:
# y_γ = √γ·y_0 + √(1-γ)·ε where γ ~ Uniform(0,1)
# The q_sample function is not needed for SR3 training.


############################################
# 5.5) CHECKPOINT LOADING
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
    
    # Resolve checkpoint path
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
# 5.7) VALIDATION SET (chr2, excluding test-eval regions)
############################################
# Ranges used by run_test_evaluation_chromosome2.sh – we exclude these so validation ≠ test eval
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
    """True if region (start, end) overlaps any (t_start, t_end) in ranges."""
    _, start, end = _parse_region(region_str)
    for t_start, t_end in ranges:
        if start < t_end and end > t_start:
            return True
    return False


def get_validation_regions_chr2(holdout_regions, n=10, seed=42):
    """
    From chr2 holdout regions, exclude those used in run_test_evaluation_chromosome2
    (44.7–45.1 Mb and 18.4–19.4 Mb), then return n regions for validation.
    """
    rng = np.random.default_rng(seed)
    # Exclude regions that overlap test-eval target ranges
    valid = [r for r in holdout_regions if not _region_overlaps_any(r, TEST_EVAL_TARGET_RANGES_CHR2)]
    if len(valid) <= n:
        return valid
    indices = rng.choice(len(valid), size=n, replace=False)
    return [valid[i] for i in indices]


############################################
# 6) TRAINING LOOP
############################################
def eval_batch_loss(model, batch, device, phase_name, generator: torch.Generator | None = None):
    """
    Compute SR3 MSE loss for one batch (no backward). Uses clean bulk (no corruption).
    """
    x0_early = batch["earlyG1"].float().to(device)
    x0_mid = batch["midG1"].float().to(device)
    x0_late = batch["lateG1"].float().to(device)
    x0_anatelo = batch["anatelo"].float().to(device)
    phase_data = {"earlyG1": x0_early, "midG1": x0_mid, "lateG1": x0_late, "anatelo": x0_anatelo}
    x0_current = phase_data[phase_name]
    x0_bulk_normalized = (x0_early + x0_mid + x0_late + x0_anatelo) / 4
    batch_size = x0_bulk_normalized.shape[0]
    bulk_for_model = x0_bulk_normalized

    chip_ctcf = batch["chip_seq_ctcf"].float().to(device)
    chip_hac = batch["chip_seq_hac"].float().to(device)
    chip_me1 = batch["chip_seq_h3k4me1"].float().to(device)
    chip_me3 = batch["chip_seq_h3k4me3"].float().to(device)

    # Use a local generator if provided so validation gamma/eps are deterministic
    if generator is not None:
        gamma_t = torch.rand(batch_size, 1, device=device, generator=generator)
        eps_true = torch.randn(x0_current.shape, device=device, generator=generator)
    else:
        gamma_t = torch.rand(batch_size, 1, device=device)
        eps_true = torch.randn_like(x0_current)
    sqrt_gamma_t = torch.sqrt(gamma_t)
    sqrt_one_minus_gamma_t = torch.sqrt(1.0 - gamma_t)
    y_gamma = sqrt_gamma_t * x0_current + sqrt_one_minus_gamma_t * eps_true

    # 4 tracks: CTCF, H3K27ac, H3K4me1, H3K4me3
    eps_pred = model(y_gamma, gamma_t.squeeze(), chip_ctcf, chip_hac, chip_me1, chip_me3, bulk_for_model)
    return F.mse_loss(eps_pred, eps_true).item()


def compute_validation_loss(model, val_dataloader, device, phase_name):
    """Average loss over validation set (model in eval mode, no grad)."""
    model.eval()
    # Local RNG for validation so gamma/eps are the same every validation call
    gen = torch.Generator(device=device)
    gen.manual_seed(12345)
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_dataloader:
            total_loss += eval_batch_loss(model, batch, device, phase_name, generator=gen)
            n_batches += 1
    model.train()
    return total_loss / n_batches if n_batches else 0.0


def train_step(model, optimizer, batch, device, phase_name, global_step=0):
    """
    Single training step for SR3-style iterative refinement.

    Model learns to predict x_{t-1} (less noisy image) from x_t (current noisy image).
    This is the core of SR3: iteratively refining from bulk Hi-C to phase-specific data.

    Args:
        model: SR3UNet for the current phase
        optimizer: optimizer for this model
        batch: dict with keys 'region', 'earlyG1', 'midG1', 'lateG1', 'anatelo', 'chip_seq'
        device: torch device
        phase_name: 'earlyG1', 'midG1', 'lateG1', or 'anatelo'
        global_step: current training step

    Returns:
        float: loss for this phase
    """
    # Extract clean x0 for each phase (needed for bulk conditioning)
    x0_early = batch['earlyG1'].float().to(device)  # (batch_size, vec_dim)
    x0_mid = batch['midG1'].float().to(device)
    x0_late = batch['lateG1'].float().to(device)
    x0_anatelo = batch['anatelo'].float().to(device)

    # Select the current phase's ground truth
    phase_data = {
        'earlyG1': x0_early,
        'midG1': x0_mid,
        'lateG1': x0_late,
        'anatelo': x0_anatelo
    }
    x0_current = phase_data[phase_name]  # (batch_size, vec_dim)

    # Compute bulk Hi-C (average of four phases) for conditioning
    x0_bulk_normalized = (x0_early + x0_mid + x0_late + x0_anatelo) / 4  # (batch_size, vec_dim)
    batch_size = x0_bulk_normalized.shape[0]

    # Get ChIP-seq conditioning: 4 tracks (CTCF, H3K27ac, H3K4me1, H3K4me3)
    chip_ctcf = batch['chip_seq_ctcf'].float().to(device)      # (batch_size, N)
    chip_hac = batch['chip_seq_hac'].float().to(device)        # (batch_size, N)
    chip_me1 = batch['chip_seq_h3k4me1'].float().to(device)    # (batch_size, N)
    chip_me3 = batch['chip_seq_h3k4me3'].float().to(device)    # (batch_size, N)
    
    # SR3 TRAINING (Algorithm 1): Sample random noise level γ ~ Uniform(0, 1)
    # γ represents the noise variance: 0 = clean, 1 = pure noise
    gamma_t = torch.rand(batch_size, 1, device=device)  # Uniform[0, 1]
    
    # Sample random noise ϵ ~ N(0, I) - THIS IS THE TARGET!
    eps_true = torch.randn_like(x0_current)  # (batch_size, vec_dim)
    
    # SR3 forward process: y_γ = √γ · y_0 + √(1-γ) · ϵ
    # Following SR3 Algorithm 1 line 5
    sqrt_gamma_t = torch.sqrt(gamma_t)
    sqrt_one_minus_gamma_t = torch.sqrt(1.0 - gamma_t)
    y_gamma = sqrt_gamma_t * x0_current + sqrt_one_minus_gamma_t * eps_true
    
    # SR3 MODEL: Predicts noise ε given (y_γ, γ, conditioning)
    # No timesteps in training - γ is passed directly!
    eps_pred = model(y_gamma, gamma_t.squeeze(), chip_ctcf, chip_hac, chip_me1, chip_me3, x0_bulk_normalized)
    
    # SR3 LOSS: MSE between predicted noise and true noise
    # Algorithm 1 line 5: minimize ||f_θ(x, √γ y_0 + √(1-γ)ε, γ) - ε||²
    loss = F.mse_loss(eps_pred, eps_true)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

############################################
# 6.5) CHECKPOINT LOADING
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
    
    # Resolve checkpoint path
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
# 7) MAIN TRAINING
############################################
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train diffusion model for Hi-C phase decomposition')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from (relative to checkpoints/ or absolute). Overrides RESUME_CHECKPOINT constant.')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of epochs to train (overrides NUM_EPOCHS constant). Default: use NUM_EPOCHS from config.')
    args = parser.parse_args()
    
    # Use command-line arguments if provided, otherwise use constants
    resume_checkpoint = args.resume_checkpoint if args.resume_checkpoint is not None else RESUME_CHECKPOINT
    num_epochs = args.num_epochs if args.num_epochs is not None else NUM_EPOCHS
    
    print("="*80)
    print(f"TRAINING PHASE: {CURRENT_PHASE}")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Vector dimension: {VEC_DIM}, Matrix size: {N}x{N}")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {num_epochs}")
    if resume_checkpoint:
        print(f"Resume checkpoint: {resume_checkpoint}")
    else:
        print("Starting from scratch (no checkpoint)")
    
    # Create noise level embedding module (for gamma)
    noise_embed_module = NoiseEmbedding(d_t, max_value=1000)
    
    # Initialize SR3-style U-Net model
    model = SR3UNet(
        vec_dim=VEC_DIM,
        n=N,
        noise_embed_module=noise_embed_module,
        base_ch=48            # Base channels for U-Net (64 -> 128 -> 256 -> 512)
    ).to(DEVICE)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print(f"Estimated memory: ~{num_params * 4 / 1e9:.2f} GB (fp32)")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Load checkpoint if specified
    start_epoch, global_step, best_loss = load_checkpoint_for_training(
        resume_checkpoint, model, optimizer, DEVICE
    )
    
    # Load data
    data_dir = Path(__file__).parent.parent / "raw_data" / "zhang_4dn"
    print(f"Loading data from: {data_dir}")
    
    # Holdout chromosome 2 for testing
    HOLD_OUT_CHROMOSOME = "2"
    
    # Shared DataLoader kwargs; we keep N=64 so VEC_DIM stays 2080 while changing resolution.
    base_loader_kwargs = dict(
        data_dir=data_dir,
        resolution=RESOLUTION_BP,
        region_size=REGION_SIZE_BP,
        normalization="KR",
        hold_out_chromosome=HOLD_OUT_CHROMOSOME,
        hic_data_type="oe",  # Use observed/expected data
        use_log_transform=True,  # Apply log1p transformation (model trained with this)
        normalization_stats_file=data_dir / "normalization_stats.csv",
    )

    # Training: randomly flip 50% of contact maps. Testing: no flip.
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
    
    # Create training dataset (excludes holdout chromosome)
    train_dataset = CellCycleDataset(cell_cycle_loader_train)
    
    # Create test dataset from holdout chromosome
    holdout_regions = cell_cycle_loader_eval.get_holdout_regions()
    if len(holdout_regions) == 0:
        raise ValueError(f"No regions found for holdout chromosome '{HOLD_OUT_CHROMOSOME}'")
    
    # Create a separate loader for holdout regions
    # We'll create a custom dataset that indexes by region string
    class HoldoutDataset(Dataset):
        """Dataset for holdout chromosome regions."""
        def __init__(self, loader, holdout_regions):
            self.loader = loader
            self.holdout_regions = holdout_regions
        
        def __len__(self):
            return len(self.holdout_regions)
        
        def __getitem__(self, idx):
            region_str = self.holdout_regions[idx]
            return self.loader[region_str]
    
    test_dataset = HoldoutDataset(cell_cycle_loader_eval, holdout_regions)

    # Validation set: 30 chr2 samples, excluding regions used in run_test_evaluation_chromosome2
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

    # Create dataloaders
    train_dataloader = TorchDataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with hicstraw
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_dataloader = TorchDataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Number of batches per epoch: {len(train_dataloader)}")
    print("="*80)

    # Training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_losses = []
        model.train()

        # Iterate through training batches
        total_epochs = start_epoch + num_epochs
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [{CURRENT_PHASE}]")
        for batch_idx, batch in enumerate(pbar):
            loss = train_step(model, optimizer, batch, DEVICE, CURRENT_PHASE, global_step)
            epoch_losses.append(loss)
            global_step += 1

            # Validation loss every 100 iterations
            if global_step % 100 == 0:
                val_loss = compute_validation_loss(model, val_dataloader, DEVICE, CURRENT_PHASE)
                print(f"  [step {global_step}] val_loss = {val_loss:.6f}")

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss:.4f}"})
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        total_epochs = start_epoch + num_epochs
        print(f"\nEpoch {epoch+1}/{total_epochs} - Average Loss: {avg_loss:.6f}")
        
        # save checkpoint with data type and log transform info
        data_type_str = cell_cycle_loader_train.hic_data_type
        log_str = "log" if cell_cycle_loader_train.use_log_transform else "nolog"
        checkpoint_path = CHECKPOINT_DIR / f"{data_type_str}_{log_str}_{CURRENT_PHASE}_epoch{epoch+1}_3-2_alpha.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'global_step': global_step,
        }, checkpoint_path)
        print(f"✓ Saved epoch checkpoint: {checkpoint_path}")
    
    # Training complete - run test inference
    print("\n" + "="*80)
    print(f"Training complete for {CURRENT_PHASE}!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    print("="*80)
    
    # Clear memory after test inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n{'='*80}")
    print(f"All tasks complete for {CURRENT_PHASE}!")
    print("="*80)
    
    # Cleanup
    cell_cycle_loader_train.close()
    cell_cycle_loader_eval.close()


if __name__ == "__main__":
    main()
