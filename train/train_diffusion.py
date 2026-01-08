"""
Cell-Cycle Hi-C Phase Decomposition via Conditional DDPM

Train 3 separate conditional denoisers (one per phase: earlyG1/midG1/lateG1)
Each denoiser predicts injected Gaussian noise ε from a noisy phase map x_t

Architecture: U-Net with Convolutional Layers
    - Converts upper triangular Hi-C vector to 2D symmetric matrix
    - Encoder-decoder with skip connections
    - FiLM conditioning (scale & shift) at each layer

Conditioning signals:
    (1) diffusion time t  -> FiLM parameters for each conv block
    (2) ChIP-seq 1D track -> FiLM parameters for each conv block

Loss: MSE between predicted noise and true noise (standard DDPM)
    loss = MSE(ε_pred_early, ε_early) + MSE(ε_pred_mid, ε_mid) + MSE(ε_pred_late, ε_late)
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
T = 1000              # diffusion steps
N = 50                # contact map size (50 x 50)
VEC_DIM = 1275        # upper triangular vector dimension (50*51/2)
L = 2                 # number of bottleneck blocks in U-Net
HIDDEN_DIM = 256      # base channel dimension for U-Net
d_t = 256             # time embedding dimension

BATCH_SIZE = 64
LR = 1e-4
NUM_EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




############################################
# 1) DIFFUSION SCHEDULE (DDPM)
############################################
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
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
        t: (batch,) normalized timestep in [0, 1]
        returns: (batch, dim)
        """
        # Scale back to original range for sinusoidal encoding
        t_scaled = t * (self.max_timesteps - 1)
        emb = self.sinusoidal(t_scaled)
        return self.mlp(emb)




############################################
# 3) CONVOLUTIONAL RESIDUAL BLOCK
############################################
class ConvResBlock(nn.Module):
    """
    Convolutional residual block with batch normalization and conditioning.
    
    Inputs:
        x: (batch, channels, H, W) feature map
        cond: (batch, cond_dim) conditioning vector (time + chip combined)
    """
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.channels = channels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Conditioning projection (FiLM: scale and shift)
        self.cond_proj = nn.Linear(cond_dim, channels * 2)
        
        self.act = nn.SiLU()
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x, cond):
        """
        x: (batch, channels, H, W)
        cond: (batch, cond_dim) - conditioning
        """
        residual = x
        
        # Project conditioning to scale and shift
        cond_params = self.cond_proj(cond)  # (batch, channels * 2)
        scale, shift = cond_params.chunk(2, dim=1)  # Each (batch, channels)
        scale = scale.unsqueeze(-1).unsqueeze(-1)  # (batch, channels, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)  # (batch, channels, 1, 1)
        
        # First conv + FiLM conditioning
        h = self.conv1(x)
        h = self.bn1(h)
        h = h * (1 + scale) + shift  # FiLM: scale and shift
        h = self.act(h)
        h = self.dropout(h)
        
        # Second conv
        h = self.conv2(h)
        h = self.bn2(h)
        
        return self.act(h + residual)


############################################
# 4) CONVOLUTIONAL EPSILON NETWORK (U-Net Style)
############################################
class EpsilonNet(nn.Module):
    """
    Convolutional U-Net denoiser with conditioning from time and ChIP-seq.
    
    Architecture:
        - Converts upper triangular vector to 2D matrix
        - U-Net encoder-decoder with skip connections
        - FiLM conditioning at each layer from time + ChIP
        - Converts back to upper triangular vector
    """
    def __init__(self, vec_dim=VEC_DIM, n=N, hidden_dim=512, num_blocks=L, time_emb_dim=d_t, max_timesteps=T):
        super().__init__()
        self.vec_dim = vec_dim
        self.n = n
        self.matrix_size = n  # 50x50 matrix
        self.hidden_dim = hidden_dim
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_emb_dim, max_timesteps=max_timesteps)
        
        # ChIP embedding: 1D signal -> embedding vector
        self.chip_embed = nn.Sequential(
            nn.Linear(n, 256),
            nn.SiLU(),
            nn.Linear(256, 512)
        )
        
        # Combined conditioning dimension
        cond_dim = time_emb_dim + 512
        
        # Initial conv: 1 channel -> 64 channels
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        
        # Encoder (downsampling)
        self.enc1 = ConvResBlock(64, cond_dim)
        self.down1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        self.enc2 = ConvResBlock(128, cond_dim)
        self.down2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvResBlock(256, cond_dim),
            ConvResBlock(256, cond_dim)
        )
        
        # Decoder (upsampling) - use interpolation + conv instead of transposed conv
        self.up1_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec1 = ConvResBlock(128 + 128, cond_dim)  # +128 from skip connection
        
        self.up2_conv = nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1)
        self.dec2 = ConvResBlock(64 + 64, cond_dim)  # +64 from skip connection
        
        # Output conv: channels -> 1
        self.output_conv = nn.Conv2d(64 + 64, 1, kernel_size=3, padding=1)
        
        # Initialize output to near zero
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
    
    def forward(self, x_t_vec, t, chip_1d):
        """
        Process noisy Hi-C vector through convolutional U-Net.
        
        x_t_vec: (batch, vec_dim) noisy phase vector in [-1, 1]
        t: (batch,) normalized timestep in [0, 1]
        chip_1d: (batch, n) 1D chip track in [-1, 1]
        
        returns: eps_hat_vec (batch, vec_dim) - predicted noise
        """
        batch_size = x_t_vec.shape[0]
        
        # Time embedding
        t_emb = self.time_embed(t)  # (batch, time_emb_dim)
        
        # ChIP embedding
        chip_emb = self.chip_embed(chip_1d)  # (batch, 512)
        
        # Combine conditioning
        cond = torch.cat([t_emb, chip_emb], dim=1)  # (batch, cond_dim)
        
        # Convert upper triangular vector to full symmetric matrix
        x_t_matrix = upper_tri_vec_to_matrix(x_t_vec, self.matrix_size)  # (batch, n, n)
        x_t_matrix = x_t_matrix.unsqueeze(1)  # (batch, 1, n, n)
        
        # Initial conv
        x = self.input_conv(x_t_matrix)  # (batch, 64, n, n)
        
        # Encoder
        enc1 = self.enc1(x, cond)  # (batch, 64, n, n)
        x = self.down1(enc1)  # (batch, 128, n/2, n/2)
        
        enc2 = self.enc2(x, cond)  # (batch, 128, n/2, n/2)
        x = self.down2(enc2)  # (batch, 256, n/4, n/4)
        
        # Bottleneck
        for block in self.bottleneck:
            x = block(x, cond)
        
        # Decoder with interpolation upsampling (matches encoder sizes exactly)
        x = F.interpolate(x, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        x = self.up1_conv(x)  # (batch, 128, n/2, n/2)
        x = torch.cat([x, enc2], dim=1)  # Skip connection
        x = self.dec1(x, cond)
        
        x = F.interpolate(x, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        x = self.up2_conv(x)  # (batch, 64, n, n)
        x = torch.cat([x, enc1], dim=1)  # Skip connection
        x = self.dec2(x, cond)
        
        # Output
        eps_matrix = self.output_conv(x)  # (batch, 1, n, n)
        eps_matrix = eps_matrix.squeeze(1)  # (batch, n, n)
        
        # Convert back to upper triangular vector
        eps_hat_vec = matrix_to_upper_tri_vec(eps_matrix)  # (batch, vec_dim)
        
        return eps_hat_vec


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
def train_step(models, optimizers, batch, device):
    """
    Single training step for all three phase denoisers.
    
    batch: dict with keys 'region', 'earlyG1', 'midG1', 'lateG1', 'chip_seq'
          Each value is already batched by PyTorch DataLoader: (batch_size, dim)
    """
    # Extract clean x0 for each phase (already batched)
    x0_early = batch['earlyG1'].float().to(device)  # (batch_size, vec_dim)
    x0_mid = batch['midG1'].float().to(device)
    x0_late = batch['lateG1'].float().to(device)
    
    # Get ChIP-seq conditioning (already batched)
    chip_1d = batch['chip_seq'].float().to(device)  # (batch_size, N)
    
    # Sample random timestep for each sample in batch
    batch_size = x0_early.shape[0]
    t = torch.randint(0, T, (batch_size,), device=device).long()  # (batch_size,)
    t_normalized = t.float() / (T - 1)  # normalize to [0, 1] for model input
    
    # Sample random noise (epsilon) for each phase
    eps_early = torch.randn_like(x0_early)  # (1, vec_dim)
    eps_mid = torch.randn_like(x0_mid)
    eps_late = torch.randn_like(x0_late)
    
    # Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    x_t_early = q_sample(x0_early, t, eps_early)
    x_t_mid = q_sample(x0_mid, t, eps_mid)
    x_t_late = q_sample(x0_late, t, eps_late)
    
    # Predict the noise that was added
    eps_hat_early = models['earlyG1'](x_t_early, t_normalized, chip_1d)
    eps_hat_mid = models['midG1'](x_t_mid, t_normalized, chip_1d)
    eps_hat_late = models['lateG1'](x_t_late, t_normalized, chip_1d)
    
    # DDPM loss: MSE between predicted noise and true noise
    loss_early = F.mse_loss(eps_hat_early, eps_early)
    loss_mid = F.mse_loss(eps_hat_mid, eps_mid)
    loss_late = F.mse_loss(eps_hat_late, eps_late)
    
    # Total loss: sum of all three phase losses
    loss = loss_early + loss_mid + loss_late
    
    # Backward pass
    for opt in optimizers.values():
        opt.zero_grad()
    loss.backward()
    for opt in optimizers.values():
        opt.step()
    
    return {
        'loss': loss.item(),
        'loss_early': loss_early.item(),
        'loss_mid': loss_mid.item(),
        'loss_late': loss_late.item(),
    }


############################################
# 7) MAIN TRAINING
############################################
def main():
    print(f"Device: {DEVICE}")
    print(f"Vector dimension: {VEC_DIM}, Matrix size: {N}x{N}")
    
    # Initialize models (one per phase)
    models = {
        'earlyG1': EpsilonNet(hidden_dim=HIDDEN_DIM, num_blocks=L).to(DEVICE),
        'midG1': EpsilonNet(hidden_dim=HIDDEN_DIM, num_blocks=L).to(DEVICE),
        'lateG1': EpsilonNet(hidden_dim=HIDDEN_DIM, num_blocks=L).to(DEVICE),
    }
    
    # Count parameters
    num_params = sum(p.numel() for p in models['earlyG1'].parameters())
    print(f"Parameters per model: {num_params:,}")
    
    # Optimizers
    optimizers = {
        phase: torch.optim.Adam(models[phase].parameters(), lr=LR)
        for phase in PHASES
    }
    
    # Load data
    data_dir = Path(__file__).parent.parent / "raw_data" / "zhang_4dn"
    print(f"Loading data from: {data_dir}")
    
    cell_cycle_loader = CellCycleDataLoader(
        data_dir=data_dir,
        resolution=10000,
        region_size=500000,
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
    
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches per epoch: {len(dataloader)}")
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        
        # Iterate through batches
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in pbar:
            losses = train_step(models, optimizers, batch, DEVICE)
            epoch_losses.append(losses)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['loss']:.4f}",
                'early': f"{losses['loss_early']:.4f}",
                'mid': f"{losses['loss_mid']:.4f}",
                'late': f"{losses['loss_late']:.4f}",
            })
        
        # Epoch summary
        avg_loss = np.mean([l['loss'] for l in epoch_losses])
        avg_early = np.mean([l['loss_early'] for l in epoch_losses])
        avg_mid = np.mean([l['loss_mid'] for l in epoch_losses])
        avg_late = np.mean([l['loss_late'] for l in epoch_losses])
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Early: {avg_early:.4f}, Mid: {avg_mid:.4f}, Late: {avg_late:.4f}")
    
    # Cleanup
    cell_cycle_loader.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
