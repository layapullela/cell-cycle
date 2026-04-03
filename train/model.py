import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocess"))
from utils import matrix_to_upper_tri_vec, upper_tri_vec_to_matrix


############################################
# NOISE LEVEL EMBEDDING (Gamma)
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
# BIGGAN RESIDUAL BLOCK (SR3-style)
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

        num_groups = min(8, in_channels)
        self.gn1 = nn.GroupNorm(num_groups, in_channels)
        self.gn2 = nn.GroupNorm(min(8, out_channels), out_channels)

        # Noise level conditioning (adaptive group norm - scale and shift)
        self.noise_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(noise_dim, out_channels * 2)
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

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
        residual = x

        if self.up:
            residual = F.interpolate(residual, scale_factor=2, mode='nearest')
        elif self.down:
            residual = F.max_pool2d(residual, kernel_size=2, stride=2)

        residual = self.residual_conv(residual)

        h = self.gn1(x)
        h = self.act(h)

        if self.up:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
        elif self.down:
            h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.conv1(h)

        # Adaptive group norm (FiLM)
        noise_params = self.noise_proj(noise_emb)           # (batch, out_channels * 2)
        scale, shift = noise_params.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        h = self.gn2(h)
        h = h * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)

        return h + residual


############################################
# SELF-ATTENTION BLOCK
############################################
class SelfAttentionBlock(nn.Module):
    """Self-attention block for exchanging information between spatial positions."""
    def __init__(self, channels, n_heads=8):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.d_head = channels // n_heads

        assert channels % n_heads == 0

        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

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

        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.n_heads, self.d_head, H * W).transpose(2, 3)
        k = k.view(B, self.n_heads, self.d_head, H * W).transpose(2, 3)
        v = v.view(B, self.n_heads, self.d_head, H * W).transpose(2, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        out = self.out_proj(out)

        return x + out


############################################
# ChIP-SEQ PAIR ENCODER (AlphaFold-style)
############################################
class AdaNorm(nn.Module):
    """
    Simple AdaNorm-style normalization:
        y = x + α * LayerNorm(x)
    where α is a learned scalar.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.ln(x)


class RowSelfAttentionWithBulkBias(nn.Module):
    """
    Multi-head self-attention over a length-L sequence with a per-head
    bias matrix derived from the bulk Hi-C map.

    Shapes:
        x:        (B*S, L, C)    # B=batch, S=tracks, L=bins, C=c_msa
        bulk_map: (B, 1, L, L)   # bulk Hi-C contact map per sample
    """
    def __init__(self, c_msa: int, n_heads: int):
        super().__init__()
        assert c_msa % n_heads == 0, "c_msa must be divisible by n_heads"
        self.c_msa = c_msa
        self.n_heads = n_heads
        self.d_head = c_msa // n_heads

        self.q_proj = nn.Linear(c_msa, c_msa)
        self.k_proj = nn.Linear(c_msa, c_msa)
        self.v_proj = nn.Linear(c_msa, c_msa)
        self.out_proj = nn.Linear(c_msa, c_msa)
        self.norm = nn.LayerNorm(c_msa)

        self.bias_from_bulk = nn.Conv2d(1, n_heads, kernel_size=1)

        self.gate_weight = nn.Parameter(
            torch.empty(n_heads, self.d_head, self.d_head)
        )
        self.gate_bias = nn.Parameter(torch.zeros(n_heads, self.d_head))
        nn.init.xavier_uniform_(self.gate_weight)

    def forward(self, x, bulk_map, B, S, L):
        """
        x:        (B*S, L, C)
        bulk_map: (B, 1, L, L)
        """
        BS, L_in, C = x.shape
        assert BS == B * S
        assert L_in == L
        assert C == self.c_msa

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B * S, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B * S, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B * S, L, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

        bias = self.bias_from_bulk(bulk_map)                     # (B, n_heads, L, L)
        bias = bias.unsqueeze(1).repeat(1, S, 1, 1, 1)          # (B, S, n_heads, L, L)
        bias = bias.view(B * S, self.n_heads, L, L)

        scores = scores + bias
        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)                              # (B*S, n_heads, L, d_head)

        gate = torch.einsum("hcd,bhld->bhlc", self.gate_weight, out) + self.gate_bias.unsqueeze(0).unsqueeze(2)
        gate = torch.sigmoid(gate)
        out = out * gate

        out = out.transpose(1, 2).contiguous().view(B * S, L, C)
        out = self.out_proj(out)
        out = self.norm(x + out)
        return out


class ChipPairEncoderAlpha(nn.Module):
    """
    AlphaFold-style encoder for 4 ChIP tracks.

    1) Stack 4 ChIP tracks into an MSA-style tensor (B, s=4, r=64, c_msa).
    2) Apply row-wise self-attention over residues (axis r) independently for each track.
    3) Compute outer-product mean over the MSA (as in AlphaFold):
           pair[i, j] = mean_s( msa[s, i] ⊗ msa[s, j] )  ∈ ℝ^{c_msa×c_msa}
       Flatten c_msa×c_msa and project to c_pair, then return (B, c_pair, 64, 64).
    """
    def __init__(self, n_bins: int = 64, c_msa: int = 32, c_pair: int = 16, n_heads: int = 4):
        super().__init__()
        self.n_bins = n_bins
        self.c_msa = c_msa
        self.c_pair = c_pair
        self.n_heads = n_heads
        self.num_tracks = 4

        self.msa_embed = nn.Linear(1, c_msa)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, n_bins, c_msa))
        self.msa_norm = nn.LayerNorm(c_msa)

        self.row_attn = RowSelfAttentionWithBulkBias(c_msa=c_msa, n_heads=n_heads)

        self.pair_proj = nn.Sequential(
            nn.Linear(c_msa * c_msa, c_pair),
            AdaNorm(c_pair),
            nn.SiLU(),
        )

    def forward(self, chip_ctcf, chip_hac, chip_me1, chip_me3, bulk_map):
        """
        Args:
            chip_ctcf: (B, 64) CTCF
            chip_hac:  (B, 64) H3K27ac
            chip_me1:  (B, 64) H3K4me1
            chip_me3:  (B, 64) H3K4me3
            bulk_map:  (B, 1, 64, 64) bulk Hi-C contact map
        Returns:
            pair_map: (B, c_pair, 64, 64)
        """
        B = chip_ctcf.shape[0]

        signals = torch.stack(
            [chip_ctcf.float(), chip_hac.float(), chip_me1.float(), chip_me3.float()],
            dim=1,
        )  # (B, 4, 64)

        msa = signals.unsqueeze(-1)   # (B, 4, 64, 1)
        msa = self.msa_embed(msa)     # (B, 4, 64, c_msa)
        msa = msa + self.pos_embed
        msa = self.msa_norm(msa)

        B, S, L, C = msa.shape
        x = msa.view(B * S, L, C)
        x = self.row_attn(x, bulk_map, B=B, S=S, L=L)
        msa = x.view(B, S, L, C)

        msa_mean = msa.max(dim=1).values                            # (B, L, C)
        pair_2d = torch.einsum("bic,bjd->bijcd", msa_mean, msa_mean)  # (B, L, L, C, C)
        pair_flat = pair_2d.reshape(B, self.n_bins, self.n_bins, C * C)
        pair_flat = F.normalize(pair_flat, dim=-1, eps=1e-6)

        pair_feat = self.pair_proj(pair_flat)        # (B, 64, 64, c_pair)
        pair_map = pair_feat.permute(0, 3, 1, 2)    # (B, c_pair, 64, 64)
        return pair_map


############################################
# SR3-STYLE U-NET (Denoised Image Predictor)
############################################
class SR3UNet(nn.Module):
    """
    SR3-style U-Net that predicts x_0 (x0-parameterization).

    Architecture:
        - Input: noisy (1) + bulk Hi-C (1) → (B, 2, 64, 64); ChIP is supplemental
        - ChIP-seq: 4 tracks → AlphaFold-style outer product → (B, c_pair, 64, 64)
        - Four downsampling stages: 64 → 32 → 16 → 8
        - BigGAN residual blocks + self-attention at 16×16 and 8×8 (bottleneck)
        - Standard U-Net skip connections (concatenation)
        - Output: Predicted clean image x_0
    """
    def __init__(self, vec_dim, n, noise_embed_module, base_ch: int = 64, c_pair: int | None = None):
        super().__init__()
        self.vec_dim = vec_dim
        self.n = n
        self.noise_embed = noise_embed_module
        self.base_ch = base_ch
        assert base_ch % 2 == 0, "base_ch must be divisible by 2 for bulk/chip split"
        self.c_pair = base_ch // 2

        noise_dim = self.noise_embed.mlp[-1].out_features

        self.input_conv = nn.Conv2d(2, base_ch // 2, kernel_size=3, padding=1)
        self.chip_pair_encoder = ChipPairEncoderAlpha(n_bins=n, c_pair=self.c_pair)

        # ---- ENCODER ----
        self.enc1 = BigGANResBlock(base_ch, base_ch, noise_dim)
        self.enc1_down = BigGANResBlock(base_ch, base_ch * 2, noise_dim, down=True)

        self.enc2 = BigGANResBlock(base_ch * 2, base_ch * 2, noise_dim)
        self.enc2_down = BigGANResBlock(base_ch * 2, base_ch * 4, noise_dim, down=True)

        self.enc3 = BigGANResBlock(base_ch * 4, base_ch * 4, noise_dim)
        self.enc3_self_attn = SelfAttentionBlock(base_ch * 4)
        self.enc3_down = BigGANResBlock(base_ch * 4, base_ch * 8, noise_dim, down=True)

        # ---- BOTTLENECK: 8x8 ----
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

        self.output_block = nn.Sequential(
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)
        )
        nn.init.zeros_(self.output_block[-1].weight)
        nn.init.zeros_(self.output_block[-1].bias)

        # Auxiliary head: predict x_0 from chip features alone
        self.chip_pred_head = nn.Conv2d(self.c_pair, 1, kernel_size=1)
        nn.init.zeros_(self.chip_pred_head.weight)
        nn.init.zeros_(self.chip_pred_head.bias)

    def chip_aux_pred(self, h_chip):
        """
        Apply the auxiliary prediction head to already-computed chip features.

        Args:
            h_chip: (B, c_pair, 64, 64) from forward()
        Returns:
            chip_pred_vec: (B, vec_dim) predicted x_0 from chip features only.
        """
        chip_pred_map = self.chip_pred_head(h_chip)           # (B, 1, 64, 64)
        return matrix_to_upper_tri_vec(chip_pred_map.squeeze(1))

    def forward(self, x_t_vec, gamma, chip_ctcf, chip_hac, chip_me1, chip_me3, bulk_vec):
        """
        SR3 forward pass: predict x_0 given noisy input y_γ.

        Args:
            x_t_vec:   (B, vec_dim) noisy Hi-C vector y_γ
            gamma:     (B,) or (B, 1) noise level γ ∈ [0, 1]
            chip_ctcf: (B, 64) CTCF ChIP-seq
            chip_hac:  (B, 64) H3K27ac
            chip_me1:  (B, 64) H3K4me1
            chip_me3:  (B, 64) H3K4me3
            bulk_vec:  (B, vec_dim) bulk Hi-C vector (conditioning)

        Returns:
            x0_vec: (B, vec_dim) predicted clean Hi-C x_0
            h_chip: (B, c_pair, 64, 64) chip pair features (reuse for aux loss)
        """
        B = x_t_vec.shape[0]
        N = self.n

        if gamma.dim() == 2:
            gamma = gamma.squeeze(-1)
        gamma_scaled = gamma * 999.0
        noise_emb = self.noise_embed(gamma_scaled)

        x_t_map = upper_tri_vec_to_matrix(x_t_vec, N).unsqueeze(1)    # (B, 1, 64, 64)
        bulk_map = upper_tri_vec_to_matrix(bulk_vec, N).unsqueeze(1)  # (B, 1, 64, 64)

        x_in = torch.cat([x_t_map, bulk_map], dim=1)                  # (B, 2, 64, 64)
        h_bulk = self.input_conv(x_in)                                 # (B, base_ch//2, 64, 64)

        h_chip = self.chip_pair_encoder(chip_ctcf, chip_hac, chip_me1, chip_me3, bulk_map)

        h = torch.cat([h_bulk, h_chip], dim=1)                        # (B, base_ch, 64, 64)

        # ========== ENCODER ==========
        h = self.enc1(h, noise_emb)
        skip1 = h
        h = self.enc1_down(h, noise_emb)                              # (B, base_ch*2, 32, 32)

        h = self.enc2(h, noise_emb)
        skip2 = h
        h = self.enc2_down(h, noise_emb)                              # (B, base_ch*4, 16, 16)

        h = self.enc3(h, noise_emb)
        h = self.enc3_self_attn(h)
        skip3 = h
        h = self.enc3_down(h, noise_emb)                              # (B, base_ch*8, 8, 8)

        # ========== BOTTLENECK ==========
        for block in self.bottleneck:
            h = block(h, noise_emb) if isinstance(block, BigGANResBlock) else block(h)

        # ========== DECODER ==========
        h = self.dec3_up(h, noise_emb)                                # (B, base_ch*4, 16, 16)
        h = torch.cat([h, skip3], dim=1)
        h = self.dec3_reduce(h)
        h = self.dec3(h, noise_emb)
        h = self.dec3_self_attn(h)

        h = self.dec2_up(h, noise_emb)                                # (B, base_ch*2, 32, 32)
        h = torch.cat([h, skip2], dim=1)
        h = self.dec2_reduce(h)
        h = self.dec2(h, noise_emb)

        h = self.dec1_up(h, noise_emb)                                # (B, base_ch, 64, 64)
        h = torch.cat([h, skip1], dim=1)
        h = self.dec1_reduce(h)
        h = self.dec1(h, noise_emb)

        x0_map = self.output_block(h)                          # (B, 1, 64, 64)
        x0_vec = matrix_to_upper_tri_vec(x0_map[:, 0])
        return x0_vec, h_chip
