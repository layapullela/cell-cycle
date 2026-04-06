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
# PHASE CROSS-ATTENTION (between decoder streams)
############################################
class PhaseStreamAttention(nn.Module):
    """
    Cross-phase attention between 4 parallel decoder streams at a given resolution.

    Each stream's feature map is summarised into one token via average pooling,
    then a 4×4 attention matrix lets each phase gather context from the others.
    The attended update is broadcast back to every spatial position via a 1×1 conv.

    This operates on feature maps, not output pixels, so the 4 phases can
    exchange semantic information (e.g. "anatelo has strong TADs here") before
    the next decoder level commits to a prediction.

    Shapes (example at 16×16, base_ch*4=256):
        streams : list of 4 × (B, C, H, W)
        tokens  : (B, 4, C)               one token per phase (global avg pool)
        Q,K,V   : (B, 4, d_model)
        A       : (B, 4, 4)               cross-phase attention weights
        update  : (B, 4, C) → broadcast → 4 × (B, C, H, W)  via 1×1 conv
    """
    def __init__(self, channels: int, d_model: int = 64, n_phases: int = 4):
        super().__init__()
        self.scale   = d_model ** -0.5
        self.norm    = nn.GroupNorm(min(8, channels), channels)
        self.to_token = nn.Linear(channels, d_model, bias=False)
        self.W_q     = nn.Linear(d_model, d_model, bias=False)
        self.W_k     = nn.Linear(d_model, d_model, bias=False)
        self.W_v     = nn.Linear(d_model, d_model, bias=False)
        self.to_feat = nn.Conv2d(d_model, channels, kernel_size=1)

        # Zero-init: module is identity at training start
        nn.init.zeros_(self.to_feat.weight)
        nn.init.zeros_(self.to_feat.bias)

    def forward(self, streams):
        """
        streams: list of n_phases tensors, each (B, C, H, W)
        returns: list of n_phases tensors, same shape, residual-updated
        """
        B, C, H, W = streams[0].shape

        # Summarise each stream into one token via global average pool
        tokens = torch.stack(
            [self.norm(s).mean(dim=(2, 3)) for s in streams], dim=1
        )                                                        # (B, 4, C)
        tokens = self.to_token(tokens)                           # (B, 4, d_model)

        Q = self.W_q(tokens)                                     # (B, 4, d_model)
        K = self.W_k(tokens)                                     # (B, 4, d_model)
        V = self.W_v(tokens)                                     # (B, 4, d_model)

        A = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)  # (B, 4, 4)
        Z = A @ V                                                          # (B, 4, d_model)

        # Broadcast update back to spatial dims via 1×1 conv
        out_streams = []
        for i, s in enumerate(streams):
            update = self.to_feat(
                Z[:, i].unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
            )                                                    # (B, C, H, W)
            out_streams.append(s + update)
        return out_streams


############################################
# SR3-STYLE U-NET — SPLIT DECODER (Denoised Image Predictor)
############################################
class SR3UNet(nn.Module):
    """
    SR3-style U-Net with a shared encoder and 4 parallel phase-specific decoder streams.

    Architecture:
        Encoder (shared):
            (B, 5, 64, 64) → [enc1 → enc2 → enc3 → bottleneck] → (B, 512, 8, 8)
            Input channels: noisy earlyG1 + midG1 + lateG1 + anatelo + bulk

        Decoder (4 parallel streams, one per phase):
            bottleneck → split into 4 streams via stream_init
            Each stream runs its own upsampling BigGAN blocks.
            Between levels, PhaseStreamAttention lets streams communicate.
            Skip connections from encoder are shared across all 4 streams.

        Output:
            Each stream → GroupNorm → SiLU → Conv2d(base_ch, 1)
            Stack → (B, 4, vec_dim)

    Cross-phase attention positions (feature-level, not pixel-level):
        After dec3 (16×16, base_ch*4 channels)
        After dec2 (32×32, base_ch*2 channels)
        After dec1 (64×64, base_ch   channels)
    """
    N_PHASES = 4

    def __init__(self, vec_dim, n, noise_embed_module, base_ch: int = 64, c_pair: int | None = None):
        super().__init__()
        self.vec_dim  = vec_dim
        self.n        = n
        self.base_ch  = base_ch
        self.noise_embed = noise_embed_module
        assert base_ch % 2 == 0
        self.c_pair   = base_ch // 2
        P             = self.N_PHASES

        noise_dim = self.noise_embed.mlp[-1].out_features

        # ---- INPUT ----
        # 5 channels: 4 noisy phases + bulk
        self.input_conv      = nn.Conv2d(5, base_ch // 2, kernel_size=3, padding=1)
        self.chip_pair_encoder = ChipPairEncoderAlpha(n_bins=n, c_pair=self.c_pair)

        # ---- SHARED ENCODER ----
        self.enc1          = BigGANResBlock(base_ch,     base_ch,     noise_dim)
        self.enc1_down     = BigGANResBlock(base_ch,     base_ch * 2, noise_dim, down=True)
        self.enc2          = BigGANResBlock(base_ch * 2, base_ch * 2, noise_dim)
        self.enc2_down     = BigGANResBlock(base_ch * 2, base_ch * 4, noise_dim, down=True)
        self.enc3          = BigGANResBlock(base_ch * 4, base_ch * 4, noise_dim)
        self.enc3_self_attn= SelfAttentionBlock(base_ch * 4)
        self.enc3_down     = BigGANResBlock(base_ch * 4, base_ch * 8, noise_dim, down=True)

        # ---- BOTTLENECK ----
        self.bottleneck = nn.ModuleList([
            BigGANResBlock(base_ch * 8, base_ch * 8, noise_dim),
            SelfAttentionBlock(base_ch * 8),
            BigGANResBlock(base_ch * 8, base_ch * 8, noise_dim),
        ])

        # ---- SPLIT: bottleneck → 4 phase streams ----
        # One 1×1 conv per phase to project (base_ch*8) → (base_ch*4) and give each stream its own start
        self.stream_init = nn.ModuleList([
            nn.Conv2d(base_ch * 8, base_ch * 4, kernel_size=1) for _ in range(P)
        ])

        # ---- PHASE-PARALLEL DECODER ----
        # Level 3: 8×8 → 16×16
        self.dec3_up     = nn.ModuleList([BigGANResBlock(base_ch * 4, base_ch * 4, noise_dim, up=True)  for _ in range(P)])
        self.dec3_reduce = nn.ModuleList([nn.Conv2d(base_ch * 8, base_ch * 4, kernel_size=1)             for _ in range(P)])
        self.dec3        = nn.ModuleList([BigGANResBlock(base_ch * 4, base_ch * 4, noise_dim)             for _ in range(P)])
        self.phase_attn3 = PhaseStreamAttention(base_ch * 4, d_model=64)

        # Level 2: 16×16 → 32×32
        self.dec2_up     = nn.ModuleList([BigGANResBlock(base_ch * 4, base_ch * 2, noise_dim, up=True)  for _ in range(P)])
        self.dec2_reduce = nn.ModuleList([nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=1)             for _ in range(P)])
        self.dec2        = nn.ModuleList([BigGANResBlock(base_ch * 2, base_ch * 2, noise_dim)             for _ in range(P)])
        self.phase_attn2 = PhaseStreamAttention(base_ch * 2, d_model=64)

        # Level 1: 32×32 → 64×64
        self.dec1_up     = nn.ModuleList([BigGANResBlock(base_ch * 2, base_ch, noise_dim, up=True)       for _ in range(P)])
        self.dec1_reduce = nn.ModuleList([nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)                  for _ in range(P)])
        self.dec1        = nn.ModuleList([BigGANResBlock(base_ch, base_ch, noise_dim)                      for _ in range(P)])
        self.phase_attn1 = PhaseStreamAttention(base_ch, d_model=64)

        # ---- PER-PHASE OUTPUT HEADS ----
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(min(8, base_ch), base_ch),
                nn.SiLU(),
                nn.Conv2d(base_ch, 1, kernel_size=3, padding=1),
            ) for _ in range(P)
        ])
        for head in self.output_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

        # ---- AUXILIARY CHIP HEAD ----
        self.chip_pred_head = nn.Conv2d(self.c_pair, 4, kernel_size=1)
        nn.init.zeros_(self.chip_pred_head.weight)
        nn.init.zeros_(self.chip_pred_head.bias)

    # ------------------------------------------------------------------
    def chip_aux_pred(self, h_chip):
        """
        Args:
            h_chip: (B, c_pair, 64, 64)
        Returns:
            (B, 4, vec_dim)
        """
        chip_pred_map = self.chip_pred_head(h_chip)                 # (B, 4, 64, 64)
        return torch.stack([
            matrix_to_upper_tri_vec(chip_pred_map[:, i]) for i in range(self.N_PHASES)
        ], dim=1)                                                    # (B, 4, vec_dim)

    # ------------------------------------------------------------------
    def forward(self, x_t_vec, gamma, chip_ctcf, chip_hac, chip_me1, chip_me3, bulk_vec):
        """
        Args:
            x_t_vec:  (B, 4, vec_dim)  noisy phases [earlyG1, midG1, lateG1, anatelo]
            gamma:    (B,)             noise level
            chip_*:   (B, 64)          ChIP-seq tracks
            bulk_vec: (B, vec_dim)     bulk Hi-C conditioning
        Returns:
            x0_vec:  (B, 4, vec_dim)
            h_chip:  (B, c_pair, 64, 64)
        """
        B  = x_t_vec.shape[0]
        N  = self.n
        P  = self.N_PHASES

        if gamma.dim() == 2:
            gamma = gamma.squeeze(-1)
        noise_emb = self.noise_embed(gamma * 999.0)

        # ---- Build input maps ----
        phase_maps = [upper_tri_vec_to_matrix(x_t_vec[:, i], N).unsqueeze(1) for i in range(P)]
        bulk_map   = upper_tri_vec_to_matrix(bulk_vec, N).unsqueeze(1)         # (B, 1, N, N)

        x_in   = torch.cat(phase_maps + [bulk_map], dim=1)                     # (B, 5, N, N)
        h_bulk = self.input_conv(x_in)                                          # (B, base_ch//2, N, N)
        h_chip = self.chip_pair_encoder(chip_ctcf, chip_hac, chip_me1, chip_me3, bulk_map)

        h = torch.cat([h_bulk, h_chip], dim=1)                                 # (B, base_ch, 64, 64)

        # ========== SHARED ENCODER ==========
        h     = self.enc1(h, noise_emb)
        skip1 = h                                                               # (B, base_ch,   64, 64)
        h     = self.enc1_down(h, noise_emb)

        h     = self.enc2(h, noise_emb)
        skip2 = h                                                               # (B, base_ch*2, 32, 32)
        h     = self.enc2_down(h, noise_emb)

        h     = self.enc3(h, noise_emb)
        h     = self.enc3_self_attn(h)
        skip3 = h                                                               # (B, base_ch*4, 16, 16)
        h     = self.enc3_down(h, noise_emb)                                   # (B, base_ch*8,  8,  8)

        # ========== BOTTLENECK ==========
        for block in self.bottleneck:
            h = block(h, noise_emb) if isinstance(block, BigGANResBlock) else block(h)
                                                                                # (B, base_ch*8, 8, 8)

        # ========== SPLIT INTO 4 PHASE STREAMS ==========
        # Each stream gets an independent linear projection of the bottleneck
        streams = [init(h) for init in self.stream_init]                       # 4 × (B, base_ch*4, 8, 8)

        # ========== PHASE-PARALLEL DECODER ==========

        # -- Level 3: 8×8 → 16×16 --
        streams = [self.dec3_up[i](streams[i], noise_emb) for i in range(P)]  # 4 × (B, base_ch*4, 16, 16)
        streams = [
            self.dec3_reduce[i](torch.cat([streams[i], skip3], dim=1))
            for i in range(P)
        ]
        streams = [self.dec3[i](streams[i], noise_emb) for i in range(P)]
        streams = self.phase_attn3(streams)                                    # cross-phase communication

        # -- Level 2: 16×16 → 32×32 --
        streams = [self.dec2_up[i](streams[i], noise_emb) for i in range(P)]  # 4 × (B, base_ch*2, 32, 32)
        streams = [
            self.dec2_reduce[i](torch.cat([streams[i], skip2], dim=1))
            for i in range(P)
        ]
        streams = [self.dec2[i](streams[i], noise_emb) for i in range(P)]
        streams = self.phase_attn2(streams)                                    # cross-phase communication

        # -- Level 1: 32×32 → 64×64 --
        streams = [self.dec1_up[i](streams[i], noise_emb) for i in range(P)]  # 4 × (B, base_ch, 64, 64)
        streams = [
            self.dec1_reduce[i](torch.cat([streams[i], skip1], dim=1))
            for i in range(P)
        ]
        streams = [self.dec1[i](streams[i], noise_emb) for i in range(P)]
        streams = self.phase_attn1(streams)                                    # cross-phase communication

        # ========== PER-PHASE OUTPUT ==========
        phase_vecs = []
        for i in range(P):
            out_map = self.output_heads[i](streams[i])                         # (B, 1, 64, 64)
            phase_vecs.append(matrix_to_upper_tri_vec(out_map[:, 0]))         # (B, vec_dim)

        x0_vec = torch.stack(phase_vecs, dim=1)                               # (B, 4, vec_dim)
        return x0_vec, h_chip
