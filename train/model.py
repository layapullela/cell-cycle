import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


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
# ChIP-SEQ PAIR ENCODER — 1-D attention then outer product
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


class ChipSelfAttention1D(nn.Module):
    """
    1-D self-attention on a chip embedding (B, N, C) with a bulk contact
    profile used as an additive key-side bias.

    The bias encodes "how globally connected is position j?" — bins inside
    TADs or at strong anchors get higher average bulk contact and will be
    attended to more by default.  The learned weight lets each head decide
    how much to trust that prior.

    Starts as identity (out projection zero-initialised).
    """
    def __init__(self, c_in: int, n_heads: int = 4):
        super().__init__()
        assert c_in % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = c_in // n_heads

        self.qkv          = nn.Linear(c_in, c_in * 3, bias=False)
        self.out          = nn.Linear(c_in, c_in, bias=False)
        self.norm         = nn.LayerNorm(c_in)
        # Maps each bin's scalar bulk-profile value → n_heads additive biases
        self.profile_bias = nn.Linear(1, n_heads, bias=False)

        nn.init.zeros_(self.out.weight)

    def forward(self, x: torch.Tensor, bulk_profile: torch.Tensor) -> torch.Tensor:
        """
        x:             (B, N, C)  chip embedding
        bulk_profile:  (B, N)     mean bulk contact per bin along this axis
        Returns:       (B, N, C)
        """
        B, N, C = x.shape
        H, D    = self.n_heads, self.d_head

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, H, D).transpose(1, 2)   # (B, H, N, D)
        k = k.view(B, N, H, D).transpose(1, 2)
        v = v.view(B, N, H, D).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (D ** 0.5)   # (B, H, N, N)

        # bias[b, h, :, j] = learned_weight[h] * bulk_profile[b, j]
        # shape: (B, N, H) → (B, H, 1, N)  → broadcasts over query positions
        bias   = self.profile_bias(bulk_profile.unsqueeze(-1))  # (B, N, H)
        bias   = bias.permute(0, 2, 1).unsqueeze(2)             # (B, H, 1, N)
        scores = scores + bias

        attn = torch.softmax(scores, dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        return self.norm(x + self.out(out))


class ChipPairEncoderAlpha(nn.Module):
    """
    ChIP-seq pair encoder: 1-D contextualisation then outer product.

    Flow:
      1) Embed 4 ChIP tracks for the row window  → max over tracks → chip_i (B, N, c_msa).
         Apply ChipSelfAttention1D biased by the bulk row-mean profile.
      2) Same for the col window → chip_j (B, N, c_msa).
      3) Outer product: pair[i,j] = chip_i[i] ⊗ chip_j[j]  → (B, N, N, c_msa²).
         Each position already carries full neighbourhood context from step 1/2.
      4) Project to c_pair, add a direct bulk residual, then AdaNorm + SiLU.
      5) Return (B, c_pair, N, N).

    Complexity: O(N²) attention (1-D passes of length N, done once per axis)
    versus O(N³) for axial attention after the outer product.

    For diagonal crops chip_*_row == chip_*_col → symmetric outer product.
    For off-diagonal crops the two sets of tracks differ → asymmetric.
    """
    def __init__(self, n_bins: int = 64, c_msa: int = 16, c_pair: int = 16, n_heads: int = 4):
        super().__init__()
        self.n_bins = n_bins
        self.c_msa  = c_msa
        self.c_pair = c_pair

        # Track embedding (shared weights for row and col)
        self.msa_embed     = nn.Linear(1, c_msa)
        self.pos_embed_row = nn.Parameter(torch.zeros(1, 1, n_bins, c_msa))
        self.pos_embed_col = nn.Parameter(torch.zeros(1, 1, n_bins, c_msa))
        self.msa_norm      = nn.LayerNorm(c_msa)

        # 1-D self-attention applied BEFORE outer product
        self.row_attn = ChipSelfAttention1D(c_msa, n_heads=n_heads)
        self.col_attn = ChipSelfAttention1D(c_msa, n_heads=n_heads)

        # Outer-product projection + direct bulk residual
        self.outer_proj   = nn.Linear(c_msa * c_msa, c_pair)
        self.bulk_to_pair = nn.Linear(1, c_pair)   # (B,N,N,1) → (B,N,N,c_pair)

        self.pair_proj = nn.Sequential(
            AdaNorm(c_pair),
            nn.SiLU(),
        )

    def _embed_tracks(self, ctcf, hac, me1, me3, pos_embed):
        """Stack 4 tracks, embed each bin to c_msa, max over tracks. Returns (B, N, c_msa)."""
        sig = torch.stack([ctcf, hac, me1, me3], dim=1).float().unsqueeze(-1)  # (B, 4, N, 1)
        msa = self.msa_embed(sig) + pos_embed                                   # (B, 4, N, c_msa)
        msa = self.msa_norm(msa)
        return msa.max(dim=1).values                                             # (B, N, c_msa)

    def forward(
        self,
        chip_ctcf_row, chip_hac_row, chip_me1_row, chip_me3_row,
        chip_ctcf_col, chip_hac_col, chip_me1_col, chip_me3_col,
        bulk_map,
    ):
        """
        Args:
            chip_*_row: (B, N) ChIP-seq for the row genomic window
            chip_*_col: (B, N) ChIP-seq for the col genomic window
            bulk_map:   (B, 1, N, N) bulk Hi-C contact map
        Returns:
            pair_map:   (B, c_pair, N, N)
        """
        B = chip_ctcf_row.shape[0]

        # 1-D bulk profiles — mean contact along each axis
        row_profile = bulk_map[:, 0].mean(dim=-1)   # (B, N)  mean over cols
        col_profile = bulk_map[:, 0].mean(dim=-2)   # (B, N)  mean over rows

        # Embed tracks then add 1-D context BEFORE outer product
        chip_i = self._embed_tracks(chip_ctcf_row, chip_hac_row, chip_me1_row, chip_me3_row,
                                    self.pos_embed_row)            # (B, N, c_msa)
        chip_i = self.row_attn(chip_i, row_profile)                # (B, N, c_msa)

        chip_j = self._embed_tracks(chip_ctcf_col, chip_hac_col, chip_me1_col, chip_me3_col,
                                    self.pos_embed_col)            # (B, N, c_msa)
        chip_j = self.col_attn(chip_j, col_profile)                # (B, N, c_msa)

        # Outer product → (B, N, N, c_msa²)
        pair_2d   = torch.einsum("bic,bjd->bijcd", chip_i, chip_j)
        pair_flat = pair_2d.reshape(B, self.n_bins, self.n_bins, self.c_msa * self.c_msa)
        pair_flat = F.normalize(pair_flat, dim=-1, eps=1e-6)

        # Project + direct bulk residual (channels-last throughout)
        pair      = self.outer_proj(pair_flat)                                   # (B, N, N, c_pair)
        bulk_feat = self.bulk_to_pair(bulk_map.permute(0, 2, 3, 1))             # (B, N, N, c_pair)
        pair      = pair + bulk_feat

        pair_feat = self.pair_proj(pair)                                         # (B, N, N, c_pair)
        return pair_feat.permute(0, 3, 1, 2)                                    # (B, c_pair, N, N)


############################################
# PHASE CROSS-ATTENTION (between decoder streams)
############################################
class PhaseStreamAttention(nn.Module):
    """
    Cross-phase attention between 4 parallel decoder streams at a given resolution.

    Each stream's feature map is summarised into one token via average pooling,
    then a 4×4 attention matrix lets each phase gather context from the others.
    The attended update is broadcast back to every spatial position via a 1×1 conv.
    """
    def __init__(self, channels: int, d_model: int = 64, n_phases: int = 4):
        super().__init__()
        self.scale    = d_model ** -0.5
        self.norm     = nn.GroupNorm(min(8, channels), channels)
        self.to_token = nn.Linear(channels, d_model, bias=False)
        self.W_q      = nn.Linear(d_model, d_model, bias=False)
        self.W_k      = nn.Linear(d_model, d_model, bias=False)
        self.W_v      = nn.Linear(d_model, d_model, bias=False)
        self.to_feat  = nn.Conv2d(d_model, channels, kernel_size=1)

        # Zero-init: module is identity at training start
        nn.init.zeros_(self.to_feat.weight)
        nn.init.zeros_(self.to_feat.bias)

    def forward(self, streams):
        """
        streams: list of n_phases tensors, each (B, C, H, W)
        returns: list of n_phases tensors, same shape, residual-updated
        """
        B, C, H, W = streams[0].shape

        tokens = torch.stack(
            [self.norm(s).mean(dim=(2, 3)) for s in streams], dim=1
        )                                        # (B, 4, C)
        tokens = self.to_token(tokens)           # (B, 4, d_model)

        Q = self.W_q(tokens)
        K = self.W_k(tokens)
        V = self.W_v(tokens)

        A = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)  # (B, 4, 4)
        Z = A @ V                                                          # (B, 4, d_model)

        out_streams = []
        for i, s in enumerate(streams):
            update = self.to_feat(
                Z[:, i].unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
            )
            out_streams.append(s + update)
        return out_streams


############################################
# SR3-STYLE U-NET — SPLIT DECODER
############################################
class SR3UNet(nn.Module):
    """
    SR3-style U-Net with shared encoder and 4 parallel phase-specific decoder streams.

    Inputs are now full 2-D contact matrices (B, 4, N, N) rather than flattened
    upper-triangular vectors, so no vec↔matrix conversion happens inside the model.

    Architecture:
        Encoder (shared):
            (B, 5, N, N) → enc1 → enc2 → enc3 → bottleneck → (B, 512, N/8, N/8)
            Input channels: 4 noisy phases + bulk (all as N×N matrices)

        Decoder (4 parallel streams, one per phase):
            bottleneck → stream_init → 3 up-sampling levels with PhaseStreamAttention

        Output:
            (B, 4, N, N) predicted denoised matrices
    """
    N_PHASES = 4

    def __init__(self, n: int, noise_embed_module: nn.Module, base_ch: int = 64):
        super().__init__()
        self.n        = n
        self.base_ch  = base_ch
        self.noise_embed = noise_embed_module
        assert base_ch % 2 == 0
        self.c_pair = base_ch // 2
        P = self.N_PHASES

        noise_dim = self.noise_embed.mlp[-1].out_features

        # ---- INPUT ----
        # 5 channels: 4 noisy phases + bulk (all N×N matrices)
        self.input_conv        = nn.Conv2d(5, base_ch // 2, kernel_size=3, padding=1)
        self.chip_pair_encoder = ChipPairEncoderAlpha(n_bins=n, c_pair=self.c_pair)

        # ---- SHARED ENCODER ----
        self.enc1           = BigGANResBlock(base_ch,     base_ch,     noise_dim)
        self.enc1_down      = BigGANResBlock(base_ch,     base_ch * 2, noise_dim, down=True)
        self.enc2           = BigGANResBlock(base_ch * 2, base_ch * 2, noise_dim)
        self.enc2_down      = BigGANResBlock(base_ch * 2, base_ch * 4, noise_dim, down=True)
        self.enc3           = BigGANResBlock(base_ch * 4, base_ch * 4, noise_dim)
        self.enc3_self_attn = SelfAttentionBlock(base_ch * 4)
        self.enc3_down      = BigGANResBlock(base_ch * 4, base_ch * 8, noise_dim, down=True)

        # ---- BOTTLENECK ----
        self.bottleneck = nn.ModuleList([
            BigGANResBlock(base_ch * 8, base_ch * 8, noise_dim),
            SelfAttentionBlock(base_ch * 8),
            BigGANResBlock(base_ch * 8, base_ch * 8, noise_dim),
        ])

        # ---- SPLIT: bottleneck → 4 phase streams ----
        self.stream_init = nn.ModuleList([
            nn.Conv2d(base_ch * 8, base_ch * 4, kernel_size=1) for _ in range(P)
        ])

        # ---- PHASE-PARALLEL DECODER ----
        # Level 3: → 16×16
        self.dec3_up     = nn.ModuleList([BigGANResBlock(base_ch * 4, base_ch * 4, noise_dim, up=True)  for _ in range(P)])
        self.dec3_reduce = nn.ModuleList([nn.Conv2d(base_ch * 8, base_ch * 4, kernel_size=1)             for _ in range(P)])
        self.dec3        = nn.ModuleList([BigGANResBlock(base_ch * 4, base_ch * 4, noise_dim)             for _ in range(P)])
        self.phase_attn3 = PhaseStreamAttention(base_ch * 4, d_model=64)

        # Level 2: → 32×32
        self.dec2_up     = nn.ModuleList([BigGANResBlock(base_ch * 4, base_ch * 2, noise_dim, up=True)  for _ in range(P)])
        self.dec2_reduce = nn.ModuleList([nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=1)             for _ in range(P)])
        self.dec2        = nn.ModuleList([BigGANResBlock(base_ch * 2, base_ch * 2, noise_dim)             for _ in range(P)])
        self.phase_attn2 = PhaseStreamAttention(base_ch * 2, d_model=64)

        # Level 1: → 64×64
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
        # Predicts all 4 phase matrices from chip features alone
        self.chip_pred_head = nn.Conv2d(self.c_pair, 4, kernel_size=1)
        nn.init.zeros_(self.chip_pred_head.weight)
        nn.init.zeros_(self.chip_pred_head.bias)

    # ------------------------------------------------------------------
    def chip_aux_pred(self, h_chip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_chip: (B, c_pair, N, N)
        Returns:
            (B, 4, N, N) predicted phase matrices from chip features alone
        """
        return self.chip_pred_head(h_chip)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_t,
        gamma,
        chip_ctcf_row, chip_hac_row, chip_me1_row, chip_me3_row,
        chip_ctcf_col, chip_hac_col, chip_me1_col, chip_me3_col,
        bulk_map,
    ):
        """
        Args:
            x_t:          (B, 4, N, N)  noisy phase matrices [earlyG1, midG1, lateG1, anatelo]
            gamma:        (B,)           noise level
            chip_*_row:   (B, N)         ChIP-seq for the row genomic window
            chip_*_col:   (B, N)         ChIP-seq for the col genomic window
            bulk_map:     (B, 1, N, N)  bulk Hi-C conditioning (already a 2-D matrix)
        Returns:
            x0:    (B, 4, N, N)  predicted clean matrices
            h_chip:(B, c_pair, N, N)  chip pair features (used for aux loss)
        """
        B = x_t.shape[0]
        P = self.N_PHASES

        if gamma.dim() == 2:
            gamma = gamma.squeeze(-1)
        noise_emb = self.noise_embed(gamma * 999.0)

        # ---- Build 2-D input feature map ----
        x_in   = torch.cat([x_t, bulk_map], dim=1)                          # (B, 5, N, N)
        h_bulk = self.input_conv(x_in)                                        # (B, base_ch//2, N, N)
        h_chip = self.chip_pair_encoder(
            chip_ctcf_row, chip_hac_row, chip_me1_row, chip_me3_row,
            chip_ctcf_col, chip_hac_col, chip_me1_col, chip_me3_col,
            bulk_map,
        )                                                                     # (B, c_pair, N, N)

        h = torch.cat([h_bulk, h_chip], dim=1)                               # (B, base_ch, N, N)

        # ========== SHARED ENCODER ==========
        h     = self.enc1(h, noise_emb)
        skip1 = h
        h     = self.enc1_down(h, noise_emb)

        h     = self.enc2(h, noise_emb)
        skip2 = h
        h     = self.enc2_down(h, noise_emb)

        h     = self.enc3(h, noise_emb)
        h     = self.enc3_self_attn(h)
        skip3 = h
        h     = self.enc3_down(h, noise_emb)

        # ========== BOTTLENECK ==========
        for block in self.bottleneck:
            h = block(h, noise_emb) if isinstance(block, BigGANResBlock) else block(h)

        # ========== SPLIT INTO 4 PHASE STREAMS ==========
        streams = [init(h) for init in self.stream_init]

        # ========== PHASE-PARALLEL DECODER ==========

        # -- Level 3 --
        streams = [self.dec3_up[i](streams[i], noise_emb) for i in range(P)]
        streams = [self.dec3_reduce[i](torch.cat([streams[i], skip3], dim=1)) for i in range(P)]
        streams = [self.dec3[i](streams[i], noise_emb) for i in range(P)]
        streams = self.phase_attn3(streams)

        # -- Level 2 --
        streams = [self.dec2_up[i](streams[i], noise_emb) for i in range(P)]
        streams = [self.dec2_reduce[i](torch.cat([streams[i], skip2], dim=1)) for i in range(P)]
        streams = [self.dec2[i](streams[i], noise_emb) for i in range(P)]
        streams = self.phase_attn2(streams)

        # -- Level 1 --
        streams = [self.dec1_up[i](streams[i], noise_emb) for i in range(P)]
        streams = [self.dec1_reduce[i](torch.cat([streams[i], skip1], dim=1)) for i in range(P)]
        streams = [self.dec1[i](streams[i], noise_emb) for i in range(P)]
        streams = self.phase_attn1(streams)

        # ========== PER-PHASE OUTPUT ==========
        phase_maps = []
        for i in range(P):
            out_map = self.output_heads[i](streams[i])   # (B, 1, N, N)
            phase_maps.append(out_map[:, 0])             # (B, N, N)

        x0 = torch.stack(phase_maps, dim=1)              # (B, 4, N, N)
        return x0, h_chip
