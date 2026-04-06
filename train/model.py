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
# ChIP-SEQ PAIR ENCODER — outer product + axial attention
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


class AxialPairAttention(nn.Module):
    """
    Axial attention (row-wise then column-wise) on a pair tensor (B, N, N, C)
    with additive bias derived from the bulk Hi-C contact map.

    Row pass : for each row i, attend over all column positions j.
    Col pass : for each col j, attend over all row positions i.

    Both passes use a per-head bias learned from bulk_map via a 1×1 conv.
    """
    def __init__(self, c_in: int, n_heads: int = 4):
        super().__init__()
        assert c_in % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = c_in // n_heads

        # Row attention
        self.row_qkv  = nn.Linear(c_in, c_in * 3, bias=False)
        self.row_out  = nn.Linear(c_in, c_in, bias=False)
        self.row_norm = nn.LayerNorm(c_in)
        self.row_bias = nn.Conv2d(1, n_heads, kernel_size=1)

        # Col attention
        self.col_qkv  = nn.Linear(c_in, c_in * 3, bias=False)
        self.col_out  = nn.Linear(c_in, c_in, bias=False)
        self.col_norm = nn.LayerNorm(c_in)
        self.col_bias = nn.Conv2d(1, n_heads, kernel_size=1)

        # Zero-init so module starts as identity
        nn.init.zeros_(self.row_out.weight)
        nn.init.zeros_(self.col_out.weight)

    def _attn_pass(self, x, bulk_bias_4d, qkv_layer, out_layer, norm_layer):
        """
        x:             (B*N, N, C)
        bulk_bias_4d:  (B*N, H, N, N)  pre-computed per-head additive bias
        """
        BN, N, C = x.shape
        H, D = self.n_heads, self.d_head

        q, k, v = qkv_layer(x).chunk(3, dim=-1)
        q = q.view(BN, N, H, D).transpose(1, 2)   # (BN, H, N, D)
        k = k.view(BN, N, H, D).transpose(1, 2)
        v = v.view(BN, N, H, D).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (D ** 0.5) + bulk_bias_4d   # (BN, H, N, N)
        attn   = torch.softmax(scores, dim=-1)
        out    = (attn @ v).transpose(1, 2).contiguous().view(BN, N, C)
        return norm_layer(x + out_layer(out))

    def forward(self, pair: torch.Tensor, bulk_map: torch.Tensor) -> torch.Tensor:
        """
        pair:     (B, N, N, C)
        bulk_map: (B, 1, N, N)
        Returns:  (B, N, N, C)
        """
        B, N, _, C = pair.shape
        H = self.n_heads

        # ---- Row attention: attend over j for each fixed row i ----
        row_bias = self.row_bias(bulk_map)                              # (B, H, N, N)
        row_bias = row_bias.unsqueeze(1).expand(B, N, H, N, N)         # (B, N, H, N, N)
        row_bias = row_bias.reshape(B * N, H, N, N)

        x    = pair.reshape(B * N, N, C)
        x    = self._attn_pass(x, row_bias, self.row_qkv, self.row_out, self.row_norm)
        pair = x.view(B, N, N, C)

        # ---- Col attention: attend over i for each fixed col j ----
        bulk_map_T = bulk_map.transpose(-2, -1).contiguous()
        col_bias   = self.col_bias(bulk_map_T)                          # (B, H, N, N)
        col_bias   = col_bias.unsqueeze(1).expand(B, N, H, N, N)
        col_bias   = col_bias.reshape(B * N, H, N, N)

        x    = pair.transpose(1, 2).contiguous().reshape(B * N, N, C)
        x    = self._attn_pass(x, col_bias, self.col_qkv, self.col_out, self.col_norm)
        pair = x.view(B, N, N, C).transpose(1, 2).contiguous()

        return pair


class ChipPairEncoderAlpha(nn.Module):
    """
    AlphaFold-inspired ChIP-seq pair encoder using outer products and axial attention.

    Flow:
      1) Embed 4 ChIP tracks for the *row* genomic window and the *col* genomic
         window independently into MSA-style tensors (B, 4, N, c_msa).
      2) Max-pool over tracks → chip_i (B, N, c_msa) and chip_j (B, N, c_msa).
      3) Outer product: pair[i, j] = chip_i[i] ⊗ chip_j[j]  → (B, N, N, c_msa²).
      4) Project to c_inner, then apply AxialPairAttention with bulk Hi-C bias.
      5) Linear + AdaNorm + SiLU → c_pair.  Return (B, c_pair, N, N).

    For diagonal crops chip_*_row == chip_*_col, so the outer product is symmetric.
    For off-diagonal crops the two sets of tracks are different.
    """
    def __init__(self, n_bins: int = 64, c_msa: int = 32, c_pair: int = 16, n_heads: int = 4):
        super().__init__()
        self.n_bins = n_bins
        self.c_msa  = c_msa
        self.c_pair = c_pair

        self.msa_embed     = nn.Linear(1, c_msa)
        self.pos_embed_row = nn.Parameter(torch.zeros(1, 1, n_bins, c_msa))
        self.pos_embed_col = nn.Parameter(torch.zeros(1, 1, n_bins, c_msa))
        self.msa_norm      = nn.LayerNorm(c_msa)

        c_inner         = c_msa   # dimension after outer-product projection
        self.outer_proj = nn.Linear(c_msa * c_msa, c_inner)
        self.axial_attn = AxialPairAttention(c_inner, n_heads=n_heads)

        self.pair_proj = nn.Sequential(
            nn.Linear(c_inner, c_pair),
            AdaNorm(c_pair),
            nn.SiLU(),
        )

    def _embed_tracks(self, ctcf, hac, me1, me3, pos_embed):
        """Stack 4 tracks, embed to c_msa, add positional embedding. Returns (B, N, c_msa)."""
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

        chip_i = self._embed_tracks(chip_ctcf_row, chip_hac_row, chip_me1_row, chip_me3_row,
                                    self.pos_embed_row)   # (B, N, c_msa)
        chip_j = self._embed_tracks(chip_ctcf_col, chip_hac_col, chip_me1_col, chip_me3_col,
                                    self.pos_embed_col)   # (B, N, c_msa)

        # Outer product → (B, N, N, c_msa²)
        pair_2d   = torch.einsum("bic,bjd->bijcd", chip_i, chip_j)
        pair_flat = pair_2d.reshape(B, self.n_bins, self.n_bins, self.c_msa * self.c_msa)
        pair_flat = F.normalize(pair_flat, dim=-1, eps=1e-6)

        pair = self.outer_proj(pair_flat)      # (B, N, N, c_inner)
        pair = self.axial_attn(pair, bulk_map) # (B, N, N, c_inner)

        pair_feat = self.pair_proj(pair)       # (B, N, N, c_pair)
        return pair_feat.permute(0, 3, 1, 2)  # (B, c_pair, N, N)


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
