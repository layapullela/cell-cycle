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
# ChIP-SEQ PAIR ENCODER — axial track attention + bulk cross-attention
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


class SpatialAxialAttention(nn.Module):
    """
    Self-attention along the spatial (N) axis of a (B, T, N, C) tensor.
    T is the number of tracks, N is the number of bins, and C is the number of channels.
    Each track attends to its own neighbors, independently of the other tracks.
    long range "is there an anchor upstream/downstream?" reasoning lives here.

    Pre-norm + zero-init out projection ⇒ block starts as identity.
    """
    def __init__(self, c_in: int, n_heads: int = 4):
        super().__init__()
        assert c_in % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = c_in // n_heads

        self.norm = nn.LayerNorm(c_in)
        self.qkv  = nn.Linear(c_in, c_in * 3, bias=False)
        self.out  = nn.Linear(c_in, c_in, bias=False)

        nn.init.zeros_(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        H, D = self.n_heads, self.d_head

        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.reshape(B * T, N, H, D).transpose(1, 2)   # (B*T, H, N, D) we do not attend across tracks here 
        k = k.reshape(B * T, N, H, D).transpose(1, 2)
        v = v.reshape(B * T, N, H, D).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (D ** 0.5)
        attn   = torch.softmax(scores, dim=-1) # scores are n x n (bins x bins)
        out    = (attn @ v).transpose(1, 2).contiguous().reshape(B, T, N, C) 
        return x + self.out(out)
        # idea is that rep for bin 5, say, is some linear comb of bins 4, 6, 10, etc. in the same track
        # where the weights are attn scores.


class TrackAxialAttention(nn.Module):
    """
    Self-attention along the track (T) axis of a (B, T, N, C) tensor.
    At each genomic bin, the 4 tracks (CTCF, H3K27ac, H3K4me1, H3K4me3) attend
    to one another so the model can learn "if CTCF AND H3K27ac co-occur, this
    bin probably anchors a loop" — the cross-mark co-occurrence reasoning.

    Pre-norm + zero-init out projection ⇒ block starts as identity.
    """
    def __init__(self, c_in: int, n_heads: int = 4):
        super().__init__()
        assert c_in % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = c_in // n_heads

        self.norm = nn.LayerNorm(c_in)
        self.qkv  = nn.Linear(c_in, c_in * 3, bias=False)
        self.out  = nn.Linear(c_in, c_in, bias=False)

        nn.init.zeros_(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, C = x.shape
        H, D = self.n_heads, self.d_head

        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        # Move N to the batch axis so attention runs along T at each bin.
        q = q.permute(0, 2, 1, 3).reshape(B * N, T, H, D).transpose(1, 2)
        k = k.permute(0, 2, 1, 3).reshape(B * N, T, H, D).transpose(1, 2)
        v = v.permute(0, 2, 1, 3).reshape(B * N, T, H, D).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (D ** 0.5) # dim of scores is T x T
        attn   = torch.softmax(scores, dim=-1) 
        out    = (attn @ v).transpose(1, 2).contiguous()  # (B*N, T, H, D) # stack heads and proj
        out    = out.reshape(B, N, T, C).permute(0, 2, 1, 3).contiguous()
        return x + self.out(out)


class ChipAxialBlock(nn.Module):
    """
    One pass of (within-track spatial attention) + (across-track attention).
    Operates on (B, T, N, C) tensors. Stack a few of these to mix information
    along both axes the way an MSA-style trunk does.
    """
    def __init__(self, c_msa: int, n_heads: int = 4):
        super().__init__()
        self.spatial = SpatialAxialAttention(c_msa, n_heads=n_heads)
        self.track   = TrackAxialAttention(c_msa, n_heads=n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        x = self.track(x)
        return x


class AxialCrossAttention(nn.Module):
    """
    Cross-attention along one axis of a 2-D pair grid.

    pair (B, N, N, C_q) provides queries.
    ctx  (B, N, N, C_kv) provides keys/values (here: encoded bulk map).

    axis="row":  pair[i, j] attends to ctx[i, j']  for j' ∈ [N]   (same row of bulk)
    axis="col":  pair[i, j] attends to ctx[i', j]  for i' ∈ [N]   (same column of bulk)

    Cost is O(N³) per axis (cheaper than the O(N⁴) of full pair-to-bulk attention)
    while still letting every pair pixel see a full row + a full column of bulk.

    A learned per-head signed relative-position bias is added to the attention
    scores: score(query @ pos_q, key @ pos_k) += rel_bias[head, pos_k - pos_q].
    This makes genomic-distance decay an explicit inductive prior — heads can
    learn near-diagonal vs. long-range biases without having to discover them
    purely through content matching.

    Returns the residual update only — caller is responsible for adding it back.
    """
    def __init__(self, c_q: int, c_kv: int, n_bins: int, n_heads: int = 4, axis: str = "row"):
        super().__init__()
        assert axis in ("row", "col")
        assert c_q % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = c_q // n_heads
        self.n_bins  = n_bins
        self.axis    = axis

        self.q_norm  = nn.LayerNorm(c_q)
        self.kv_norm = nn.LayerNorm(c_kv)
        self.to_q    = nn.Linear(c_q, c_q, bias=False)
        self.to_kv   = nn.Linear(c_kv, 2 * c_q, bias=False)
        self.out     = nn.Linear(c_q, c_q, bias=False)

        nn.init.zeros_(self.out.weight)

        # Learned per-head bias indexed by signed offset (pos_k - pos_q) ∈ [-N+1, N-1].
        # Zero-initialised so the block is identity at start (matches `out` zero-init).
        self.rel_bias = nn.Embedding(2 * n_bins - 1, n_heads)
        nn.init.zeros_(self.rel_bias.weight)

        # Precomputed lookup of (pos_k - pos_q) + (N - 1)  ∈ [0, 2N-2] # TODO: we do not need to care about sign if we have flipping augmentation. 
        positions = torch.arange(n_bins)
        rel_idx   = (positions[None, :] - positions[:, None]) + (n_bins - 1)   # (N, N)
        self.register_buffer("rel_idx", rel_idx, persistent=False)

    def forward(self, pair: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        if self.axis == "col":
            pair = pair.transpose(1, 2)
            ctx  = ctx.transpose(1, 2)

        B, H, W, C = pair.shape
        nh, dh = self.n_heads, self.d_head

        q  = self.to_q(self.q_norm(pair))                    # (B, H, W, C)
        kv = self.to_kv(self.kv_norm(ctx))                   # (B, H, W, 2C)
        k, v = kv.chunk(2, dim=-1)

        q = q.reshape(B * H, W, nh, dh).transpose(1, 2)      # (B*H, nh, W, dh)
        k = k.reshape(B * H, W, nh, dh).transpose(1, 2)
        v = v.reshape(B * H, W, nh, dh).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (dh ** 0.5)     # (B*H, nh, W, W)

        # Add learned per-head signed relative-position bias.
        # rel_bias(rel_idx): (W, W, nh) → (nh, W, W) broadcasts over (B*H).
        rel_pos_bias = self.rel_bias(self.rel_idx).permute(2, 0, 1)
        scores = scores + rel_pos_bias

        attn = torch.softmax(scores, dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().reshape(B, H, W, C)
        out  = self.out(out)

        if self.axis == "col":
            out = out.transpose(1, 2)
        return out


class PairBulkCrossAttention(nn.Module):
    """
    Axial cross-attention block: pair features query encoded bulk-map features
    along rows then along columns.  After this, every pair pixel (i, j) has
    pulled in information from the entire bulk row i AND the entire bulk col j.

    Each axis carries its own learned per-head relative-position bias so the
    model has an explicit "decay-with-distance" inductive prior.
    """
    def __init__(self, c_pair: int, n_bins: int, n_heads: int = 4):
        super().__init__()
        self.row = AxialCrossAttention(c_pair, c_pair, n_bins=n_bins, n_heads=n_heads, axis="row")
        self.col = AxialCrossAttention(c_pair, c_pair, n_bins=n_bins, n_heads=n_heads, axis="col")

    def forward(self, pair: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        pair = pair + self.row(pair, ctx)
        pair = pair + self.col(pair, ctx)
        return pair


class ChipPairEncoderAlpha(nn.Module):
    """
    ChIP-seq pair encoder: per-track embedding → axial track attention →
    outer product → bulk-map cross-attention.

    Flow:
      1) Per-track embedding.  Each of the 4 tracks (CTCF, H3K27ac, H3K4me1,
         H3K4me3) gets its own learned identity vector (nn.Embedding) which is
         added to a value embedding of the per-bin signal.  Output shape
         (B, T=4, N, c_msa) — track identity is preserved everywhere.
      2) `n_axial_blocks` rounds of axial attention:
            - SpatialAxialAttention along N: each track sees its neighbours
              (long-range "is there an anchor a few bins away?")
            - TrackAxialAttention along T: at each bin, the 4 tracks attend
              to one another (cross-mark co-occurrence: "CTCF + H3K27ac → loop")
      3) Gated softmax-weighted sum across tracks  → (B, N, c_msa).  No max,
         so co-binding strength is preserved.
      4) Same pipeline (shared weights) for row and col genomic windows, with
         separate row / col positional embeddings.
      5) Outer product → (B, N, N, c_msa²) → linear project → (B, N, N, c_pair).
         No L2 normalisation: magnitude (= co-binding strength) flows through.
      6) `bulk_encoder` (small 2-D conv stack) maps the bulk map to
         (B, c_pair, N, N) features that genuinely see 2-D structure.
      7) `pair_bulk_xattn`: axial cross-attention from pair (Q) to bulk (K, V)
         along rows then cols — every pair pixel queries a full bulk row + col.
      8) AdaNorm + SiLU → (B, c_pair, N, N).

    For diagonal crops the row and col tracks are identical → symmetric outer
    product.  For off-diagonal crops they differ → asymmetric, as expected.
    """
    N_TRACKS = 4

    def __init__(
        self,
        n_bins: int = 64,
        c_msa: int = 32,
        c_pair: int = 16,
        n_heads: int = 4,
        n_axial_blocks: int = 2,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.c_msa  = c_msa
        self.c_pair = c_pair

        T = self.N_TRACKS

        # ---- per-track embedding ----
        self.track_id_embed = nn.Embedding(T, c_msa)                         # track identity
        self.value_proj     = nn.Linear(1, c_msa)                            # signal magnitude
        self.pos_embed_row  = nn.Parameter(torch.zeros(1, 1, n_bins, c_msa)) # row positions
        self.pos_embed_col  = nn.Parameter(torch.zeros(1, 1, n_bins, c_msa)) # col positions
        self.msa_norm       = nn.LayerNorm(c_msa)

        # ---- axial attention trunk (within-track + across-track) ----
        self.axial_blocks = nn.ModuleList([
            ChipAxialBlock(c_msa, n_heads=n_heads) for _ in range(n_axial_blocks)
        ])

        # ---- gated softmax aggregation over tracks ----
        self.track_gate = nn.Linear(c_msa, 1)

        # ---- outer-product projection (no L2 normalisation) ----
        self.outer_proj = nn.Linear(c_msa * c_msa, c_pair)

        # ---- bulk-map encoder ----
        self.bulk_encoder = nn.Sequential(
            nn.Conv2d(1,      c_pair, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(c_pair, c_pair, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(c_pair, c_pair, kernel_size=3, padding=1),
        )

        # ---- pair ↔ bulk cross-attention (with per-head relative-position bias) ----
        self.pair_bulk_xattn = PairBulkCrossAttention(c_pair, n_bins=n_bins, n_heads=n_heads)

        # ---- final pair refinement ----
        self.pair_proj = nn.Sequential(
            AdaNorm(c_pair),
            nn.SiLU(),
        )

    def _embed_axis(self, ctcf, hac, me1, me3, pos_embed):
        """
        Run the per-track-embedding + axial-trunk + gated-aggregation pipeline
        on one genomic window (either row or col).

        Inputs:  4 tensors of shape (B, N) — CTCF, H3K27ac (HAC), H3K4me1, H3K4me3
        Returns: (B, N, c_msa)
        """
        T = self.N_TRACKS

        sig = torch.stack([ctcf, hac, me1, me3], dim=1).float().unsqueeze(-1)   # (B, T, N, 1)
        ids = torch.arange(T, device=sig.device)
        h_id = self.track_id_embed(ids).view(1, T, 1, self.c_msa)               # (1, T, 1, c)

        x = self.value_proj(sig) + h_id + pos_embed                              # (B, T, N, c) # combine track id. vector with actual chip signal and pos embed (bin idx)
        x = self.msa_norm(x)

        for block in self.axial_blocks:
            x = block(x)

        # softmax-weighted sum over tracks (preserves co-occurrence magnitude)
        w = torch.softmax(self.track_gate(x), dim=1)                             # (B, T, N, 1)
        return (w * x).sum(dim=1)                                                # (B, N, c) # we get rid of track here and have representations of each bin only.

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
        N = self.n_bins

        # 1)–3) per-axis embedding + axial attention + track aggregation
        chip_i = self._embed_axis(chip_ctcf_row, chip_hac_row, chip_me1_row, chip_me3_row,
                                  self.pos_embed_row)                            # (B, N, c_msa)
        chip_j = self._embed_axis(chip_ctcf_col, chip_hac_col, chip_me1_col, chip_me3_col,
                                  self.pos_embed_col)                            # (B, N, c_msa)

        # 4) outer product (no L2 normalisation — keeps co-binding magnitude)
        pair_2d   = torch.einsum("bic,bjd->bijcd", chip_i, chip_j)
        pair_flat = pair_2d.reshape(B, N, N, self.c_msa * self.c_msa)
        pair      = self.outer_proj(pair_flat)                                   # (B, N, N, c_pair)

        # 5) encode bulk map → channels-last
        bulk_feat = self.bulk_encoder(bulk_map).permute(0, 2, 3, 1)              # (B, N, N, c_pair)

        # 6) pair queries bulk along rows then cols
        pair = self.pair_bulk_xattn(pair, bulk_feat)                             # (B, N, N, c_pair)

        # 7) final refinement
        pair = self.pair_proj(pair)                                              # (B, N, N, c_pair)
        return pair.permute(0, 3, 1, 2)                                          # (B, c_pair, N, N)


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
        # Predicts an auxiliary (B,4,N,N) target from chip features.
        self.chip_pred_head = nn.Conv2d(self.c_pair, 4, kernel_size=1)
        nn.init.zeros_(self.chip_pred_head.weight)
        nn.init.zeros_(self.chip_pred_head.bias)

    # ------------------------------------------------------------------
    def chip_aux_pred(self, h_chip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_chip: (B, c_pair, N, N)
        Returns:
            (B, 4, N, N) predicted auxiliary target (MSE target in training)
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
