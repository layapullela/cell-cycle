"""
SR3 Inference for Cell-Cycle Hi-C Phase Decomposition

Clean, focused implementation of SR3 sampling with optional visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocess"))


class Inference:
    """
    SR3 inference engine for sampling phase-specific Hi-C from trained models.

    Implements iterative refinement using x0-prediction (model predicts the
    clean image directly, then the implied noise is used for the SR3 update).
    """

    def __init__(self, model, device, T=1000, gamma_min=1e-4, gamma_max=1.0):
        """
        Args:
            model: Trained SR3UNet model (predicts x_0 directly)
            device: torch device
            T: Number of diffusion timesteps
            gamma_min: Minimum noise level (almost clean) - default 1e-4
            gamma_max: Maximum noise level (pure noise) - default 1.0
        """
        self.model = model
        self.device = device
        self.T = T
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        # Load noise schedule from training (must match training script)
        from train_diffusion_alpha import gammas, alphas
        self.gammas = gammas.to(device)
        self.alphas = alphas.to(device)

        self.model.eval()
        
    @torch.no_grad()
    def sample(self, bulk_vec, chip_ctcf, chip_hac, chip_me1, chip_me3):
        """
        SR3 sampling: Generate anaphase and G1 Hi-C from bulk using iterative refinement.

        Args:
            bulk_vec:  (B, vec_dim) bulk Hi-C conditioning = average(anaphase, G1)
            chip_ctcf: (B, N) CTCF ChIP-seq conditioning
            chip_hac:  (B, N) H3K27ac ChIP-seq conditioning
            chip_me1:  (B, N) H3K4me1 ChIP-seq conditioning
            chip_me3:  (B, N) H3K4me3 ChIP-seq conditioning

        Returns:
            y_0: (B, 2, vec_dim)  channel 0=anaphase, channel 1=G1
        """
        batch_size, vec_dim = bulk_vec.shape

        # SR3 Algorithm 2 Line 1: Start from pure Gaussian noise (both channels)
        y_t = torch.randn(batch_size, 2, vec_dim, device=self.device)

        # SR3 Algorithm 2: iterative refinement with stochastic noise re-injection.
        # Model predicts ε directly (SR3 Algorithm 1 training).
        #
        #   y_{t-1} = (1/√α_t)(y_t - (1-α_t)/√(1-γ_t) · ε_θ(y_t, γ_t)) + √(1-α_t) · z
        #
        # where z ~ N(0,I) for t > 1, z = 0 for t = 1 (final step is deterministic).
        for t_idx in range(self.T - 1, 0, -1):
            gamma_t = self.gammas[t_idx]
            alpha_t = self.alphas[t_idx]

            sqrt_one_minus_gamma_t = torch.sqrt(1.0 - gamma_t)

            # Model predicts ε for both channels; discard h_chip
            gamma_batch = torch.full((batch_size,), gamma_t, device=self.device)
            eps_pred, _ = self.model(
                y_t,
                gamma_batch,
                chip_ctcf,
                chip_hac,
                chip_me1,
                chip_me3,
                bulk_vec,
            )

            # SR3 Algorithm 2 update (applied identically to both channels)
            z = torch.randn_like(y_t) if t_idx > 1 else torch.zeros_like(y_t)
            y_t = (1.0 / torch.sqrt(alpha_t)) * (
                y_t - (1.0 - alpha_t) / sqrt_one_minus_gamma_t * eps_pred
            ) + torch.sqrt(1.0 - alpha_t) * z

        # ---- Commented-out DDIM x0-prediction update (deviation from SR3) ----
        # for t_idx in range(self.T - 1, 0, -1):
        #     gamma_t    = self.gammas[t_idx]
        #     gamma_prev = self.gammas[t_idx - 1]
        #     sqrt_gamma_t           = torch.sqrt(gamma_t)
        #     sqrt_one_minus_gamma_t = torch.sqrt(torch.clamp(1.0 - gamma_t, min=0.0))
        #     gamma_batch = torch.full((batch_size,), gamma_t, device=self.device)
        #     x0_pred, _ = self.model(y_t, gamma_batch, chip_ctcf, chip_hac, chip_me1, chip_me3, bulk_vec)
        #     eps_implied = (y_t - sqrt_gamma_t * x0_pred) / (sqrt_one_minus_gamma_t + 1e-8)
        #     sqrt_gamma_prev           = torch.sqrt(gamma_prev)
        #     sqrt_one_minus_gamma_prev = torch.sqrt(torch.clamp(1.0 - gamma_prev, min=0.0))
        #     y_t = sqrt_gamma_prev * x0_pred + sqrt_one_minus_gamma_prev * eps_implied

        return y_t  # (B, 2, vec_dim)
    
    def visualize(self, batch, output_path=None, n=64, vmin=None, vmax=None):
        """
        Run inference and visualize results for both output channels (anaphase and G1).

        Args:
            batch: Dict with keys 'earlyG1', 'midG1', 'lateG1', 'anatelo',
                   'chip_seq_ctcf', 'chip_seq_hac', 'chip_seq_h3k4me1',
                   'chip_seq_h3k4me3', and optionally 'region'
            output_path: Where to save plot (if None, just display)
            n: Matrix size (default 64)
            vmin/vmax: Optional fixed color scale (default: -1.0 / 1.0)

        Returns:
            sampled_vec: (B, 2, vec_dim)  channel 0=anaphase, channel 1=G1
        """
        from train_diffusion_alpha import upper_tri_vec_to_matrix

        x0_early   = batch['earlyG1'].float().to(self.device)
        x0_mid     = batch['midG1'].float().to(self.device)
        x0_late    = batch['lateG1'].float().to(self.device)
        x0_anatelo = batch['anatelo'].float().to(self.device)
        chip_ctcf  = batch['chip_seq_ctcf'].float().to(self.device)
        chip_hac   = batch['chip_seq_hac'].float().to(self.device)
        chip_me1   = batch.get('chip_seq_h3k4me1', batch['chip_seq_hac']).float().to(self.device)
        chip_me3   = batch.get('chip_seq_h3k4me3', batch['chip_seq_hac']).float().to(self.device)

        # Bulk = average(anaphase, G1), matching training
        x0_anaphase_gt = x0_anatelo
        x0_G1_gt       = (x0_early + x0_mid + x0_late) / 3.0
        bulk_vec       = (x0_anaphase_gt + x0_G1_gt) / 2.0

        chip_histone_1d = chip_hac[0].detach().cpu().numpy()

        # Run sampling → (B, 2, vec_dim)
        sampled_vec = self.sample(bulk_vec, chip_ctcf, chip_hac, chip_me1, chip_me3)

        # Convert to matrices (first sample in batch)
        def to_mat(vec_1d):
            return upper_tri_vec_to_matrix(vec_1d[0:1], n)[0].cpu().numpy()

        anaphase_gt_mat = to_mat(x0_anaphase_gt)
        G1_gt_mat       = to_mat(x0_G1_gt)
        bulk_mat        = to_mat(bulk_vec)
        anaphase_pred_mat = upper_tri_vec_to_matrix(sampled_vec[0:1, 0], n)[0].cpu().numpy()
        G1_pred_mat       = upper_tri_vec_to_matrix(sampled_vec[0:1, 1], n)[0].cpu().numpy()

        region = batch.get('region', ['unknown'])[0]
        print(f"\n  [{region}] value ranges:", flush=True)
        for name, mat in [('anaphase_gt', anaphase_gt_mat), ('G1_gt', G1_gt_mat),
                          ('bulk', bulk_mat), ('anaphase_pred', anaphase_pred_mat),
                          ('G1_pred', G1_pred_mat)]:
            print(f"    {name:14s}: [{mat.min():.4f}, {mat.max():.4f}]  mean={mat.mean():.4f}", flush=True)

        if vmin is None:
            vmin = -1.0
        if vmax is None:
            vmax = 1.0

        # 2 rows (anaphase, G1) × 3 cols (bulk | predicted | ground truth)
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        def _plot_row(row_axes, bulk, pred, gt, row_label):
            im0 = row_axes[0].imshow(bulk, cmap='Reds', vmin=vmin, vmax=vmax)
            row_axes[0].set_title(f'Bulk (conditioning)', fontsize=12, fontweight='bold')
            row_axes[0].axis('off')
            plt.colorbar(im0, ax=row_axes[0], fraction=0.046)

            im1 = row_axes[1].imshow(pred, cmap='Reds', vmin=vmin, vmax=vmax)
            row_axes[1].set_title(f'Predicted: {row_label}', fontsize=12, fontweight='bold')
            row_axes[1].axis('off')
            plt.colorbar(im1, ax=row_axes[1], fraction=0.046)

            im2 = row_axes[2].imshow(gt, cmap='Reds', vmin=vmin, vmax=vmax)
            row_axes[2].set_title(f'Ground Truth: {row_label}', fontsize=12, fontweight='bold')
            row_axes[2].axis('off')
            plt.colorbar(im2, ax=row_axes[2], fraction=0.046)

            # ChIP-seq track inset
            chip_len = chip_histone_1d.shape[0]
            chip_plot = (np.interp(np.linspace(0, 1, n), np.linspace(0, 1, chip_len), chip_histone_1d)
                         if chip_len != n else chip_histone_1d)
            for ax in row_axes:
                ymin_ax, ymax_ax = ax.get_ylim()
                y_pos = np.linspace(ymax_ax, ymin_ax, n)
                ins = inset_axes(ax, width="8%", height="100%", loc="right",
                                 bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes,
                                 borderpad=0)
                ins.plot(chip_plot, y_pos, color="black", linewidth=1.0)
                ins.set_ylim(ymin_ax, ymax_ax)
                ins.set_xticks([]); ins.set_yticks([])
                ins.set_xlim(chip_plot.min(), chip_plot.max())

            try:
                corr = np.corrcoef(gt.flatten(), pred.flatten())[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            except Exception:
                corr = 0.0
            mse = np.mean((gt - pred) ** 2)
            return mse, corr

        mse_a, corr_a = _plot_row(axes[0], bulk_mat, anaphase_pred_mat, anaphase_gt_mat, 'Anaphase')
        mse_g, corr_g = _plot_row(axes[1], bulk_mat, G1_pred_mat,       G1_gt_mat,       'G1')

        fig.suptitle(
            f'SR3 Inference | Region: {region}\n'
            f'Anaphase — MSE: {mse_a:.6f}  Corr: {corr_a:.4f} | '
            f'G1 — MSE: {mse_g:.6f}  Corr: {corr_g:.4f}',
            fontsize=14, fontweight='bold', y=1.01,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {output_path}")
        else:
            plt.show()
        plt.close(fig)

        return sampled_vec


def run_inference_and_visualize(model, batch, device, step, output_dir="./inference_visualizations",
                                vmin=None, vmax=None):
    """
    Convenience function for running inference during training.

    Args:
        model: Trained SR3UNet model
        batch: Training batch
        device: torch device
        step: Current training step (for filename)
        output_dir: Where to save visualization
        vmin/vmax: Optional fixed color scale

    Returns:
        output_path: Path to saved visualization
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    inference = Inference(model, device, T=1000)

    save_path = output_path / f"inference_anaphase_G1_step_{step}_alpha.png"
    inference.visualize(batch, output_path=save_path, vmin=vmin, vmax=vmax)

    return save_path
