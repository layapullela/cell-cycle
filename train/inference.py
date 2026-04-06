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
        SR3 sampling: Generate all four phases Hi-C from bulk using iterative refinement.

        Args:
            bulk_vec:  (B, vec_dim) bulk Hi-C conditioning = average(anaphase, G1)
            chip_ctcf: (B, N) CTCF ChIP-seq conditioning
            chip_hac:  (B, N) H3K27ac ChIP-seq conditioning
            chip_me1:  (B, N) H3K4me1 ChIP-seq conditioning
            chip_me3:  (B, N) H3K4me3 ChIP-seq conditioning

        Returns:
            y_0: (B, 4, vec_dim)  channels: earlyG1, midG1, lateG1, anatelo
        """
        batch_size, vec_dim = bulk_vec.shape

        # SR3 Algorithm 2 Line 1: Start from pure Gaussian noise (all four channels)
        y_t = torch.randn(batch_size, 4, vec_dim, device=self.device)

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

            # clamp
            y_t = torch.clamp(y_t, min=-1.0, max=1.0)

            # Project y_t onto the constraint: mean(y_t[:,0..3]) == bulk_vec.
            # Adding residual to every channel shifts the mean to bulk without
            # changing any pairwise differences between channels.
            # current_mean = y_t.mean(dim=1)                   # (B, vec_dim)
            # residual     = bulk_vec - current_mean            # (B, vec_dim)
            # y_t = y_t + residual.unsqueeze(1)                 # broadcast to (B, 4, vec_dim)

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
        return y_t  # (B, 4, vec_dim)
    
    def visualize(self, batch, output_path=None, n=64, vmin=None, vmax=None):
        """
        Run inference and visualize results for all four output channels.

        Args:
            batch: Dict with keys 'earlyG1', 'midG1', 'lateG1', 'anatelo',
                   'chip_seq_ctcf', 'chip_seq_hac', 'chip_seq_h3k4me1',
                   'chip_seq_h3k4me3', and optionally 'region'
            output_path: Where to save plot (if None, just display)
            n: Matrix size (default 64)
            vmin/vmax: Optional fixed color scale (default: -1.0 / 1.0)

        Returns:
            sampled_vec: (B, 4, vec_dim)  channels: earlyG1, midG1, lateG1, anatelo
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

        bulk_vec = (x0_early + x0_mid + x0_late + x0_anatelo) / 4.0
        chip_histone_1d = chip_hac[0].detach().cpu().numpy()

        # Run sampling → (B, 4, vec_dim)
        sampled_vec = self.sample(bulk_vec, chip_ctcf, chip_hac, chip_me1, chip_me3)

        def to_mat(vec):
            return upper_tri_vec_to_matrix(vec[0:1], n)[0].cpu().numpy()

        gt_mats   = [to_mat(x) for x in [x0_early, x0_mid, x0_late, x0_anatelo]]
        pred_mats = [upper_tri_vec_to_matrix(sampled_vec[0:1, i], n)[0].cpu().numpy() for i in range(4)]
        bulk_mat  = to_mat(bulk_vec)
        phase_labels = ['earlyG1', 'midG1', 'lateG1', 'anatelo']

        region = batch.get('region', ['unknown'])[0]
        print(f"\n  [{region}] value ranges:", flush=True)
        print(f"    {'bulk':12s}: [{bulk_mat.min():.4f}, {bulk_mat.max():.4f}]  mean={bulk_mat.mean():.4f}", flush=True)
        for label, gt, pred in zip(phase_labels, gt_mats, pred_mats):
            print(f"    {label+'_gt':12s}: [{gt.min():.4f}, {gt.max():.4f}]  mean={gt.mean():.4f}", flush=True)
            print(f"    {label+'_pred':12s}: [{pred.min():.4f}, {pred.max():.4f}]  mean={pred.mean():.4f}", flush=True)

        if vmin is None:
            vmin = -1.0
        if vmax is None:
            vmax = 1.0

        # ChIP track (interpolated if needed)
        chip_len = chip_histone_1d.shape[0]
        chip_plot = (np.interp(np.linspace(0, 1, n), np.linspace(0, 1, chip_len), chip_histone_1d)
                     if chip_len != n else chip_histone_1d)

        def _add_chip_inset(ax):
            ymin_ax, ymax_ax = ax.get_ylim()
            ins = inset_axes(ax, width="8%", height="100%", loc="right",
                             bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
            ins.plot(chip_plot, np.linspace(ymax_ax, ymin_ax, n), color="black", linewidth=1.0)
            ins.set_ylim(ymin_ax, ymax_ax)
            ins.set_xticks([]); ins.set_yticks([])
            ins.set_xlim(chip_plot.min(), chip_plot.max())

        # 4 rows × 3 cols: bulk | predicted | ground truth
        fig, axes = plt.subplots(4, 3, figsize=(16, 20))
        metrics = []
        for row, (label, gt, pred) in enumerate(zip(phase_labels, gt_mats, pred_mats)):
            axes[row, 0].imshow(bulk_mat, cmap='Reds', vmin=vmin, vmax=vmax)
            axes[row, 0].set_title('Bulk (conditioning)', fontsize=11, fontweight='bold')
            axes[row, 0].axis('off')

            axes[row, 1].imshow(pred, cmap='Reds', vmin=vmin, vmax=vmax)
            axes[row, 1].set_title(f'Predicted: {label}', fontsize=11, fontweight='bold')
            axes[row, 1].axis('off')

            im = axes[row, 2].imshow(gt, cmap='Reds', vmin=vmin, vmax=vmax)
            axes[row, 2].set_title(f'Ground Truth: {label}', fontsize=11, fontweight='bold')
            axes[row, 2].axis('off')
            plt.colorbar(im, ax=axes[row, 2], fraction=0.046)

            for ax in axes[row]:
                _add_chip_inset(ax)

            try:
                corr = np.corrcoef(gt.flatten(), pred.flatten())[0, 1]
                corr = 0.0 if np.isnan(corr) else corr
            except Exception:
                corr = 0.0
            mse = np.mean((gt - pred) ** 2)
            metrics.append((label, mse, corr))

        metric_str = '  |  '.join(f"{l}: MSE={m:.4f} r={c:.3f}" for l, m, c in metrics)
        fig.suptitle(f'SR3 Inference | Region: {region}\n{metric_str}',
                     fontsize=12, fontweight='bold', y=1.005)
        plt.tight_layout(rect=[0, 0, 1, 0.995])

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

    save_path = output_path / f"inference_4phase_step_{step}_alpha.png"
    inference.visualize(batch, output_path=save_path, vmin=vmin, vmax=vmax)

    return save_path
