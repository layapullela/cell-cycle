"""
SR3 Inference for Cell-Cycle Hi-C Phase Decomposition

Inputs and outputs are full 2-D contact matrices (B, 4, N, N).
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

    Implements iterative refinement using ε-prediction (SR3 Algorithm 1/2).
    """

    def __init__(self, model, device, T=1000, gamma_min=1e-4, gamma_max=1.0):
        """
        Args:
            model: Trained SR3UNet model
            device: torch device
            T: Number of diffusion timesteps
        """
        self.model  = model
        self.device = device
        self.T      = T

        from train_diffusion_alpha import gammas, alphas
        self.gammas = gammas.to(device)
        self.alphas = alphas.to(device)

        self.model.eval()

    @torch.no_grad()
    def sample(
        self,
        bulk_map,
        chip_ctcf_row, chip_hac_row, chip_me1_row, chip_me3_row,
        chip_ctcf_col, chip_hac_col, chip_me1_col, chip_me3_col,
    ):
        """
        SR3 sampling: generate all four phase Hi-C matrices from bulk.

        Args:
            bulk_map:     (B, 1, N, N) bulk Hi-C conditioning
            chip_*_row:   (B, N)       ChIP-seq for the row genomic window
            chip_*_col:   (B, N)       ChIP-seq for the col genomic window

        Returns:
            y_0: (B, 4, N, N)  channels: earlyG1, midG1, lateG1, anatelo
        """
        B, _, N, N2 = bulk_map.shape
        y_t = torch.randn(B, 4, N, N2, device=self.device)

        for t_idx in range(self.T - 1, 0, -1):
            gamma_t = self.gammas[t_idx]
            alpha_t = self.alphas[t_idx]

            sqrt_one_minus_gamma_t = torch.sqrt(1.0 - gamma_t)
            gamma_batch = torch.full((B,), gamma_t, device=self.device)

            eps_pred, _ = self.model(
                y_t, gamma_batch,
                chip_ctcf_row, chip_hac_row, chip_me1_row, chip_me3_row,
                chip_ctcf_col, chip_hac_col, chip_me1_col, chip_me3_col,
                bulk_map,
            )

            z   = torch.randn_like(y_t) if t_idx > 1 else torch.zeros_like(y_t)
            y_t = (1.0 / torch.sqrt(alpha_t)) * (
                y_t - (1.0 - alpha_t) / sqrt_one_minus_gamma_t * eps_pred
            ) + torch.sqrt(1.0 - alpha_t) * z

            y_t = torch.clamp(y_t, min=-1.0, max=1.0)

        return y_t   # (B, 4, N, N)

    def visualize(self, batch, output_path=None, n=64, vmin=None, vmax=None):
        """
        Run inference and visualize results for all four output channels.

        Args:
            batch: Dict with keys 'earlyG1', 'midG1', 'lateG1', 'anatelo' as (B, N, N),
                   'chip_seq_*_row' / '_col' as (B, N), and optionally 'region'
            output_path: Where to save plot (None → display)
            n: Matrix size (default 64)
            vmin/vmax: Optional fixed colour scale

        Returns:
            sampled: (B, 4, N, N)
        """
        x0_early   = batch['earlyG1'].float().to(self.device)           # (B, N, N)
        x0_mid     = batch['midG1'].float().to(self.device)
        x0_late    = batch['lateG1'].float().to(self.device)
        x0_anatelo = batch['anatelo'].float().to(self.device)

        chip_ctcf_row = batch['chip_seq_ctcf_row'].float().to(self.device)
        chip_hac_row  = batch['chip_seq_hac_row'].float().to(self.device)
        chip_me1_row  = batch.get('chip_seq_h3k4me1_row', batch['chip_seq_hac_row']).float().to(self.device)
        chip_me3_row  = batch.get('chip_seq_h3k4me3_row', batch['chip_seq_hac_row']).float().to(self.device)
        chip_ctcf_col = batch.get('chip_seq_ctcf_col', batch['chip_seq_ctcf_row']).float().to(self.device)
        chip_hac_col  = batch.get('chip_seq_hac_col',  batch['chip_seq_hac_row']).float().to(self.device)
        chip_me1_col  = batch.get('chip_seq_h3k4me1_col', chip_me1_row).float().to(self.device)
        chip_me3_col  = batch.get('chip_seq_h3k4me3_col', chip_me3_row).float().to(self.device)

        bulk_map = (x0_early + x0_mid + x0_late + x0_anatelo).mul(0.25).unsqueeze(1)  # (B, 1, N, N)

        chip_histone_1d = chip_hac_row[0].detach().cpu().numpy()

        # Run sampling → (B, 4, N, N)
        sampled = self.sample(
            bulk_map,
            chip_ctcf_row, chip_hac_row, chip_me1_row, chip_me3_row,
            chip_ctcf_col, chip_hac_col, chip_me1_col, chip_me3_col,
        )

        gt_mats   = [x[0].cpu().numpy() for x in [x0_early, x0_mid, x0_late, x0_anatelo]]
        pred_mats = [sampled[0, i].cpu().numpy() for i in range(4)]
        bulk_mat  = bulk_map[0, 0].cpu().numpy()
        phase_labels = ['earlyG1', 'midG1', 'lateG1', 'anatelo']

        region = batch.get('region', ['unknown'])
        if isinstance(region, (list, tuple)):
            region = region[0]
        print(f"\n  [{region}] value ranges:", flush=True)
        print(f"    {'bulk':12s}: [{bulk_mat.min():.4f}, {bulk_mat.max():.4f}]  mean={bulk_mat.mean():.4f}", flush=True)
        for label, gt, pred in zip(phase_labels, gt_mats, pred_mats):
            print(f"    {label+'_gt':12s}: [{gt.min():.4f}, {gt.max():.4f}]  mean={gt.mean():.4f}", flush=True)
            print(f"    {label+'_pred':12s}: [{pred.min():.4f}, {pred.max():.4f}]  mean={pred.mean():.4f}", flush=True)

        if vmin is None:
            vmin = -1.0
        if vmax is None:
            vmax = 1.0

        chip_len  = chip_histone_1d.shape[0]
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

        return sampled


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

    Returns:
        output_path: Path to saved visualization
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    inference = Inference(model, device, T=1000)

    save_path = output_path / f"inference_4phase_step_{step}_alpha.png"
    inference.visualize(batch, output_path=save_path, vmin=vmin, vmax=vmax)

    return save_path
