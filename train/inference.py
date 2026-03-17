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

# Add preprocess dir to path for undo_normalization
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocess"))
from undo_normalization import reverse_normalization_to_log_scale, quantile_normalize_across_samples


class Inference:
    """
    SR3 inference engine for sampling phase-specific Hi-C from trained models.

    Implements SR3 Algorithm 2: iterative refinement using noise prediction.
    """

    def __init__(self, model, device, T=1000, gamma_min=1e-4, gamma_max=1.0):
        """
        Args:
            model: Trained SR3UNet model (predicts noise ε)
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

        # Load noise schedule from training (must match script used for training)
        from train_diffusion_single_tracks import gammas, alphas
        self.gammas = gammas.to(device)
        self.alphas = alphas.to(device)

        self.model.eval()
        
    @torch.no_grad()
    def sample(self, bulk_vec, chip_ctcf, chip_hac, chip_me1, chip_me3):
        """
        SR3 sampling: Generate phase-specific Hi-C from bulk using iterative refinement.
        
        Args:
            bulk_vec:  (B, vec_dim) bulk Hi-C conditioning
            chip_ctcf: (B, N) CTCF ChIP-seq conditioning
            chip_hac:  (B, N) H3K27ac ChIP-seq conditioning
            chip_me1:  (B, N) H3K4me1 ChIP-seq conditioning
            chip_me3:  (B, N) H3K4me3 ChIP-seq conditioning
        
        Returns:
            y_0: (B, vec_dim) sampled phase-specific Hi-C
        """
        batch_size, vec_dim = bulk_vec.shape
        
        # SR3 Algorithm 2 Line 1: Start from pure Gaussian noise
        y_t = torch.randn(batch_size, vec_dim, device=self.device)
        
        # Iteratively denoise: t = T-1, T-2, ..., 1
        # SR3 Algorithm 2: same formula for all steps, z=0 only at final step (t=1)
        for t_idx in range(self.T - 1, 0, -1):
            # Get noise levels from schedule
            gamma_t = self.gammas[t_idx]
            # Get alpha_t from schedule
            alpha_t = self.alphas[t_idx]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_gamma_t = torch.sqrt(1.0 - gamma_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

            # Predict noise ε - pass gamma directly to model
            gamma_batch = torch.full((batch_size,), gamma_t, device=self.device)
            # For alpha model: 4 ChIP tracks (ctcf, hac, me1, me3)
            eps_pred = self.model(
                y_t,
                gamma_batch,
                chip_ctcf,
                chip_hac,
                chip_me1,
                chip_me3,
                bulk_vec,
            )

            # #SR3 Algorithm 2: z ~ N(0,I) if t > 1, else z = 0
            if t_idx > 1:
                z = torch.randn_like(y_t)
            else:
                z = torch.zeros_like(y_t)  # Final step: t=1→0, no noise

            # for now, let's make z = 0 for all steps (hicDiff)
            #z = torch.zeros_like(y_t)
            
            # SR3 Algorithm 2 formula (same for all steps)
            y_prev = (1.0 / sqrt_alpha_t) * (y_t - ((1.0 - alpha_t) / sqrt_one_minus_gamma_t) * eps_pred) + sqrt_one_minus_alpha_t * z

            #y_prev = torch.clamp(y_prev, -1.0, 1.0) # clip to -1, 1 #TODO: check this
            
            y_t = y_prev
        
        # After loop, y_t is y_0 (fully denoised)
        return y_t
    
    def visualize(self, batch, phase_name, output_path=None, n=64, vmin=None, vmax=None):
        """
        Run inference and visualize results.

        Args:
            batch: Dict with keys
                   'earlyG1', 'midG1', 'lateG1', 'anatelo',
                   'chip_seq_ctcf', 'chip_seq_hac',
                   optionally 'chip_seq_h3k4me1', 'chip_seq_h3k4me3',
                   and 'region'
            phase_name: Which phase to visualize ('earlyG1', 'midG1', 'lateG1', or 'anatelo')
            output_path: Where to save plot (if None, just display)
            n: Matrix size (default 64)
            vmin: Optional fixed vmin for color scale (if None, computed from this batch)
            vmax: Optional fixed vmax for color scale (if None, computed from this batch)

        Returns:
            sampled: (B, vec_dim) sampled phase-specific Hi-C
        """
        # Import matrix conversion utilities (must match training script: N=64)
        from train_diffusion_single_tracks import upper_tri_vec_to_matrix
        
        # Extract ground truth and conditioning
        x0_early = batch['earlyG1'].float().to(self.device)
        x0_mid = batch['midG1'].float().to(self.device)
        x0_late = batch['lateG1'].float().to(self.device)
        x0_anatelo = batch['anatelo'].float().to(self.device)
        chip_ctcf = batch['chip_seq_ctcf'].float().to(self.device)
        chip_hac = batch['chip_seq_hac'].float().to(self.device)
        # Backward-compatible: if me1/me3 not present (old models), reuse hac
        chip_me1 = batch.get('chip_seq_h3k4me1', batch['chip_seq_hac']).float().to(self.device)
        chip_me3 = batch.get('chip_seq_h3k4me3', batch['chip_seq_hac']).float().to(self.device)
        
        # Compute bulk conditioning (average of four phases)
        bulk_vec = (x0_early + x0_mid + x0_late + x0_anatelo) / 4.0
        # Use HAC (H3K27ac) as the 1D track for visualization
        chip_histone_1d = chip_hac[0].detach().cpu().numpy()
        
        # Get ground truth for current phase
        phase_to_gt = {
            'earlyG1': x0_early,
            'midG1': x0_mid,
            'lateG1': x0_late,
            'anatelo': x0_anatelo
        }
        gt_vec = phase_to_gt[phase_name]
        
        # Run sampling
        sampled_vec = self.sample(bulk_vec, chip_ctcf, chip_hac, chip_me1, chip_me3)

        # Convert to matrices for visualization (use first sample in batch)
        # Convert ALL phase ground truths to enable cross-phase quantile normalization
        early_matrix = upper_tri_vec_to_matrix(x0_early[0:1], n)[0].cpu().numpy()
        mid_matrix = upper_tri_vec_to_matrix(x0_mid[0:1], n)[0].cpu().numpy()
        late_matrix = upper_tri_vec_to_matrix(x0_late[0:1], n)[0].cpu().numpy()
        anatelo_matrix = upper_tri_vec_to_matrix(x0_anatelo[0:1], n)[0].cpu().numpy()
        sampled_matrix = upper_tri_vec_to_matrix(sampled_vec[0:1], n)[0].cpu().numpy()
        bulk_matrix = upper_tri_vec_to_matrix(bulk_vec[0:1], n)[0].cpu().numpy()

        # Get region for display
        region = batch.get('region', ['unknown'])[0]

        # DEBUG: Check raw value ranges BEFORE quantile normalization
        import sys
        print(f"\n  DEBUG [{region}] - Raw value ranges (in [-1, 1] space, BEFORE quantile norm):", flush=True)
        print(f"    earlyG1:  [{early_matrix.min():.4f}, {early_matrix.max():.4f}], mean={early_matrix.mean():.4f}, std={early_matrix.std():.4f}", flush=True)
        print(f"    midG1:    [{mid_matrix.min():.4f}, {mid_matrix.max():.4f}], mean={mid_matrix.mean():.4f}, std={mid_matrix.std():.4f}", flush=True)
        print(f"    lateG1:   [{late_matrix.min():.4f}, {late_matrix.max():.4f}], mean={late_matrix.mean():.4f}, std={late_matrix.std():.4f}", flush=True)
        print(f"    anatelo:  [{anatelo_matrix.min():.4f}, {anatelo_matrix.max():.4f}], mean={anatelo_matrix.mean():.4f}, std={anatelo_matrix.std():.4f}", flush=True)
        print(f"    bulk:     [{bulk_matrix.min():.4f}, {bulk_matrix.max():.4f}], mean={bulk_matrix.mean():.4f}, std={bulk_matrix.std():.4f}", flush=True)
        print(f"    SAMPLED:  [{sampled_matrix.min():.4f}, {sampled_matrix.max():.4f}], mean={sampled_matrix.mean():.4f}, std={sampled_matrix.std():.4f}", flush=True)
        # Apply quantile normalization ACROSS all phases for the same region
        # This allows fair comparison: same color = same relative contact frequency across phases
        # Pool all 4 phase ground truths + sampled + bulk, compute global ranks
        # normalized_matrices = quantile_normalize_across_samples(
        #     [early_matrix, mid_matrix, late_matrix, anatelo_matrix, sampled_matrix, bulk_matrix],
        #     output_min=0.0,
        #     output_max=40.0
        # )

        normalized_matrices = [early_matrix, mid_matrix, late_matrix, anatelo_matrix, sampled_matrix, bulk_matrix]

        # Extract normalized matrices
        early_norm, mid_norm, late_norm, anatelo_norm, sampled_matrix, bulk_matrix = normalized_matrices

        # Get the ground truth for the current phase
        phase_to_normalized = {
            'earlyG1': early_norm,
            'midG1': mid_norm,
            'lateG1': late_norm,
            'anatelo': anatelo_norm
        }
        gt_matrix = phase_to_normalized[phase_name]

        # Use fixed [0, 40] scale for q-normed data (matching papers)
        # This can be overridden with vmin/vmax parameters if needed
        if vmin is None:
            vmin = -1.0
        if vmax is None:
            vmax = 1.0
        
        # Create visualization: 1 row x 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Input: Bulk Hi-C (conditioning)
        im1 = axes[0].imshow(bulk_matrix, cmap='Reds', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Input: Bulk Hi-C\n(Conditioning)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # Output: Sampled phase-specific
        im2 = axes[1].imshow(sampled_matrix, cmap='Reds', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'Output: Sampled {phase_name}\n(SR3 Inference)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        # Ground truth: Target phase
        im3 = axes[2].imshow(gt_matrix, cmap='Reds', vmin=vmin, vmax=vmax)
        axes[2].set_title(f'Ground Truth: {phase_name}', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)

        # Add histone ChIP-seq track to the right side of each map, aligned with the heatmap y-axis
        # Assume chip_histone_1d length matches matrix size n; if not, interpolate to n bins
        chip_len = chip_histone_1d.shape[0]
        if chip_len != n:
            # Simple interpolation to n points
            x_orig = np.linspace(0, 1, chip_len)
            x_new = np.linspace(0, 1, n)
            chip_histone_plot = np.interp(x_new, x_orig, chip_histone_1d)
        else:
            chip_histone_plot = chip_histone_1d

        for ax in axes:
            # Get the exact y-limits used by the heatmap so we can match them
            ymin, ymax = ax.get_ylim()
            y_positions = np.linspace(ymax, ymin, n)

            # Create a narrow inset axis on the right side of each heatmap
            inset_ax = inset_axes(
                ax,
                width="8%",
                height="100%",
                loc="right",
                bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
            inset_ax.plot(chip_histone_plot, y_positions, color="black", linewidth=1.0)
            inset_ax.set_ylim(ymin, ymax)  # Match heatmap y-axis exactly
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_xlim(chip_histone_plot.min(), chip_histone_plot.max())
        
        # Compute metrics
        mse = np.mean((gt_matrix - sampled_matrix) ** 2)
        
        # Correlation (handle degenerate cases early in training)
        try:
            corr = np.corrcoef(gt_matrix.flatten(), sampled_matrix.flatten())[0, 1]
            if np.isnan(corr):
                corr = 0.0  # Zero variance or invalid data
        except:
            corr = 0.0  # Fallback for any issues
        
        # Move region extraction to earlier (already done above)
        # region = batch.get('region', ['unknown'])[0]  # Already extracted above
        fig.suptitle(
            f'SR3 Inference: {phase_name} | Region: {region}\n'
            f'MSE: {mse:.6f} | Correlation: {corr:.4f}',
            fontsize=16,
            fontweight='bold',
            y=1.02  # Move title slightly above to avoid overlap
        )

        # Leave space at top for suptitle (rect=[left, bottom, right, top])
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {output_path}")
        else:
            plt.show()
        
        plt.close(fig)
        
        return sampled_vec


def run_inference_and_visualize(model, batch, phase_name, device, step, output_dir="./inference_visualizations",
                                vmin=None, vmax=None):
    """
    Convenience function for running inference during training.

    Args:
        model: Trained SR3UNet model
        batch: Training batch
        phase_name: Phase to sample ('earlyG1', 'midG1', 'lateG1', or 'anatelo')
        device: torch device
        step: Current training step (for filename)
        output_dir: Where to save visualization
        vmin: Optional fixed vmin for color scale
        vmax: Optional fixed vmax for color scale

    Returns:
        output_path: Path to saved visualization
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Initialize inference
    inference = Inference(model, device, T=1000)

    # Run and visualize
    save_path = output_path / f"inference_{phase_name}_step_{step}_alpha.png"
    inference.visualize(batch, phase_name, output_path=save_path, vmin=vmin, vmax=vmax)

    return save_path
