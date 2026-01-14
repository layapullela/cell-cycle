"""
SR3 Inference for Cell-Cycle Hi-C Phase Decomposition

Clean, focused implementation of SR3 sampling with optional visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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
        
        # Load noise schedule from training
        from train_diffusion import gammas, alphas
        self.gammas = gammas.to(device)
        self.alphas = alphas.to(device)
        
        self.model.eval()
    
    @torch.no_grad()
    def sample(self, bulk_vec, chip_1d):
        """
        SR3 sampling: Generate phase-specific Hi-C from bulk using iterative refinement.
        
        Args:
            bulk_vec: (B, vec_dim) bulk Hi-C conditioning
            chip_1d: (B, N) ChIP-seq conditioning
        
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
            eps_pred = self.model(y_t, gamma_batch, chip_1d, bulk_vec)

            # SR3 Algorithm 2: z ~ N(0,I) if t > 1, else z = 0
            if t_idx > 1:
                z = torch.randn_like(y_t)
            else:
                z = torch.zeros_like(y_t)  # Final step: t=1→0, no noise
            
            # SR3 Algorithm 2 formula (same for all steps)
            y_prev = (1.0 / sqrt_alpha_t) * (y_t - ((1.0 - alpha_t) / sqrt_one_minus_gamma_t) * eps_pred) + sqrt_one_minus_alpha_t * z
            
            y_t = y_prev
        
        # After loop, y_t is y_0 (fully denoised)
        return y_t
    
    def visualize(self, batch, phase_name, output_path=None, n=64):
        """
        Run inference and visualize results.
        
        Args:
            batch: Dict with keys 'earlyG1', 'midG1', 'lateG1', 'chip_seq', 'region'
            phase_name: Which phase to visualize ('earlyG1', 'midG1', or 'lateG1')
            output_path: Where to save plot (if None, just display)
            n: Matrix size (default 64)
        
        Returns:
            sampled: (B, vec_dim) sampled phase-specific Hi-C
        """
        # Import matrix conversion utilities
        from train_diffusion import upper_tri_vec_to_matrix
        
        # Extract ground truth and conditioning
        x0_early = batch['earlyG1'].float().to(self.device)
        x0_mid = batch['midG1'].float().to(self.device)
        x0_late = batch['lateG1'].float().to(self.device)
        chip_1d = batch['chip_seq'].float().to(self.device)
        
        # Compute bulk conditioning
        bulk_vec = (x0_early + x0_mid + x0_late) / 3.0
        
        # Get ground truth for current phase
        phase_to_gt = {
            'earlyG1': x0_early,
            'midG1': x0_mid,
            'lateG1': x0_late
        }
        gt_vec = phase_to_gt[phase_name]
        
        # Run sampling
        sampled_vec = self.sample(bulk_vec, chip_1d)
        
        # Convert to matrices for visualization (use first sample in batch)
        gt_matrix = upper_tri_vec_to_matrix(gt_vec[0:1], n)[0].cpu().numpy()
        sampled_matrix = upper_tri_vec_to_matrix(sampled_vec[0:1], n)[0].cpu().numpy()
        bulk_matrix = upper_tri_vec_to_matrix(bulk_vec[0:1], n)[0].cpu().numpy()
        
        # Create visualization: 1 row x 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        vmin, vmax = -1, 1
        
        # Input: Bulk Hi-C (conditioning)
        im1 = axes[0].imshow(bulk_matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Input: Bulk Hi-C\n(Conditioning)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # Output: Sampled phase-specific
        im2 = axes[1].imshow(sampled_matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'Output: Sampled {phase_name}\n(SR3 Inference)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        # Ground truth: Target phase
        im3 = axes[2].imshow(gt_matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[2].set_title(f'Ground Truth: {phase_name}', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
        
        # Compute metrics
        mse = np.mean((gt_matrix - sampled_matrix) ** 2)
        
        # Correlation (handle degenerate cases early in training)
        try:
            corr = np.corrcoef(gt_matrix.flatten(), sampled_matrix.flatten())[0, 1]
            if np.isnan(corr):
                corr = 0.0  # Zero variance or invalid data
        except:
            corr = 0.0  # Fallback for any issues
        
        region = batch.get('region', ['unknown'])[0]
        fig.suptitle(
            f'SR3 Inference: {phase_name} | Region: {region}\n'
            f'MSE: {mse:.6f} | Correlation: {corr:.4f}',
            fontsize=16,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved: {output_path}")
        else:
            plt.show()
        
        plt.close(fig)
        
        return sampled_vec


def run_inference_and_visualize(model, batch, phase_name, device, step, output_dir="./inference_visualizations"):
    """
    Convenience function for running inference during training.
    
    Args:
        model: Trained SR3UNet model
        batch: Training batch
        phase_name: Phase to sample ('earlyG1', 'midG1', or 'lateG1')
        device: torch device
        step: Current training step (for filename)
        output_dir: Where to save visualization
    
    Returns:
        output_path: Path to saved visualization
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize inference
    inference = Inference(model, device, T=1000)
    
    # Run and visualize
    save_path = output_path / f"inference_{phase_name}_step_{step}.png"
    inference.visualize(batch, phase_name, output_path=save_path)
    
    return save_path
