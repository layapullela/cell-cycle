"""
Visualization utilities for diffusion model training.
Samples from trained models and visualizes predicted phases.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_ddpm_schedules(T, beta_start=1e-4, beta_end=0.02, device=None):
    """
    Match training schedule exactly:
      betas, alphas, alphas_cumprod
    and also provide:
      sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
      sqrt_alphas
    """
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas = torch.sqrt(alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    if device is not None:
        betas = betas.to(device)
        alphas = alphas.to(device)
        alphas_cumprod = alphas_cumprod.to(device)
        sqrt_alphas = sqrt_alphas.to(device)
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas": sqrt_alphas,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }


class DiffusionVisualizer:
    """
    Visualizes diffusion model outputs during training.
    Takes bulk Hi-C + ChIP-seq, runs sampling, visualizes predicted phases.
    """

    def __init__(self, models, device, T=1000, output_dir="./visualizations",
                 beta_start=1e-4, beta_end=0.02):
        self.models = models
        self.device = device
        self.T = T
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Schedules that match training
        sch = get_ddpm_schedules(T, beta_start=beta_start, beta_end=beta_end, device=device)
        self.betas = sch["betas"]  # (T,)
        self.alphas = sch["alphas"]
        self.alphas_cumprod = sch["alphas_cumprod"]
        self.sqrt_alphas = sch["sqrt_alphas"]
        self.sqrt_alphas_cumprod = sch["sqrt_alphas_cumprod"]
        self.sqrt_one_minus_alphas_cumprod = sch["sqrt_one_minus_alphas_cumprod"]

        # eval mode for sampling
        for model in self.models.values():
            model.eval()

    @torch.no_grad()
    def _ddpm_step_eps_pred(self, x_t, t_idx, eps_pred, deterministic=False):
        """
        One DDPM reverse step that is consistent with training's q(x_t|x0).

        Training uses:
          x_t = sqrt(a_bar_t) * x0 + sqrt(1-a_bar_t) * eps

        Given eps_pred ~ eps_theta(x_t, t), we do:
          x0_hat = (x_t - sqrt(1-a_bar_t)*eps_pred) / sqrt(a_bar_t)

        DDPM mean (paper / common implementation):
          mu = 1/sqrt(alpha_t) * ( x_t - (beta_t / sqrt(1-a_bar_t)) * eps_pred )

        Posterior variance:
          beta_tilde = beta_t * (1-a_bar_{t-1})/(1-a_bar_t)

        Then:
          x_{t-1} = mu + sqrt(beta_tilde)*z    (z=0 if deterministic or t==0)
        """
        beta_t = self.betas[t_idx]
        alpha_t = self.alphas[t_idx]
        a_bar_t = self.alphas_cumprod[t_idx]

        # coefficients matching your sampling formula, but using the same a_bar_t as training
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - a_bar_t)

        mu = coef1 * (x_t - coef2 * eps_pred)

        if t_idx == 0:
            return mu  # final, no noise

        # posterior variance beta_tilde (more correct than sqrt(beta_t))
        a_bar_prev = self.alphas_cumprod[t_idx - 1]
        beta_tilde = beta_t * (1.0 - a_bar_prev) / (1.0 - a_bar_t)
        sigma = torch.sqrt(beta_tilde.clamp(min=1e-20))

        if deterministic:
            z = torch.zeros_like(x_t)
        else:
            z = torch.randn_like(x_t)

        return mu + sigma * z

    @torch.no_grad()
    def sample_phases(self, bulk_vec, chip_1d, num_steps=None, deterministic=False):
        """
        True DDPM sampling requires contiguous timesteps t=T-1,...,0.

        If num_steps is provided, we will *still* run contiguously but only
        for the last `num_steps` steps (i.e., start at t=num_steps-1).
        (Skipping steps via linspace is NOT valid for DDPM; use DDIM for that.)
        """
        vec_dim = bulk_vec.shape[1]

        x_t_early = torch.randn(1, vec_dim, device=self.device)
        x_t_mid   = torch.randn(1, vec_dim, device=self.device)
        x_t_late  = torch.randn(1, vec_dim, device=self.device)

        if num_steps is None:
            start = self.T - 1
        else:
            # run only a contiguous suffix: start at t=num_steps-1
            start = min(self.T - 1, int(num_steps) - 1)

        for t_idx in range(start, -1, -1):
            t = torch.tensor([t_idx], device=self.device).long()
            t_normalized = t.float() / (self.T - 1)

            eps_pred_early = self.models["earlyG1"](x_t_early, t_normalized, chip_1d, bulk_vec)
            eps_pred_mid   = self.models["midG1"]  (x_t_mid,   t_normalized, chip_1d, bulk_vec)
            eps_pred_late  = self.models["lateG1"] (x_t_late,  t_normalized, chip_1d, bulk_vec)

            x_t_early = self._ddpm_step_eps_pred(x_t_early, t_idx, eps_pred_early, deterministic=deterministic)
            x_t_mid   = self._ddpm_step_eps_pred(x_t_mid,   t_idx, eps_pred_mid,   deterministic=deterministic)
            x_t_late  = self._ddpm_step_eps_pred(x_t_late,  t_idx, eps_pred_late,  deterministic=deterministic)

        return {"earlyG1": x_t_early, "midG1": x_t_mid, "lateG1": x_t_late}

    def visualize_and_save(self, bulk_vec, chip_1d, step, region_name="unknown"):
        """
        Sample phases and create visualization grid.
        """
        # IMPORTANT: for DDPM sanity checks, don't skip steps.
        # Use full sampling first; later move to DDIM if you want 50-step sampling.
        sampled_phases = self.sample_phases(bulk_vec, chip_1d, num_steps=None, deterministic=False)

        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from train_diffusion import upper_tri_vec_to_matrix

        n = chip_1d.shape[1]

        early_matrix = upper_tri_vec_to_matrix(sampled_phases["earlyG1"], n)[0].cpu().numpy()
        mid_matrix   = upper_tri_vec_to_matrix(sampled_phases["midG1"],   n)[0].cpu().numpy()
        late_matrix  = upper_tri_vec_to_matrix(sampled_phases["lateG1"],  n)[0].cpu().numpy()
        bulk_matrix  = upper_tri_vec_to_matrix(bulk_vec, n)[0].cpu().numpy()

        predicted_sum = early_matrix + mid_matrix + late_matrix

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        vmin, vmax = -1, 1
        im1 = axes[0, 0].imshow(early_matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[0, 0].set_title("Predicted Early G1"); axes[0, 0].axis("off"); plt.colorbar(im1, ax=axes[0, 0])

        im2 = axes[0, 1].imshow(mid_matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[0, 1].set_title("Predicted Mid G1"); axes[0, 1].axis("off"); plt.colorbar(im2, ax=axes[0, 1])

        im3 = axes[0, 2].imshow(late_matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[0, 2].set_title("Predicted Late G1"); axes[0, 2].axis("off"); plt.colorbar(im3, ax=axes[0, 2])

        im4 = axes[1, 0].imshow(bulk_matrix, cmap="RdBu_r")
        axes[1, 0].set_title("Ground Truth Bulk"); axes[1, 0].axis("off"); plt.colorbar(im4, ax=axes[1, 0])

        im5 = axes[1, 1].imshow(predicted_sum, cmap="RdBu_r")
        axes[1, 1].set_title("Predicted Sum"); axes[1, 1].axis("off"); plt.colorbar(im5, ax=axes[1, 1])

        axes[1, 2].plot(chip_1d[0].cpu().numpy())
        axes[1, 2].set_title("ChIP-seq Signal")
        axes[1, 2].set_xlabel("Genomic Position")
        axes[1, 2].set_ylabel("Signal")
        axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle(f"Step {step} - Region: {region_name}", fontsize=16)
        plt.tight_layout()

        save_path = self.output_dir / "training_sanity_check.png"
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()

        for model in self.models.values():
            model.train()

        return save_path


def visualize_training_step(models, batch, device, step, T=1000, output_dir="./visualizations"):
    x0_early = batch["earlyG1"][0:1].float().to(device)
    x0_mid   = batch["midG1"][0:1].float().to(device)
    x0_late  = batch["lateG1"][0:1].float().to(device)
    chip_1d  = batch["chip_seq"][0:1].float().to(device)
    region   = batch["region"][0] if "region" in batch else "unknown"

    bulk_vec = x0_early + x0_mid + x0_late

    visualizer = DiffusionVisualizer(models, device, T, output_dir)
    return visualizer.visualize_and_save(bulk_vec, chip_1d, step, region)
