import torch

T = 1000  # diffusion steps

# SR3 training: sample γ ~ Uniform(γ_min, γ_max) continuously.
# No discrete schedule needed during training — these bounds are used in train_step.
GAMMA_MIN = 1e-4
GAMMA_MAX = 1.0 - 1e-4

def sr3_noise_schedule(timesteps, gamma_min=1e-4, gamma_max=1.0):
    """
    SR3 inference schedule: γ decreases from 1.0 (pure noise) to ~0 (clean).
    
    During inference, we start at t=T (γ=1.0, pure noise) and denoise
    down to t=0 (γ≈0, clean image).
    
    Args:
        timesteps: Number of diffusion steps T
        gamma_min: Minimum noise level (end, almost clean) - default 1e-4
        gamma_max: Maximum noise level (start, pure noise) - default 1.0
    
    Returns:
        gammas: (T,) tensor of noise levels, decreasing from gamma_max to gamma_min
        alphas: (T,) tensor of step-wise alphas α_t = γ_{t-1} / γ_t
        alphas_cumprod: (T,) tensor (kept for compatibility, not used in SR3)
    """
    # at inference, we start with gamma close to 0 at T - 1 and end with gamma close to 1 at time step 0
    gammas = torch.linspace(gamma_max, gamma_min, T)

    alphas = torch.ones(T)
    
    # Compute α_t = γ_{t-1} / γ_t for reverse process
    alphas[1:] = gammas[1:] / (gammas[:-1] + 1e-10)
    
    # Cumulative product (not used in SR3, kept for compatibility)
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return gammas, alphas, alphas_cumprod

# Initialize SR3 inference schedule (only used during inference, not training)
gammas, alphas, alphas_cumprod = sr3_noise_schedule(T, gamma_min=1e-4, gamma_max=1.0)
# Note: Training does NOT use this schedule - it samples γ ~ Uniform(0,1) directly

# ---- Commented-out variants ----
# import math
# def cosine_schedule(T, s=0.008):
#     """Cosine schedule (Nichol & Dhariwal 2021) — designed for discrete-timestep DDPM, not SR3."""
#     steps = torch.arange(T + 1, dtype=torch.float64)
#     f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
#     gammas = (f / f[0]).float()
#     return torch.clamp(gammas[1:], min=0.0, max=1.0 - 1e-4)
# gammas = cosine_schedule(T)
#
# def enforce_zero_terminal_snr(gammas):
#     """Rescale schedule so gammas[-1] = 0 exactly (Lin et al. 2023)."""
#     sqrt_gammas = torch.sqrt(gammas)
#     sqrt_gammas = (sqrt_gammas - sqrt_gammas[-1]) / (sqrt_gammas[0] - sqrt_gammas[-1]) * sqrt_gammas[0]
#     return sqrt_gammas ** 2
# gammas = enforce_zero_terminal_snr(cosine_schedule(T))
