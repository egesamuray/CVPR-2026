# improved_diffusion/losses_wavelet.py
import torch

@torch.no_grad()
def channel_moments(x: torch.Tensor):
    """
    x: (B, C, H, W)
    returns per-channel mean and variance over batch+space: (C,), (C,)
    """
    B, C, H, W = x.shape
    m = x.mean(dim=(0,2,3))
    v = x.var(dim=(0,2,3), unbiased=False).clamp_min(1e-8)
    return m, v

def gaussian_kl_to_unit(mu: torch.Tensor, var: torch.Tensor):
    """
    KL(N(mu,var) || N(0,1)) for diagonal per-channel Gaussians.
    Returns per-channel KL, shape (C,)
    """
    return 0.5 * (mu.pow(2) + var - var.log() - 1.0)

def wavelet_hf_kl_regularizer(x0_hat_whitened: torch.Tensor, weights: torch.Tensor = None):
    """
    Encourage predicted HF (whitened) to match N(0,I) per channel (matching real data HF stats).
    x0_hat_whitened: (B, C=9, H, W)
    weights (optional): (C,) multipliers per HF band.
    """
    mu, var = channel_moments(x0_hat_whitened)
    kl_c = gaussian_kl_to_unit(mu, var)  # (C,)
    if weights is not None:
        kl_c = kl_c * weights
    return kl_c.mean()
