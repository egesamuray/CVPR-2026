# improved_diffusion/cond_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNetModel  # your existing UNet backbone

class CondUNetModel(nn.Module):
    """
    SR3-style conditional UNet.
    Concatenates conditioning (e.g., upsampled LR or LL band) with the noised HF input
    and forwards through your existing UNetModel.

    Args:
        in_channels_hf:  number of HF channels we diffuse (e.g., 9 = 3 colors x {LH,HL,HH})
        cond_channels:   number of conditioning channels (e.g., 3 for RGB LR↑ or 3 for LL)
        unet_kwargs:     kwargs forwarded to UNetModel (must set in_channels=in_channels_hf+cond_channels,
                         out_channels=in_channels_hf if predicting eps in HF space)
    """
    def __init__(self, in_channels_hf: int, cond_channels: int, **unet_kwargs):
        super().__init__()
        self.in_channels_hf = in_channels_hf
        self.cond_channels = cond_channels
        total_in = in_channels_hf + cond_channels
        # Ensure UNet sees (HF + cond) channels and outputs HF-shaped prediction
        self.unet = UNetModel(in_channels=total_in, out_channels=in_channels_hf, **unet_kwargs)

    def forward(self, x, timesteps, conditioning=None, **kwargs):
        """
        x:           (B, in_channels_hf, H, W)    - noised HF at time t
        conditioning:(B, cond_channels, H, W)     - LR↑ or LL aligned to HF spatial size
        timesteps:   (B,) or scalar tensor
        """
        if conditioning is None:
            raise ValueError("conditioning is required for SR3 wavelet training")
        if conditioning.shape[-2:] != x.shape[-2:]:
            conditioning = F.interpolate(conditioning, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x_cat = torch.cat([x, conditioning], dim=1)
        return self.unet(x_cat, timesteps, **kwargs)
