# codes/models/modules/diffusion_net.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(timesteps, embed_dim):
    """
    Sinusoidal timestep embeddings as in DDPM/Transformers.
    timesteps: [B] float or int; we expect RAW integer steps (0..T-1) for a healthy spectrum.
    returns: [B, embed_dim]
    """
    device = timesteps.device
    half = embed_dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / half)
    angles = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if embed_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class ResBlock(nn.Module):
    """Residual block with FiLM-like time/class injection."""
    def __init__(self, n_channels, time_embed_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=n_channels)
        self.conv1 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, n_channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=n_channels)
        self.conv2 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        y = self.act(self.norm1(x))
        y = self.conv1(y)
        y = y + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        y = self.act(self.norm2(y))
        y = self.conv2(y)
        return x + y

class SR3UNet(nn.Module):
    """
    SR3 UNet (x0-parameterization): input is concat(noisy_HR, upsampled_LR) => 6 channels.
    Predicts x0 at timestep t. Optional class_id allows one model to handle multiple classes.
    """
    def __init__(self, in_ch=3, out_ch=3, base_nf=64, num_res_blocks=2, num_classes=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.base_nf = base_nf
        self.num_res_blocks = num_res_blocks
        self.num_classes = num_classes

        time_embed_dim = base_nf * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_nf, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        if num_classes is not None and num_classes > 0:
            self.class_embed = nn.Embedding(num_classes, base_nf)
            self.class_proj = nn.Linear(base_nf, time_embed_dim)
        else:
            self.class_embed = None
            self.class_proj = None

        # encoder
        self.conv_in = nn.Conv2d(in_ch * 2, base_nf, 3, padding=1)
        self.down1 = nn.ModuleList([ResBlock(base_nf, time_embed_dim) for _ in range(num_res_blocks)])
        self.down2_conv = nn.Conv2d(base_nf, base_nf * 2, 3, stride=2, padding=1)
        self.down2 = nn.ModuleList([ResBlock(base_nf * 2, time_embed_dim) for _ in range(num_res_blocks)])
        self.down3_conv = nn.Conv2d(base_nf * 2, base_nf * 4, 3, stride=2, padding=1)
        self.down3 = nn.ModuleList([ResBlock(base_nf * 4, time_embed_dim) for _ in range(num_res_blocks)])

        # bottleneck
        self.mid = ResBlock(base_nf * 4, time_embed_dim)

        # decoder
        self.up3_conv = nn.ConvTranspose2d(base_nf * 4, base_nf * 2, 4, stride=2, padding=1)
        self.up3 = nn.ModuleList([ResBlock(base_nf * 2, time_embed_dim) for _ in range(num_res_blocks)])
        self.up2_conv = nn.ConvTranspose2d(base_nf * 2, base_nf, 4, stride=2, padding=1)
        self.up2 = nn.ModuleList([ResBlock(base_nf, time_embed_dim) for _ in range(num_res_blocks)])

        self.conv_out = nn.Conv2d(base_nf, out_ch, 3, padding=1)

    def forward(self, x_noisy, x_lowres, t_raw, class_id=None):
        """
        x_noisy:  [B,3,H,W]  (noisy HR)
        x_lowres: [B,3,h,w]  (LR conditioning; will be upsampled to HxW)
        t_raw:    [B] int/float in [0, T-1]
        class_id: [B] optional integer labels for class-conditioning
        """
        B, _, H, W = x_noisy.shape
        # upsample LR to match HR spatial size
        x_lr_up = F.interpolate(x_lowres, size=(H, W), mode='bilinear', align_corners=False)

        # time / class embeddings
        t_emb = timestep_embedding(t_raw, self.base_nf)
        t_emb = self.time_mlp(t_emb)
        if self.class_embed is not None and class_id is not None:
            ce = self.class_embed(class_id.long())
            t_emb = t_emb + self.class_proj(ce)

        # encoder
        x = torch.cat([x_noisy, x_lr_up], dim=1)       # [B, 6, H, W]
        x = self.conv_in(x)
        for block in self.down1:
            x = block(x, t_emb)
        skip1 = x

        x = self.down2_conv(x)
        for block in self.down2:
            x = block(x, t_emb)
        skip2 = x

        x = self.down3_conv(x)
        for block in self.down3:
            x = block(x, t_emb)

        # bottleneck
        x = self.mid(x, t_emb)

        # decoder
        x = self.up3_conv(x)
        x = x + skip2
        for block in self.up3:
            x = block(x, t_emb)

        x = self.up2_conv(x)
        x = x + skip1
        for block in self.up2:
            x = block(x, t_emb)

        return self.conv_out(x)

