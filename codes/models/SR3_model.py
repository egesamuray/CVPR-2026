# codes/models/SR3_model.py
import torch
import torch.nn as nn
import numpy as np
import pywt

from .base_model import BaseModel
from models.modules.diffusion_net import SR3UNet
import models.SWT as SWT  # repo's Stationary Wavelet Transform module (SWTForward)

class SR3Model(BaseModel):
    """
    Diffusion training with a wavelet-domain objective.
    Predict \hat{x}_0 from x_t and minimize:
      L = λ_LL * L2(LL( \hat{x}_0 ), LL( x_0 ))  +
          Σ_{b∈{LH,HL,HH}} λ_b * [ (μ_b( \hat{x}_0 )-μ_b( x_0 ))^2 + (σ_b( \hat{x}_0 )-σ_b( x_0 ))^2 ].
    """
    def __init__(self, opt):
        super().__init__(opt)
        train_opt = opt['train']
        net_opt = opt['network_G']

        # generator
        self.netG = SR3UNet(
            in_ch=net_opt['in_nc'],
            out_ch=net_opt['out_nc'],
            base_nf=net_opt.get('nf', 64),
            num_res_blocks=net_opt.get('num_res_blocks', 2)
        ).to(self.device)
        self.netG.train()

        # pixel criterion for LL / full-image if enabled
        self.cri_pix = None
        if train_opt.get('pixel_weight', 0) > 0:
            if train_opt['pixel_criterion'].lower() == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif train_opt['pixel_criterion'].lower() == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            else:
                raise ValueError("pixel_criterion must be l1 or l2")

        # wavelet settings (normalized filters like WGSR)
        self.wavelet_filter = train_opt.get('wavelet_filter', 'sym7')
        self.wavelet_level  = train_opt.get('wavelet_level', 1)
        wavelet = pywt.Wavelet(self.wavelet_filter)
        dec_lo = np.array(wavelet.dec_lo, dtype=np.float64); dec_lo = dec_lo / dec_lo.sum()
        dec_hi = np.array(wavelet.dec_hi, dtype=np.float64)
        rec_lo = np.array(wavelet.rec_lo, dtype=np.float64); rec_lo = 2 * rec_lo / rec_lo.sum()
        rec_hi = np.array(wavelet.rec_hi, dtype=np.float64)
        filt = pywt.Wavelet('norm_wave', [dec_lo, dec_hi, rec_lo, rec_hi])

        self.swt_forward = SWT.SWTForward(self.wavelet_level, filt, mode='periodic').to(self.device)

        # diffusion schedule (linear β)
        self.num_steps = int(train_opt.get('diffusion_steps', 1000))
        beta0 = float(train_opt.get('beta_start', 1e-4))
        beta1 = float(train_opt.get('beta_end', 2e-2))
        self.beta_schedule = torch.linspace(beta0, beta1, self.num_steps, device=self.device)
        alphas = 1.0 - self.beta_schedule
        self.alpha_bars = torch.cumprod(alphas, dim=0)

        # optimizer & schedulers (use G settings from options)
        wd_G = train_opt.get('weight_decay_G', 0)
        self.optimizer_G = torch.optim.Adam(
            [p for p in self.netG.parameters() if p.requires_grad],
            lr=train_opt['lr_G'],
            weight_decay=wd_G,
            betas=(train_opt['beta1_G'], 0.999)
        )
        self.optimizers.append(self.optimizer_G)

        if train_opt.get('lr_scheme') == 'MultiStepLR':
            self.schedulers.append(torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_G,
                milestones=train_opt['lr_steps'],
                gamma=train_opt['lr_gamma']
            ))

        # weights for subbands
        self.w_LL = float(train_opt.get('pixel_weight', 0.1))
        self.w_LH = float(train_opt.get('pixel_weight_lh', 0.05))
        self.w_HL = float(train_opt.get('pixel_weight_hl', 0.05))
        self.w_HH = float(train_opt.get('pixel_weight_hh', 0.10))

        self.log_dict = {}

    # ---------- required hooks ----------
    def feed_data(self, data):
        self.var_L = data['LR'].to(self.device)  # low-res conditioning
        self.var_H = data['HR'].to(self.device)  # ground-truth HR x0

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        B = self.var_H.size(0)
        t_int  = torch.randint(0, self.num_steps, (B,), device=self.device)
        t_norm = t_int.float() / (self.num_steps - 1)

        noise = torch.randn_like(self.var_H)
        alpha_bar_t = self.alpha_bars[t_int].view(-1, 1, 1, 1)
        x_t = (alpha_bar_t.sqrt() * self.var_H) + ((1 - alpha_bar_t).sqrt() * noise)

        # predict x0
        x0_hat = self.netG(x_t, self.var_L, t_norm)

        # luminance (Y) as in WGSR before SWT
        fake_y = 16.0 + (x0_hat[:, 0:1] * 65.481 + x0_hat[:, 1:2] * 128.553 + x0_hat[:, 2:3] * 24.966)
        real_y = 16.0 + (self.var_H[:, 0:1] * 65.481 + self.var_H[:, 1:2] * 128.553 + self.var_H[:, 2:3] * 24.966)

        coeffs_fake = self.swt_forward(fake_y)[0]  # [B,4,H,W]
        coeffs_real = self.swt_forward(real_y)[0]

        LL_f, LH_f, HL_f, HH_f = coeffs_fake[:,0:1], coeffs_fake[:,1:2], coeffs_fake[:,2:3], coeffs_fake[:,3:]
        LL_r, LH_r, HL_r, HH_r = coeffs_real[:,0:1], coeffs_real[:,1:2], coeffs_real[:,2:3], coeffs_real[:,3:]

        # subband distribution loss (moment matching)
        def band_moment_loss(a, b):
            mu_a, mu_b = a.mean(), b.mean()
            sd_a, sd_b = a.std(unbiased=False), b.std(unbiased=False)
            return (mu_a - mu_b).pow(2) + (sd_a - sd_b).pow(2)

        loss = 0.0
        # low-frequency structure (optional)
        if self.cri_pix is not None and self.w_LL > 0:
            loss_LL = self.cri_pix(LL_f, LL_r)
            loss += self.w_LL * loss_LL
            self.log_dict['l_LL'] = loss_LL.detach().item()

        # high-frequency distribution alignment
        loss_LH = band_moment_loss(LH_f, LH_r); loss += self.w_LH * loss_LH
        loss_HL = band_moment_loss(HL_f, HL_r); loss += self.w_HL * loss_HL
        loss_HH = band_moment_loss(HH_f, HH_r); loss += self.w_HH * loss_HH

        loss.backward()
        self.optimizer_G.step()

        self.log_dict['l_total'] = loss.detach().item()
        self.log_dict['l_LH'] = loss_LH.detach().item()
        self.log_dict['l_HL'] = loss_HL.detach().item()
        self.log_dict['l_HH'] = loss_HH.detach().item()

    def get_current_log(self):
        return self.log_dict

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            # deterministic eval: t=0 => denoise from x0 (identity).
            # For proper sampling, use a sampling script with iterative reverse diffusion.
            x_t = self.var_H  # or run a few denoise steps if you wish
            zeros = torch.zeros(x_t.size(0), device=self.device)
            self.fake_H = self.netG(x_t, self.var_L, zeros)
        self.netG.train()

    def get_current_visuals(self, need_HR=True):
        out = {'LR': self.var_L.detach()[0].float().cpu(),
               'SR': self.fake_H.detach()[0].float().cpu()}
        if need_HR:
            out['HR'] = self.var_H.detach()[0].float().cpu()
        return out
