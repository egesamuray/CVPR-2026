# codes/models/SR3_model.py
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt

from .base_model import BaseModel
from models.modules.diffusion_net import SR3UNet
import models.SWT as SWT  # SWTForward/SWTInverse

# -------------------- utils --------------------

def y_from_rgb01(x):  # x in [0,1]
    # BT.601 luma from normalized RGB
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

def soft_histogram(x, bins, xmin, xmax, tau=0.1, eps=1e-12):
    """
    Differentiable soft-hist via Gaussian kernels over centers in [xmin,xmax].
    x: [B,1,H,W] or [B,N]  -> probs [B,bins]
    """
    if x.dim() > 2:
        x = x.flatten(1)
    B, N = x.shape
    centers = torch.linspace(xmin, xmax, bins, device=x.device)[None, :]  # [1,K]
    x = x[:, :, None]  # [B,N,1]
    k = torch.exp(-(x - centers)**2 / (2*(tau**2)))
    k_sum = k.sum(dim=1) + eps  # [B,K]
    return k_sum / k_sum.sum(dim=1, keepdim=True).clamp_min(eps)

class EMABandPrior:
    """
    EMA histogram prior per HF band and (optional) class.
    Keeps p_bar in simplex; update: p_bar <- m p_bar + (1-m) mean(P_batch)
    """
    def __init__(self, bins=128, momentum=0.99, n_classes=2, device='cpu'):
        self.bins = bins
        self.momentum = momentum
        self.device = device
        self.n_classes = n_classes
        self.store = {}  # keys: ('LH',c), ('HL',c), ('HH',c)

    def get(self, band, c):
        key = (band, int(c))
        if key not in self.store:
            self.store[key] = torch.full((self.bins,), 1.0/self.bins, device=self.device)
        return self.store[key]

    def update(self, band, c, p_batch):
        key = (band, int(c))
        p_bar = self.get(band, c)
        with torch.no_grad():
            p_new = p_batch.mean(dim=0)
            self.store[key] = self.momentum * p_bar + (1.0 - self.momentum) * p_new


class SR3Model(BaseModel):
    r"""
    Diffusion training + wavelet-domain distribution objective.

    Total loss:
      L = w_eps * E[ ||eps_hat - eps||^2 ]  +  gamma(t) * Σ_b λ_b * D_b( W_b(\hat x_0), W_b(x_0) )
    where D_b is KL-to-prior (soft histogram) or MMD; b ∈ {LH,HL,HH}.
    """
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        train_opt = opt['train']
        net_opt   = opt['network_G']

        # scale (for LR-consistency)
        self.scale = int(opt.get('scale', 4))

        # generator (class-conditional if requested)
        num_classes = train_opt.get('num_classes', None) if train_opt.get('use_class_prior', False) else None
        self.netG = SR3UNet(
            in_ch=net_opt['in_nc'],
            out_ch=net_opt['out_nc'],
            base_nf=net_opt.get('nf', 64),
            num_res_blocks=net_opt.get('num_res_blocks', 2),
            num_classes=num_classes
        ).to(self.device)
        # EMA (recommended for eval/sampling)
        from copy import deepcopy
        self.use_ema = bool(train_opt.get('use_ema', False))
        self.ema_decay = float(train_opt.get('ema_decay', 0.999))
        if opt.get('gpu_ids') and len(opt['gpu_ids']) > 1:
            self.netG = nn.DataParallel(self.netG, device_ids=opt['gpu_ids'])
        if self.use_ema:
            self.netG_ema = deepcopy(self.netG).eval()
            for p in self.netG_ema.parameters(): p.requires_grad_(False)
        self.netG.train()

        # pixel criterion for LL anchor (optional)
        self.cri_pix = None
        if float(train_opt.get('pixel_weight', 0)) > 0:
            crit = train_opt.get('pixel_criterion', 'l2').lower()
            self.cri_pix = (nn.MSELoss() if crit == 'l2' else nn.L1Loss()).to(self.device)

        # wavelet settings (normalized filters)
        self.wavelet_filter = train_opt.get('wavelet_filter', 'sym7')
        self.wavelet_level  = int(train_opt.get('wavelet_level', 1))
        wavelet = pywt.Wavelet(self.wavelet_filter)
        dec_lo = np.array(wavelet.dec_lo, dtype=np.float64); dec_lo = dec_lo / max(dec_lo.sum(), 1e-12)
        dec_hi = np.array(wavelet.dec_hi, dtype=np.float64)
        rec_lo = np.array(wavelet.rec_lo, dtype=np.float64); rec_lo = 2 * rec_lo / max(rec_lo.sum(), 1e-12)
        rec_hi = np.array(wavelet.rec_hi, dtype=np.float64)
        filt = pywt.Wavelet('norm_wave', [dec_lo, dec_hi, rec_lo, rec_hi])
        self.swt_forward = SWT.SWTForward(self.wavelet_level, filt, mode='periodic').to(self.device)

        # diffusion schedule
        self.num_steps = int(train_opt.get('diffusion_steps', 1000))
        schedule = train_opt.get('beta_schedule', 'linear')
        self.beta_schedule = self._build_beta_schedule(schedule, self.num_steps, device=self.device)
        alphas = 1.0 - self.beta_schedule
        self.alpha_bars = torch.cumprod(alphas, dim=0)

        # optimizer & schedulers
        wd_G = float(train_opt.get('weight_decay_G', 0))
        self.optimizer_G = torch.optim.Adam(
            [p for p in self.netG.parameters() if p.requires_grad],
            lr=float(train_opt['lr_G']),
            weight_decay=wd_G,
            betas=(float(train_opt.get('beta1_G', 0.9)), 0.999)
        )
        self.optimizers.append(self.optimizer_G)
        if train_opt.get('lr_scheme') == 'MultiStepLR':
            self.schedulers.append(torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_G,
                milestones=train_opt['lr_steps'],
                gamma=train_opt['lr_gamma']
            ))

        # weights for subbands and eps
        self.w_LL = float(train_opt.get('pixel_weight',    0.10))
        self.w_LH = float(train_opt.get('pixel_weight_lh', 0.05))
        self.w_HL = float(train_opt.get('pixel_weight_hl', 0.05))
        self.w_HH = float(train_opt.get('pixel_weight_hh', 0.10))
        self.w_eps = float(train_opt.get('eps_weight', 1.0))

        # wavelet distribution config
        self.wavelet_loss = train_opt.get('wavelet_loss', 'softkl')  # 'softkl' | 'mmd' | 'moment'
        self.gamma_a = float(train_opt.get('gamma_a', 2.0))
        self.gamma_b = float(train_opt.get('gamma_b', 4.0))
        self.t_weight_eps = train_opt.get('t_weight_eps', 'sigma_inv_sq')

        # MMD memory budget
        self.mmd_max_points = int(train_opt.get('mmd_max_points', 4096))
        self.mmd_dtype      = train_opt.get('mmd_dtype', 'float16')
        self.mmd_sigmas     = train_opt.get('mmd_sigmas', [0.5, 1.0, 2.0])

        # LR-consistency (optional)
        self.lr_cons_w = float(train_opt.get('lr_consistency_weight', 0.0))

        # soft-hist / prior options
        self.use_class_prior = bool(train_opt.get('use_class_prior', False))
        self.num_classes     = int(train_opt.get('num_classes', 2)) if self.use_class_prior else None
        self.hist_bins       = int(train_opt.get('hist_bins', 128))
        self.hist_min        = float(train_opt.get('hist_min', -0.5))
        self.hist_max        = float(train_opt.get('hist_max',  0.5))
        self.hist_tau        = float(train_opt.get('hist_tau',  0.10))
        self.prior_momentum  = float(train_opt.get('prior_momentum', 0.99))
        self.band_prior = EMABandPrior(self.hist_bins, self.prior_momentum,
                                       n_classes=self.num_classes or 2, device=self.device) if self.use_class_prior else None

        # try loading persisted EMA priors (no train.py edits needed)
        self.experiments_root = opt['path']['experiments_root']
        self.prior_path = os.path.join(self.experiments_root, 'band_prior.pt')
        if self.use_class_prior and os.path.exists(self.prior_path):
            try:
                ckpt = torch.load(self.prior_path, map_location='cpu')
                self._load_persistent_state(ckpt)
            except Exception:
                pass

        self.log_dict = {}

    # ---------- beta schedules ----------
    def _build_beta_schedule(self, kind, T, device):
        if kind == 'cosine':
            # Improved DDPM (Nichol & Dhariwal)
            s = 0.008
            t = torch.linspace(0, T, T+1, device=device) / T
            alphas_bar = torch.cos((t + s)/(1+s) * math.pi/2)**2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            return betas.clamp(1e-8, 0.999)
        else:
            # linear
            beta0 = 1e-4
            beta1 = 2e-2
            return torch.linspace(beta0, beta1, T, device=device)

    # --- inside class SR3Model(BaseModel) ---

    def feed_data(self, data):
        self.var_L = data['LR'].to(self.device)
        self.var_H = data['HR'].to(self.device)

        self.class_id = None
        if 'class_id' in data:
            self.class_id = (data['class_id'].to(self.device)
                             if isinstance(data['class_id'], torch.Tensor)
                             else torch.tensor(data['class_id'], device=self.device, dtype=torch.long))
        elif self.use_class_prior:
            paths = data.get('HR_path') or data.get('LR_path')
            if paths is not None:
                if isinstance(paths, (list, tuple)):
                    ids = []
                    for p in paths:
                        p_low = str(p).lower()
                    # fallback mapping BEFORE clamping
                        ids.append(0 if 'text' in p_low else 1 if 'face' in p_low else 0)
                    self.class_id = torch.tensor(ids, device=self.device, dtype=torch.long)
                else:
                    p_low = str(paths).lower()
                    cid = 0 if 'text' in p_low else 1 if 'face' in p_low else 0
                    self.class_id = torch.tensor([cid], device=self.device, dtype=torch.long)

    # ---- NEW: sanitize to valid embedding range ----
        if self.use_class_prior and self.class_id is not None:
            if getattr(self, 'num_classes', None) is not None:
                if self.num_classes == 1:
                # faces-only training: collapse to class 0
                    self.class_id = torch.zeros_like(self.class_id)
                else:
                    self.class_id = self.class_id.clamp(0, self.num_classes - 1)

    def _gamma_t(self, t_int):
        t_norm = t_int.float() / max(1, (self.num_steps - 1))
        return torch.sigmoid(self.gamma_a - self.gamma_b * t_norm)

    def _wavelet_bands_all_levels(self, x01):
        """
        Return list of (LL, LH, HL, HH) for levels 1..L.
        Handles SWTForward outputs that are:
          - Tensor [B, 4*L, H, W]
          - List/tuple of Tensors [B,4,H,W] per level
          - Tuple with a stacked Tensor at [0]
        """
        y = y_from_rgb01(x01)
        out = self.swt_forward(y)
        levels = []

        def _split_tensor(t):
            L = int(t.shape[1] // 4)
            res = []
            for lev in range(L):
                base = 4 * lev
                res.append((
                    t[:, base+0:base+1],
                    t[:, base+1:base+2],
                    t[:, base+2:base+3],
                    t[:, base+3:base+4],
                ))
            return res

        if torch.is_tensor(out):
            levels = _split_tensor(out)
        elif isinstance(out, (list, tuple)):
            if len(out) == 1 and torch.is_tensor(out[0]):
                levels = _split_tensor(out[0])
            else:
                for c in out:
                    t = c[0] if isinstance(c, (list, tuple)) and torch.is_tensor(c[0]) else c
                    if torch.is_tensor(t) and t.dim() == 4 and t.size(1) >= 4:
                        levels.append((t[:,0:1], t[:,1:2], t[:,2:3], t[:,3:4]))
        if not levels:
            t = out[0] if isinstance(out, (list, tuple)) else out
            levels = [(t[:,0:1], t[:,1:2], t[:,2:3], t[:,3:4])]
        return levels

    # ---------- memory-friendly MMD ----------
    def _mmd_loss_band(self, A, B):
        Bsz = A.size(0)
        losses = []
        use_half = (self.mmd_dtype == 'float16' and A.is_cuda)
        for i in range(Bsz):
            x = A[i].reshape(-1, 1)
            y = B[i].reshape(-1, 1)
            if x.size(0) > self.mmd_max_points:
                idx = torch.randperm(x.size(0), device=A.device)[:self.mmd_max_points]
                x = x[idx]
            if y.size(0) > self.mmd_max_points:
                idy = torch.randperm(y.size(0), device=A.device)[:self.mmd_max_points]
                y = y[idy]
            kdtype = torch.float16 if use_half else torch.float32
            xk = x.to(kdtype); yk = y.to(kdtype)
            dxx = torch.cdist(xk, xk, p=2).pow(2)
            dyy = torch.cdist(yk, yk, p=2).pow(2)
            dxy = torch.cdist(xk, yk, p=2).pow(2)
            mmd2 = 0.0
            for s in self.mmd_sigmas:
                gamma = 1.0 / (2.0 * s * s + 1e-8)
                kxx = torch.exp(-gamma * dxx).mean()
                kyy = torch.exp(-gamma * dyy).mean()
                kxy = torch.exp(-gamma * dxy).mean()
                mmd2 = mmd2 + (kxx + kyy - 2.0 * kxy)
            losses.append(mmd2 / float(len(self.mmd_sigmas)))
        return torch.stack(losses).mean().to(A.dtype)

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        B = self.var_H.size(0)
        t_int = torch.randint(0, self.num_steps, (B,), device=self.device)
        alpha_bar_t = self.alpha_bars[t_int].view(-1, 1, 1, 1)
        sigma_t = (1 - alpha_bar_t).sqrt()
        noise = torch.randn_like(self.var_H)
        x_t = alpha_bar_t.sqrt() * self.var_H + sigma_t * noise

        # predict x0 using RAW integer t for embedding
        x0_hat = self.netG(x_t, self.var_L, t_int.float(), class_id=self.class_id)

        # ε-loss backbone (derive eps_hat from x0_hat)
        eps_hat = (x_t - alpha_bar_t.sqrt() * x0_hat) / sigma_t.clamp_min(1e-8)
        loss_eps = (eps_hat - noise).pow(2).mean()
        if self.t_weight_eps == 'sigma_inv_sq':
            w_eps_t = (1.0 / (sigma_t**2 + 1e-6)).mean()
            loss_backbone = self.w_eps * w_eps_t * loss_eps
        else:
            loss_backbone = self.w_eps * loss_eps

        # wavelet distribution objective on x0_hat vs x0 (multi-level)
        bands_f = self._wavelet_bands_all_levels(x0_hat.clamp(0,1))
        bands_r = self._wavelet_bands_all_levels(self.var_H.clamp(0,1))
        decays = [1.0 / (2**i) for i in range(len(bands_f))]  # level1=1.0, level2=0.5, ...

        loss_LL = torch.tensor(0.0, device=self.device)
        loss_LH = torch.tensor(0.0, device=self.device)
        loss_HL = torch.tensor(0.0, device=self.device)
        loss_HH = torch.tensor(0.0, device=self.device)

        def _kl_qp(q, p, eps=1e-12):
            return (p * (p.add(eps).log() - q.add(eps).log())).sum(dim=1).mean()
        def _band_softkl(f, r, band_name):
            P = soft_histogram(r, self.hist_bins, self.hist_min, self.hist_max, self.hist_tau)
            Q = soft_histogram(f, self.hist_bins, self.hist_min, self.hist_max, self.hist_tau)
            if self.use_class_prior and (self.class_id is not None):
                for c in self.class_id.unique().tolist():
                    mask = (self.class_id == c)
                    if mask.any():
                        self.band_prior.update(band_name, c, P[mask])
                priors = []
                for i in range(P.size(0)):
                    c_i = int(self.class_id[i].item()) if self.class_id is not None else 0
                    priors.append(self.band_prior.get(band_name, c_i)[None, :])
                P_bar = torch.cat(priors, dim=0)
                return _kl_qp(Q, P_bar)  # KL(prior || Q)
            else:
                return _kl_qp(Q, P)     # KL(P || Q) per-image

        for (LLf, LHf, HLf, HHf), (LLr, LHr, HLr, HHr), w in zip(bands_f, bands_r, decays):
            if self.cri_pix is not None and self.w_LL > 0:
                loss_LL = loss_LL + w * self.cri_pix(LLf, LLr)

            if self.wavelet_loss == 'softkl':
                loss_LH = loss_LH + w * _band_softkl(LHf, LHr, 'LH')
                loss_HL = loss_HL + w * _band_softkl(HLf, HLr, 'HL')
                loss_HH = loss_HH + w * _band_softkl(HHf, HHr, 'HH')
            elif self.wavelet_loss == 'mmd':
                loss_LH = loss_LH + w * self._mmd_loss_band(LHf, LHr)
                loss_HL = loss_HL + w * self._mmd_loss_band(HLf, HLr)
                loss_HH = loss_HH + w * self._mmd_loss_band(HHf, HHr)
            else:
                # ablation: simple moment loss
                def band_moment_loss(a, b, eps=1e-6):
                    B = a.size(0)
                    mu_a = a.view(B,-1).mean(dim=1); mu_b = b.view(B,-1).mean(dim=1)
                    sd_a = a.view(B,-1).std(dim=1, unbiased=False).clamp_min(eps)
                    sd_b = b.view(B,-1).std(dim=1, unbiased=False).clamp_min(eps)
                    return ((mu_a - mu_b)**2 + (sd_a - sd_b)**2).mean()
                loss_LH = loss_LH + w * band_moment_loss(LHf, LHr)
                loss_HL = loss_HL + w * band_moment_loss(HLf, HLr)
                loss_HH = loss_HH + w * band_moment_loss(HHf, HHr)

        loss_wav = (self.w_LL * loss_LL) + (self.w_LH * loss_LH) + (self.w_HL * loss_HL) + (self.w_HH * loss_HH)

        # optional LR-consistency (downsample with bicubic antialias to match dataset LR creation)
        if self.lr_cons_w > 0:
            down = F.interpolate(x0_hat.clamp(0,1),
                                 scale_factor=1.0/self.scale,
                                 mode='bicubic', align_corners=False, antialias=True)
            loss_lr = F.mse_loss(down, self.var_L)
            loss_wav = loss_wav + self.lr_cons_w * loss_lr
            self.log_dict['l_LR'] = float(loss_lr.detach().item())

        gamma = self._gamma_t(t_int).mean()
        loss = loss_backbone + gamma * loss_wav

        loss.backward()
        self.optimizer_G.step()

        # EMA update
        if self.use_ema:
            def _ema_update(ema, online, m):
                for p_ema, p in zip(ema.parameters(), online.parameters()):
                    p_ema.data.mul_(m).add_(p.data, alpha=1-m)
            G_ema = self.netG_ema.module if hasattr(self.netG_ema, 'module') else self.netG_ema
            G_online = self.netG.module if hasattr(self.netG, 'module') else self.netG
            _ema_update(G_ema, G_online, self.ema_decay)

        # persist EMA priors periodically (no train.py edits)
        try:
            save_freq = int(self.opt['logger'].get('save_checkpoint_freq', 20000))
            if self.use_class_prior and save_freq > 0 and step > 0 and (step % save_freq == 0):
                os.makedirs(self.experiments_root, exist_ok=True)
                torch.save(self._get_persistent_state(), self.prior_path)
        except Exception:
            pass

        self.log_dict['l_total'] = float(loss.detach().item())
        self.log_dict['l_eps']   = float(loss_eps.detach().item())
        self.log_dict['l_LL']    = float(loss_LL.detach().item())
        self.log_dict['l_LH']    = float(loss_LH.detach().item())
        self.log_dict['l_HL']    = float(loss_HL.detach().item())
        self.log_dict['l_HH']    = float(loss_HH.detach().item())
        self.log_dict['gamma']   = float(gamma.detach().item())

    def get_current_log(self):
        return self.log_dict

    def test(self):
        G = self.netG_ema if getattr(self, 'use_ema', False) else self.netG
        G.eval()
        with torch.no_grad():
            zeros = torch.zeros(self.var_H.size(0), device=self.device)
            self.fake_H = G(self.var_H, self.var_L, zeros, class_id=self.class_id)
        G.train()

    def get_current_visuals(self, need_HR=True):
        out = {'LR': self.var_L.detach()[0].float().cpu(),
               'SR': self.fake_H.detach()[0].float().cpu()}
        if need_HR:
            out['HR'] = self.var_H.detach()[0].float().cpu()
        return out

    # ---------- EMA prior persistence helpers ----------
    def _get_persistent_state(self):
        if not self.use_class_prior or self.band_prior is None:
            return {}
        state = {str(k): v.detach().float().cpu() for k, v in self.band_prior.store.items()}
        return {'band_prior_store': state}

    def _load_persistent_state(self, obj):
        if not obj or 'band_prior_store' not in obj or self.band_prior is None:
            return
        store = obj['band_prior_store']
        self.band_prior.store = {eval(k): v.to(self.device) for k, v in store.items()}

    # ---------- checkpoints ----------
    def save(self, iter_label):
        """Save generator weights to experiments/<name>/models/ as G_<iter>.pth"""
        self.save_network(self.netG, 'G', iter_label)

    def load(self):
        """Load pretrain_model_G if provided in the options."""
        load_path_G = self.opt['path'].get('pretrain_model_G', None)
        if load_path_G:
            self.load_network(load_path_G, self.netG, strict=True)
