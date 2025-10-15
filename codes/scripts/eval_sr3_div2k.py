# codes/scripts/eval_sr3_div2k.py
import os, sys, glob, json, math, argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# repo-local imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from models.SR3_model import SR3Model  # uses the same UNet/schedule you trained

# ---------- metrics ----------
def to_y(img01_t):  # img in [0,1], CHW torch
    r,g,b = img01_t[0:1], img01_t[1:2], img01_t[2:3]
    return 0.299*r + 0.587*g + 0.114*b  # BT.601

def psnr_y(pred01, gt01, crop=4):
    y_p = to_y(pred01); y_g = to_y(gt01)
    if crop > 0:
        y_p = y_p[..., crop:-crop, crop:-crop]
        y_g = y_g[..., crop:-crop, crop:-crop]
    mse = F.mse_loss(y_p, y_g).item()
    return 100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)

def ssim_y(pred01, gt01, crop=4):
    # simple torch SSIM on Y; window 11x11 gaussian-ish weight
    y_p = to_y(pred01); y_g = to_y(gt01)
    if crop > 0:
        y_p = y_p[..., crop:-crop, crop:-crop]
        y_g = y_g[..., crop:-crop, crop:-crop]
    C1, C2 = (0.01**2), (0.03**2)
    mu1 = F.avg_pool2d(y_p, 11, 1, 5)
    mu2 = F.avg_pool2d(y_g, 11, 1, 5)
    mu1_sq, mu2_sq, mu12 = mu1*mu1, mu2*mu2, mu1*mu2
    sigma1_sq = F.avg_pool2d(y_p*y_p, 11,1,5) - mu1_sq
    sigma2_sq = F.avg_pool2d(y_g*y_g, 11,1,5) - mu2_sq
    sigma12   = F.avg_pool2d(y_p*y_g, 11,1,5) - mu12
    ssim_map = ((2*mu12 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())

# ---------- IO ----------
def imread_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def to_tensor01(arr):
    t = torch.from_numpy(arr).permute(2,0,1).float() / 255.0
    return t

def to_uint8_img(t01):
    t = t01.clamp(0,1).permute(1,2,0).cpu().numpy()
    return Image.fromarray((t*255.0 + 0.5).astype(np.uint8))

# ---------- sampling (x0-parameterization) ----------
@torch.no_grad()
def ddpm_sample_x0param(model_obj, net, lr01, steps=None, class_id=None):
    """
    model_obj: SR3Model instance (has beta schedule, alpha_bars)
    net:       model_obj.netG_ema or model_obj.netG
    lr01:      (1,3,h,w) low-res in [0,1] (will be bicubic-upsampled inside UNet)
    """
    device = lr01.device
    T = int(model_obj.num_steps if steps is None else steps)
    betas = model_obj.beta_schedule[:T]                 # (T,)
    alphas = (1.0 - betas)
    alpha_bars = torch.cumprod(alphas, dim=0)          # same as model_obj.alpha_bars

    # start from pure noise
    h, w = lr01.shape[-2]*model_obj.scale, lr01.shape[-1]*model_obj.scale
    x_t = torch.randn(1, 3, h, w, device=device)

    # reverse diffusion
    for t in reversed(range(T)):
        t_tensor = torch.full((1,), float(t), device=device)
        x0_hat = net(x_t, lr01, t_tensor, class_id=class_id)  # predict x0
        alpha_t     = alphas[t]
        abar_t      = alpha_bars[t]
        abar_tm1    = alpha_bars[t-1] if t > 0 else torch.tensor(1.0, device=device)
        beta_t      = betas[t]
        # posterior q(x_{t-1}|x_t, x0)
        coef1 = (beta_t * torch.sqrt(abar_tm1)) / (1.0 - abar_t + 1e-8)
        coef2 = (torch.sqrt(alpha_t) * (1.0 - abar_tm1)) / (1.0 - abar_t + 1e-8)
        mean  = coef1 * x0_hat + coef2 * x_t
        if t > 0:
            var  = (beta_t * (1.0 - abar_tm1)) / (1.0 - abar_t + 1e-8)
            x_t = mean + torch.sqrt(var).view(1,1,1,1) * torch.randn_like(x_t)
        else:
            x_t = mean
    return x_t.clamp(0, 1)

def find_latest_ckpt(models_dir):
    cands = sorted(glob.glob(os.path.join(models_dir, "G_*.pth")))
    if not cands:  # if you saved raw state_dicts differently, add a fallback here
        raise FileNotFoundError(f"No checkpoints under: {models_dir}")
    return cands[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt",  default=os.path.join(REPO_ROOT, "options/train/train_SR3_wavelet_div2k_x4.json"))
    ap.add_argument("--ckpt", default="")  # optional explicit path to G_*.pth
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_examples", action="store_true")
    args = ap.parse_args()

    # load options and wire in checkpoint path
    with open(args.opt, "r") as f:
        opt = json.load(f)
    exp_root = opt["path"]["experiments_root"]
    models_dir = opt["path"]["models"]
    ckpt = args.ckpt if args.ckpt else find_latest_ckpt(models_dir)

    # make sure scheduling/log ints are ints
    for k in ("val_freq",):  # no-op if not present
        if k in opt.get("train", {}): opt["train"][k] = int(opt["train"][k])
    # plug checkpoint for loading
    opt2 = json.loads(json.dumps(opt))  # deep copy
    opt2["path"]["pretrain_model_G"] = ckpt

    # build model (uses EMA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SR3Model(opt2)     # builds UNet, schedule, EMA, etc.
    model.load()               # load weights
    G = model.netG_ema if getattr(model, "use_ema", False) else model.netG
    G.eval()

    # list val pairs
    hr_dir = opt["datasets"]["val"]["dataroot_HR"]
    lr_dir = opt["datasets"]["val"]["dataroot_LR"]
    hr_list = sorted([p for p in glob.glob(os.path.join(hr_dir, "*.png")) + glob.glob(os.path.join(hr_dir, "*.jpg"))])
    lr_list = [os.path.join(lr_dir, os.path.basename(p)) for p in hr_list]
    assert all(os.path.exists(p) for p in lr_list), "LR/HR mismatch."

    psnrs, ssims = [], []
    save_dir = os.path.join(exp_root, "eval_samples")
    os.makedirs(save_dir, exist_ok=True)

    for idx, (hr_p, lr_p) in enumerate(zip(hr_list, lr_list)):
        if args.limit and idx >= args.limit: break
        hr = to_tensor01(imread_rgb(hr_p)).unsqueeze(0).to(device)      # (1,3,H,W), [0,1]
        lr = to_tensor01(imread_rgb(lr_p)).unsqueeze(0).to(device)      # (1,3,h,w), [0,1]

        # class_id (faces/text) — for DIV2K we do single domain (no class prior)
        class_id = None
        if model.use_class_prior and getattr(model, "num_classes", None) == 1:
            class_id = torch.zeros(1, dtype=torch.long, device=device)

        with torch.no_grad():
            sr = ddpm_sample_x0param(model, G, lr, steps=model.num_steps, class_id=class_id)

        psnrs.append(psnr_y(sr[0], hr[0], crop=4))
        ssims.append(ssim_y(sr[0], hr[0], crop=4))

        if args.save_examples and idx < 10:
            to_uint8_img(sr[0]).save(os.path.join(save_dir, f"sr_{idx:03d}.png"))

    print(f"DIV2K x4  |  PSNR-Y: {np.mean(psnrs):.3f} dB  |  SSIM-Y: {np.mean(ssims):.4f}")
    print(f"Count: {len(psnrs)}  Examples saved to: {save_dir}")

if __name__ == "__main__":
    # seed for deterministic eval
    import random
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    main()
