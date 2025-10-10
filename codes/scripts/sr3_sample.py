# codes/scripts/sr3_sample.py
import os, sys, re, glob, json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F

# repo imports
sys.path.insert(0, os.path.join("/content/CVPR-2026", "codes"))  # adjust if needed
import options.options as option
from models import create_model

def pick_latest_checkpoint(models_dir: str) -> str:
    finals = ["G_final.pth", "final_G.pth"]
    for fn in finals:
        p = os.path.join(models_dir, fn)
        if os.path.exists(p): return p
    cand = []
    for p in glob.glob(os.path.join(models_dir, "G_*.pth")):
        m = re.search(r"G_(\d+)\.pth$", os.path.basename(p))
        if m: cand.append((int(m.group(1)), p))
    for p in glob.glob(os.path.join(models_dir, "*_G.pth")):
        m = re.search(r"(\d+)_G\.pth$", os.path.basename(p))
        if m: cand.append((int(m.group(1)), p))
    if not cand:
        raise FileNotFoundError(f"No checkpoints in {models_dir}")
    cand.sort(key=lambda x: x[0])
    return cand[-1][1]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--opt', required=True)
    ap.add_argument('--lr_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    opt = option.parse(args.opt, is_train=False)
    exp_root  = opt.get('path', {}).get('experiments_root') or os.path.join("/content/CVPR-2026", "experiments", opt['name'])
    opt.setdefault('path', {})['experiments_root'] = exp_root
    models_dir = opt['path'].get('models') or os.path.join(exp_root, "models")

    model = create_model(opt)
    ckpt_path = pick_latest_checkpoint(models_dir)
    print("Loading checkpoint:", ckpt_path)
    model.load_network(ckpt_path, model.netG, strict=True)

    G = model.netG
    if hasattr(model, 'netG_ema'):
        try:
            model.netG_ema.load_state_dict(model.netG.state_dict(), strict=False)
            G = model.netG_ema
            print("Using EMA for sampling.")
        except Exception:
            G = model.netG
    G.eval()
    device = model.device

    betas = model.beta_schedule.to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    scale = int(opt.get('scale', 8))

    @torch.no_grad()
    def p_sample(x_t, lr, t):
        # faces-only => class_id = 0
        class_id = torch.zeros(x_t.size(0), device=device, dtype=torch.long)
        x0_hat = G(x_t, lr, t.float(), class_id=class_id)
        ab = alpha_bars[t.long()].view(-1,1,1,1)
        sig = (1 - ab).sqrt()
        eps_hat = (x_t - ab.sqrt()*x0_hat) / sig.clamp_min(1e-8)
        a = alphas[t.long()].sqrt().view(-1,1,1,1)
        b = betas[t.long()].view(-1,1,1,1)
        mean = (1.0/a) * (x_t - b / sig.clamp_min(1e-8) * eps_hat)
        z = torch.randn_like(x_t) if (t.item() > 0) else torch.zeros_like(x_t)
        return mean + b.sqrt()*z

    @torch.no_grad()
    def sample_chain(lr_img_u8):
        lr = torch.from_numpy(lr_img_u8).permute(2,0,1).float().unsqueeze(0).div(255.0).to(device)
        B, _, h, w = lr.shape
        H, W = h * scale, w * scale
        x_t = torch.randn(B, 3, H, W, device=device)
        for i in reversed(range(len(betas))):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            x_t = p_sample(x_t, lr, t)
        sr = x_t.clamp(0,1)[0].permute(1,2,0).cpu().numpy()
        return (sr*255.0 + 0.5).astype(np.uint8)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    lr_paths = sorted(glob.glob(os.path.join(args.lr_dir, "*.png")))[:100]
    for p in tqdm(lr_paths, desc="Sampling"):
        lr = np.array(Image.open(p).convert("RGB"))
        sr = sample_chain(lr)
        Image.fromarray(sr).save(os.path.join(args.out_dir, os.path.basename(p)))
    print("Saved SR ->", args.out_dir)

if __name__ == "__main__":
    main()
