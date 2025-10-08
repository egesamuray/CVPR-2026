# codes/scripts/sr3_sample.py
import os, glob, argparse
import torch
import numpy as np
from PIL import Image
from options.options import parse
from models import create_model

@torch.no_grad()
def p_sample(model, x_t, lr, t, alphas, alpha_bars, betas, G):
    # predict x0 and sample x_{t-1}
    x0_hat = G(x_t, lr, t.float(), class_id=None)
    ab = alpha_bars[t.long()].view(-1,1,1,1)
    sig = (1 - ab).sqrt()
    eps_hat = (x_t - ab.sqrt()*x0_hat) / sig.clamp_min(1e-8)
    a = alphas[t.long()].sqrt().view(-1,1,1,1)
    b = betas[t.long()].view(-1,1,1,1)
    mean = (1.0/a) * (x_t - b / sig.clamp_min(1e-8) * eps_hat)
    z = torch.randn_like(x_t) if (t.item() > 0) else torch.zeros_like(x_t)
    return mean + b.sqrt()*z

@torch.no_grad()
def sample_chain(model, lr, steps=None, save_every=None):
    device = model.device
    betas = model.beta_schedule
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    B, _, h, w = lr.shape
    H, W = h*model.scale, w*model.scale
    x_t = torch.randn(B, 3, H, W, device=device)
    G = model.netG_ema if getattr(model, 'use_ema', False) else model.netG
    G.eval()
    seq = range(len(betas)) if steps is None else range(steps)
    for i in reversed(seq):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        x_t = p_sample(model, x_t, lr, t, alphas, alpha_bars, betas, G)
    return x_t.clamp(0,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opt', required=True)
    ap.add_argument('--lr_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    opt = parse(args.opt, is_train=False)
    model = create_model(opt)
    # load latest G_*.pth
    models_dir = os.path.join(opt['path']['experiments_root'], 'models')
    ckpts = sorted(glob.glob(os.path.join(models_dir, "G_*.pth")))
    assert ckpts, "No G_*.pth found"
    model.load_network(ckpts[-1], model.netG, strict=True)
    model.netG.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    lr_paths = sorted(glob.glob(os.path.join(args.lr_dir, "*.png")))
    for p in lr_paths:
        lr = np.array(Image.open(p).convert('RGB'))
        lr = torch.from_numpy(lr).permute(2,0,1).unsqueeze(0).float().div(255.).to(model.device)
        sr = sample_chain(model, lr)[0].permute(1,2,0).cpu().numpy()
        Image.fromarray((sr*255).round().astype(np.uint8)).save(os.path.join(args.out_dir, os.path.basename(p)))

if __name__ == "__main__":
    main()
