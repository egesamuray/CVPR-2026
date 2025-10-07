# codes/scripts/sr3_sample.py
import torch
from utils import img_write  # whatever you use
from models import create_model
from options.options import parse

@torch.no_grad()
def p_sample(model, x_t, lr, t, alpha, alpha_bar, beta):
    # predict x0 and sample x_{t-1}
    x0_hat = model.netG(x_t, lr, t.float(), class_id=None)  # or pass class_id
    eps_hat = (x_t - alpha_bar[t].sqrt().view(-1,1,1,1) * x0_hat) / (1-alpha_bar[t]).sqrt().view(-1,1,1,1)
    mean = (1.0/alpha[t].sqrt()).view(-1,1,1,1) * (x_t - beta[t].view(-1,1,1,1) / (1-alpha_bar[t]).sqrt().view(-1,1,1,1) * eps_hat)
    if t.item() == 0:
        return mean
    z = torch.randn_like(x_t)
    return mean + beta[t].sqrt().view(-1,1,1,1) * z

def sample_chain(model, lr, steps, betas):
    device = lr.device
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    B, C, H, W = lr.shape[0], 3, lr.shape[2]*4, lr.shape[3]*4  # scale=4
    x_t = torch.randn(B, C, H, W, device=device)
    for i in reversed(range(steps)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        x_t = p_sample(model, x_t, lr, t, alphas, alpha_bar, betas)
    return x_t.clamp(0,1)

# usage:
# opt = parse() ; model = create_model(opt) ; model.load_network(...)
# prepare LR batch ; run sample_chain(model, LR, steps=1000, betas=model.beta_schedule)
