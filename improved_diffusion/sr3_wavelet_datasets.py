# improved_diffusion/sr3_wavelet_datasets.py
import os, glob, math, random
from typing import Optional, Tuple, Dict, Iterable, List
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import pywt  # wavelet transforms
from .wavelet_datasets import wavelet_stats  # reuse your stats cache format  # noqa
# You already ship wavelet stats/utilities for CIFAR/CelebA and others.  :contentReference[oaicite:7]{index=7}

# -------------------------
# Helpers: DWT / pack HF
# -------------------------
def _to_tensor_rgb(im: Image.Image, size: Optional[int]) -> torch.Tensor:
    if size is not None:
        im = im.resize((size, size), Image.BICUBIC)
    ar = torch.from_numpy(np.asarray(im.convert("RGB"))).float() / 127.5 - 1.0  # [-1,1]
    return ar.permute(2, 0, 1)  # C,H,W

def dwt_level1_rgb(x: torch.Tensor, wavelet: str = "haar") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (3,H,W) in [-1,1]
    returns:
        LL: (3,H/2,W/2)
        HF: (9,H/2,W/2) stacking LH, HL, HH for each color
    """
    C, H, W = x.shape
    assert C == 3
    LLs, HFs = [], []
    for c in range(C):
        arr = x[c].cpu().numpy()
        (LL, (LH, HL, HH)) = pywt.dwt2(arr, wavelet=wavelet, mode="periodization")
        LLs.append(torch.from_numpy(LL).float())
        HFs.extend([torch.from_numpy(LH).float(),
                    torch.from_numpy(HL).float(),
                    torch.from_numpy(HH).float()])
    LL = torch.stack(LLs, dim=0)
    HF = torch.stack(HFs, dim=0)  # 9 x H/2 x W/2
    return LL, HF

def idwt_level1_rgb(LL: torch.Tensor, HF: torch.Tensor, wavelet: str = "haar") -> torch.Tensor:
    """
    LL: (3,h,w)
    HF: (9,h,w) as [LH_R, HL_R, HH_R, LH_G, HL_G, HH_G, LH_B, HL_B, HH_B]
    returns RGB (3, 2h, 2w) in [-1,1] (clamped)
    """
    outs = []
    for c in range(3):
        LH, HL, HH = HF[3*c+0].cpu().numpy(), HF[3*c+1].cpu().numpy(), HF[3*c+2].cpu().numpy()
        rec = pywt.idwt2((LL[c].cpu().numpy(), (LH, HL, HH)), wavelet=wavelet, mode="periodization")
        outs.append(torch.from_numpy(rec).float())
    out = torch.stack(outs, dim=0)
    return out.clamp_(-1, 1)

# -------------------------
# Dataset
# -------------------------
class SR3WaveletDataset(Dataset):
    """
    Produces:
        HF_target_whitened: (9,h,w)     - training target in HF space (whitened)
        KW={"conditioning": LR_up_rgb}: (3,h,w)  - SR3 conditioning
        Optional label y in {text, face} for class-conditional runs (0=text/number, 1=face)
    """
    def __init__(
        self,
        root: str,
        large_size: int = 256,
        scale: int = 4,                 # SR factor; level-1 DWT maps natural 2x; we still use LR↑ at large_size
        wavelet: str = "haar",
        stats_dir_for_whitening: Optional[str] = None,  # directory hint to reuse your wavelet_stats()
        class_cond: bool = False,
    ):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.large_size = large_size
        self.scale = scale
        self.wavelet = wavelet
        self.class_cond = class_cond

        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        self.paths: List[str] = []
        for ext in exts:
            self.paths.extend(glob.glob(os.path.join(self.root, "**", f"*{ext}"), recursive=True))
        if not self.paths:
            raise FileNotFoundError(f"No images found under {self.root}")

        # transforms
        self.to_hr = T.Compose([
            T.Resize((large_size, large_size), interpolation=T.InterpolationMode.BICUBIC),
        ])

        # Whitening stats for (LL, HF) → reuse your stats format: first 3 entries are LL (coarse), last 9 are HF
        # If no directory hint is provided, just compute per-batch on the fly (fallback).
        self.mean12 = None
        self.std12 = None
        if stats_dir_for_whitening is not None:
            # we piggyback on your cached stats logic
            # choose j=1 (LL + 3 HF bands), consistent with 12-channel convention in your wavelet code.  :contentReference[oaicite:8]{index=8}
            try:
                m, s = wavelet_stats(j=1, dir_name=stats_dir_for_whitening)
                self.mean12 = m.float()  # (12,)
                self.std12 = s.float().clamp_min(1e-6)
            except Exception:
                self.mean12 = None
                self.std12 = None

    def __len__(self): return len(self.paths)

    def _class_label(self, path: str) -> int:
        # Simple heuristic: folder name contains "text"/"number" vs "face"
        p = path.lower()
        if ("text" in p) or ("number" in p) or ("ocr" in p):
            return 0
        return 1  # face as default

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        im = Image.open(p).convert("RGB")
        hr = _to_tensor_rgb(im, self.large_size)       # (3,H,W) in [-1,1]

        # build LR (bicubic) and upsample back to HR for conditioning (classic SR3)
        small = self.large_size // self.scale
        lr = F.interpolate(hr.unsqueeze(0), size=(small, small), mode="bicubic", align_corners=False).squeeze(0)
        lr_up = F.interpolate(lr.unsqueeze(0), size=(self.large_size, self.large_size), mode="bicubic", align_corners=False).squeeze(0)

        # DWT level-1 on HR → (LL, HF) at (H/2, W/2)
        LL, HF = dwt_level1_rgb(hr, wavelet=self.wavelet)  # (3,h,w), (9,h,w)

        # Whitening in the classic 12-channel order: [LL(R,G,B), HF(9)]
        if (self.mean12 is not None) and (self.std12 is not None):
            LLw = (LL - self.mean12[:3].view(3,1,1)) / self.std12[:3].view(3,1,1)
            HFw = (HF - self.mean12[3:].view(9,1,1)) / self.std12[3:].view(9,1,1)
        else:
            # fallback: per-image moments
            LLw = (LL - LL.mean(dim=(1,2), keepdim=True)) / (LL.std(dim=(1,2), keepdim=True).clamp_min(1e-6))
            HFw = (HF - HF.mean(dim=(1,2), keepdim=True)) / (HF.std(dim=(1,2), keepdim=True).clamp_min(1e-6))

        KW: Dict[str, torch.Tensor] = {"conditioning": F.interpolate(lr_up.unsqueeze(0), size=LLw.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)}
        if self.class_cond:
            KW["y"] = torch.tensor(self._class_label(p), dtype=torch.long)
        return HFw, KW  # HF whitened is the diffusion target; SR3 will predict it given KW.conditioning

def load_data_sr3_wavelet(
    *,
    data_dir: str,
    batch_size: int,
    large_size: int = 256,
    scale: int = 4,
    wavelet: str = "haar",
    class_cond: bool = False,
    stats_dir_for_whitening: Optional[str] = None,
    deterministic: bool = False,
    num_workers: int = 4,
):
    ds = SR3WaveletDataset(
        data_dir, large_size=large_size, scale=scale, wavelet=wavelet,
        class_cond=class_cond, stats_dir_for_whitening=stats_dir_for_whitening
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=not deterministic, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )
    while True:
        for X, KW in loader:
            yield X, {k: v for k, v in KW.items()}
