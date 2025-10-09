# codes/data/LRHR_dataset.py
import os
import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util


class LRHRDataset(data.Dataset):
    """
    LR/HR paired dataset with optional on-the-fly LR generation.

    This revision is robust to missing keys in the dataset options:
      - Uses opt.get('subset_file'), opt.get('phase'), opt.get('color'), etc.
      - Emits class_id per sample (0=text/number, 1=face) for wavelet class-prior.
      - Forces 3-channel images; handles grayscale inputs.
      - In val/test, if LR size mismatches HR/scale, LR is regenerated from HR.
      - Fixes color-conversion bug by passing the correct channels to channel_convert.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.paths_LR = None
        self.paths_HR = None
        self.LR_env = None
        self.HR_env = None

        phase = self.opt.get('phase', 'train')
        subset_file = self.opt.get('subset_file', None)
        dataroot_HR = self.opt.get('dataroot_HR', None)
        dataroot_LR = self.opt.get('dataroot_LR', None)
        data_type = self.opt.get('data_type', 'img')

        # read image list from subset list txt
        if subset_file is not None and phase == 'train':
            with open(subset_file) as f:
                self.paths_HR = sorted([os.path.join(dataroot_HR, line.rstrip('\n'))
                                        for line in f])
            if dataroot_LR is not None:
                raise NotImplementedError('subset only supports generating LR on-the-fly.')
        else:  # read image list from lmdb or image files
            self.HR_env, self.paths_HR = util.get_image_paths(data_type, dataroot_HR)
            self.LR_env, self.paths_LR = util.get_image_paths(data_type, dataroot_LR)

        assert self.paths_HR, 'Error: HR path is empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                f'HR and LR datasets have different number of images - {len(self.paths_LR)}, {len(self.paths_HR)}.'

        self.random_scale_list = [1]
        self.scale = int(self.opt.get('scale', 4))
        self.HR_size = self.opt.get('HR_size', None)

    def __len__(self):
        return len(self.paths_HR)

    # ---------- helpers ----------

    @staticmethod
    def _ensure_3ch(img):
        """Ensure HWC 3-channel image."""
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        return img

    @staticmethod
    def _infer_class_id(hr_path, lr_path):
        """
        0 = text/number, 1 = face; default 0.
        Lightweight heuristic; model also sanitizes when num_classes==1.
        """
        p = (hr_path or lr_path or "").lower()
        if ("face" in p) or ("ffhq" in p) or ("celeba" in p): return 1
        if ("text" in p) or ("number" in p) or ("word" in p) or ("textzoom" in p): return 0
        return 0

    # ---------- main ----------

    def __getitem__(self, index):
        phase = self.opt.get('phase', 'train')
        color = self.opt.get('color', None)
        use_flip = bool(self.opt.get('use_flip', False))
        use_rot = bool(self.opt.get('use_rot', False))

        HR_path = self.paths_HR[index]
        LR_path = self.paths_LR[index] if self.paths_LR else None

        # ----- HR -----
        img_HR = util.read_img(self.HR_env, HR_path)
        img_HR = self._ensure_3ch(img_HR)

        # modcrop in val/test
        if phase != 'train':
            img_HR = util.modcrop(img_HR, self.scale)

        # change color space for HR (if requested)
        if color:
            inC = img_HR.shape[2]
            img_HR = util.channel_convert(inC, color, [img_HR])[0]

        # ----- LR -----
        if LR_path:
            img_LR = util.read_img(self.LR_env, LR_path)
            img_LR = self._ensure_3ch(img_LR)

            # In val/test ensure LR size matches HR/scale; if not, regenerate LR from HR
            if phase != 'train':
                HrH, HrW = img_HR.shape[:2]
                exp = (HrH // self.scale, HrW // self.scale)
                if img_LR.shape[0] != exp[0] or img_LR.shape[1] != exp[1]:
                    img_LR = util.imresize_np(img_HR, 1 / self.scale, True)
                    if img_LR.ndim == 2:
                        img_LR = np.expand_dims(img_LR, axis=2)
                    img_LR = self._ensure_3ch(img_LR)
        else:
            # Generate LR on-the-fly from HR
            if phase == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_HR.shape

                def _mod(n, r, sc, th):
                    rlt = int(n * r)
                    rlt = (rlt // sc) * sc
                    return th if (self.HR_size and rlt < th) else rlt

                if self.HR_size:
                    H_s = _mod(H_s, random_scale, self.scale, self.HR_size)
                    W_s = _mod(W_s, random_scale, self.scale, self.HR_size)
                    img_HR = cv2.resize(np.copy(img_HR), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                    img_HR = self._ensure_3ch(img_HR)

            img_LR = util.imresize_np(img_HR, 1 / self.scale, True)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)
            img_LR = self._ensure_3ch(img_LR)

        # ----- Training crops & aug -----
        if phase == 'train' and self.HR_size:
            H, W, _ = img_HR.shape
            if H < self.HR_size or W < self.HR_size:
                # upscale HR to HR_size and regenerate LR
                img_HR = cv2.resize(np.copy(img_HR), (self.HR_size, self.HR_size), interpolation=cv2.INTER_LINEAR)
                img_HR = self._ensure_3ch(img_HR)
                img_LR = util.imresize_np(img_HR, 1 / self.scale, True)
                if img_LR.ndim == 2:
                    img_LR = np.expand_dims(img_LR, axis=2)
                img_LR = self._ensure_3ch(img_LR)

            # paired random crop
            H, W, _ = img_LR.shape
            LR_size = self.HR_size // self.scale
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            img_HR = img_HR[rnd_h*self.scale:rnd_h*self.scale + self.HR_size,
                            rnd_w*self.scale:rnd_w*self.scale + self.HR_size, :]

            # augmentation
            img_LR, img_HR = util.augment([img_LR, img_HR], use_flip, use_rot)

        # optional color conversion for LR (match HR)
        if color:
            inC_lr = img_LR.shape[2]
            img_LR = util.channel_convert(inC_lr, color, [img_LR])[0]

        # BGR->RGB (if util.read_img returns BGR), HWC->CHW, to float32 tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        # default LR_path if missing
        if LR_path is None:
            LR_path = HR_path

        # attach class_id
        class_id = self._infer_class_id(HR_path, LR_path)

        return {
            'LR': img_LR,
            'HR': img_HR,
            'LR_path': LR_path,
            'HR_path': HR_path,
            'class_id': class_id
        }
