# -*- coding: utf-8 -*-
import random
import torch

class VideoAugmentor:
    """Augment (3, T, H, W): hflip / gaussian noise / light frame-drop / small time roll."""
    def __init__(self, p_hflip=0.5, p_pix=0.5, p_drop=0.25,
                 noise_std=0.005, max_drop_ratio=0.05, max_roll=2):
        self.p_hflip = p_hflip
        self.p_pix = p_pix
        self.p_drop = p_drop
        self.noise_std = noise_std
        self.max_drop_ratio = max_drop_ratio
        self.max_roll = max_roll

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p_hflip:
            x = torch.flip(x, dims=[-1])
        if random.random() < self.p_pix and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        if random.random() < self.p_drop:
            x = self._drop_and_pad(x)
        if self.max_roll > 0:
            shift = random.randint(-self.max_roll, self.max_roll)
            if shift != 0:
                x = x.roll(shifts=shift, dims=1)
        return x

    def _drop_and_pad(self, x: torch.Tensor) -> torch.Tensor:
        C, T, H, W = x.shape
        if T <= 1: return x
        k = max(1, int(T * self.max_drop_ratio))
        drop_k = random.randint(1, k)
        keep_idx = sorted(random.sample(range(T), k=T - drop_k))
        kept = x[:, keep_idx]
        pad = kept[:, -1:].repeat(1, drop_k, 1, 1)
        return torch.cat([kept, pad], dim=1)
