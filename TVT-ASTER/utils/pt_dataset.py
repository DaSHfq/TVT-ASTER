# -*- coding: utf-8 -*-
import os
import glob
import math
import random
from typing import List, Iterable, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from .augment import VideoAugmentor


class PTVideoDataset(Dataset):
    """Read *.pt packed samples: {'video': FloatTensor(3,T,224,224), 'label': int}."""
    def __init__(self, root_dir: str, split: str = "train", augmentor: Optional[VideoAugmentor] = None):
        assert split in {"train", "val", "test"}
        self.split = split
        self.paths = sorted(glob.glob(os.path.join(root_dir, split, "*.pt")))
        if not self.paths:
            raise FileNotFoundError(f"No .pt files in {root_dir}/{split}")
        self.augmentor = augmentor if split == "train" else None

        self.labels: List[int] = []
        for p in self.paths:
            try:
                self.labels.append(int(torch.load(p, map_location="cpu")["label"]))
            except Exception:
                self.labels.append(0)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.paths[idx]
        try:
            pkg = torch.load(p, map_location="cpu")
            video = pkg["video"]
            label = int(pkg["label"])
        except Exception:
            video = torch.zeros(3, 32, 224, 224, dtype=torch.float32)
            label = 0
        if self.augmentor is not None:
            video = self.augmentor(video)
        return video, torch.tensor(label, dtype=torch.long)


class BalancedBatchSampler(Sampler[List[int]]):
    """Binary balanced batches (0/1 half-half). batch_size must be even."""
    def __init__(self, labels: List[int], batch_size: int, seed: int = 42):
        assert batch_size % 2 == 0
        self.labels = labels
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.seed = seed
        self.epoch = 0

        self.pos = [i for i, y in enumerate(labels) if y == 1]
        self.neg = [i for i, y in enumerate(labels) if y == 0]
        if not self.pos or not self.neg:
            raise ValueError("Both classes required for BalancedBatchSampler.")
        self.num_batches = math.floor(min(len(self.pos), len(self.neg)) / self.half)

    def set_epoch(self, epoch: int): self.epoch = int(epoch)

    def __iter__(self) -> Iterable[List[int]]:
        rng = random.Random(self.seed ^ self.epoch)
        pos, neg = self.pos[:], self.neg[:]
        rng.shuffle(pos); rng.shuffle(neg)
        p_ptr = n_ptr = 0
        for _ in range(self.num_batches):
            if p_ptr + self.half > len(pos): rng.shuffle(pos); p_ptr = 0
            if n_ptr + self.half > len(neg): rng.shuffle(neg); n_ptr = 0
            batch = pos[p_ptr:p_ptr+self.half] + neg[n_ptr:n_ptr+self.half]
            rng.shuffle(batch); p_ptr += self.half; n_ptr += self.half
            yield batch

    def __len__(self): return self.num_batches


def make_loader(root_dir: str, split: str, batch_size: int, num_workers: int = 4, seed: int = 42):
    aug = VideoAugmentor(p_hflip=0.5, p_pix=0.5, p_drop=0.25,
                         noise_std=0.005, max_drop_ratio=0.05, max_roll=2) if split == "train" else None
    ds = PTVideoDataset(root_dir, split, augmentor=aug)

    if split == "train":
        sampler = BalancedBatchSampler(ds.labels, batch_size=batch_size, seed=seed)
        dl = DataLoader(ds, batch_sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True, drop_last=False)
    return dl, ds
