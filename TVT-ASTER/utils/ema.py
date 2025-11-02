# -*- coding: utf-8 -*-
from typing import List, Dict
import torch
import torch.nn as nn

class EMA:
    """EMA for a list of modules; swap in/out for eval."""
    def __init__(self, modules: List[nn.Module], decay: float = 0.999):
        self.modules = modules
        self.decay = decay
        self.ema_states = [{k: v.clone().detach() for k, v in m.state_dict().items()} for m in modules]
        self.backups = None

    @torch.no_grad()
    def update(self):
        for m, ema in zip(self.modules, self.ema_states):
            for k, v in m.state_dict().items():
                ema[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def swap_to_ema(self):
        self.backups = [{k: v.clone().detach() for k, v in m.state_dict().items()} for m in self.modules]
        for m, ema in zip(self.modules, self.ema_states):
            m.load_state_dict(ema, strict=True)

    @torch.no_grad()
    def swap_back(self):
        if self.backups is None: return
        for m, backup in zip(self.modules, self.backups):
            m.load_state_dict(backup, strict=True)
        self.backups = None

    def get_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        keys = ["vit", "ets", "head"]
        return {k: s for k, s in zip(keys, self.ema_states)}
