# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .ema import EMA
from .metrics import metrics_from_probs


def train_one_epoch(
    vit, ets, head, loader, optimizer, scaler, device: str,
    clip_norm: float = 1.0, logger=None, ema: Optional[EMA] = None
) -> Dict[str, float]:
    vit.train(); ets.train(); head.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    loss_sum = acc_sum = steps = 0

    pbar = tqdm(loader, desc="[Train]", ncols=100)
    for video, label in pbar:
        video = video.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            feat = ets(vit(video))
            logits = head(feat)
            loss = criterion(logits, label)

        scaler.scale(loss).backward()
        if clip_norm and clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(vit.parameters(),  clip_norm)
            torch.nn.utils.clip_grad_norm_(ets.parameters(),  clip_norm)
            torch.nn.utils.clip_grad_norm_(head.parameters(), clip_norm)

        scaler.step(optimizer); scaler.update()
        if ema is not None: ema.update()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == label).float().mean()

        steps += 1; loss_sum += loss.item(); acc_sum += acc.item()
        pbar.set_postfix({"loss": f"{loss_sum/steps:.4f}", "acc": f"{(acc_sum/steps)*100:5.2f}%"})

    stats = {"loss": loss_sum / max(steps, 1), "acc": acc_sum / max(steps, 1)}
    if logger is not None:
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        logger.info(f"[Train][epoch] loss={stats['loss']:.4f} | acc={stats['acc']*100:.2f}% | lrs={lrs}")
    return stats


@torch.no_grad()
def evaluate(
    vit, ets, head, loader, device: str, desc: str = "[Eval]", logger=None,
    ema: Optional[EMA] = None, threshold: float = 0.5,
    return_probs_targets: bool = False,
    tta: Optional[Dict] = None
):
    if ema is not None: ema.swap_to_ema()
    vit.eval(); ets.eval(); head.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    loss_sum = steps = 0

    use_tta = tta is not None
    hflip = bool(tta.get("hflip", True)) if use_tta else False
    time_rolls = tta.get("time_rolls", [0]) if use_tta else [0]

    def forward_probs(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            p = torch.softmax(head(ets(vit(v))), dim=1)[:, 1]
            lg = torch.logit(p.clamp(1e-6, 1 - 1e-6))
        return p, lg

    pbar = tqdm(loader, desc=desc, ncols=100)
    for video, label in pbar:
        video = video.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True).long()

        p_base, logits_base = forward_probs(video)
        with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            loss = criterion(torch.stack([1-p_base, p_base], dim=1), label)

        if use_tta:
            probs_list = []
            for r in time_rolls:
                v = video if r == 0 else video.roll(shifts=r, dims=2)
                p_r, _ = forward_probs(v); probs_list.append(p_r)
                if hflip:
                    p_h, _ = forward_probs(torch.flip(v, dims=[-1])); probs_list.append(p_h)
            probs = torch.stack(probs_list, dim=0).mean(dim=0)
        else:
            probs = p_base

        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(label.detach().cpu().numpy())

        steps += 1; loss_sum += loss.item()
        acc = (probs >= 0.5).long().eq(label).float().mean().item()
        pbar.set_postfix({"loss": f"{loss_sum/steps:.4f}", "acc@0.5": f"{acc*100:5.2f}%"})

    probs_all = np.concatenate(all_probs, axis=0)
    targets_all = np.concatenate(all_targets, axis=0).astype(np.int64)

    metr = metrics_from_probs(probs_all, targets_all, threshold)
    metr["loss"] = loss_sum / max(steps, 1)

    if logger is not None:
        tag = "TTA" if use_tta else "plain"
        logger.info(f"{desc} ({tag}, τ={threshold:.2f}) "
                    f"loss={metr['loss']:.4f} | acc={metr['acc']*100:.2f}% | "
                    f"prec={metr['prec']*100:.2f}% | rec={metr['rec']*100:.2f}% | f1={metr['f1']*100:.2f}% | "
                    f"TP={metr['tp']} FP={metr['fp']} FN={metr['fn']} TN={metr['tn']} | N={metr['n']}")

    if ema is not None: ema.swap_back()
    if return_probs_targets:
        return metr, probs_all, targets_all
    return metr
