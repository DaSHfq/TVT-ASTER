# -*- coding: utf-8 -*-
import os
import math
import logging
import argparse
import torch
import torch.backends.cudnn as cudnn

from utils.logger import get_logger
from utils.pt_dataset import make_loader
from utils.ema import EMA
from utils.metrics import find_best_threshold, metrics_from_probs
from utils.engine import train_one_epoch, evaluate

from models.vit import ViTBackbone
from models.etsformer import ETSConfigs, ETSFormerAdapter
from models.bilstm import BiLSTMHead


def set_seed(seed: int):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_models(device: str, weights_path: str):
    vit = ViTBackbone(
        pretrained=False, freeze_blocks=6, d_model=512, use_ln=True,
        weights_path=weights_path, frames_per_gpu_batch=64
    ).to(device)
    ets = ETSFormerAdapter(ETSConfigs()).to(device)
    head = BiLSTMHead(
        input_dim=512, hidden=256, num_layers=3, dropout=0.5,
        num_classes=2, pooling='mean_last', ln_input=False, ln_output=False
    ).to(device)
    return vit, ets, head


def build_optimizer_scheduler(vit, ets, head, epochs, args):
    vit_params  = [p for p in vit.parameters()  if p.requires_grad]
    ets_params  = [p for p in ets.parameters()  if p.requires_grad]
    head_params = [p for p in head.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": vit_params,  "lr": args.lr_vit},
            {"params": ets_params,  "lr": args.lr_head_ets},
            {"params": head_params, "lr": args.lr_head_ets},
        ],
        weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )

    def lr_lambda(epoch_idx: int):
        if epoch_idx < args.warmup_epochs:
            return float(epoch_idx + 1) / float(max(1, args.warmup_epochs))
        progress = float(epoch_idx - args.warmup_epochs) / float(max(1, epochs - args.warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_factor + (1.0 - args.min_lr_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def save_best_ckpt(path, epoch, best_threshold, val_star, val_05, vit, ets, head, ema, optimizer, args):
    obj = {
        "epoch": epoch,
        "best_threshold": best_threshold,
        "val_metrics_tau_star": val_star,
        "val_metrics_tau_05": val_05,
        "model_state_dict": {
            "vit": vit.state_dict(), "ets": ets.state_dict(), "head": head.state_dict()
        },
        "ema_state_dict": ema.get_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
    }
    torch.save(obj, path)


def load_best_ckpt(path, device, vit, ets, head, logger: logging.Logger):
    best_t = 0.5
    if not os.path.exists(path):
        logger.info(f"[TEST] No ckpt at {path}, use current EMA + τ*={best_t:.2f}")
        return best_t

    ckpt = torch.load(path, map_location=device)
    if "ema_state_dict" in ckpt or "vit" in ckpt.get("ema_state_dict", {}):
        sd = ckpt["ema_state_dict"]
        vit.load_state_dict(sd["vit"], strict=True)
        ets.load_state_dict(sd["ets"], strict=True)
        head.load_state_dict(sd["head"], strict=True)
        logger.info("[TEST] Loaded EMA weights.")
    else:
        vit.load_state_dict(ckpt["model_state_dict"]["vit"], strict=True)
        ets.load_state_dict(ckpt["model_state_dict"]["ets"], strict=True)
        head.load_state_dict(ckpt["model_state_dict"]["head"], strict=True)
        logger.info("[TEST] Loaded raw weights.")

    best_t = float(ckpt.get("best_threshold", best_t))
    logger.info(f"[TEST] τ* from ckpt: {best_t:.2f}")
    return best_t


def parse_args():
    p = argparse.ArgumentParser("ViT → ETSFormer → BiLSTM training")
    p.add_argument("--data_root", default=os.path.join("data", "pt-HockeyFight"))
    p.add_argument("--weights_path", default=os.path.join("weights", "vit_b_16_imagenet1k_v1.pth"))
    p.add_argument("--ckpt_dir", default="ckpts")
    p.add_argument("--logfile", default="train.log")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--lr_head_ets", type=float, default=5e-4)
    p.add_argument("--lr_vit", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_norm", type=float, default=1.0)

    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--min_lr_factor", type=float, default=0.01)

    # TTA
    p.add_argument("--tta_hflip", action="store_true", help="Enable test-time horizontal flip")
    p.add_argument("--tta_time_shifts", type=int, nargs="*", default=[0], help="e.g. 0 1 -1")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    set_seed(args.seed)
    cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    logger = get_logger(args.logfile)
    logger.info(f"[Device] {args.device} | [Seed] {args.seed} | [Batch] {args.batch_size} | [Epochs] {args.epochs}")

    # Data
    train_loader, train_ds = make_loader(args.data_root, "train", args.batch_size, args.num_workers, seed=args.seed)
    val_loader,   val_ds   = make_loader(args.data_root, "val",   args.batch_size, args.num_workers, seed=args.seed)
    test_loader,  test_ds  = make_loader(args.data_root, "test",  args.batch_size, args.num_workers, seed=args.seed)
    logger.info(f"[Data] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")

    # Models
    vit, ets, head = build_models(args.device, args.weights_path)

    # Optimizers / schedulers / AMP / EMA
    optimizer, scheduler = build_optimizer_scheduler(vit, ets, head, args.epochs, args)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device.startswith("cuda")))
    ema = EMA([vit, ets, head], decay=0.999)

    best_val_f1, best_t = -1.0, 0.5
    best_path = os.path.join(args.ckpt_dir, "best.pt")

    # Train
    for epoch in range(1, args.epochs + 1):
        logger.info(f"========== Epoch {epoch}/{args.epochs} ==========")
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        _ = train_one_epoch(
            vit, ets, head, train_loader, optimizer, scaler, args.device,
            clip_norm=args.clip_norm, logger=logger, ema=ema
        )

        # Val@0.5 & collect probs
        stats_05, probs, targets = evaluate(
            vit, ets, head, val_loader, args.device, desc="[Val@0.5]",
            logger=logger, ema=ema, threshold=0.5, return_probs_targets=True, tta=None
        )
        # grid search τ*
        t_star, f1_star = find_best_threshold(probs, targets, t_min=0.30, t_max=0.70, num=41)
        stats_star = metrics_from_probs(probs, targets, t_star)
        logger.info(f"[Val@τ*] τ*={t_star:.2f} | f1={stats_star['f1']*100:.2f}% | "
                    f"acc={stats_star['acc']*100:.2f}% | prec={stats_star['prec']*100:.2f}% | rec={stats_star['rec']*100:.2f}%")

        if stats_star["f1"] > best_val_f1:
            best_val_f1, best_t = stats_star["f1"], float(t_star)
            save_best_ckpt(best_path, epoch, best_t, stats_star, stats_05, vit, ets, head, ema, optimizer, args)
            logger.info(f"[CKPT] Saved best to {best_path} (F1@τ*={best_val_f1*100:.2f}%, τ*={best_t:.2f})")

        scheduler.step()
        lrs = [pg["lr"] for pg in optimizer.param_groups]
        logger.info(f"[LR] {lrs}")

    # Test
    logger.info("========== Test ==========")
    best_t = load_best_ckpt(best_path, args.device, vit, ets, head, logger)

    _ = evaluate(
        vit, ets, head, test_loader, args.device,
        desc=f"[Test@τ*={best_t:.2f}]", logger=logger, ema=None,
        threshold=best_t, return_probs_targets=False, tta=None
    )

    tta_cfg = {"hflip": args.tta_hflip, "time_rolls": args.tta_time_shifts}
    _ = evaluate(
        vit, ets, head, test_loader, args.device,
        desc=f"[Test-TTA@τ*={best_t:.2f}]", logger=logger, ema=None,
        threshold=best_t, return_probs_targets=False, tta=tta_cfg
    )

    # Shape/params sanity
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(args.device.startswith("cuda"))):
        B = max(1, min(args.batch_size, 4))
        from models.etsformer import ETSConfigs
        x = torch.randn(B, 3, ETSConfigs().seq_len, 224, 224, device=args.device)
        y = head(ets(vit(x)))
    logger.info(f"[Shapes] logits: {tuple(y.shape)}")

    ets_total = sum(p.numel() for p in ets.parameters())
    ets_train = sum(p.numel() for p in ets.parameters() if p.requires_grad)
    head_total = sum(p.numel() for p in head.parameters())
    head_train = sum(p.numel() for p in head.parameters() if p.requires_grad)
    vit_total = sum(p.numel() for p in vit.parameters())
    vit_train = sum(p.numel() for p in vit.parameters() if p.requires_grad)

    logger.info(f"[Params] ViT total={vit_total:,} trainable={vit_train:,}")
    logger.info(f"[Params] ETS total={ets_total:,} trainable={ets_train:,}")
    logger.info(f"[Params] Head total={head_total:,} trainable={head_train:,}")
    logger.info("[Done]")


if __name__ == "__main__":
    main()
