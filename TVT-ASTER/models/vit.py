# models/vit.py
# -*- coding: utf-8 -*-
import os
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torchvision.models import vit_b_16


class ViTBackbone(nn.Module):
    """
    ViT-Base/16 as a per-frame spatial backbone (offline friendly).

    Features:
      - Offline weights loading via `weights_path` (no internet).
      - Freeze the first N encoder blocks (default: 6).
      - Project CLS feature (768) -> d_model (default: 512) + optional LayerNorm.
      - Accepts (B, C, T, H, W) and outputs (B, T, d_model).
      - NEW: temporal micro-batching via `frames_per_gpu_batch` to reduce peak GPU memory.

    Args:
      pretrained (bool): Keep False in offline mode. If True and no weights_path,
                         would use torchvision's default (may require internet).
      freeze_blocks (int): Number of encoder blocks (from the beginning) to freeze.
      d_model (int): Output feature dimension after projection.
      use_ln (bool): Whether to append a LayerNorm after projection.
      weights_path (Optional[str]): Local path to ViT-B/16 ImageNet-1k weights (.pth).
                                    If provided, it will be used with strict=False to ignore head mismatch.
      frames_per_gpu_batch (int): Number of frames to process per GPU micro-batch along time.
                                  Reduces peak memory for B*T frames. Default: 64.
    """

    def __init__(
        self,
        pretrained: bool = False,
        freeze_blocks: int = 6,
        d_model: int = 512,
        use_ln: bool = True,
        weights_path: Optional[str] = None,
        frames_per_gpu_batch: int = 64,
    ):
        super().__init__()

        # 1) Offline-friendly vit_b_16 (no auto-download)
        self.vit = vit_b_16(weights=None)
        self.hidden_dim = self.vit.hidden_dim  # 768 for ViT-B/16
        self.frames_per_gpu_batch = max(1, int(frames_per_gpu_batch))

        # 2) Load local weights if provided
        if weights_path is not None:
            self._load_offline_weights(weights_path)
        elif pretrained:
            raise RuntimeError(
                "pretrained=True without weights_path is not offline-friendly. "
                "Please provide a local weights_path."
            )

        # 3) Freeze first `freeze_blocks` encoder layers
        total_blocks = len(self.vit.encoder.layers)
        assert 0 <= freeze_blocks <= total_blocks, f"freeze_blocks must be in [0, {total_blocks}]"
        for i, block in enumerate(self.vit.encoder.layers):
            if i < freeze_blocks:
                for p in block.parameters():
                    p.requires_grad = False

        # 4) Output projection to d_model (+ LayerNorm if enabled)
        out_layers = [nn.Linear(self.hidden_dim, d_model)]
        if use_ln:
            out_layers.append(nn.LayerNorm(d_model))
        self.proj = nn.Sequential(*out_layers)

    # ---------------------- Weights utilities ---------------------- #
    def _clean_state_dict_keys(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize common checkpoint formats to torchvision's ViT keys."""
        clean_sd: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module.") :]
            if k.startswith(("head.", "heads.", "classifier.", "fc.")):
                continue
            clean_sd[k] = v
        return clean_sd

    def _extract_state_dict(self, ckpt: Any) -> Dict[str, torch.Tensor]:
        """Extract state_dict from various wrappers."""
        if isinstance(ckpt, dict):
            for key in ["state_dict", "model", "net", "weights", "params"]:
                if key in ckpt and isinstance(ckpt[key], dict):
                    return ckpt[key]
            if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                return ckpt
        raise RuntimeError(
            "Unsupported checkpoint structure. Provide a plain state_dict or a dict "
            "with one of keys: 'state_dict', 'model', 'net', 'weights', 'params'."
        )

    def _load_offline_weights(self, weights_path: str) -> None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"weights_path not found: {weights_path}")
        ckpt = torch.load(weights_path, map_location="cpu")
        state_dict = self._extract_state_dict(ckpt)
        state_dict = self._clean_state_dict_keys(state_dict)

        missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[ViT] Missing keys: {len(missing)} (ok if only classifier/head related)")
        if unexpected:
            print(f"[ViT] Unexpected keys: {len(unexpected)} (ignored)")

    # ---------------------- Frame-wise forward ---------------------- #
    def _frames_forward(self, x_bchw: torch.Tensor) -> torch.Tensor:
        """
        Per-frame forward mirroring torchvision's ViT forward until CLS extraction.
        Args:
          x_bchw: (N, 3, H, W)
        Returns:
          cls_feats: (N, hidden_dim)
        """
        # Patch embedding -> (N, num_patches, hidden_dim)
        x = self.vit._process_input(x_bchw)

        # Prepend CLS token
        n = x.shape[0]
        class_tok = self.vit.class_token.expand(n, -1, -1)  # (N, 1, hidden_dim)
        x = torch.cat([class_tok, x], dim=1)                # (N, 1+num_patches, hidden_dim)

        # IMPORTANT: Do NOT add pos_embedding manually.
        # torchvision's encoder(x) handles: +pos_embedding, dropout, layers, final LN.
        x = self.vit.encoder(x)                             # (N, 1+num_patches, hidden_dim)
        cls_feats = x[:, 0]                                 # (N, hidden_dim)
        return cls_feats

    # ---------------------- Micro-batched forward ---------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (B, C, T, H, W)
        Returns:
          feats: (B, T, d_model)
        """
        assert x.ndim == 5, f"Expected input (B,C,T,H,W), got {x.shape}"
        B, C, T, H, W = x.shape

        # (B,C,T,H,W) -> (B*T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)                 # (B*T, C, H, W)

        # Temporal micro-batching to reduce peak GPU memory
        bt = x.shape[0]
        chunk = max(1, self.frames_per_gpu_batch)
        cls_chunks = []
        for s in range(0, bt, chunk):
            e = min(s + chunk, bt)
            x_chunk = x[s:e]                                # (chunk, C, H, W)
            cls_feats_chunk = self._frames_forward(x_chunk) # (chunk, hidden_dim)
            cls_chunks.append(cls_feats_chunk)
            # release chunk references ASAP
            del x_chunk, cls_feats_chunk

        cls_feats = torch.cat(cls_chunks, dim=0)            # (B*T, hidden_dim)
        del cls_chunks

        # Projection (+ optional LayerNorm)
        proj = self.proj(cls_feats)                         # (B*T, d_model)

        # (B*T, d_model) -> (B, T, d_model)
        proj = proj.view(B, T, -1).contiguous()
        return proj


if __name__ == "__main__":
    # Quick sanity check (kept minimal and offline-friendly)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTBackbone(
        pretrained=False,
        freeze_blocks=6,
        d_model=512,
        use_ln=True,
        weights_path=os.path.join("weights", "vit_b_16_imagenet1k_v1.pth"),
        frames_per_gpu_batch=64,  # adjust to control peak memory
    ).to(device)

    dummy = torch.randn(2, 3, 32, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)
    print("Output shape:", tuple(out.shape))  # expected: (2, 32, 512)
