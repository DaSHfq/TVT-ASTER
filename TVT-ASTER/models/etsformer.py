# models/etsformer.py
# -*- coding: utf-8 -*-
"""
ETSformer 适配器（接收 ViT 输出）+ BiLSTM 分类头

输入： (B, T, 512) —— 来自 models/vit.py 的 ViTBackbone
流程： ViT(空间) -> ETSformer(时序/频域/指数平滑) -> BiLSTM -> 线性分类
输出： (B, num_classes)

说明：
- 不修改 ETSformer 核心算法（models/model.py 等文件保持原样），仅做“形状与配置适配”。
- 关键超参数在 __main__ 中集中罗列与打印，使用 dataclass 默认值实例化（避免重复定义数值）。
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from .model import ETSformer as _ETSCore  # 相对导入：使用迁移到 models/ 下的核心实现


@dataclass
class ETSConfigs:
    # 基本长度配置
    seq_len: int = 32        # 输入序列长度（和 ViT 帧数一致）
    label_len: int = 0       # 本实现不使用，可保留 0
    pred_len: int = 32       # 输出序列长度（保持与输入等长，方便后续 BiLSTM）

    # 通道与维度
    enc_in: int = 512        # ETSformer 输入通道（与 ViT 输出一致）
    d_model: int = 512       # ETSformer 内部维度
    c_out: int = 512         # ETSformer 输出通道（保持 512，便于直接送 BiLSTM）

    # Transformer 结构
    n_heads: int = 8         # 注意：需整除 d_model
    e_layers: int = 3
    d_layers: int = 3
    d_ff: int = 2048

    # 频域主频数量（T=32 时可用频点为 15，故 K ≤ 15，这里取 12 更稳妥）
    K: int = 12

    # 训练细节
    dropout: float = 0.1
    activation: str = 'gelu'
    output_attention: bool = False

    # 轻量增强噪声（训练期在 ETSformer 内部生效）
    std: float = 0.02


class ETSFormerAdapter(nn.Module):
    """
    包装 models/model.py 的 ETSformer，使其输入/输出与 ViT 接口一致：
      in : (B, T, 512)
      out: (B, T, 512)
    """
    def __init__(self, configs: ETSConfigs):
        super().__init__()
        # 合法性校验
        assert configs.d_model % configs.n_heads == 0, \
            f"d_model ({configs.d_model}) must be divisible by n_heads ({configs.n_heads})"
        assert configs.enc_in == configs.d_model == configs.c_out == 512, \
            "适配器默认 enc_in=d_model=c_out=512，以无缝衔接 ViT 与 BiLSTM"

        # 记录配置并构建核心 ETSformer
        self.configs = configs
        self.core = _ETSCore(configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 512)
        return: (B, T, 512)
        """
        assert x.ndim == 3, f"Expected (B,T,512), got {x.shape}"
        B, T, C = x.shape
        assert T == self.configs.seq_len, f"seq_len mismatch: got {T}, expect {self.configs.seq_len}"
        assert C == self.configs.enc_in, f"enc_in mismatch: got {C}, expect {self.configs.enc_in}"

        # ETSformer forward 签名含有 x_mark_enc/x_dec/x_mark_dec 等，这里均置 None
        y = self.core(
            x_enc=x,
            x_mark_enc=None, x_dec=None, x_mark_dec=None,
            enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
            decomposed=False, attention=False
        )
        # y 形状：(B, pred_len=seq_len, c_out=512)
        return y


class TemporalHead(nn.Module):
    """
    BiLSTM + 分类头
      in : (B, T, 512)
      out: (B, num_classes)
    """
    def __init__(self,
                 input_dim: int = 512,
                 hidden: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 num_classes: int = 2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 512)
        return: (B, num_classes)
        """
        out, _ = self.bilstm(x)   # (B,T,2*hidden)
        last = out[:, -1, :]      # (B, 2*hidden) 也可改 mean-pool：out.mean(dim=1)
        last = self.dropout(last)
        logits = self.classifier(last)
        return logits


def count_params(module: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


# --------------------- 实验超参数（集中展示/打印） --------------------- #
if __name__ == "__main__":
    # 随机种子与批大小（仅用于自检打印/最小前向）
    seed = 42
    batch_size = 8
    torch.manual_seed(seed)

    # 使用 dataclass 默认值实例化（不重复写数值）
    ets_cfg = ETSConfigs()

    # 打印关键信息
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[Seed] {seed}")
    print(f"[BatchSize] {batch_size}")
    print(f"[ETSformerCfg] {ets_cfg}")

    # 可整除与 K 合法性校验
    ok_div = (ets_cfg.d_model % ets_cfg.n_heads == 0)
    # T 偶数时可用频点 = T/2 - 1；T 奇数时 = floor(T/2)
    allowed_k = (ets_cfg.seq_len // 2 - 1) if (ets_cfg.seq_len % 2 == 0) else (ets_cfg.seq_len // 2)
    print(f"[Check] d_model % n_heads == 0 ? {ok_div}")
    print(f"[Check] K <= allowed_k({allowed_k}) ? {ets_cfg.K <= allowed_k}")

    # 仅做形状冒烟：随机特征输入 (B,T,512) -> ETS -> Head
    ets = ETSFormerAdapter(ets_cfg).to(device)
    head = TemporalHead(input_dim=ets_cfg.c_out, hidden=256, num_layers=3, dropout=0.5, num_classes=2).to(device)

    x = torch.randn(batch_size, ets_cfg.seq_len, ets_cfg.enc_in, device=device)
    with torch.no_grad():
        y_ets = ets(x)      # (B,T,512)
        y_cls = head(y_ets) # (B,2)

    print(f"[Forward] ETSformer out: {tuple(y_ets.shape)} ; Head out: {tuple(y_cls.shape)}")
    ets_total, ets_trainable = count_params(ets)
    head_total, head_trainable = count_params(head)
    print(f"[Params] ETSformer total={ets_total:,} trainable={ets_trainable:,}")
    print(f"[Params] TemporalHead total={head_total:,} trainable={head_trainable:,}")
