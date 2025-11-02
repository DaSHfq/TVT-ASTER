# models/bilstm.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Literal, Optional


class AttentionPool(nn.Module):
    """
    轻量注意力池化：对时序输出 (B,T,D) 计算可学习权重并加权求和得到 (B,D)。
    """
    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        score = self.mlp(x)                        # (B,T,1)
        attn = torch.softmax(score, dim=1)         # (B,T,1)
        pooled = (attn * x).sum(dim=1)             # (B,D)
        return pooled


class BiLSTMHead(nn.Module):
    """
    BiLSTM 分类头
      输入:  (B, T, input_dim)  —— 与 ETSformer 输出一致
      输出:  (B, num_classes)

    参数:
      input_dim:      输入通道（默认 512）
      hidden:         LSTM 隐藏维（默认 256）
      num_layers:     LSTM 层数（默认 3）
      dropout:        LSTM 内层 dropout（默认 0.5）
      num_classes:    分类数（默认 2）
      pooling:        'last' | 'mean' | 'max' | 'mean_last' | 'attn'
      ln_input:       是否在 LSTM 前做 LayerNorm（默认 False）
      ln_output:      是否在池化后做 LayerNorm（默认 False）
      attn_hidden:    注意力池化的隐层大小（仅 pooling='attn' 时有效）
    """
    def __init__(self,
                 input_dim: int = 512,
                 hidden: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 num_classes: int = 2,
                 pooling: Literal['last', 'mean', 'max', 'mean_last', 'attn'] = 'mean_last',
                 ln_input: bool = False,
                 ln_output: bool = False,
                 attn_hidden: int = 128):
        super().__init__()
        self.pooling = pooling
        self.ln_in = nn.LayerNorm(input_dim) if ln_input else None
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)

        out_dim = hidden * 2  # 双向
        if pooling == 'mean_last':
            out_dim = out_dim * 2  # mean(2h) + last(2h)
        elif pooling == 'attn':
            self.attn = AttentionPool(out_dim, hidden=attn_hidden, dropout=dropout)

        self.ln_out = nn.LayerNorm(out_dim) if ln_output else None
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        if self.ln_in is not None:
            x = self.ln_in(x)

        y, _ = self.bilstm(x)  # y: (B,T,2h)

        if self.pooling == 'last':
            pooled = y[:, -1, :]
        elif self.pooling == 'mean':
            pooled = y.mean(dim=1)
        elif self.pooling == 'max':
            pooled, _ = y.max(dim=1)
        elif self.pooling == 'mean_last':
            pooled = torch.cat([y.mean(dim=1), y[:, -1, :]], dim=-1)  # (B,4h)
        elif self.pooling == 'attn':
            pooled = self.attn(y)  # (B,2h)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        if self.ln_out is not None:
            pooled = self.ln_out(pooled)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
