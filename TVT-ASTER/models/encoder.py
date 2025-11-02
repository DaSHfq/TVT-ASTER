# models/encoder.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from einops import rearrange, reduce, repeat

from .modules import Feedforward
from .exponential_smoothing import ExponentialSmoothing


class GrowthLayer(nn.Module):

    def __init__(self, d_model, nhead, d_head=None, dropout=0.1, output_attention=False):
        super().__init__()
        self.d_head = d_head or (d_model // nhead)
        self.d_model = d_model
        self.nhead = nhead
        self.output_attention = output_attention

        self.z0 = nn.Parameter(torch.randn(self.nhead, self.d_head))
        self.in_proj = nn.Linear(self.d_model, self.d_head * self.nhead)
        self.es = ExponentialSmoothing(self.d_head, self.nhead, dropout=dropout)
        self.out_proj = nn.Linear(self.d_head * self.nhead, self.d_model)

        assert self.d_head * self.nhead == self.d_model, "d_model must be divisible by nhead"

    def forward(self, inputs):
        """
        :param inputs: shape: (batch, seq_len, dim)
        :return: shape: (batch, seq_len, dim)
        """
        b, t, d = inputs.shape
        values = self.in_proj(inputs).view(b, t, self.nhead, -1)
        values = torch.cat([repeat(self.z0, 'h d -> b 1 h d', b=b), values], dim=1)
        values = values[:, 1:] - values[:, :-1]
        out = self.es(values)
        out = torch.cat([repeat(self.es.v0, '1 1 h d -> b 1 h d', b=b), out], dim=1)
        out = rearrange(out, 'b t h d -> b t (h d)')
        out = self.out_proj(out)

        if self.output_attention:
            return out, self.es.get_exponential_weight(t)[1]
        return out, None


class FourierLayer(nn.Module):

    def __init__(self, d_model, pred_len, k=None, low_freq=1, output_attention=False):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq
        self.output_attention = output_attention

    def forward(self, x):
        """x: (b, t, d)"""
        if self.output_attention:
            return self.dft_forward(x)

        b, t, d = x.shape

        with torch.autocast(x.device.type, enabled=False):
            x32 = x.to(torch.float32)
            x_freq = fft.rfft(x32, dim=1)  # (b, t//2+1, d) on x.device

            if t % 2 == 0:
                x_freq = x_freq[:, self.low_freq:-1]
                f = fft.rfftfreq(t)[self.low_freq:-1]
            else:
                x_freq = x_freq[:, self.low_freq:]
                f = fft.rfftfreq(t)[self.low_freq:]

        device = x_freq.device
        f = f.to(device=device)

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))
        f = rearrange(f[index_tuple], 'b f d -> b f () d')

        return self.extrapolate(x_freq, f, t), None

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_val = rearrange(torch.arange(t + self.pred_len, dtype=torch.float, device=x_freq.device),
                          't -> () () t ()')

        amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')

        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)

        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        # x_freq: (b, f, d) on same device as input
        b, f, d = x_freq.shape
        device = x_freq.device

        k = min(self.k, f) if self.k is not None else f
        values, indices = torch.topk(x_freq.abs(), k, dim=1, largest=True, sorted=True)

        a = torch.arange(b, device=device, dtype=torch.long)
        c = torch.arange(d, device=device, dtype=torch.long)
        mesh_a, mesh_c = torch.meshgrid(a, c, indexing='ij')

        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_c.unsqueeze(1))
        x_freq = x_freq[index_tuple]

        return x_freq, index_tuple

    def dft_forward(self, x):
        T = x.size(1)
        device = x.device
        d = x.size(-1)

        # DFT/IDFT 矩阵（纯 torch、驻留 GPU）
        dft_mat = fft.fft(torch.eye(T, device=device))  # (T, T) complex

        i = torch.arange(self.pred_len + T, device=device, dtype=torch.long)
        j = torch.arange(T, device=device, dtype=torch.long)
        i, j = torch.meshgrid(i, j, indexing='ij')
        phase = (2 * math.pi / T) * (i * j).to(dtype=torch.float32)
        idft_mat = torch.exp(1j * phase).to(dtype=torch.complex64) / T  # (T+pred_len, T) complex

        x_freq = torch.einsum('ft,btd->bfd', [dft_mat, x.cfloat()])

        if T % 2 == 0:
            x_freq = x_freq[:, self.low_freq:T // 2]
        else:
            x_freq = x_freq[:, self.low_freq:T // 2 + 1]

        k = min(self.k, x_freq.size(1)) if self.k is not None else x_freq.size(1)
        _, indices = torch.topk(x_freq.abs(), k, dim=1, largest=True, sorted=True)
        indices = indices + self.low_freq
        indices = torch.cat([indices, -indices], dim=1)

        dft_mat = repeat(dft_mat, 'f t -> b f t d', b=x.shape[0], d=d)
        idft_mat = repeat(idft_mat, 't f -> b t f d', b=x.shape[0], d=d)

        a = torch.arange(x.size(0), device=device, dtype=torch.long)
        c = torch.arange(d, device=device, dtype=torch.long)
        mesh_a, mesh_c = torch.meshgrid(a, c, indexing='ij')

        dft_mask = torch.zeros_like(dft_mat, device=device)
        dft_mask[mesh_a, indices, :, mesh_c] = 1
        dft_mat = dft_mat * dft_mask

        idft_mask = torch.zeros_like(idft_mat, device=device)
        idft_mask[mesh_a, :, indices, mesh_c] = 1
        idft_mat = idft_mat * idft_mask

        attn = torch.einsum('bofd,bftd->botd', [idft_mat, dft_mat]).real
        return torch.einsum('botd,btd->bod', [attn, x]), rearrange(attn, 'b d o t -> b d o t')


class LevelLayer(nn.Module):

    def __init__(self, d_model, c_out, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.c_out = c_out

        self.es = ExponentialSmoothing(1, self.c_out, dropout=dropout, aux=True)
        self.growth_pred = nn.Linear(self.d_model, self.c_out)
        self.season_pred = nn.Linear(self.d_model, self.c_out)

    def forward(self, level, growth, season):
        b, t, _ = level.shape
        growth = self.growth_pred(growth).view(b, t, self.c_out, 1)
        season = self.season_pred(season).view(b, t, self.c_out, 1)
        growth = growth.view(b, t, self.c_out, 1)
        season = season.view(b, t, self.c_out, 1)
        level = level.view(b, t, self.c_out, 1)
        out = self.es(level - season, aux_values=growth)
        out = rearrange(out, 'b t h d -> b t (h d)')
        return out


class EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, c_out, seq_len, pred_len, k, dim_feedforward=None, dropout=0.1,
                 activation='sigmoid', layer_norm_eps=1e-5, output_attention=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.c_out = c_out
        self.seq_len = seq_len
        self.pred_len = pred_len
        dim_feedforward = dim_feedforward or 4 * d_model
        self.dim_feedforward = dim_feedforward

        self.growth_layer = GrowthLayer(d_model, nhead, dropout=dropout, output_attention=output_attention)
        self.seasonal_layer = FourierLayer(d_model, pred_len, k=k, output_attention=output_attention)
        self.level_layer = LevelLayer(d_model, c_out, dropout=dropout)

        # Implementation of Feedforward model
        self.ff = Feedforward(d_model, dim_feedforward, dropout=dropout, activation=activation)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, res, level, attn_mask=None):
        season, season_attn = self._season_block(res)
        res = res - season[:, :-self.pred_len]
        growth, growth_attn = self._growth_block(res)
        res = self.norm1(res - growth[:, 1:])
        res = self.norm2(res + self.ff(res))

        level = self.level_layer(level, growth[:, :-1], season[:, :-self.pred_len])

        return res, level, growth, season, season_attn, growth_attn

    def _growth_block(self, x):
        x, growth_attn = self.growth_layer(x)
        return self.dropout1(x), growth_attn

    def _season_block(self, x):
        x, season_attn = self.seasonal_layer(x)
        return self.dropout2(x), season_attn


class Encoder(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, res, level, attn_mask=None):
        growths = []
        seasons = []
        season_attns = []
        growth_attns = []
        for layer in self.layers:
            res, level, growth, season, season_attn, growth_attn = layer(res, level, attn_mask=None)
            growths.append(growth)
            seasons.append(season)
            season_attns.append(season_attn)
            growth_attns.append(growth_attn)

        return level, growths, seasons, season_attns, growth_attns
