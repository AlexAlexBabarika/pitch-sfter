import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from kernels import get_kernel

activation = get_kernel("kernels-community/activation")


class F0Encoder(nn.Module):
    def __init__(self, out_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            activation.layers.Silu(),
            nn.Conv1d(32, 64, 5, padding=2),
            activation.layers.Silu(),
            nn.Conv1d(64, out_dim, 1),
        )

    def forward(self, f0: torch.Tensor, n_mels: int):
        # f0: [B, T]
        h = self.encoder(f0.unsqueeze(1))  # [B, C, T]
        h = h.unsqueeze(2).expand(-1, -1, n_mels, -1)  # broadcast to mel bins
        return h  # [B, C, n_mels, T]


class ShiftEncoder(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, out_dim), activation.layers.Silu(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, shift: torch.Tensor):
        return self.encoder(shift.unsqueeze(-1))


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], cond: [B, cond_dim]
        gb = self.proj(cond)  # [B, 2C]
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * x + beta


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.film = FiLM(cond_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(activation.layers.Silu(self.norm1(x)))
        h = self.film(h, cond)
        h = self.conv2(activation.layers.Silu(self.norm2(h)))
        return h + self.skip(x)


class GatedSkip(nn.Module):
    def __init__(self, channels: int, f0_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(f0_dim, channels, 1),
            activation.layers.Silu(),
            nn.Conv1d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, skip: torch.Tensor, f0_feat: torch.Tensor) -> torch.Tensor:
        # skip: [B, C, H, W]
        # f0_feat: [B, C_f0, n_mels, T_full] — pool to skip's W (time) length
        B, C, H, W = skip.shape
        f = F.adaptive_avg_pool2d(f0_feat, (1, W)).squeeze(2)  # [B, C_f0, W]
        g = self.gate(f).unsqueeze(2)  # [B, C, 1, W]
        return skip * g


class SelfAttn(nn.Module):
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.heads = heads

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv.unbind(1)  # [B, h, d, HW]
        q, k, v = (t.transpose(-1, -2).contiguous() for t in (q, k, v))  # [B, h, HW, d]
        out = F.scaled_dot_product_attention(q, k, v)  # [B, h, HW, d]
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)
