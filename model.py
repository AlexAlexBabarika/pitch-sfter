import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from kernels import get_kernel
from config import ModelConfig

activation = get_kernel("kernels-community/activation")


class F0Encoder(nn.Module):
    def __init__(self, out_dim: int = 64):
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
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, H, W]
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
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film(h, cond)
        h = self.conv2(F.silu(self.norm2(h)))
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
        f = f0_feat.mean(dim=2)  # [B, C_f0, T_full] — collapse mel
        f = F.interpolate(f, size=W, mode="linear", align_corners=False)  # [B, C_f0, W]
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


class PitchUNet(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = ModelConfig()
        base = cfg.base_channels
        mults = cfg.channel_mults
        shift_emb_dim = cfg.shift_emb_dim
        f0_dim = cfg.f0_emb_dim

        # shift, f0 embs
        self.shift_mlp = ShiftEncoder(out_dim=shift_emb_dim)
        self.f0_enc = F0Encoder(out_dim=f0_dim)
        in_ch = 1 + f0_dim  # mel(1) + f0 features
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)

        # encoder
        chs = [base * m for m in mults]  # e.g. [64,128,256,384]
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = base
        for c in chs:
            self.enc_blocks.append(ResBlock(prev, c, shift_emb_dim))
            self.downs.append(nn.Conv2d(c, c, kernel_size=4, stride=2, padding=1))
            prev = c

        # bottleneck
        self.bot1 = ResBlock(prev, prev, shift_emb_dim)
        self.attn = SelfAttn(prev) if cfg.attention_in_bottleneck else nn.Identity()
        self.bot2 = ResBlock(prev, prev, shift_emb_dim)

        # decoder (mirrors encoder)
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.skip_gates = nn.ModuleList()
        for c in reversed(chs):
            self.ups.append(
                nn.ConvTranspose2d(prev, c, kernel_size=4, stride=2, padding=1)
            )
            # input to dec block = upsampled (c) + gated skip (c) = 2c
            self.dec_blocks.append(ResBlock(2 * c, c, shift_emb_dim))
            self.skip_gates.append(GatedSkip(c, f0_dim))
            prev = c

        # output head
        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

        # zero-init output so initial output ≈ 0; loss starts as |target|
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        mel: torch.Tensor,
        f0: torch.Tensor,
        shift: torch.Tensor,
        skip_dropout_p: float = 0.0,
    ):
        B, _, n_mels, T = mel.shape
        cond = self.shift_mlp(shift)  # [B, cond_dim]
        f0_feat = self.f0_enc(f0, n_mels)  # [B, C_f0, n_mels, T]

        x = torch.cat([mel, f0_feat], dim=1)  # [B, 1+C_f0, n_mels, T]
        x = self.stem(x)

        skips = []
        for block, down in zip(self.enc_blocks, self.downs):
            x = block(x, cond)
            skips.append(x)
            x = down(x)

        x = self.bot1(x, cond)
        x = self.attn(x)
        x = self.bot2(x, cond)

        for up, block, gate, skip in zip(
            self.ups, self.dec_blocks, self.skip_gates, reversed(skips)
        ):
            x = up(x)
            gated = gate(skip, f0_feat)
            # random skip dropout: occasionally force decoder to rely on
            # bottleneck + F0 only. Strengthens disentanglement.
            if self.training and skip_dropout_p > 0:
                mask = (
                    torch.rand(B, 1, 1, 1, device=x.device) > skip_dropout_p
                ).float()
                gated = gated * mask
            # match spatial size in case of off-by-one from odd input dims
            if gated.shape[-2:] != x.shape[-2:]:
                gated = F.interpolate(gated, size=x.shape[-2:], mode="nearest")
            x = torch.cat([x, gated], dim=1)
            x = block(x, cond)

        x = self.out_conv(F.silu(self.out_norm(x)))
        return x  # [B, 1, n_mels, T]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = PitchUNet()
    print(f"Params: {count_params(m) / 1e6:.2f} M")
    mel = torch.randn(2, 1, 80, 220)
    f0 = torch.randn(2, 220)
    sh = torch.zeros(2)
    out = m(mel, f0, sh, skip_dropout_p=0.2)
    print("out:", out.shape)
