"""MiniDance video DiT, written from scratch.

latent [B, 48, T, 16, 16] --patch(1,2,2)--> T*8*8 tokens --N x Block--> velocity [B, 48, T, 16, 16]

Block = self-attention over all space-time tokens (3D RoPE, QK-Norm)
      + cross-attention to the text tokens
      + SwiGLU MLP,
with timestep conditioning via adaLN-single (one shared modulation MLP + a small per-block table).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiTConfig:
    in_channels: int = 48         # Wan2.2 VAE latent channels
    patch: tuple = (1, 2, 2)      # (t, h, w)
    dim: int = 384
    depth: int = 8
    heads: int = 6
    text_dim: int = 1024          # Qwen3-0.6B hidden size
    mlp_ratio: float = 4.0
    rope_theta: float = 10000.0


PRESETS = {  # S3 overfit test uses "small"; "base" is the release-size target
    "tiny": dict(dim=256, depth=6, heads=4),
    "small": dict(dim=384, depth=8, heads=6),
    "base": dict(dim=768, depth=12, heads=12),
}


# ------------------------------------------------------------------ 3D RoPE
def rope_3d(T, H, W, head_dim, theta, device):
    """cos/sin [T*H*W, head_dim/2]. head_dim is split between the t/h/w axes (t gets 1/4)."""
    dt = head_dim // 4 // 2 * 2
    dh = (head_dim - dt) // 2 // 2 * 2
    dw = head_dim - dt - dh
    t, h, w = torch.meshgrid(torch.arange(T, device=device), torch.arange(H, device=device),
                             torch.arange(W, device=device), indexing="ij")
    angles = []
    for pos, d in ((t, dt), (h, dh), (w, dw)):
        freqs = 1.0 / theta ** (torch.arange(0, d, 2, device=device).float() / d)
        angles.append(pos.reshape(-1, 1).float() * freqs)
    angles = torch.cat(angles, dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):  # x: [B, heads, N, head_dim], rotate consecutive pairs
    x1, x2 = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.type_as(x)


# ------------------------------------------------------------------ layers
class RMSNorm(nn.RMSNorm):
    """Casts its weight to the input dtype so bf16 autocast can use the fused kernel."""

    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.eps)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, cos, sin):
        q, k, v = self.qkv(x).unflatten(-1, (3, self.heads, -1)).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin)
        return self.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2))


class CrossAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.q, self.kv = nn.Linear(dim, dim), nn.Linear(dim, dim * 2)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, y, mask):  # y: text tokens [B, L, dim], mask: [B, L] True = keep
        q = self.q_norm(self.q(x).unflatten(-1, (self.heads, -1)).transpose(1, 2))
        k, v = self.kv(y).unflatten(-1, (2, self.heads, -1)).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, self.k_norm(k), v, attn_mask=mask[:, None, None, :])
        return self.out(o.transpose(1, 2).flatten(2))


class SwiGLU(nn.Module):
    def __init__(self, dim, ratio):
        super().__init__()
        hidden = int(dim * ratio * 2 / 3) // 64 * 64
        self.w12, self.w3 = nn.Linear(dim, hidden * 2), nn.Linear(hidden, dim)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.dim
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(d, cfg.heads)
        self.norm2 = nn.LayerNorm(d, eps=1e-6)
        self.cross = CrossAttention(d, cfg.heads)
        self.norm3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLU(d, cfg.mlp_ratio)
        self.mod_table = nn.Parameter(torch.randn(6, d) / d ** 0.5)  # adaLN-single per-block offset

    def forward(self, x, t_mod, y, mask, cos, sin):
        shift1, scale1, gate1, shift2, scale2, gate2 = (t_mod + self.mod_table).unbind(1)
        h = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        x = x + gate1[:, None] * self.attn(h, cos, sin)
        x = x + self.cross(self.norm2(x), y, mask)
        h = self.norm3(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + gate2[:, None] * self.mlp(h)


def timestep_embedding(t, dim=256, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * 1000 * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


# ------------------------------------------------------------------ model
class DiT(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        d, (pt, ph, pw) = cfg.dim, cfg.patch
        self.patch_dim = cfg.in_channels * pt * ph * pw
        self.x_embed = nn.Linear(self.patch_dim, d)
        self.t_embed = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.t_mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))  # adaLN-single shared MLP
        self.y_embed = nn.Sequential(RMSNorm(cfg.text_dim), nn.Linear(cfg.text_dim, d),
                                     nn.GELU(approximate="tanh"), nn.Linear(d, d))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.out_table = nn.Parameter(torch.randn(2, d) / d ** 0.5)
        self.out = nn.Linear(d, self.patch_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # every block starts as identity: zero the residual-branch outputs
        for b in self.blocks:
            for lin in (b.attn.out, b.cross.out, b.mlp.w3):
                nn.init.zeros_(lin.weight)
        nn.init.zeros_(self.out.weight)
        nn.init.normal_(self.t_embed[0].weight, std=0.02)
        nn.init.normal_(self.t_embed[2].weight, std=0.02)

    def patchify(self, x):  # [B, C, T, H, W] -> [B, N, C*pt*ph*pw]
        pt, ph, pw = self.cfg.patch
        B, C, T, H, W = x.shape
        x = x.reshape(B, C, T // pt, pt, H // ph, ph, W // pw, pw)
        return x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, -1, self.patch_dim)

    def unpatchify(self, x, T, H, W):
        pt, ph, pw = self.cfg.patch
        B, C = x.shape[0], self.cfg.in_channels
        x = x.reshape(B, T // pt, H // ph, W // pw, C, pt, ph, pw)
        return x.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(B, C, T, H, W)

    def forward(self, x, t, y, mask):
        """x: noisy latent [B, C, T, H, W]; t: [B] in [0, 1]; y: text [B, L, text_dim]; mask: [B, L]."""
        B, C, T, H, W = x.shape
        pt, ph, pw = self.cfg.patch
        cos, sin = rope_3d(T // pt, H // ph, W // pw, self.cfg.dim // self.cfg.heads,
                           self.cfg.rope_theta, x.device)
        h = self.x_embed(self.patchify(x))
        t_emb = self.t_embed(timestep_embedding(t))
        t_mod = self.t_mod(t_emb).unflatten(1, (6, -1))
        y = self.y_embed(y)
        for block in self.blocks:
            h = block(h, t_mod, y, mask, cos, sin)
        shift, scale = (t_emb[:, None] + self.out_table).unbind(1)
        h = self.norm_out(h) * (1 + scale[:, None]) + shift[:, None]
        return self.unpatchify(self.out(h), T, H, W)


def build_dit(preset="small", **overrides):
    return DiT(DiTConfig(**{**PRESETS[preset], **overrides}))
