"""DiT backbone: channel-concatenated source conditioning, 2D RoPE, QK-norm, adaLN timestep modulation."""
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mtdiff.config import ModelConfig

ROPE_THETA = 10_000.0
TIME_FREQ_DIM = 256
TIME_SCALE = 1000.0
MODULATION_SLOTS = 6


class RMSNorm(nn.Module):
    """RMS normalisation in the input dtype (kept in bf16 under autocast for speed)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.scale = math.sqrt(dim)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=-1, eps=self.eps) * (self.scale * self.weight).to(x.dtype)


def axial_rope(height: int, width: int, head_dim: int, device: torch.device) -> Tensor:
    """Rotation table of shape (2, height*width, head_dim/2): cos and sin of the axial angles."""
    quarter = head_dim // 4
    freqs = 1.0 / (ROPE_THETA ** (torch.arange(quarter, device=device, dtype=torch.float32) / quarter))
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    angles = torch.cat([ys.reshape(-1, 1) * freqs, xs.reshape(-1, 1) * freqs], dim=1)
    return torch.stack([angles.cos(), angles.sin()])


def apply_rope(x: Tensor, rope: Tensor) -> Tensor:
    """Rotate (batch, heads, tokens, head_dim) features; channel i pairs with channel i + head_dim/2."""
    cos, sin = rope.to(x.dtype)
    first, second = x.chunk(2, dim=-1)
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)


def timestep_embedding(t: Tensor, dim: int = TIME_FREQ_DIM) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * TIME_SCALE * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads or (dim // heads) % 4:
            raise ValueError("dim must be divisible by heads, and head_dim by 4 (axial RoPE)")
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.q_norm = RMSNorm(dim // heads)
        self.k_norm = RMSNorm(dim // heads)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, rope: Tensor) -> Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        q = apply_rope(self.q_norm(qkv[0]), rope)
        k = apply_rope(self.k_norm(qkv[1]), rope)
        out = F.scaled_dot_product_attention(q, k, qkv[2])
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, dim))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        self.table = nn.Parameter(torch.randn(MODULATION_SLOTS, dim) / math.sqrt(dim))

    def forward(self, x: Tensor, modulation: Tensor, rope: Tensor) -> Tensor:
        slots = (self.table[None] + modulation).unbind(dim=1)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (s[:, None] for s in slots)
        x = x + gate_a * self.attn(self.norm1(x) * (1 + scale_a) + shift_a, rope)
        return x + gate_m * self.mlp(self.norm2(x) * (1 + scale_m) + shift_m)


class DiT(nn.Module):
    """Predicts the flow velocity for the target latent given the noisy target and the source latent."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_channels = cfg.latent_channels * 2 + cfg.extra_channels
        self.patch_embed = nn.Conv2d(in_channels, cfg.dim, cfg.patch, cfg.patch)
        self.time_mlp = nn.Sequential(nn.Linear(TIME_FREQ_DIM, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
        self.time_to_modulation = nn.Linear(cfg.dim, MODULATION_SLOTS * cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg.dim, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth))
        self.final_norm = nn.LayerNorm(cfg.dim, elementwise_affine=False, eps=1e-6)
        self.final_table = nn.Parameter(torch.randn(2, cfg.dim) / math.sqrt(cfg.dim))
        self.final_time = nn.Linear(cfg.dim, 2 * cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.patch * cfg.patch * cfg.latent_channels)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.patch_embed.weight.view(self.patch_embed.weight.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.zeros_(self.head.weight)  # start by predicting zero velocity

    def unpatchify(self, tokens: Tensor, height: int, width: int) -> Tensor:
        patch, channels = self.cfg.patch, self.cfg.latent_channels
        x = tokens.reshape(tokens.shape[0], height, width, patch, patch, channels)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(tokens.shape[0], channels, height * patch, width * patch)

    def forward(self, noisy: Tensor, source: Tensor, t: Tensor, extra: Tensor | None = None) -> Tensor:
        inputs = [noisy, source] if extra is None else [noisy, source, extra]
        x = self.patch_embed(torch.cat(inputs, dim=1))
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        rope = axial_rope(height, width, self.cfg.dim // self.cfg.heads, x.device)
        temb = self.time_mlp(timestep_embedding(t).to(x.dtype))
        modulation = self.time_to_modulation(temb).reshape(-1, MODULATION_SLOTS, self.cfg.dim)
        for block in self.blocks:
            x = block(x, modulation, rope)
        shift, scale = (self.final_table[None] + self.final_time(temb).reshape(-1, 2, self.cfg.dim)).unbind(dim=1)
        x = self.head(self.final_norm(x) * (1 + scale[:, None]) + shift[:, None])
        return self.unpatchify(x, height, width)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
