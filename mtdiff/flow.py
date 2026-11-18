"""Rectified-flow training targets, mask-weighted loss, and Euler sampling.

Convention: x_t = (1 - t) * data + t * start, velocity = start - data, sampling runs t from 1 to 0.
The start point is pure noise (baseline) or the source latent plus scaled noise (v2), so regions
that do not change need almost no transport.
"""
import torch
from torch import Tensor, nn


def sample_timesteps(batch: int, device: torch.device, generator: torch.Generator | None = None) -> Tensor:
    """Logit-normal timesteps, which concentrate training on mid-noise levels."""
    return torch.sigmoid(torch.randn(batch, device=device, generator=generator))


def start_point(source: Tensor, noise: Tensor, from_source: bool, noise_scale: float) -> Tensor:
    return source + noise_scale * noise if from_source else noise


def make_noisy(target: Tensor, start: Tensor, t: Tensor) -> Tensor:
    t = t.view(-1, 1, 1, 1)
    return (1 - t) * target + t * start


def velocity_target(target: Tensor, start: Tensor) -> Tensor:
    return start - target


def weighted_flow_loss(pred: Tensor, velocity: Tensor, mask: Tensor, mask_weight: float) -> Tensor:
    """MSE where pixels inside `mask` (batch, 1, h, w in [0, 1]) count `1 + mask_weight` times."""
    weight = 1.0 + mask_weight * mask
    squared = (pred.float() - velocity.float()).pow(2)
    return (squared * weight).sum() / (weight.sum() * pred.shape[1])


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    source: Tensor,
    steps: int,
    generator: torch.Generator | None = None,
    extra: Tensor | None = None,
    from_source: bool = False,
    noise_scale: float = 1.0,
) -> Tensor:
    noise = torch.randn(source.shape, device=source.device, generator=generator, dtype=source.dtype)
    x = start_point(source, noise, from_source, noise_scale)
    times = torch.linspace(1.0, 0.0, steps + 1, device=source.device)
    for current, following in zip(times[:-1], times[1:]):
        t = current.expand(source.shape[0])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=source.is_cuda):
            velocity = model(x, source, t, extra)
        x = x + (following - current) * velocity.to(x.dtype)
    return x
