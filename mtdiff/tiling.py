"""Tiled inference: evaluate the model on training-sized windows and average the velocities.

The model is trained on fixed-size latent crops, so whole pages are processed as overlapping
tiles at every sampling step (the page shares one noise field, so tiles agree in the overlaps).
"""
import torch
from torch import Tensor, nn


def tile_origins(length: int, tile: int, stride: int) -> list[int]:
    """Start offsets that cover `length` with windows of `tile`, the last one flush with the end."""
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile, stride))
    return origins + [length - tile]


def tile_weight(tile: int, device: torch.device) -> Tensor:
    """Separable triangular window so tile centres dominate the average."""
    ramp = 1.0 - (torch.arange(tile, device=device) + 0.5 - tile / 2).abs() / (tile / 2 + 1)
    return ramp[:, None] * ramp[None, :]


class TiledModel(nn.Module):
    """Wraps a model so that `forward` runs on overlapping tiles and returns the blended output."""

    def __init__(self, model: nn.Module, tile: int, stride: int, batch: int = 16) -> None:
        super().__init__()
        self.model = model
        self.tile = tile
        self.stride = stride
        self.batch = batch

    def forward(self, noisy: Tensor, source: Tensor, t: Tensor, extra: Tensor | None = None) -> Tensor:
        height, width = noisy.shape[-2:]
        tile = self.tile
        if height < tile or width < tile:
            return self.model(noisy, source, t, extra)
        corners = [
            (y, x) for y in tile_origins(height, tile, self.stride) for x in tile_origins(width, tile, self.stride)
        ]
        weight = tile_weight(tile, noisy.device)
        total = torch.zeros_like(noisy, dtype=torch.float32)
        norm = torch.zeros((1, 1, height, width), device=noisy.device)
        for start in range(0, len(corners), self.batch):
            chunk = corners[start : start + self.batch]

            def windows(tensor: Tensor) -> Tensor:
                return torch.cat([tensor[..., y : y + tile, x : x + tile] for y, x in chunk])

            out = self.model(
                windows(noisy),
                windows(source),
                t.repeat(len(chunk)),
                None if extra is None else windows(extra),
            ).float()
            pieces = out.split(noisy.shape[0])
            for (y, x), piece in zip(chunk, pieces):
                total[..., y : y + tile, x : x + tile] += piece * weight
                norm[..., y : y + tile, x : x + tile] += weight
        return total / norm
