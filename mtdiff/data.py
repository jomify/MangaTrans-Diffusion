"""Dataset of cached VAE latents with crops biased towards changed (text) regions."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from mtdiff.vae import LATENT_STRIDE


def read_manifest(path: str, split: str, domains: tuple[str, ...]) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r for r in rows if r["ok"] and r["split"] == split and r["domain"] in domains]


def sampling_weights(rows: list[dict], synthetic_fraction: float) -> list[float]:
    """Per-row weights so synthetic rows are drawn with total probability `synthetic_fraction`."""
    synthetic = sum(bool(r.get("synthetic")) for r in rows)
    real = len(rows) - synthetic
    if synthetic == 0 or real == 0:
        return [1.0] * len(rows)
    return [
        synthetic_fraction / synthetic if r.get("synthetic") else (1 - synthetic_fraction) / real for r in rows
    ]


def pick_crop_origin(mask: Tensor, crop: int, text_center_prob: float, rng: torch.Generator) -> tuple[int, int]:
    """Top-left corner of a crop; with probability `text_center_prob` it is centred near a masked pixel."""
    height, width = mask.shape
    max_y, max_x = height - crop, width - crop
    points = mask.nonzero()
    use_text = len(points) > 0 and torch.rand(1, generator=rng).item() < text_center_prob
    if not use_text:
        y = int(torch.randint(0, max_y + 1, (1,), generator=rng))
        x = int(torch.randint(0, max_x + 1, (1,), generator=rng))
        return y, x
    cy, cx = points[int(torch.randint(0, len(points), (1,), generator=rng))].tolist()
    jitter = torch.randint(-crop // 4, crop // 4 + 1, (2,), generator=rng).tolist()
    y = min(max(cy - crop // 2 + jitter[0], 0), max_y)
    x = min(max(cx - crop // 2 + jitter[1], 0), max_x)
    return y, x


def glyph_channels(glyph: Tensor, y: int, x: int, crop: int) -> Tensor:
    """Crop the full-resolution glyph image (uint8, H x W) and fold it to latent resolution."""
    stride = LATENT_STRIDE
    window = glyph[y * stride : (y + crop) * stride, x * stride : (x + crop) * stride]
    return F.pixel_unshuffle(window.float()[None] / 255.0, stride)


class LatentPairDataset(Dataset):
    """Yields {source, target, mask} latent crops. `deterministic=True` fixes the crop per index."""

    def __init__(
        self, cache_dir: str, rows: list[dict], crop: int, text_center_prob: float, deterministic: bool = False
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.rows = [r for r in rows if (self.cache_dir / f"{r['name']}.pt").exists()]
        if not self.rows:
            raise FileNotFoundError(f"no cached latents for the requested rows under {cache_dir}")
        self.crop = crop
        self.text_center_prob = text_center_prob
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = torch.load(self.cache_dir / f"{self.rows[index]['name']}.pt", weights_only=True)
        rng = torch.Generator()
        if self.deterministic:
            rng.manual_seed(index)
        else:
            rng.seed()
        y, x = pick_crop_origin(item["mask"], self.crop, self.text_center_prob, rng)
        window = (slice(y, y + self.crop), slice(x, x + self.crop))
        sample = {
            "source": item["src"][:, window[0], window[1]].float(),
            "target": item["tgt"][:, window[0], window[1]].float(),
            "mask": item["mask"][window[0], window[1]].float()[None],
        }
        if "glyph" in item:
            sample["glyph"] = glyph_channels(item["glyph"], y, x, self.crop)
        return sample
