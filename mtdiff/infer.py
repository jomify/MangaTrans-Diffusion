"""End-to-end page rendering: Japanese page + (box, Chinese text) pairs -> translated page.

The model only writes text; detecting the original text, translating it and choosing the
layout boxes are inputs here (see README for how to obtain them).
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from mtdiff.config import ModelConfig
from mtdiff.flow import euler_sample
from mtdiff.glyph import line_rect, render_glyphs
from mtdiff.model import DiT
from mtdiff.tiling import TiledModel
from mtdiff.vae import LATENT_STRIDE, LatentCodec

PAD_MULTIPLE = 16
DEFAULT_WIDTH = 1280  # working width of the training pages
DEFAULT_MARGIN = 24
FEATHER = 15
MODEL_FILE = "model.safetensors"
CONFIG_FILE = "config.json"


@dataclass(frozen=True)
class Region:
    """One Chinese text line: where it is typeset and (optionally) a larger area to repaint."""

    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in input-image pixels
    text: str
    erase: tuple[float, float, float, float] | None = None


def _parse_rectangle(values: object, name: str) -> tuple[float, float, float, float]:
    message = f"invalid {name} {values!r}: expected four finite coordinates [x0, y0, x1, y1] with x1 > x0 and y1 > y0"
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(message)
    try:
        x0, y0, x1, y1 = (float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
        raise ValueError(message)
    return x0, y0, x1, y1


def parse_regions(data: list | dict) -> list[Region]:
    items = data["regions"] if isinstance(data, dict) else data
    regions = []
    for item in items:
        box = _parse_rectangle(item["box"], "box")
        if not item.get("text"):
            raise ValueError(f"region {item['box']} has no text")
        erase = _parse_rectangle(item["erase"], "erase") if item.get("erase") is not None else None
        regions.append(Region(box, item["text"], erase))
    return regions


def scale_regions(regions: list[Region], factor: float) -> list[Region]:
    def scaled(box):
        return None if box is None else tuple(v * factor for v in box)

    return [Region(scaled(r.box), r.text, scaled(r.erase)) for r in regions]


def to_lines(regions: list[Region]) -> list[dict]:
    lines = []
    for region in regions:
        x0, y0, x1, y1 = region.box
        lines.append({"box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "text": region.text})
    return lines


def paste_mask(regions: list[Region], height: int, width: int, margin: int) -> np.ndarray:
    """Soft mask (H x W, 0..1) of the area where model output replaces the source page."""
    mask = np.zeros((height, width), np.float32)
    for region in regions:
        box = region.erase or region.box
        x0, y0, x1, y1 = line_rect({"box": [[box[0], box[1]], [box[2], box[3]]]}, height, width)
        mask[max(y0 - margin, 0) : y1 + margin, max(x0 - margin, 0) : x1 + margin] = 1.0
    return cv2.GaussianBlur(mask, (0, 0), FEATHER / 3) if mask.any() else mask


def pad_edges(array: np.ndarray, multiple: int = PAD_MULTIPLE) -> np.ndarray:
    height, width = array.shape[:2]
    pads = ((0, (-height) % multiple), (0, (-width) % multiple)) + ((0, 0),) * (array.ndim - 2)
    return np.pad(array, pads, mode="edge")


def load_model(weights_dir: Path, device: torch.device) -> tuple[DiT, dict]:
    config = json.loads((weights_dir / CONFIG_FILE).read_text())
    model = DiT(ModelConfig(**config["model"]))
    model.load_state_dict(load_file(weights_dir / MODEL_FILE))
    return model.to(device).eval(), config


class PageTranslator:
    def __init__(self, weights_dir: str, vae: str | None = None, device: str = "cuda") -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA GPU is required (the VAE encoder runs under CUDA autocast)")
        self.device = torch.device(device)
        model, self.config = load_model(Path(weights_dir), self.device)
        self.model = TiledModel(model, tile=self.config["tile"], stride=self.config["tile_stride"])
        vae_cfg = self.config["vae"]
        self.codec = LatentCodec(
            vae or vae_cfg["repo"], device=str(self.device), subfolder=None if vae else vae_cfg["subfolder"]
        )

    @torch.no_grad()
    def translate(
        self,
        page: np.ndarray,
        regions: list[Region],
        steps: int = 30,
        seed: int = 0,
        margin: int = DEFAULT_MARGIN,
        full_page: bool = False,
    ) -> np.ndarray:
        """`page` is an RGB uint8 image already at the working resolution; returns an image of the same size."""
        height, width = page.shape[:2]
        padded = pad_edges(page)
        source_px = torch.from_numpy(padded).permute(2, 0, 1)[None].float() / 127.5 - 1.0
        source = self.codec.encode(source_px)
        glyph = pad_edges(render_glyphs(to_lines(regions), height, width))
        extra = F.pixel_unshuffle(torch.from_numpy(glyph).float()[None, None] / 255.0, LATENT_STRIDE).to(self.device)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        latent = euler_sample(
            self.model, source, steps, generator, extra,
            self.config["start_from_source"], self.config["start_noise"],
        )
        decoded = self.codec.decode(latent)[0, :, :height, :width]
        output = ((decoded.permute(1, 2, 0) + 1) * 127.5).round().clamp(0, 255).byte().cpu().numpy()
        if full_page:
            return output
        alpha = paste_mask(regions, height, width, margin)[..., None]
        blended = alpha * output.astype(np.float32) + (1 - alpha) * page.astype(np.float32)
        return blended.round().astype(np.uint8)
