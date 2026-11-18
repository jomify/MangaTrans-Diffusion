"""Validation: fixed-noise flow loss, sampled crops, and text-region PSNR against the copy-source baseline."""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from mtdiff.config import TrainConfig
from mtdiff.flow import euler_sample, make_noisy, start_point, velocity_target, weighted_flow_loss
from mtdiff.vae import LATENT_STRIDE, LatentCodec

EVAL_SEED = 1234
_CODECS: dict[str, LatentCodec] = {}


def get_codec(vae_path: str) -> LatentCodec:
    if vae_path not in _CODECS:
        _CODECS[vae_path] = LatentCodec(vae_path)
    return _CODECS[vae_path]


def masked_psnr(pred: Tensor, reference: Tensor, mask: Tensor) -> float:
    """PSNR over masked pixels for images in [-1, 1]; mask is (batch, 1, H, W)."""
    weight = mask.expand_as(pred)
    if weight.sum() == 0:
        return float("nan")
    mse = ((pred - reference).pow(2) * weight).sum() / weight.sum()
    return float(10 * torch.log10(4.0 / mse.clamp_min(1e-10)))


@torch.no_grad()
def validation_loss(model: nn.Module, dataset, cfg: TrainConfig, device: torch.device) -> float:
    count = min(len(dataset), cfg.batch_size * cfg.eval_loss_batches)
    loader = DataLoader(Subset(dataset, range(count)), batch_size=cfg.batch_size, num_workers=4)
    generator = torch.Generator(device=device).manual_seed(EVAL_SEED)
    total, batches = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        target = batch["target"]
        noise = torch.randn(target.shape, device=device, generator=generator)
        start = start_point(batch["source"], noise, cfg.start_from_source, cfg.start_noise)
        t = torch.rand(target.shape[0], device=device, generator=generator)
        extra = batch["glyph"] if cfg.use_glyph else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(make_noisy(target, start, t), batch["source"], t, extra)
        total += weighted_flow_loss(pred, velocity_target(target, start), batch["mask"], cfg.mask_weight).item()
        batches += 1
    return total / max(batches, 1)


def to_uint8(images: Tensor) -> np.ndarray:
    return ((images.permute(0, 2, 3, 1) + 1) * 127.5).round().clamp(0, 255).byte().cpu().numpy()


def save_grid(path: Path, columns: list[Tensor]) -> None:
    """One row per item; columns are source | (glyph) | prediction | target."""
    rows = [np.concatenate(tiles, axis=1) for tiles in zip(*(to_uint8(c) for c in columns))]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))


@torch.no_grad()
def evaluate(model: nn.Module, dataset, cfg: TrainConfig, sample_path: Path, device: torch.device) -> dict[str, float]:
    codec = get_codec(cfg.vae_path)
    step = max(1, len(dataset) // cfg.eval_items)
    items = [dataset[i] for i in range(0, len(dataset), step)][: cfg.eval_items]
    batch = {k: torch.stack([item[k] for item in items]).to(device) for k in items[0]}
    generator = torch.Generator(device=device).manual_seed(EVAL_SEED)
    extra = batch["glyph"] if cfg.use_glyph else None
    predicted = euler_sample(
        model, batch["source"], cfg.sample_steps, generator, extra, cfg.start_from_source, cfg.start_noise
    )

    source_px = codec.decode(batch["source"])
    predicted_px = codec.decode(predicted)
    target_px = codec.decode(batch["target"])
    mask_px = F.interpolate(batch["mask"], scale_factor=LATENT_STRIDE, mode="nearest")
    columns = [source_px, predicted_px, target_px]
    if extra is not None:
        glyph_px = F.pixel_shuffle(extra, LATENT_STRIDE).expand(-1, 3, -1, -1) * 2 - 1
        columns.insert(1, glyph_px)
    save_grid(sample_path, columns)
    return {
        "val_loss": validation_loss(model, dataset, cfg, device),
        "text_psnr": masked_psnr(predicted_px, target_px, mask_px),
        "text_psnr_copy_source": masked_psnr(source_px, target_px, mask_px),
        "background_psnr": masked_psnr(predicted_px, target_px, 1 - mask_px),
        "background_psnr_copy_source": masked_psnr(source_px, target_px, 1 - mask_px),
    }
