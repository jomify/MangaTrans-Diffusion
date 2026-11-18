"""Run a trained checkpoint over whole cached pages and save the decoded images for OCR scoring.

Usage: PYTHONPATH=/workspace/mtdiff python scripts/predict_pages.py --checkpoint RUN/latest.pt --out DIR
Modes: pred (model output), target (decoded target latent, the ceiling), source (decoded source, the floor).
"""
import argparse
import logging
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from mtdiff.config import ModelConfig
from mtdiff.data import read_manifest
from mtdiff.flow import euler_sample
from mtdiff.model import DiT
from mtdiff.tiling import TiledModel
from mtdiff.vae import LATENT_STRIDE, LatentCodec

SEED = 1234
LOG = logging.getLogger("predict_pages")


def load_model(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = DiT(ModelConfig(**state["config"]["model"])).to(device)
    model.load_state_dict(state["ema"])
    return model.eval(), state["config"]


def page_latent(model: DiT, cfg: dict, item: dict, mode: str, steps: int, device: torch.device) -> torch.Tensor:
    if mode == "target":
        return item["tgt"][None].float().to(device)
    source = item["src"][None].float().to(device)
    if mode == "source":
        return source
    glyph = None
    if cfg["use_glyph"]:
        glyph = F.pixel_unshuffle(item["glyph"].float()[None, None].to(device) / 255.0, LATENT_STRIDE)
    generator = torch.Generator(device=device).manual_seed(SEED)
    return euler_sample(model, source, steps, generator, glyph, cfg["start_from_source"], cfg["start_noise"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--mode", choices=("pred", "target", "source"), default="pred")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tile", type=int, default=64, help="latent tile size; 0 runs the whole page at once")
    parser.add_argument("--tile-stride", type=int, default=48)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    device = torch.device("cuda")
    model, cfg = load_model(args.checkpoint, device)
    if args.tile:
        model = TiledModel(model, args.tile, args.tile_stride)
    codec = LatentCodec(cfg["vae_path"])
    rows = read_manifest(cfg["manifest"], args.split, tuple(cfg["domains"]))
    rows = [r for r in rows if not r.get("synthetic")]
    if args.limit:
        rows = rows[:: max(1, len(rows) // args.limit)][: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows):
        item = torch.load(Path(cfg["cache_dir"]) / f"{row['name']}.pt", weights_only=True)
        latent = page_latent(model, cfg, item, args.mode, args.steps, device)
        image = codec.decode(latent)[0, :, : row["h"], : row["w"]]
        rgb = ((image.permute(1, 2, 0) + 1) * 127.5).round().clamp(0, 255).byte().cpu().numpy()
        cv2.imwrite(str(args.out / f"{row['name']}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if index % 25 == 0:
            LOG.info("%d / %d", index, len(rows))
    LOG.info("done: %d pages", len(rows))


if __name__ == "__main__":
    main()
