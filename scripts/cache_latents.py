"""Encode cleaned page pairs to VAE latents once, so training never touches the VAE.

Usage: PYTHONPATH=/workspace/mtdiff python scripts/cache_latents.py --proc /workspace/data/proc --out /workspace/data/latents
Each output file holds: src, tgt (float16, C x h x w) and mask (uint8, h x w) at latent resolution.
"""
import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mtdiff.vae import LATENT_STRIDE, LatentCodec

PAD_MULTIPLE = 16  # keeps latent sides even for 2x2 patches
DEFAULT_VAE = "/model/HuggingFace/black-forest-labs/FLUX.1-dev/vae"
LOG = logging.getLogger("cache_latents")


def pad_edges(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    pads = ((0, (-height) % PAD_MULTIPLE), (0, (-width) % PAD_MULTIPLE)) + ((0, 0),) * (image.ndim - 2)
    return np.pad(image, pads, mode="edge")


def load_image_tensor(path: Path) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = pad_edges(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 127.5 - 1.0


def load_latent_mask(path: Path) -> torch.Tensor:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    tensor = torch.from_numpy(pad_edges(mask) > 0)[None, None].float()
    return F.max_pool2d(tensor, LATENT_STRIDE)[0, 0].to(torch.uint8)


def encode_pair(codec: LatentCodec, proc: Path, name: str) -> dict[str, torch.Tensor]:
    return {
        "src": codec.encode(load_image_tensor(proc / "src" / f"{name}.png"))[0].half().cpu(),
        "tgt": codec.encode(load_image_tensor(proc / "tgt" / f"{name}.png"))[0].half().cpu(),
        "mask": load_latent_mask(proc / "mask" / f"{name}.png"),
    }


def encode_pair_v2(codec: LatentCodec, proc: Path, name: str, reuse: Path, base: str) -> dict[str, torch.Tensor]:
    """v2 entry: composite target, OCR-filtered mask, full-resolution glyph.

    The source latent is reused from the cache entry `base` (the page itself, or the real page a
    synthetic variant was derived from).
    """
    glyph = cv2.imread(str(proc / "glyph" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    if glyph is None:
        raise FileNotFoundError(proc / "glyph" / f"{name}.png")
    return {
        "src": torch.load(reuse / f"{base}.pt", weights_only=True)["src"],
        "tgt": codec.encode(load_image_tensor(proc / "tgt2" / f"{name}.png"))[0].half().cpu(),
        "mask": load_latent_mask(proc / "mask2" / f"{name}.png"),
        "glyph": torch.from_numpy(np.pad(glyph, [(0, (-s) % PAD_MULTIPLE) for s in glyph.shape])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vae", default=DEFAULT_VAE)
    parser.add_argument("--manifest", default="manifest.jsonl", help="manifest file name inside --proc")
    parser.add_argument("--reuse-src", type=Path, help="v1 cache dir; enables v2 mode (tgt2, mask2, glyph)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rows = [json.loads(line) for line in (args.proc / args.manifest).read_text().splitlines() if line.strip()]
    names = [r["name"] for r in rows if r["ok"]]
    bases = {r["name"]: r.get("base", r["name"]) for r in rows}
    args.out.mkdir(parents=True, exist_ok=True)
    codec = LatentCodec(args.vae)

    failures = 0
    for index, name in enumerate(names):
        target = args.out / f"{name}.pt"
        if target.exists():
            continue
        try:
            entry = (
                encode_pair_v2(codec, args.proc, name, args.reuse_src, bases[name])
                if args.reuse_src
                else encode_pair(codec, args.proc, name)
            )
            torch.save(entry, target)
        except Exception:
            failures += 1
            LOG.exception("failed on %s", name)
        if index % 250 == 0:
            LOG.info("%d / %d", index, len(names))
    LOG.info("done: %d pairs, %d failures", len(names), failures)


if __name__ == "__main__":
    main()
