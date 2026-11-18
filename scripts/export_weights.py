"""Export the EMA weights of a training checkpoint as a small inference package.

Usage: python scripts/export_weights.py --checkpoint runs/v3/latest.pt --out weights/mangatrans-v3
Writes model.safetensors (fp16) and config.json.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

VAE_REPO = "black-forest-labs/FLUX.1-dev"
VAE_SUBFOLDER = "vae"
TILE = 64
TILE_STRIDE = 48


def export(checkpoint: Path, out: Path, dtype: torch.dtype = torch.float16) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = state["config"]
    tensors = {k: v.detach().to(dtype).contiguous() for k, v in state["ema"].items()}
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out / "model.safetensors")
    config = {
        "model": cfg["model"],
        "start_from_source": cfg["start_from_source"],
        "start_noise": cfg["start_noise"],
        "use_glyph": cfg["use_glyph"],
        "vae": {"repo": VAE_REPO, "subfolder": VAE_SUBFOLDER, "stride": 8},
        "tile": TILE,
        "tile_stride": TILE_STRIDE,
        "training_step": state["step"],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = export(args.checkpoint, args.out)
    size = (args.out / "model.safetensors").stat().st_size / 1e6
    print(f"exported step {config['training_step']} -> {args.out} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
