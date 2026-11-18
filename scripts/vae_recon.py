"""Compare off-the-shelf VAEs on manga pages: whole-page and text-region reconstruction.

Usage: python vae_recon.py --root /workspace/data/COMIC --out /workspace/out/vae --n 120
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from pair_utils import list_pair_names, load_pair, match_size, psnr, text_change_mask, to_gray

VAE_PATHS = {
    "sdxl_4ch": "/model/HuggingFace/stabilityai/sdxl-vae",
    "flux_16ch": "/model/HuggingFace/black-forest-labs/FLUX.1-dev/vae",
}
QWEN_VAE_PATH = "/model/ModelScope/Qwen/Qwen-Image-2512/vae"
MULTIPLE = 16
CROP = 256
N_CROPS_SAVED = 12


def load_vaes(device: str) -> dict:
    from diffusers import AutoencoderKL

    vaes = {}
    for label, path in VAE_PATHS.items():
        if Path(path).exists():
            vaes[label] = AutoencoderKL.from_pretrained(path, torch_dtype=torch.float32).to(device).eval()
        else:
            print(f"skip {label}: {path} not found")
    if Path(QWEN_VAE_PATH).exists():
        try:
            from diffusers import AutoencoderKLQwenImage

            vaes["qwen_16ch"] = (
                AutoencoderKLQwenImage.from_pretrained(QWEN_VAE_PATH, torch_dtype=torch.float32).to(device).eval()
            )
        except Exception as exc:
            print(f"skip qwen_16ch: {exc!r}")
    return vaes


def pad_to_multiple(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    pad_h, pad_w = (-height) % MULTIPLE, (-width) % MULTIPLE
    return np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


@torch.no_grad()
def reconstruct(vae, rgb: np.ndarray, device: str) -> np.ndarray:
    height, width = rgb.shape[:2]
    padded = pad_to_multiple(rgb)
    x = torch.from_numpy(padded).to(device).permute(2, 0, 1)[None].float() / 127.5 - 1.0
    is_video_vae = vae.__class__.__name__ == "AutoencoderKLQwenImage"
    if is_video_vae:
        x = x[:, :, None]
    latent = vae.encode(x).latent_dist.mode()
    y = vae.decode(latent).sample
    if is_video_vae:
        y = y[:, :, 0]
    out = ((y[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
    return out[:height, :width]


def largest_region_crop(mask: np.ndarray) -> tuple[int, int] | None:
    count, _, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    if count <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = centroids[idx]
    height, width = mask.shape
    x0 = int(np.clip(cx - CROP // 2, 0, max(0, width - CROP)))
    y0 = int(np.clip(cy - CROP // 2, 0, max(0, height - CROP)))
    return x0, y0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=120)
    args = parser.parse_args()

    device = "cuda"
    crop_dir = args.out / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    vaes = load_vaes(device)
    if not vaes:
        raise SystemExit("no VAE could be loaded")

    names = list_pair_names(args.root)
    picked = names[:: max(1, len(names) // args.n)][: args.n]
    scores = {label: {"page": [], "text": []} for label in vaes}
    saved = 0

    for name in picked:
        src, tgt = load_pair(args.root, name)
        mask = text_change_mask(to_gray(match_size(src, tgt)), to_gray(tgt)) > 0
        recons = {label: reconstruct(vae, tgt, device) for label, vae in vaes.items()}
        for label, recon in recons.items():
            scores[label]["page"].append(psnr(tgt, recon))
            scores[label]["text"].append(psnr(tgt, recon, np.repeat(mask[..., None], 3, axis=2)))
        origin = largest_region_crop(mask.astype(np.uint8))
        if origin is not None and saved < N_CROPS_SAVED:
            x0, y0 = origin
            tiles = [tgt[y0 : y0 + CROP, x0 : x0 + CROP]]
            tiles += [recons[label][y0 : y0 + CROP, x0 : x0 + CROP] for label in vaes]
            panel = cv2.resize(np.concatenate(tiles, axis=1), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(crop_dir / f"{Path(name).stem}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            saved += 1

    summary = {
        "pages": len(picked),
        "crop_columns": ["original", *vaes.keys()],
        "psnr": {
            label: {
                "page_mean": round(float(np.mean(v["page"])), 2),
                "text_region_mean": round(float(np.nanmean(v["text"])), 2),
                "text_region_p10": round(float(np.nanpercentile(v["text"], 10)), 2),
            }
            for label, v in scores.items()
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
