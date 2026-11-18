"""Frozen VAE wrapper that maps images in [-1, 1] to normalised latents and back."""
import torch
from torch import Tensor

LATENT_STRIDE = 8


class LatentCodec:
    def __init__(self, vae_path: str, device: str = "cuda", subfolder: str | None = None) -> None:
        from diffusers import AutoencoderKL

        self.vae = (
            AutoencoderKL.from_pretrained(vae_path, subfolder=subfolder, torch_dtype=torch.float32).to(device).eval()
        )
        self.vae.requires_grad_(False)
        self.shift = float(getattr(self.vae.config, "shift_factor", 0.0) or 0.0)
        self.scale = float(self.vae.config.scaling_factor)
        self.device = device

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        # bf16 roughly halves encode time on full pages; decoding stays in float32.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latents = self.vae.encode(images.to(self.device, torch.float32)).latent_dist.mode()
        return (latents.float() - self.shift) * self.scale

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        raw = latents.to(self.device, torch.float32) / self.scale + self.shift
        return self.vae.decode(raw).sample.clamp(-1, 1)
