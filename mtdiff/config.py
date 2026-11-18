"""Immutable configuration objects for the manga-translation DiT."""
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    latent_channels: int = 16
    extra_channels: int = 64  # glyph image, pixel-unshuffled by the latent stride (8 * 8)
    patch: int = 2
    dim: int = 768
    depth: int = 16
    heads: int = 12
    mlp_ratio: float = 4.0


@dataclass(frozen=True)
class TrainConfig:
    cache_dir: str = "/workspace/data/latents_v2"
    manifest: str = "/workspace/data/proc/manifest_v3.jsonl"  # real pairs plus synthetic text variants
    run_dir: str = "/workspace/runs/v3"
    init_from: str = ""  # checkpoint whose EMA weights initialise the model (fresh optimizer and step count)
    synthetic_fraction: float = 0.5  # share of training samples drawn from synthetic rows
    vae_path: str = "/model/HuggingFace/black-forest-labs/FLUX.1-dev/vae"
    domains: tuple[str, ...] = ("manga",)
    crop: int = 64  # latent units; 64 -> 512 px
    text_center_prob: float = 0.8
    batch_size: int = 32
    workers: int = 12
    steps: int = 100_000
    lr: float = 1e-4
    warmup_steps: int = 2_000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    mask_weight: float = 5.0  # extra loss weight on changed (text) regions
    use_glyph: bool = True  # feed the rendered target text as extra input channels
    start_from_source: bool = True  # flow starts at source + noise instead of pure noise
    start_noise: float = 0.5  # noise scale at the start point when starting from the source
    ema_decay: float = 0.9999
    log_every: int = 100
    eval_every: int = 2_500
    save_every: int = 5_000
    eval_items: int = 8
    eval_loss_batches: int = 4
    sample_steps: int = 30
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict:
        return asdict(self)
