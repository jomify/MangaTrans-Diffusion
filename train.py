"""Train the manga-translation DiT with rectified flow on cached latents.

Usage: python train.py --run-dir /workspace/runs/baseline [--steps N] [--batch-size N] [--resume]
"""
import argparse
import copy
import dataclasses
import json
import logging
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from mtdiff.config import TrainConfig
from mtdiff.data import LatentPairDataset, read_manifest, sampling_weights
from mtdiff.evaluation import evaluate
from mtdiff.flow import make_noisy, sample_timesteps, start_point, velocity_target, weighted_flow_loss
from mtdiff.model import DiT, count_parameters

LOG = logging.getLogger("train")
EMA_WARMUP = 10.0


def parse_config() -> tuple[TrainConfig, bool, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir")
    parser.add_argument("--manifest")
    parser.add_argument("--cache-dir")
    parser.add_argument("--vae-path")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--init-from")
    parser.add_argument("--synthetic-fraction", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    args = vars(parser.parse_args())
    resume = args.pop("resume")
    compile_model = not args.pop("no_compile")
    overrides = {k: (tuple(v) if k == "domains" else v) for k, v in args.items() if v is not None}
    return dataclasses.replace(TrainConfig(), **overrides), resume, compile_model


def infinite(loader: DataLoader):
    while True:
        yield from loader


def lr_at(step: int, cfg: TrainConfig) -> float:
    return cfg.lr * min(1.0, (step + 1) / cfg.warmup_steps)


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, step: int, max_decay: float) -> None:
    decay = min(max_decay, (1 + step) / (EMA_WARMUP + step))
    for ema_param, param in zip(ema.parameters(), model.parameters()):
        ema_param.lerp_(param.detach(), 1 - decay)


def training_loss(model: nn.Module, batch: dict[str, Tensor], cfg: TrainConfig) -> Tensor:
    target = batch["target"]
    start = start_point(batch["source"], torch.randn_like(target), cfg.start_from_source, cfg.start_noise)
    t = sample_timesteps(target.shape[0], target.device)
    extra = batch["glyph"] if cfg.use_glyph else None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(make_noisy(target, start, t), batch["source"], t, extra)
    return weighted_flow_loss(pred, velocity_target(target, start), batch["mask"], cfg.mask_weight)


def save_checkpoint(path: Path, model: nn.Module, ema: nn.Module, optimizer, step: int, cfg: TrainConfig) -> None:
    state = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": cfg.to_dict(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader, LatentPairDataset]:
    train_rows = read_manifest(cfg.manifest, "train", cfg.domains)
    val_rows = read_manifest(cfg.manifest, "val", cfg.domains)
    train_set = LatentPairDataset(cfg.cache_dir, train_rows, cfg.crop, cfg.text_center_prob)
    val_set = LatentPairDataset(cfg.cache_dir, val_rows, cfg.crop, text_center_prob=1.0, deterministic=True)
    LOG.info("train pages: %d, val pages: %d", len(train_set), len(val_set))
    synthetic = sum(bool(r.get("synthetic")) for r in train_set.rows)
    LOG.info("synthetic train pages: %d, drawn with probability %.2f", synthetic, cfg.synthetic_fraction if synthetic else 0)
    weights = sampling_weights(train_set.rows, cfg.synthetic_fraction)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    loader = DataLoader(
        train_set, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.workers,
        drop_last=True, pin_memory=True, persistent_workers=True,
    )
    return loader, val_set


def main() -> None:
    cfg, resume, compile_model = parse_config()
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log")],
    )
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    loader, val_set = build_loaders(cfg)
    model = DiT(cfg.model).to(device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.99), weight_decay=cfg.weight_decay)
    LOG.info("parameters: %.1fM", count_parameters(model) / 1e6)

    step = 0
    latest = run_dir / "latest.pt"
    if resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        LOG.info("resumed from step %d", step)
    elif cfg.init_from:
        weights = torch.load(cfg.init_from, map_location=device, weights_only=True)["ema"]
        model.load_state_dict(weights)
        ema.load_state_dict(weights)
        LOG.info("initialised from %s", cfg.init_from)

    # Compiling roughly doubles training throughput; checkpoints always store the uncompiled weights.
    train_model = torch.compile(model) if compile_model else model
    writer = SummaryWriter(str(run_dir / "tb"))
    batches = infinite(loader)
    running, window_start = 0.0, time.time()
    while step < cfg.steps:
        batch = {k: v.to(device, non_blocking=True) for k, v in next(batches).items()}
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, cfg)
        loss = training_loss(train_model, batch, cfg)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        update_ema(ema, model, step, cfg.ema_decay)
        step += 1
        running += loss.item()

        if step % cfg.log_every == 0:
            rate = cfg.log_every / (time.time() - window_start)
            mean_loss = running / cfg.log_every
            LOG.info("step %d loss %.4f grad %.2f lr %.2e %.2f it/s", step, mean_loss, grad_norm, lr_at(step, cfg), rate)
            writer.add_scalar("train/loss", mean_loss, step)
            writer.add_scalar("train/grad_norm", float(grad_norm), step)
            running, window_start = 0.0, time.time()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            metrics = evaluate(ema, val_set, cfg, run_dir / "samples" / f"{step:07d}.jpg", device)
            LOG.info("eval step %d %s", step, json.dumps(metrics))
            for key, value in metrics.items():
                writer.add_scalar(f"eval/{key}", value, step)
            window_start = time.time()
        if step % cfg.save_every == 0 or step == cfg.steps:
            save_checkpoint(latest, model, ema, optimizer, step, cfg)
    writer.close()


if __name__ == "__main__":
    main()
