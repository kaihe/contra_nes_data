"""Reconstruction and entity baseline for the published 512-D frame token."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from datahouse.encoder import EntityFrameEncoder
from datahouse.projectile_probe import _average_precision
from datahouse.frame_training import (ConvBlock, ENTITY_NAMES, FRAME_HW, frame_loader,
                                      frame_loss, learning_rate, prepare_targets,
                                      wsd_learning_rate)


class OneTokenDecoder(nn.Module):
    """Decode the frozen 512-D production token to a native 224x240 RGB frame."""

    def __init__(self, token_dim: int = 512):
        super().__init__()
        self.seed = nn.Linear(token_dim, 256 * 2 * 2)
        self.blocks = nn.ModuleList([
            ConvBlock(256, 192), ConvBlock(192, 128), ConvBlock(128, 96),
            ConvBlock(96, 64), ConvBlock(64, 32)])
        self.sizes = ((7, 8), (14, 15), (28, 30), (56, 60), (112, 120))
        self.output = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        x = self.seed(token.float()).view(len(token), 256, 2, 2)
        for size, block in zip(self.sizes, self.blocks):
            x = block(F.interpolate(x, size=size, mode="bilinear", align_corners=False))
        x = F.interpolate(x, size=FRAME_HW, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.output(x))


class OneTokenAutoencoder(nn.Module):
    """Native-resolution one-token encoder initialized and trained from scratch."""

    def __init__(self, config: dict, *, token_dim: int | None = None):
        super().__init__()
        config = dict(config)
        if token_dim is not None:
            config["hiddim"] = token_dim
        config.update({"input_height": FRAME_HW[0], "input_width": FRAME_HW[1],
                       "n_layers": 6})
        config["entity_classes"] = 3
        self.encoder = EntityFrameEncoder(config)
        self.decoder = OneTokenDecoder(int(config["hiddim"]))

    def forward(self, images: torch.Tensor):
        token = self.encoder.encode(_encoder_input(images, self.encoder.input_hw))
        return self.decoder(token), self.encoder.entity_head(token), token


def _encoder_input(images: torch.Tensor, input_hw: tuple[int, int] = FRAME_HW) -> torch.Tensor:
    if tuple(images.shape[2:]) != input_hw:
        raise ValueError(f"training images must have native shape {input_hw}")
    return images.mul(255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)


def _architecture_config(checkpoint_path: str) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return dict(payload["config"])


def train_decoder(corpus: str, encoder_checkpoint: str, output: str, *,
                  steps: int = 20_000, micro_batch: int = 32,
                  effective_batch: int = 128, workers: int = 4,
                  device: str = "cuda", save_every: int = 1000,
                  log_every: int = 20, schedule: str = "cosine",
                  warmup: int | None = None, decay_start: int | None = None,
                  init_from: str | None = None, token_dim: int = 512) -> None:
    if effective_batch % micro_batch:
        raise ValueError("effective batch must be divisible by micro batch")
    if schedule not in ("cosine", "wsd"):
        raise ValueError(f"unknown schedule {schedule}")
    warmup = min(2000, max(1, steps // 10)) if warmup is None else warmup
    decay_start = steps if decay_start is None else decay_start
    if schedule == "wsd" and not warmup <= decay_start <= steps:
        raise ValueError("wsd needs warmup <= decay-start <= steps")
    torch.manual_seed(0); np.random.seed(0)
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    if token_dim <= 0:
        raise ValueError("token dimension must be positive")
    model = OneTokenAutoencoder(_architecture_config(encoder_checkpoint),
                                token_dim=token_dim).to(device)
    parameters = sum(value.numel() for value in model.parameters())
    trainable_parameters = sum(value.numel() for value in model.parameters()
                               if value.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    config = {"experiment": "0010" if token_dim == 512 else "0015",
              "kind": "one-token-reconstruction-baseline",
              "steps": steps, "micro_batch": micro_batch,
              "effective_batch": effective_batch, "learning_rate": 3e-4,
              "schedule": schedule, "warmup_steps": warmup,
              "decay_start_step": decay_start if schedule == "wsd" else warmup,
              "initialized_from": init_from, "corpus": str(corpus), "seed": 0,
              "architecture_source": encoder_checkpoint,
              "token_dim": token_dim, "parameters": parameters,
              "trainable_parameters": trainable_parameters,
              "encoder_initialization": "random", "decoder_output_hw": list(FRAME_HW),
              "encoder_input_hw": list(FRAME_HW), "resize": None,
              "loss": "weighted_mse+bce+soft_dice"}
    (root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    latest = root / "latest.pt"; start_step = 0
    # This run's own checkpoint wins, so a branch is itself resumable; `init_from`
    # only seeds a fresh directory, carrying the step counter so the branch decays
    # over the window its schedule declares rather than restarting at zero.
    source = latest if latest.exists() else Path(init_from) if init_from else None
    if source is not None:
        checkpoint = torch.load(source, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
        print(json.dumps({"resumed_from": str(source), "start_step": start_step}),
              flush=True)
    loader = frame_loader(corpus, "train", batch=micro_batch, workers=workers)
    iterator = iter(loader); accumulation = effective_batch // micro_batch
    metrics_path = root / "metrics.jsonl"; started = time.time()
    model.train()
    for step in range(start_step, steps):
        lr = wsd_learning_rate(step, base=3e-4, warmup=warmup,
                               decay_start=decay_start, total=steps) \
            if schedule == "wsd" else \
            learning_rate(step, base=3e-4, warmup=warmup, total=steps)
        for group in optimizer.param_groups: group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        sums = {name: 0.0 for name in ("loss", "pixel", "bce", "dice")}
        for _ in range(accumulation):
            raw, supervision, _keys = next(iterator)
            images = raw.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
            targets, weights = prepare_targets(supervision, device=images.device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.startswith("cuda")):
                reconstruction, logits, _ = model(images)
                loss, parts = frame_loss(reconstruction, images, logits, targets, weights)
                scaled = loss / accumulation
            scaler.scale(scaled).backward()
            for name in sums: sums[name] += float(parts[name]) / accumulation
        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); completed = step + 1
        if completed % log_every == 0 or completed == 1:
            row = {"step": completed, "lr": lr, **sums,
                   "elapsed_seconds": time.time() - started}
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
        if completed % save_every == 0 or completed == steps:
            payload = {"step": completed, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                       "config": config}
            for path in (root / f"checkpoint-{completed:06d}.pt", latest):
                temporary = path.with_suffix(path.suffix + ".tmp")
                torch.save(payload, temporary); os.replace(temporary, path)


class BaselineMetrics:
    def __init__(self):
        self.frames = self.channels = self.exact_channels = self.exact_pixels = 0
        self.squared_error = self.weighted_squared_error = self.weight_sum = 0.0
        self.dice_sum = np.zeros(3)
        self.presence = {name: [] for name in ENTITY_NAMES}
        self.scores = {name: [] for name in ENTITY_NAMES}

    def update(self, images, reconstruction, entity_probability, targets, weights,
               presence) -> None:
        batch = len(images); prediction = reconstruction.mul(255).round().clamp(
            0, 255).to(torch.uint8); truth = images.mul(255).round().to(torch.uint8)
        equal = prediction == truth
        self.frames += batch; self.channels += truth.numel()
        self.exact_channels += int(equal.sum()); self.exact_pixels += int(equal.all(1).sum())
        error = (reconstruction.float() - images.float()).square()
        self.squared_error += float(error.sum())
        self.weighted_squared_error += float((error * weights).sum())
        self.weight_sum += float(weights.sum()) * images.shape[1]
        numerator = 2 * (entity_probability * targets).sum((2, 3)) + 1e-6
        denominator = (entity_probability.sum((2, 3)) + targets.sum((2, 3)) + 1e-6)
        self.dice_sum += (numerator / denominator).sum(0).cpu().numpy()
        presence = np.asarray(presence, dtype=bool)
        if presence.shape != (batch, len(ENTITY_NAMES)):
            raise ValueError(f"presence must have shape {(batch, len(ENTITY_NAMES))}")
        scores = entity_probability.amax((2, 3)).cpu().numpy()
        for index, name in enumerate(ENTITY_NAMES):
            self.presence[name].extend(presence[:, index].tolist())
            self.scores[name].extend(scores[:, index].tolist())

    def result(self) -> dict:
        mse = self.squared_error / self.channels
        result = {"frames": self.frames,
                "exact_rgb_channel_accuracy": self.exact_channels / self.channels,
                "exact_rgb_pixel_accuracy": self.exact_pixels / (self.channels // 3),
                "unweighted_mse": mse,
                "weighted_mse": self.weighted_squared_error / self.weight_sum,
                "psnr_db": float("inf") if mse == 0 else -10 * math.log10(mse),
                "soft_dice": {name: float(value / self.frames) for name, value in zip(
                    ENTITY_NAMES, self.dice_sum)}}
        for name in ENTITY_NAMES:
            presence = np.asarray(self.presence[name], dtype=bool)
            scores = np.asarray(self.scores[name])
            empty = ~presence
            average_precision = _average_precision(presence, scores)
            result.update({
                f"{name}_presence_ap": (
                    average_precision if math.isfinite(average_precision) else None),
                f"{name}_positive_frames": int(presence.sum()),
                f"{name}_empty_frames": int(empty.sum()),
                f"{name}_empty_fpr_0.5": (
                    float((scores[empty] >= 0.5).mean()) if empty.any() else None),
            })
        return result


@torch.no_grad()
def evaluate(corpus: str, encoder_checkpoint: str, decoder_checkpoint: str,
             output: str, *, split: str = "validation", limit: int | None = None,
             batch: int = 32, workers: int = 4, device: str = "cuda") -> dict:
    checkpoint = torch.load(decoder_checkpoint, map_location=device, weights_only=False)
    run_config = checkpoint.get("config", {})
    token_dim = int(run_config.get("token_dim", 512))
    model = OneTokenAutoencoder(_architecture_config(encoder_checkpoint),
                                token_dim=token_dim).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    loader = frame_loader(corpus, split, batch=batch, workers=workers)
    metrics = BaselineMetrics()
    for raw, supervision, _keys in loader:
        if limit is not None and metrics.frames >= limit: break
        if limit is not None and metrics.frames + len(raw) > limit:
            take = limit - metrics.frames
            raw = raw[:take]
            supervision = supervision[:take]
        images = raw.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        targets, weights = prepare_targets(supervision, device=images.device)
        if isinstance(supervision, torch.Tensor):
            presence = supervision.amax((2, 3)) > 0
        else:
            presence = [[bool(row[name]) for name in ENTITY_NAMES]
                        for row in supervision]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.startswith("cuda")):
            reconstruction, logits, _ = model(images)
        probability = logits.float().sigmoid()
        metrics.update(images, reconstruction, probability, targets, weights, presence)
    result = {"schema_version": 2,
              "experiment": run_config.get("experiment", "0010"), "split": split,
              "architecture_source": encoder_checkpoint,
              "decoder_checkpoint": decoder_checkpoint, "token_dim": token_dim,
              **metrics.result()}
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    evaluate_parser = sub.add_parser("evaluate")
    for item in (train, evaluate_parser):
        item.add_argument("--corpus", default="tmp/0012-vq-codebook/corpus-1k-all")
        item.add_argument("--encoder", default=(
            "game_trace/datahouse/encoder/f36041bc69f1ce20781d5200bc89970b1b305e12bff5ae826b23581ca0f1923c/encoder.pt"))
        item.add_argument("--workers", type=int, default=4)
        item.add_argument("--device", default="cuda")
    train.add_argument("--output", default="runs/encoder-baseline/one-token-reconstruction")
    train.add_argument("--steps", type=int, default=20_000)
    train.add_argument("--micro-batch", type=int, default=32)
    train.add_argument("--effective-batch", type=int, default=128)
    train.add_argument("--save-every", type=int, default=1000)
    train.add_argument("--log-every", type=int, default=20)
    train.add_argument("--schedule", choices=("cosine", "wsd"), default="cosine")
    train.add_argument("--warmup", type=int,
                       help="default: 10%% of steps, capped at 2000")
    train.add_argument("--decay-start", type=int,
                       help="wsd: first decaying step; default steps (stable only)")
    train.add_argument("--init-from",
                       help="seed weights, optimizer and step counter from a checkpoint")
    train.add_argument("--token-dim", type=int, default=512)
    evaluate_parser.add_argument("--decoder", default=(
        "runs/encoder-baseline/one-token-reconstruction/checkpoint-020000.pt"))
    evaluate_parser.add_argument("--output", default=(
        "runs/encoder-baseline/one-token-reconstruction/validation.json"))
    evaluate_parser.add_argument("--split", default="validation")
    evaluate_parser.add_argument("--limit", type=int)
    evaluate_parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    if args.command == "train":
        train_decoder(args.corpus, args.encoder, args.output, steps=args.steps,
                      micro_batch=args.micro_batch, effective_batch=args.effective_batch,
                      workers=args.workers, device=args.device,
                      save_every=args.save_every, log_every=args.log_every,
                      schedule=args.schedule, warmup=args.warmup,
                      decay_start=args.decay_start, init_from=args.init_from,
                      token_dim=args.token_dim)
    else:
        evaluate(args.corpus, args.encoder, args.decoder, args.output, split=args.split,
                 limit=args.limit, batch=args.batch, workers=args.workers,
                 device=args.device)


if __name__ == "__main__":
    main()
