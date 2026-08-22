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
from datahouse.vq_train import (ConvBlock, FRAME_HW, frame_loader, learning_rate,
                                prepare_targets, warmup_loss)


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

    def __init__(self, config: dict):
        super().__init__()
        config = dict(config)
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
                  log_every: int = 20) -> None:
    if effective_batch % micro_batch:
        raise ValueError("effective batch must be divisible by micro batch")
    torch.manual_seed(0); np.random.seed(0)
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    model = OneTokenAutoencoder(_architecture_config(encoder_checkpoint)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    config = {"experiment": "0010", "kind": "one-token-reconstruction-baseline",
              "steps": steps, "micro_batch": micro_batch,
              "effective_batch": effective_batch, "learning_rate": 3e-4,
              "warmup_steps": min(2000, steps), "seed": 0,
              "architecture_source": encoder_checkpoint,
              "encoder_initialization": "random", "decoder_output_hw": list(FRAME_HW),
              "encoder_input_hw": list(FRAME_HW), "resize": None,
              "loss": "weighted_mse+bce+soft_dice"}
    (root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    latest = root / "latest.pt"; start_step = 0
    if latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
    loader = frame_loader(corpus, "train", batch=micro_batch, workers=workers)
    iterator = iter(loader); accumulation = effective_batch // micro_batch
    metrics_path = root / "metrics.jsonl"; started = time.time()
    model.train()
    for step in range(start_step, steps):
        lr = learning_rate(step, base=3e-4, warmup=min(2000, steps), total=steps)
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
                loss, parts = warmup_loss(reconstruction, images, logits, targets, weights)
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
        self.dice_sum = np.zeros(3); self.presence = []; self.scores = []

    def update(self, images, reconstruction, entity_probability, targets, weights,
               projectile_presence) -> None:
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
        self.presence.extend(bool(value) for value in projectile_presence)
        self.scores.extend(entity_probability[:, 2].amax((1, 2)).cpu().tolist())

    def result(self) -> dict:
        mse = self.squared_error / self.channels
        presence = np.asarray(self.presence, dtype=bool); scores = np.asarray(self.scores)
        empty = ~presence
        return {"frames": self.frames,
                "exact_rgb_channel_accuracy": self.exact_channels / self.channels,
                "exact_rgb_pixel_accuracy": self.exact_pixels / (self.channels // 3),
                "unweighted_mse": mse,
                "weighted_mse": self.weighted_squared_error / self.weight_sum,
                "psnr_db": float("inf") if mse == 0 else -10 * math.log10(mse),
                "soft_dice": {name: float(value / self.frames) for name, value in zip(
                    ("player", "enemy", "projectile"), self.dice_sum)},
                "projectile_presence_ap": _average_precision(presence, scores),
                "projectile_positive_frames": int(presence.sum()),
                "projectile_empty_frames": int(empty.sum()),
                "projectile_empty_fpr_0.5": float((scores[empty] >= 0.5).mean())}


@torch.no_grad()
def evaluate(corpus: str, encoder_checkpoint: str, decoder_checkpoint: str,
             output: str, *, split: str = "validation", limit: int | None = None,
             batch: int = 32, workers: int = 4, device: str = "cuda") -> dict:
    checkpoint = torch.load(decoder_checkpoint, map_location=device, weights_only=False)
    model = OneTokenAutoencoder(_architecture_config(encoder_checkpoint)).to(device).eval()
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
            presence = supervision[:, 2].amax((1, 2)) > 0
        else:
            presence = [bool(row["projectile"]) for row in supervision]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.startswith("cuda")):
            reconstruction, logits, _ = model(images)
        probability = logits.float().sigmoid()
        metrics.update(images, reconstruction, probability, targets, weights, presence)
    result = {"schema_version": 1, "experiment": "0010", "split": split,
              "architecture_source": encoder_checkpoint,
              "decoder_checkpoint": decoder_checkpoint, **metrics.result()}
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
                      save_every=args.save_every, log_every=args.log_every)
    else:
        evaluate(args.corpus, args.encoder, args.decoder, args.output, split=args.split,
                 limit=args.limit, batch=args.batch, workers=args.workers,
                 device=args.device)


if __name__ == "__main__":
    main()
