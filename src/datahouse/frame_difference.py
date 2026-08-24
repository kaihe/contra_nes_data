"""Train and evaluate the 0019 signed-frame-difference one-token encoder."""

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
from torch.utils.data import DataLoader

from datahouse.compressed_loader import CompressedFramePairDataset
from datahouse.frame_training import (ENTITY_NAMES, frame_loss, prepare_targets,
                                      wsd_learning_rate)
from datahouse.one_token_baseline import (BaselineMetrics, OneTokenAutoencoder,
                                          _architecture_config)


PARAMETERS = 17_667_078


def collate_pairs(rows):
    previous = torch.from_numpy(np.stack([row[0] for row in rows]))
    current = torch.from_numpy(np.stack([row[1] for row in rows]))
    metadata = [row[2] for row in rows]
    return previous, current, metadata, [row["key"] for row in metadata]


def pair_loader(corpus: str | Path, split: str, *, batch: int, workers: int,
                seed: int = 0) -> DataLoader:
    dataset = CompressedFramePairDataset(corpus, split, seed=seed)
    return DataLoader(dataset, batch_size=batch, num_workers=workers,
                      collate_fn=collate_pairs, pin_memory=True,
                      persistent_workers=workers > 0,
                      prefetch_factor=2 if workers else None)


class FrameDifferenceAutoencoder(OneTokenAutoencoder):
    """0010 model with current RGB and signed temporal delta fused at conv 1."""

    def __init__(self, config: dict):
        super().__init__(config, token_dim=512)
        old = self.encoder.view_backbone.convs[0]
        expanded = nn.Conv2d(6, old.out_channels, old.kernel_size,
                             stride=old.stride, padding=old.padding,
                             dilation=old.dilation, groups=old.groups,
                             bias=old.bias is not None, padding_mode=old.padding_mode)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :3].copy_(old.weight)
            if old.bias is not None:
                expanded.bias.copy_(old.bias)
        self.encoder.view_backbone.convs[0] = expanded
        parameters = sum(value.numel() for value in self.parameters())
        production = (int(config["depth"]) == 32 and int(config["proj_ch"]) == 256
                      and int(config.get("hiddim", 512)) == 512
                      and int(config["head_depth"]) == 32)
        if production and parameters != PARAMETERS:
            raise RuntimeError(f"0019 parameter contract changed: {parameters}")

    def encode_pair(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        if current.shape != previous.shape or current.ndim != 4 or current.shape[1] != 3:
            raise ValueError("current and previous must be equal-shape BCHW RGB tensors")
        if tuple(current.shape[2:]) != self.encoder.input_hw:
            raise ValueError(f"frames must have spatial shape {self.encoder.input_hw}")
        fused = torch.cat((current, current - previous), dim=1)
        features = self.encoder.view_backbone.forward_features(fused)
        reduced = self.encoder.reduce(features)
        return self.encoder.token_ln(self.encoder.proj(reduced.flatten(1)))

    def forward(self, current: torch.Tensor, previous: torch.Tensor):
        token = self.encode_pair(current, previous)
        return self.decoder(token), self.encoder.entity_head(token), token


def _images(raw: torch.Tensor, device: str) -> torch.Tensor:
    return raw.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)


def train(corpus: str, encoder_checkpoint: str, output: str, *,
          steps: int = 20_000, micro_batch: int = 32, effective_batch: int = 128,
          workers: int = 2, device: str = "cuda", save_every: int = 1000,
          log_every: int = 20, warmup: int = 200, decay_start: int = 16_000) -> None:
    if effective_batch % micro_batch:
        raise ValueError("effective batch must be divisible by micro batch")
    if not warmup <= decay_start <= steps:
        raise ValueError("need warmup <= decay_start <= steps")
    torch.manual_seed(0)
    np.random.seed(0)
    model = FrameDifferenceAutoencoder(
        _architecture_config(encoder_checkpoint)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "0019", "kind": "frame-difference-one-token",
        "corpus": corpus, "architecture_source": encoder_checkpoint,
        "steps": steps, "micro_batch": micro_batch,
        "effective_batch": effective_batch, "learning_rate": 3e-4,
        "schedule": "wsd", "warmup_steps": warmup,
        "decay_start_step": decay_start, "seed": 0, "token_dim": 512,
        "input_channels": 6, "parameters": PARAMETERS,
        "trainable_parameters": PARAMETERS,
        "delta": "current_rgb_minus_previous_rgb_float",
        "first_frame_delta": "zero", "loss": "weighted_mse+bce+soft_dice",
    }
    (root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    latest = root / "latest.pt"
    start_step = 0
    if latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
        print(json.dumps({"resumed_from": str(latest), "start_step": start_step}),
              flush=True)
    loader = pair_loader(corpus, "train", batch=micro_batch, workers=workers)
    iterator = iter(loader)
    accumulation = effective_batch // micro_batch
    metrics_path = root / "metrics.jsonl"
    started = time.time()
    model.train()
    for step in range(start_step, steps):
        lr = wsd_learning_rate(step, base=3e-4, warmup=warmup,
                               decay_start=decay_start, total=steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        sums = {name: 0.0 for name in ("loss", "pixel", "bce", "dice")}
        for _ in range(accumulation):
            previous_raw, current_raw, supervision, _keys = next(iterator)
            previous, current = _images(previous_raw, device), _images(current_raw, device)
            targets, weights = prepare_targets(supervision, device=current.device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.startswith("cuda")):
                reconstruction, logits, _ = model(current, previous)
                loss, parts = frame_loss(reconstruction, current, logits, targets, weights)
                scaled = loss / accumulation
            scaler.scale(scaled).backward()
            for name in sums:
                sums[name] += float(parts[name]) / accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        completed = step + 1
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
                torch.save(payload, temporary)
                os.replace(temporary, path)


class FrameDifferenceMetrics(BaselineMetrics):
    def __init__(self):
        super().__init__()
        self.projectile_localization = []

    def update_localization(self, probability: torch.Tensor, metadata: list[dict]) -> None:
        projectile = probability[:, 2]
        width_scale, height_scale = 240 / 32, 224 / 32
        for heatmap, row in zip(projectile, metadata):
            points = row["projectile"]
            if not points:
                continue
            flat = int(heatmap.argmax())
            predicted_x = (flat % 32 + .5) * width_scale
            predicted_y = (flat // 32 + .5) * height_scale
            distance = min(math.hypot(predicted_x - x, predicted_y - y)
                           for x, y in points)
            self.projectile_localization.append(distance)


@torch.no_grad()
def evaluate(corpus: str, encoder_checkpoint: str, checkpoint_path: str,
             output: str, *, split: str = "validation", batch: int = 32,
             workers: int = 2, device: str = "cuda", limit: int | None = None) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FrameDifferenceAutoencoder(
        _architecture_config(encoder_checkpoint)).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    metrics = FrameDifferenceMetrics()
    for previous_raw, current_raw, supervision, _keys in pair_loader(
            corpus, split, batch=batch, workers=workers):
        if limit is not None and metrics.frames >= limit:
            break
        if limit is not None and metrics.frames + len(current_raw) > limit:
            take = limit - metrics.frames
            previous_raw, current_raw = previous_raw[:take], current_raw[:take]
            supervision = supervision[:take]
        previous, current = _images(previous_raw, device), _images(current_raw, device)
        targets, weights = prepare_targets(supervision, device=current.device)
        presence = [[bool(row[name]) for name in ENTITY_NAMES] for row in supervision]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.startswith("cuda")):
            reconstruction, logits, _ = model(current, previous)
        probability = logits.float().sigmoid()
        metrics.update(current, reconstruction, probability, targets, weights, presence)
        metrics.update_localization(probability, supervision)
    distances = np.asarray(metrics.projectile_localization)
    result = {"schema_version": 1, "experiment": "0019", "split": split,
              "architecture_source": encoder_checkpoint,
              "checkpoint": checkpoint_path, **metrics.result(),
              "projectile_localization_positive_frames": len(distances),
              "projectile_peak_error_pixels_mean": float(distances.mean()),
              "projectile_peak_error_pixels_median": float(np.median(distances)),
              "projectile_peak_error_pixels_p90": float(np.quantile(distances, .9))}
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", default="tmp/0010-one-token-compressed-1k")
    common.add_argument("--encoder", default=(
        "game_trace/datahouse/encoder/f36041bc69f1ce20781d5200bc89970b1b305e12bff5ae826b23581ca0f1923c/encoder.pt"))
    common.add_argument("--workers", type=int, default=2)
    common.add_argument("--device", default="cuda")
    train_parser = sub.add_parser("train", parents=[common])
    train_parser.add_argument("--output", default="runs/encoder-motion/frame-difference-wsd-20000")
    train_parser.add_argument("--steps", type=int, default=20_000)
    train_parser.add_argument("--micro-batch", type=int, default=32)
    train_parser.add_argument("--effective-batch", type=int, default=128)
    train_parser.add_argument("--save-every", type=int, default=1000)
    train_parser.add_argument("--log-every", type=int, default=20)
    train_parser.add_argument("--warmup", type=int, default=200)
    train_parser.add_argument("--decay-start", type=int, default=16_000)
    eval_parser = sub.add_parser("evaluate", parents=[common])
    eval_parser.add_argument("--checkpoint", default=(
        "runs/encoder-motion/frame-difference-wsd-20000/latest.pt"))
    eval_parser.add_argument("--output", default=(
        "runs/encoder-motion/frame-difference-wsd-20000/validation.json"))
    eval_parser.add_argument("--split", default="validation")
    eval_parser.add_argument("--batch", type=int, default=32)
    eval_parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "train":
        train(args.corpus, args.encoder, args.output, steps=args.steps,
              micro_batch=args.micro_batch, effective_batch=args.effective_batch,
              workers=args.workers, device=args.device, save_every=args.save_every,
              log_every=args.log_every, warmup=args.warmup,
              decay_start=args.decay_start)
    else:
        evaluate(args.corpus, args.encoder, args.checkpoint, args.output,
                 split=args.split, batch=args.batch, workers=args.workers,
                 device=args.device, limit=args.limit)


if __name__ == "__main__":
    main()
