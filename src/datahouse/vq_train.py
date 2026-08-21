"""Train the continuous warmup and VQ cells declared by experiment 0012."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


FRAME_HW = (224, 240)
GRID = 32
LATENT_DIM = 256
ENTITY_NAMES = ("player", "enemy", "projectile")
SIGMAS = (6.0, 6.0, 4.0)
PIXEL_WEIGHTS = (3.0, 3.0, 15.0)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, *, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
            nn.GroupNorm(min(16, cout), cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(min(16, cout), cout), nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContinuousAutoencoder(nn.Module):
    """Native-frame autoencoder with the experiment's four 256-D latents."""

    def __init__(self):
        super().__init__()
        channels = (32, 64, 96, 128, 192, LATENT_DIM)
        blocks = []
        cin = 3
        for cout in channels:
            blocks.append(ConvBlock(cin, cout, stride=2))
            cin = cout
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((2, 2)))
        self.decoder_blocks = nn.ModuleList([
            ConvBlock(256, 192), ConvBlock(192, 128), ConvBlock(128, 96),
            ConvBlock(96, 64), ConvBlock(64, 32)])
        self.decoder_sizes = ((7, 8), (14, 15), (28, 30), (56, 60), (112, 120))
        self.reconstruction = nn.Conv2d(32, 3, 3, padding=1)
        self.entity_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.SiLU(),
            nn.Upsample(size=(8, 8), mode="bilinear", align_corners=False),
            ConvBlock(128, 64),
            nn.Upsample(size=(32, 32), mode="bilinear", align_corners=False),
            nn.Conv2d(64, 3, 1))

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        x = latent
        for size, block in zip(self.decoder_sizes, self.decoder_blocks):
            x = block(F.interpolate(x, size=size, mode="bilinear",
                                    align_corners=False))
        x = F.interpolate(x, size=FRAME_HW, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.reconstruction(x))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(images)
        return self.decode(latent), self.entity_head(latent)


class FrameTarDataset(IterableDataset):
    """Stream paired PNG/JSON samples with bounded shuffle memory."""

    def __init__(self, root: str | Path, split: str, *, seed: int = 0,
                 shuffle_buffer: int = 2048):
        self.root = Path(root)
        self.split = split
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer

    def _shards(self) -> list[Path]:
        shards = []
        for marker_path in sorted(self.root.glob("shard-*.json")):
            marker = json.loads(marker_path.read_text())
            if marker["splits"].get(self.split, 0):
                shards.append(self.root / marker["file"])
        if not shards:
            raise RuntimeError(f"no {self.split} shards under {self.root}")
        return shards

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        rng = random.Random(self.seed + worker_id)
        shards = self._shards()[worker_id::worker_count]
        buffer = []
        epoch = 0
        while True:
            rng.shuffle(shards)
            for path in shards:
                with tarfile.open(path) as archive:
                    pending: dict[str, dict] = {}
                    for member in archive:
                        if not member.isfile():
                            continue
                        key, suffix = member.name.rsplit(".", 1)
                        row = pending.setdefault(key, {})
                        payload = archive.extractfile(member).read()
                        row[suffix] = payload
                        if "png" not in row or "json" not in row:
                            continue
                        meta = json.loads(row["json"])
                        image = np.asarray(Image.open(io.BytesIO(row["png"])).convert("RGB"))
                        buffer.append((image, meta))
                        del pending[key]
                        if len(buffer) >= self.shuffle_buffer:
                            yield buffer.pop(rng.randrange(len(buffer)))
            while buffer:
                yield buffer.pop(rng.randrange(len(buffer)))
            epoch += 1
            rng.seed(self.seed + worker_id + epoch * 1009)


def collate_frames(rows):
    images = np.stack([row[0] for row in rows])
    return torch.from_numpy(images), [row[1] for row in rows]


def entity_targets(metadata: list[dict], *, device: torch.device,
                   dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Create 32-grid targets and native-resolution entity pixel weights."""
    batch = len(metadata)
    yy, xx = torch.meshgrid(torch.arange(GRID, device=device, dtype=dtype),
                            torch.arange(GRID, device=device, dtype=dtype), indexing="ij")
    targets = torch.zeros((batch, 3, GRID, GRID), device=device, dtype=dtype)
    for bi, meta in enumerate(metadata):
        for ci, (name, sigma) in enumerate(zip(ENTITY_NAMES, SIGMAS)):
            grid_sigma = sigma * GRID / FRAME_HW[0]
            for px, py in meta[name]:
                gx = float(px) * GRID / FRAME_HW[1]
                gy = float(py) * GRID / FRAME_HW[0]
                blob = torch.exp(-((xx - gx).square() + (yy - gy).square()) /
                                 (2 * grid_sigma * grid_sigma))
                targets[bi, ci] = torch.maximum(targets[bi, ci], blob)
    coarse_weights = 1.0 + sum(weight * targets[:, ci:ci + 1]
                               for ci, weight in enumerate(PIXEL_WEIGHTS))
    coarse_weights.clamp_(max=16.0)
    pixel_weights = F.interpolate(coarse_weights, size=FRAME_HW, mode="bilinear",
                                  align_corners=False)
    return targets, pixel_weights


def warmup_loss(reconstruction: torch.Tensor, images: torch.Tensor,
                entity_logits: torch.Tensor, targets: torch.Tensor,
                pixel_weights: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pixel = ((reconstruction - images).square() * pixel_weights).sum() / \
            (pixel_weights.sum() * images.shape[1])
    bce = F.binary_cross_entropy_with_logits(entity_logits, targets)
    probs = entity_logits.sigmoid()
    numerator = 2 * (probs * targets).sum(dim=(2, 3)) + 1e-6
    denominator = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + 1e-6
    dice = 1 - (numerator / denominator).mean()
    total = pixel + bce + dice
    return total, {"loss": total.detach(), "pixel": pixel.detach(),
                   "bce": bce.detach(), "dice": dice.detach()}


def learning_rate(step: int, *, base: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_warmup(corpus: str, output: str, *, steps: int = 20_000,
                 micro_batch: int = 32, effective_batch: int = 128,
                 workers: int = 4, device: str = "cuda", save_every: int = 1000,
                 log_every: int = 20) -> None:
    if effective_batch % micro_batch:
        raise ValueError("effective batch must be divisible by micro batch")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    config = {"experiment": "0012", "kind": "continuous-warmup", "steps": steps,
              "micro_batch": micro_batch, "effective_batch": effective_batch,
              "latent_shape": [256, 2, 2], "learning_rate": 3e-4,
              "warmup_steps": min(2000, steps), "seed": 0,
              "loss": "weighted_mse+bce+soft_dice", "corpus": str(corpus)}
    (output_path / "config.json").write_text(json.dumps(config, indent=2,
                                                         sort_keys=True) + "\n")
    model = ContinuousAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    start_step = 0
    latest = output_path / "latest.pt"
    if latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
        print(f"resuming at step {start_step}", flush=True)
    dataset = FrameTarDataset(corpus, "train", seed=0)
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=workers,
                        collate_fn=collate_frames, pin_memory=True,
                        persistent_workers=workers > 0, prefetch_factor=2 if workers else None)
    iterator = iter(loader)
    accumulation = effective_batch // micro_batch
    metrics_path = output_path / "metrics.jsonl"
    model.train()
    started = time.time()
    for step in range(start_step, steps):
        lr = learning_rate(step, base=3e-4, warmup=min(2000, steps), total=steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        sums = {name: 0.0 for name in ("loss", "pixel", "bce", "dice")}
        for _ in range(accumulation):
            raw_images, metadata = next(iterator)
            images = raw_images.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
            targets, weights = entity_targets(metadata, device=images.device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.startswith("cuda")):
                reconstruction, logits = model(images)
                loss, parts = warmup_loss(reconstruction, images, logits, targets, weights)
                loss = loss / accumulation
            scaler.scale(loss).backward()
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
            print(" ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in row.items()), flush=True)
        if completed % save_every == 0 or completed == steps:
            payload = {"step": completed, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                       "config": config}
            numbered = output_path / f"checkpoint-{completed:06d}.pt"
            _atomic_torch_save(payload, numbered)
            _atomic_torch_save(payload, latest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    warmup = sub.add_parser("warmup")
    warmup.add_argument("--corpus", default="tmp/0012-vq-codebook/corpus-1k-all")
    warmup.add_argument("--output", default="runs/0012-vq-codebook/warmup")
    warmup.add_argument("--steps", type=int, default=20_000)
    warmup.add_argument("--micro-batch", type=int, default=32)
    warmup.add_argument("--effective-batch", type=int, default=128)
    warmup.add_argument("--workers", type=int, default=4)
    warmup.add_argument("--device", default="cuda")
    warmup.add_argument("--save-every", type=int, default=1000)
    warmup.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.command == "warmup":
        train_warmup(args.corpus, args.output, steps=args.steps,
                     micro_batch=args.micro_batch, effective_batch=args.effective_batch,
                     workers=args.workers, device=args.device, save_every=args.save_every,
                     log_every=args.log_every)


if __name__ == "__main__":
    main()
