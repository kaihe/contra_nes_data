"""Train the continuous warmup and VQ cells declared by experiment 0012."""

from __future__ import annotations

import argparse
import hashlib
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

from datahouse.indexed_chunks import IndexedChunkDataset, collate_indexed


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


class VectorQuantizer(nn.Module):
    """Shared nearest-neighbour codebook with straight-through gradients."""

    def __init__(self, entries: int, dimension: int = LATENT_DIM):
        super().__init__()
        self.entries = entries
        self.embedding = nn.Parameter(torch.empty(entries, dimension))
        nn.init.normal_(self.embedding, std=dimension ** -0.5)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                      torch.Tensor]:
        # [B,C,2,2] -> [B*4,C], with distances evaluated in float32.
        flat = latent.permute(0, 2, 3, 1).reshape(-1, latent.shape[1]).float()
        codebook = self.embedding.float()
        distances = (flat.square().sum(1, keepdim=True)
                     + codebook.square().sum(1).unsqueeze(0)
                     - 2 * flat @ codebook.T)
        indices = distances.argmin(1)
        raw = F.embedding(indices, self.embedding).view(
            latent.shape[0], 2, 2, latent.shape[1]).permute(0, 3, 1, 2)
        straight_through = latent + (raw - latent).detach()
        return straight_through, raw, indices.view(latent.shape[0], 2, 2)


class VQAutoencoder(nn.Module):
    def __init__(self, entries: int):
        super().__init__()
        self.autoencoder = ContinuousAutoencoder()
        self.quantizer = VectorQuantizer(entries)

    def forward(self, images: torch.Tensor):
        latent = self.autoencoder.encode(images)
        quantized, raw_quantized, indices = self.quantizer(latent)
        reconstruction = self.autoencoder.decode(quantized)
        entity_logits = self.autoencoder.entity_head(quantized)
        return reconstruction, entity_logits, latent, raw_quantized, indices


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
    metadata = [row[1] for row in rows]
    return torch.from_numpy(images), metadata, [row["key"] for row in metadata]


def frame_loader(corpus: str | Path, split: str, *, batch: int, workers: int,
                 seed: int = 0) -> DataLoader:
    """Use indexed chunks when present, retaining tar compatibility."""
    root = Path(corpus)
    indexed = any(root.glob("chunk-*/manifest.json"))
    dataset = IndexedChunkDataset(root, split, seed=seed) if indexed else \
        FrameTarDataset(root, split, seed=seed)
    return DataLoader(dataset, batch_size=batch, num_workers=workers,
                      collate_fn=collate_indexed if indexed else collate_frames,
                      pin_memory=True, persistent_workers=workers > 0,
                      prefetch_factor=2 if workers else None)


def pixel_weights_from_targets(targets: torch.Tensor) -> torch.Tensor:
    coarse = 1.0 + sum(weight * targets[:, channel:channel + 1]
                       for channel, weight in enumerate(PIXEL_WEIGHTS))
    return F.interpolate(coarse.clamp(max=16.0), size=FRAME_HW, mode="bilinear",
                         align_corners=False)


def prepare_targets(supervision, *, device: torch.device) -> tuple[torch.Tensor,
                                                                    torch.Tensor]:
    if isinstance(supervision, torch.Tensor):
        targets = supervision.to(device, non_blocking=True).float()
        return targets, pixel_weights_from_targets(targets)
    return entity_targets(supervision, device=device)


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
    return targets, pixel_weights_from_targets(targets)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_latents(corpus: str, warmup_checkpoint: str, output: str, *,
                    frames: int = 100_000, batch: int = 128, workers: int = 4,
                    device: str = "cuda") -> None:
    """Freeze four-position warmup latents from a deterministic train sample."""
    destination = Path(output)
    manifest_path = destination.with_suffix(".json")
    if destination.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("frames") == frames and manifest.get("sha256") == _sha256(destination):
            print(f"using existing latent sample {destination}", flush=True)
            return
        raise RuntimeError(f"incomplete or mismatched latent artifact: {destination}")
    checkpoint_path = Path(warmup_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if int(checkpoint["step"]) != 20_000:
        raise ValueError("VQ initialization requires the completed 20,000-step warmup")
    model = ContinuousAutoencoder().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    loader = frame_loader(corpus, "train", batch=batch, workers=workers)
    temporary = destination.with_suffix(".tmp.npy")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                      shape=(frames * 4, LATENT_DIM))
    keys, written = [], 0
    started = time.time()
    with torch.inference_mode():
        for raw_images, _supervision, batch_keys in loader:
            take = min(len(raw_images), frames - written)
            images = raw_images[:take].to(device, non_blocking=True).permute(
                0, 3, 1, 2).float().div_(255)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.startswith("cuda")):
                latent = model.encode(images)
            values = latent.permute(0, 2, 3, 1).reshape(-1, LATENT_DIM).float().cpu().numpy()
            array[written * 4:(written + take) * 4] = values.astype(np.float16)
            keys.extend(batch_keys[:take])
            written += take
            if written % 10_000 < take:
                print(f"latents: {written}/{frames} frames", flush=True)
            if written == frames:
                break
    array.flush()
    del array
    os.replace(temporary, destination)
    manifest = {"schema_version": 1, "frames": frames, "positions": 4,
                "dimension": LATENT_DIM, "dtype": "float16", "keys": keys,
                "warmup_checkpoint": str(checkpoint_path),
                "warmup_sha256": _sha256(checkpoint_path),
                "sha256": _sha256(destination),
                "elapsed_seconds": time.time() - started}
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    os.replace(manifest_tmp, manifest_path)


def initialize_kmeans(latents_path: str, output: str, *, entries: int,
                      iterations: int = 20, chunk: int = 16_384,
                      device: str = "cuda") -> None:
    """Run deterministic Lloyd k-means over the frozen four-position latents."""
    destination = Path(output)
    if destination.exists():
        existing = torch.load(destination, map_location="cpu", weights_only=False)
        if existing.get("entries") == entries:
            print(f"using existing k-means codebook {destination}", flush=True)
            return
        raise RuntimeError(f"codebook entry mismatch: {destination}")
    values = np.load(latents_path, mmap_mode="r")
    generator = torch.Generator().manual_seed(0)
    initial = torch.randperm(len(values), generator=generator)[:entries].numpy()
    centroids = torch.from_numpy(np.asarray(values[initial], dtype=np.float32)).to(device)
    for iteration in range(iterations):
        sums = torch.zeros_like(centroids)
        counts = torch.zeros(entries, device=device, dtype=torch.long)
        inertia = 0.0
        for start in range(0, len(values), chunk):
            points = torch.from_numpy(np.asarray(values[start:start + chunk],
                                                  dtype=np.float32)).to(device)
            distances = (points.square().sum(1, keepdim=True)
                         + centroids.square().sum(1).unsqueeze(0)
                         - 2 * points @ centroids.T)
            assignments = distances.argmin(1)
            inertia += float(distances.gather(1, assignments[:, None]).sum())
            sums.index_add_(0, assignments, points)
            counts.add_(torch.bincount(assignments, minlength=entries))
        nonempty = counts > 0
        updated = centroids.clone()
        updated[nonempty] = sums[nonempty] / counts[nonempty, None]
        empty = (~nonempty).nonzero().flatten()
        if len(empty):
            replacement = torch.randperm(len(values), generator=generator)[:len(empty)].numpy()
            updated[empty] = torch.from_numpy(np.asarray(values[replacement],
                                                          dtype=np.float32)).to(device)
        shift = float((updated - centroids).square().sum().sqrt())
        centroids = updated
        print(f"kmeans iteration={iteration + 1} inertia={inertia / len(values):.6g} "
              f"shift={shift:.6g} empty={len(empty)}", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save({"entries": entries, "dimension": LATENT_DIM,
                        "iterations": iterations, "seed": 0,
                        "latents_sha256": _sha256(Path(latents_path)),
                        "centroids": centroids.cpu()}, destination)


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
    loader = frame_loader(corpus, "train", batch=micro_batch, workers=workers)
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
            raw_images, supervision, _keys = next(iterator)
            images = raw_images.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
            targets, weights = prepare_targets(supervision, device=images.device)
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


def train_vq(corpus: str, warmup_checkpoint: str, codebook_path: str, output: str,
             *, entries: int, steps: int = 100_000, micro_batch: int = 32,
             effective_batch: int = 128, workers: int = 4, device: str = "cuda",
             save_every: int = 1000, log_every: int = 20) -> None:
    """Train one straight-through VQ cell from the common warmup and k-means."""
    if effective_batch % micro_batch:
        raise ValueError("effective batch must be divisible by micro batch")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    warmup = torch.load(warmup_checkpoint, map_location=device, weights_only=False)
    if int(warmup["step"]) != 20_000:
        raise ValueError("VQ training requires the completed 20,000-step warmup")
    initialization = torch.load(codebook_path, map_location="cpu", weights_only=False)
    if int(initialization["entries"]) != entries:
        raise ValueError("k-means codebook size does not match requested entries")
    config = {"experiment": "0012", "kind": f"vq-k{entries}", "entries": entries,
              "steps": steps, "micro_batch": micro_batch,
              "effective_batch": effective_batch, "latent_shape": [256, 2, 2],
              "learning_rate": 3e-4, "warmup_steps": min(2000, steps), "seed": 0,
              "commitment_weight": 0.25,
              "loss": "vq_codebook+0.25*commitment+weighted_mse+bce+soft_dice",
              "corpus": str(corpus), "warmup_checkpoint": str(warmup_checkpoint),
              "warmup_sha256": _sha256(Path(warmup_checkpoint)),
              "codebook": str(codebook_path),
              "codebook_sha256": _sha256(Path(codebook_path))}
    (output_path / "config.json").write_text(json.dumps(config, indent=2,
                                                         sort_keys=True) + "\n")
    model = VQAutoencoder(entries).to(device)
    model.autoencoder.load_state_dict(warmup["model"])
    with torch.no_grad():
        model.quantizer.embedding.copy_(initialization["centroids"].to(device))
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
    loader = frame_loader(corpus, "train", batch=micro_batch, workers=workers)
    iterator = iter(loader)
    accumulation = effective_batch // micro_batch
    metrics_path = output_path / "metrics.jsonl"
    model.train()
    started = time.time()
    metric_names = ("loss", "pixel", "bce", "dice", "codebook", "commitment",
                    "perplexity", "used_codes")
    for step in range(start_step, steps):
        lr = learning_rate(step, base=3e-4, warmup=min(2000, steps), total=steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        sums = {name: 0.0 for name in metric_names}
        for _ in range(accumulation):
            raw_images, supervision, _keys = next(iterator)
            images = raw_images.to(device, non_blocking=True).permute(
                0, 3, 1, 2).float().div_(255)
            targets, weights = prepare_targets(supervision, device=images.device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.startswith("cuda")):
                reconstruction, logits, latent, raw_quantized, indices = model(images)
                task_loss, parts = warmup_loss(reconstruction, images, logits,
                                               targets, weights)
                codebook_loss = F.mse_loss(raw_quantized, latent.detach())
                commitment = F.mse_loss(latent, raw_quantized.detach())
                total = task_loss + codebook_loss + 0.25 * commitment
                scaled_loss = total / accumulation
            scaler.scale(scaled_loss).backward()
            counts = torch.bincount(indices.flatten(), minlength=entries).float()
            probabilities = counts[counts > 0] / counts.sum()
            values = {**parts, "loss": total.detach(),
                      "codebook": codebook_loss.detach(),
                      "commitment": commitment.detach(),
                      "perplexity": torch.exp(-(probabilities * probabilities.log()).sum()),
                      "used_codes": (counts > 0).sum()}
            for name in sums:
                sums[name] += float(values[name]) / accumulation
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
    latents = sub.add_parser("extract-latents")
    latents.add_argument("--corpus", default="tmp/0012-vq-codebook/corpus-1k-all")
    latents.add_argument("--warmup", default=
                         "runs/0012-vq-codebook/warmup/checkpoint-020000.pt")
    latents.add_argument("--output", default=
                         "runs/0012-vq-codebook/initialization/latents-100k.npy")
    latents.add_argument("--frames", type=int, default=100_000)
    latents.add_argument("--batch", type=int, default=128)
    latents.add_argument("--workers", type=int, default=4)
    latents.add_argument("--device", default="cuda")
    kmeans = sub.add_parser("kmeans")
    kmeans.add_argument("--latents", default=
                        "runs/0012-vq-codebook/initialization/latents-100k.npy")
    kmeans.add_argument("--output", required=True)
    kmeans.add_argument("--entries", type=int, required=True)
    kmeans.add_argument("--iterations", type=int, default=20)
    kmeans.add_argument("--chunk", type=int, default=16_384)
    kmeans.add_argument("--device", default="cuda")
    vq = sub.add_parser("vq")
    vq.add_argument("--corpus", default="tmp/0012-vq-codebook/corpus-1k-all")
    vq.add_argument("--warmup", default=
                    "runs/0012-vq-codebook/warmup/checkpoint-020000.pt")
    vq.add_argument("--codebook", required=True)
    vq.add_argument("--output", required=True)
    vq.add_argument("--entries", type=int, required=True)
    vq.add_argument("--steps", type=int, default=100_000)
    vq.add_argument("--micro-batch", type=int, default=32)
    vq.add_argument("--effective-batch", type=int, default=128)
    vq.add_argument("--workers", type=int, default=4)
    vq.add_argument("--device", default="cuda")
    vq.add_argument("--save-every", type=int, default=1000)
    vq.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.command == "warmup":
        train_warmup(args.corpus, args.output, steps=args.steps,
                     micro_batch=args.micro_batch, effective_batch=args.effective_batch,
                     workers=args.workers, device=args.device, save_every=args.save_every,
                     log_every=args.log_every)
    elif args.command == "extract-latents":
        extract_latents(args.corpus, args.warmup, args.output, frames=args.frames,
                        batch=args.batch, workers=args.workers, device=args.device)
    elif args.command == "kmeans":
        initialize_kmeans(args.latents, args.output, entries=args.entries,
                          iterations=args.iterations, chunk=args.chunk, device=args.device)
    elif args.command == "vq":
        train_vq(args.corpus, args.warmup, args.codebook, args.output,
                 entries=args.entries, steps=args.steps, micro_batch=args.micro_batch,
                 effective_batch=args.effective_batch, workers=args.workers,
                 device=args.device, save_every=args.save_every,
                 log_every=args.log_every)


if __name__ == "__main__":
    main()
