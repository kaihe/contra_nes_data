"""Shared frame-training toolkit: readers, entity targets, losses, schedules.

Every frame-level trainer in the datahouse reads its corpus and builds its entity
supervision through this module, so a decoder trained on one representation is
comparable to a decoder trained on another. Representation-specific models live with
their experiment; nothing here knows what a token is.
"""

from __future__ import annotations

import io
import json
import math
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from datahouse.compressed_loader import CompressedEpisodeDataset, is_compressed_corpus
from datahouse.indexed_chunks import IndexedChunkDataset, collate_indexed


FRAME_HW = (224, 240)
GRID = 32
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
            if "decode_window" in marker:
                raise RuntimeError(f"{self.root} holds episode shards, not per-frame "
                                   "tars; use CompressedEpisodeDataset")
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
    """Select the reader the corpus layout implies, retaining tar compatibility."""
    root = Path(corpus)
    if any(root.glob("chunk-*/manifest.json")):
        dataset, collate = IndexedChunkDataset(root, split, seed=seed), collate_indexed
    elif is_compressed_corpus(root):
        dataset, collate = CompressedEpisodeDataset(root, split, seed=seed), collate_frames
    else:
        dataset, collate = FrameTarDataset(root, split, seed=seed), collate_frames
    return DataLoader(dataset, batch_size=batch, num_workers=workers,
                      collate_fn=collate, pin_memory=True,
                      persistent_workers=workers > 0,
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


def frame_loss(reconstruction: torch.Tensor, images: torch.Tensor,
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
    """Linear warmup, then cosine decay to zero across the declared budget."""
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

def wsd_learning_rate(step: int, *, base: float, warmup: int, decay_start: int,
                      total: int) -> float:
    """Warmup, a constant stable phase, then cosine decay to zero.

    Cosine ties its shape to a total budget chosen before the first step, so a run
    can neither be extended nor read at an intermediate step without measuring a
    model whose learning rate never annealed. The stable phase has no such horizon:
    one run can be stopped, resumed, or extended, and a short decay branched off any
    stable checkpoint yields a properly annealed model at that budget. A compute
    ladder is then a set of branches off one run rather than a set of full retrains.
    """
    if step < warmup:
        return base * (step + 1) / warmup
    if step < decay_start:
        return base
    progress = (step - decay_start) / max(1, total - decay_start)
    return base * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
