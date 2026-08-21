"""Balanced Spread/Laser RGB-versus-token projectile localization experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import tempfile
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datahouse.encoder import (EncoderSpec, HeatmapHead, load_entity_encoder)
from datahouse.encoder_baseline import ENTITY_SIGMA_CELLS
from datahouse.full_level import _download_archive, _extract_selected
from env.entity import entity_heatmaps
from util.replay import make_env, rewind_state, step_env


WEAPONS = ("Spread", "Laser")
PLAYER_BULLET_CHANNEL = 1
_ENV = None


def _canonical_prefix(root: str, weapon: str) -> str:
    return (root.rstrip("/") + "/batches/level1-boss-"
            + weapon.lower() + "-canonical-40k/")


def freeze_snapshot(gcs_root: str, output: str, *, per_weapon: int = 1000,
                    train_per_weapon: int = 800, client=None) -> dict:
    """Freeze verified canonical objects and balanced fingerprint selections."""
    if client is None:
        from google.cloud import storage
        client = storage.Client()
    bucket_name, _, root = gcs_root[5:].partition("/")
    bucket = client.bucket(bucket_name)
    batches, candidates = [], {weapon: {} for weapon in WEAPONS}

    def read_marker(blob):
        marker = json.loads(blob.download_as_bytes())
        base = blob.name.rsplit("/", 1)[0]
        generation = int(marker["object_generations"]["manifest.json"])
        manifest_blob = bucket.blob(base + "/manifest.json", generation=generation)
        payload = manifest_blob.download_as_bytes()
        if hashlib.sha256(payload).hexdigest() != marker["manifest_sha256"]:
            raise RuntimeError(f"manifest hash mismatch: {base}")
        return base, marker, json.loads(payload)

    markers = []
    for weapon in WEAPONS:
        prefix = _canonical_prefix(root, weapon)
        markers.extend((weapon, blob) for blob in client.list_blobs(bucket, prefix=prefix)
                       if blob.name.endswith("/COMMITTED.json"))
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [(weapon, executor.submit(read_marker, blob))
                   for weapon, blob in markers]
        for weapon, future in futures:
            base, marker, manifest = future.result()
            batch_index = len(batches)
            batches.append({
                "weapon": weapon,
                "archive_uri": f"gs://{bucket_name}/{base}/traces.tar.zst",
                "archive_generation": int(marker["object_generations"]["traces.tar.zst"]),
                "archive_sha256": marker["archive_sha256"],
                "manifest_uri": f"gs://{bucket_name}/{base}/manifest.json",
                "manifest_generation": int(marker["object_generations"]["manifest.json"]),
                "manifest_sha256": marker["manifest_sha256"],
            })
            for row in manifest["traces"]:
                if row.get("boss_weapon") != weapon:
                    continue
                candidate = dict(row)
                candidate["batch_index"] = batch_index
                candidates[weapon].setdefault(row["fingerprint"], candidate)
    selected = []
    for weapon in WEAPONS:
        fingerprints = sorted(candidates[weapon])[:per_weapon]
        if len(fingerprints) != per_weapon:
            raise RuntimeError(f"{weapon}: need {per_weapon}, found {len(fingerprints)}")
        for ordinal, fingerprint in enumerate(fingerprints):
            row = dict(candidates[weapon][fingerprint])
            row.update(weapon=weapon, weapon_ordinal=ordinal,
                       split="train" if ordinal < train_per_weapon else "validation")
            selected.append(row)
    snapshot = {
        "schema_version": 1, "collection": "l1-boss-projectile-probe-v1",
        "gcs_root": gcs_root, "created_at": time.time(),
        "selection": "smallest-fingerprint-per-weapon",
        "per_weapon": per_weapon, "train_per_weapon": train_per_weapon,
        "eligible": {weapon: len(candidates[weapon]) for weapon in WEAPONS},
        "batches": batches, "selected": selected,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(f"frozen {len(selected)} traces: {snapshot['eligible']}", flush=True)
    return snapshot


def _init_worker() -> None:
    global _ENV
    cv2.setNumThreads(1)
    _ENV = make_env()


def _stage_trace(job: tuple[str, str, int]) -> tuple[str, int]:
    source, destination, image_size = job
    with np.load(source, allow_pickle=False) as trace:
        actions = np.asarray(trace["actions"], dtype=np.uint8)
        state = bytes(np.asarray(trace["initial_state"], dtype=np.uint8))
        skip = int(trace["skip"]) if "skip" in trace else 4
    rewind_state(_ENV, state)
    images, targets = [], []

    def capture() -> None:
        images.append(cv2.resize(_ENV.em.get_screen().copy(),
                                 (image_size, image_size), interpolation=cv2.INTER_AREA))
        targets.append(entity_heatmaps(
            _ENV.unwrapped.get_ram(), grid=32,
            sigma=ENTITY_SIGMA_CELLS)[PLAYER_BULLET_CHANNEL])

    capture()
    for action in actions:
        step_env(_ENV, action, skip)
        capture()
    temporary = Path(destination).with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez(output, images=np.asarray(images, dtype=np.uint8),
                 targets=np.asarray(targets, dtype=np.float16))
    os.replace(temporary, destination)
    return destination, len(images)


def materialize(snapshot_path: str, checkpoint: str, output_dir: str, *, workers: int = 6,
                chunk: int = 512, device: str = "cuda", client=None,
                limit_per_weapon: int | None = None) -> dict:
    """Replay frozen traces and materialize RGB, target, and production-token arrays."""
    if client is None:
        from google.cloud import storage
        client = storage.Client()
    snapshot = json.loads(Path(snapshot_path).read_text())
    rows = [row for row in snapshot["selected"]
            if limit_per_weapon is None or row["weapon_ordinal"] < limit_per_weapon]
    rows = [dict(row) for row in rows]
    if limit_per_weapon is not None:
        # A canary still needs both sides of the loader contract.
        for row in rows:
            row["split"] = ("validation" if row["weapon_ordinal"] == limit_per_weapon - 1
                            else "train")
    rows.sort(key=lambda row: (WEAPONS.index(row["weapon"]), row["weapon_ordinal"]))
    counts = [int(row["trace_steps"]) + 1 for row in rows]
    offsets = np.cumsum([0] + counts)
    total = int(offsets[-1])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    spec = EncoderSpec.from_checkpoint(checkpoint)
    frames = np.lib.format.open_memmap(output / "frames.npy", mode="w+", dtype=np.uint8,
                                       shape=(total, spec.image_size, spec.image_size, 3))
    targets = np.lib.format.open_memmap(output / "targets.npy", mode="w+", dtype=np.float16,
                                        shape=(total, 32, 32))
    tokens = np.lib.format.open_memmap(output / "tokens.npy", mode="w+", dtype=np.float16,
                                       shape=(total, spec.token_dim))
    control = np.lib.format.open_memmap(output / "control_logits.npy", mode="w+",
                                        dtype=np.float16, shape=(total, 32, 32))
    episode = np.lib.format.open_memmap(output / "episode.npy", mode="w+", dtype=np.int32,
                                        shape=(total,))
    weapon_array = np.lib.format.open_memmap(output / "weapon.npy", mode="w+", dtype=np.uint8,
                                             shape=(total,))
    split_array = np.lib.format.open_memmap(output / "split.npy", mode="w+", dtype=np.uint8,
                                            shape=(total,))
    model = load_entity_encoder(checkpoint).to(device).eval()
    row_index = {row["fingerprint"]: index for index, row in enumerate(rows)}
    stage_root = output / "stage"
    stage_root.mkdir(exist_ok=True)
    executor = ProcessPoolExecutor(max_workers=workers,
                                   mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_init_worker)
    completed = 0
    try:
        by_batch = {}
        for row in rows:
            by_batch.setdefault(row["batch_index"], []).append(row)
        for batch_index, batch_rows in sorted(by_batch.items()):
            batch = snapshot["batches"][batch_index]
            with tempfile.TemporaryDirectory(dir=stage_root) as temporary:
                temporary = Path(temporary)
                archive = temporary / "traces.tar.zst"
                _download_archive(client, batch, archive)
                sources = _extract_selected(archive, batch_rows, temporary)
                jobs = [(str(source), str(stage_root / f"{source.stem}.npz"),
                         spec.image_size) for source in sources]
                for staged_path, observed in executor.map(_stage_trace, jobs, chunksize=1):
                    fingerprint = Path(staged_path).stem
                    index = row_index[fingerprint]
                    start, stop = int(offsets[index]), int(offsets[index + 1])
                    if observed != stop - start:
                        raise RuntimeError(f"trace length mismatch: {fingerprint}")
                    with np.load(staged_path, allow_pickle=False) as staged:
                        image = np.asarray(staged["images"], dtype=np.uint8)
                        target = np.asarray(staged["targets"], dtype=np.float16)
                    frames[start:stop] = image
                    targets[start:stop] = target
                    with torch.inference_mode():
                        for cursor in range(0, observed, chunk):
                            value = torch.from_numpy(image[cursor:cursor + chunk]).to(device)
                            token = model.encode(value)
                            logits = model.entity_head(token)[:, PLAYER_BULLET_CHANNEL]
                            tokens[start + cursor:start + cursor + len(value)] = (
                                token.cpu().numpy().astype(np.float16))
                            control[start + cursor:start + cursor + len(value)] = (
                                logits.cpu().numpy().astype(np.float16))
                    row = rows[index]
                    episode[start:stop] = index
                    weapon_array[start:stop] = WEAPONS.index(row["weapon"])
                    split_array[start:stop] = 0 if row["split"] == "train" else 1
                    Path(staged_path).unlink()
                    completed += 1
                    if completed % 50 == 0:
                        print(f"materialize: {completed}/{len(rows)} episodes", flush=True)
    finally:
        executor.shutdown(cancel_futures=True)
    for array in (frames, targets, tokens, control, episode, weapon_array, split_array):
        array.flush()
    metadata = {"collection": snapshot["collection"], "checkpoint_sha256":
                spec.checkpoint_sha256, "episodes": len(rows), "observations": total,
                "rows": rows, "offsets": offsets.tolist(), "image_size": spec.image_size}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"materialized {len(rows)} episodes / {total} observations", flush=True)
    return metadata


class DirectImageCNN(nn.Module):
    """Fully convolutional 256x256 RGB to 32x32 projectile logits."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(16, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(32, 128), nn.SiLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(16, 64), nn.SiLU(),
            nn.Conv2d(64, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image.permute(0, 3, 1, 2).float().div(255)).squeeze(1)


class TokenProbe(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.head = HeatmapHead(dim=dim, grid=32, n_classes=1, depth=32)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        return self.head(token.float()).squeeze(1)


def _loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 10.0):
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    weight = 1 + pos_weight * target.float()
    return (bce * weight).mean() / (1 + pos_weight * 0.5)


def _metric_rows(logits: torch.Tensor, target: torch.Tensor) -> dict[str, np.ndarray]:
    pred = torch.sigmoid(logits.float())
    truth = target.float()
    present = truth.flatten(1).max(-1).values > 0.5
    numerator = 2 * (pred * truth).flatten(1).sum(-1)
    denominator = ((pred ** 2).flatten(1).sum(-1)
                   + (truth ** 2).flatten(1).sum(-1)).clamp_min(1e-9)
    mse = ((pred - truth) ** 2).flatten(1).mean(-1)
    base = (truth ** 2).flatten(1).mean(-1).clamp_min(1e-9)
    peak = pred.flatten(1).argmax(-1)
    hit = truth.flatten(1).gather(1, peak[:, None]).squeeze(1) > 0.5
    return {"present": present.cpu().numpy(), "dice": (numerator / denominator).cpu().numpy(),
            "mse_skill": (1 - mse / base).cpu().numpy(),
            "peak_hit": hit.float().cpu().numpy()}


def _training_buckets(targets, weapon, split) -> list[np.ndarray]:
    train = split == 0
    present = np.asarray(targets).reshape(len(targets), -1).max(1) > 0.5
    buckets = [np.flatnonzero(train & (weapon == w) & (present == p))
               for w in range(2) for p in (False, True)]
    if any(len(bucket) == 0 for bucket in buckets):
        raise RuntimeError("a weapon/presence training bucket is empty")
    return buckets


def _balanced_indices(buckets: list[np.ndarray], rng, batch: int) -> np.ndarray:
    count = batch // 4
    return np.concatenate([rng.choice(bucket, count, replace=True) for bucket in buckets])


@torch.no_grad()
def evaluate_model(model, arm: str, arrays: dict, device: str, batch: int = 256) -> dict:
    model.eval()
    result = {}
    for weapon_index, weapon_name in enumerate(WEAPONS):
        indices = np.flatnonzero((arrays["split"] == 1) &
                                 (arrays["weapon"] == weapon_index))
        values = {key: [] for key in ("present", "dice", "mse_skill", "peak_hit")}
        episode_values = {}
        for start in range(0, len(indices), batch):
            chosen = indices[start:start + batch]
            if arm == "direct_image":
                value = torch.from_numpy(np.asarray(arrays["frames"][chosen])).to(device)
            elif arm == "token_probe":
                value = torch.from_numpy(np.asarray(arrays["tokens"][chosen])).to(device)
            else:
                logits = torch.from_numpy(np.asarray(arrays["control"][chosen])).to(device)
                target = torch.from_numpy(np.asarray(arrays["targets"][chosen])).to(device)
                rows = _metric_rows(logits, target)
                for key in values: values[key].append(rows[key])
                continue
            target = torch.from_numpy(np.asarray(arrays["targets"][chosen])).to(device)
            rows = _metric_rows(model(value), target)
            for key in values: values[key].append(rows[key])
        merged = {key: np.concatenate(parts) for key, parts in values.items()}
        present = merged["present"]
        ep = np.asarray(arrays["episode"][indices])
        for episode_id in np.unique(ep[present]):
            mask = present & (ep == episode_id)
            episode_values[str(int(episode_id))] = float(merged["dice"][mask].mean())
        result[weapon_name] = {
            "observations": len(indices), "positive_observations": int(present.sum()),
            "dice": float(merged["dice"][present].mean()),
            "mse_skill": float(merged["mse_skill"][present].mean()),
            "peak_hit": float(merged["peak_hit"][present].mean()),
            "episode_dice": episode_values,
        }
    return result


def _open_arrays(data_dir: str) -> dict:
    root = Path(data_dir)
    return {"frames": np.load(root / "frames.npy", mmap_mode="r"),
            "targets": np.load(root / "targets.npy", mmap_mode="r"),
            "tokens": np.load(root / "tokens.npy", mmap_mode="r"),
            "control": np.load(root / "control_logits.npy", mmap_mode="r"),
            "episode": np.load(root / "episode.npy", mmap_mode="r"),
            "weapon": np.load(root / "weapon.npy", mmap_mode="r"),
            "split": np.load(root / "split.npy", mmap_mode="r")}


def train(data_dir: str, output_dir: str, *, arm: str, seed: int, steps: int = 20_000,
          batch: int = 64, device: str = "cuda") -> dict:
    """Train one matched probe and evaluate the fixed per-weapon validation frames."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    arrays = _open_arrays(data_dir)
    if arm == "published_control":
        model = nn.Identity().to(device)
    elif arm == "token_probe":
        model = TokenProbe(arrays["tokens"].shape[1]).to(device)
    elif arm == "direct_image":
        model = DirectImageCNN().to(device)
    else:
        raise ValueError(f"unknown arm: {arm}")
    started = time.time()
    if arm != "published_control":
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        rng = np.random.default_rng(seed)
        buckets = _training_buckets(arrays["targets"], arrays["weapon"], arrays["split"])
        model.train()
        for step in range(steps):
            indices = _balanced_indices(buckets, rng, batch)
            rng.shuffle(indices)
            key = "frames" if arm == "direct_image" else "tokens"
            value = torch.from_numpy(np.asarray(arrays[key][indices])).to(device)
            target = torch.from_numpy(np.asarray(arrays["targets"][indices])).to(device)
            loss = _loss(model(value), target)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            warmup = min(1.0, (step + 1) / 500)
            cosine = 0.5 * (1 + math.cos(math.pi * step / max(1, steps)))
            for group in optimizer.param_groups: group["lr"] = 3e-4 * warmup * cosine
            if (step + 1) % 250 == 0:
                print(f"{arm}/seed{seed}: {step+1}/{steps} loss={loss.item():.5f}", flush=True)
    metrics = evaluate_model(model, arm, arrays, device)
    result = {"arm": arm, "seed": seed, "steps": 0 if arm == "published_control" else steps,
              "elapsed_s": time.time() - started, "metrics": metrics}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{arm}-seed{seed}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    if arm != "published_control":
        torch.save(model.state_dict(), output / f"{arm}-seed{seed}.pt")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--gcs-root", required=True); freeze.add_argument("--output", required=True)
    mat = sub.add_parser("materialize")
    mat.add_argument("--snapshot", required=True); mat.add_argument("--checkpoint", required=True)
    mat.add_argument("--output-dir", required=True); mat.add_argument("--workers", type=int, default=6)
    mat.add_argument("--limit-per-weapon", type=int); mat.add_argument("--device", default="cuda")
    run = sub.add_parser("train")
    run.add_argument("--data-dir", required=True); run.add_argument("--output-dir", required=True)
    run.add_argument("--arm", choices=("published_control", "token_probe", "direct_image"), required=True)
    run.add_argument("--seed", type=int, default=0); run.add_argument("--steps", type=int, default=20_000)
    run.add_argument("--batch", type=int, default=64); run.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "freeze": freeze_snapshot(args.gcs_root, args.output)
    elif args.command == "materialize":
        materialize(args.snapshot, args.checkpoint, args.output_dir, workers=args.workers,
                    limit_per_weapon=args.limit_per_weapon, device=args.device)
    else: train(args.data_dir, args.output_dir, arm=args.arm, seed=args.seed,
                steps=args.steps, batch=args.batch, device=args.device)


if __name__ == "__main__":
    main()
