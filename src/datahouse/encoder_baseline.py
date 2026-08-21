"""Measure the frozen entity encoder on a deterministic full-trace holdout."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import queue
import tempfile
import threading
import time

import cv2
import numpy as np
import torch

from datahouse.encoder import EncoderSpec, load_entity_encoder
from datahouse.full_level import _download_archive, _extract_selected
from env.entity import HEATMAP_CLASSES, entity_heatmaps
from util.replay import make_env, rewind_state, step_env


_ENV = None
# The historical target specifies sigma in source-screen pixels. Convert it to
# the 32x32 target grid using the mean x/y scale, matching policy goal_mask.
ENTITY_SIGMA_CELLS = 6.0 * (32.0 / 240.0 + 32.0 / 224.0) * 0.5


def _init_worker() -> None:
    global _ENV
    cv2.setNumThreads(1)
    _ENV = make_env()


def _stage(job: tuple[str, str, int]) -> tuple[str, int]:
    source, destination, image_size = job
    with np.load(source, allow_pickle=False) as trace:
        actions = np.asarray(trace["actions"], dtype=np.uint8)
        initial_state = bytes(np.asarray(trace["initial_state"], dtype=np.uint8))
        skip = int(trace["skip"]) if "skip" in trace else 4
    rewind_state(_ENV, initial_state)
    images, targets = [], []

    def capture() -> None:
        images.append(cv2.resize(_ENV.em.get_screen().copy(),
                                 (image_size, image_size),
                                 interpolation=cv2.INTER_AREA))
        targets.append(entity_heatmaps(_ENV.unwrapped.get_ram(), grid=32,
                                       sigma=ENTITY_SIGMA_CELLS))

    capture()
    for action in actions:
        step_env(_ENV, action, skip)
        capture()
    temporary = Path(destination).with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez(output, images=np.asarray(images, dtype=np.uint8),
                 targets=np.asarray(targets, dtype=np.float32))
    os.replace(temporary, destination)
    return destination, len(images)


class Metrics:
    def __init__(self):
        self.sums = {name: {metric: 0.0 for metric in
                            ("dice", "mse_skill", "peak_hit")}
                     for name in HEATMAP_CLASSES}
        self.counts = {name: 0 for name in HEATMAP_CLASSES}
        self.frames = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        probability = torch.sigmoid(logits.float())
        for index, name in enumerate(HEATMAP_CLASSES):
            truth = target[:, index]
            present = truth.flatten(1).max(-1).values > 0.5
            if not bool(present.any()):
                continue
            pred, truth = probability[:, index][present], truth[present]
            numerator = 2 * (pred * truth).flatten(1).sum(-1)
            denominator = ((pred ** 2).flatten(1).sum(-1)
                           + (truth ** 2).flatten(1).sum(-1)).clamp_min(1e-9)
            mse = ((pred - truth) ** 2).flatten(1).mean(-1)
            zero_mse = (truth ** 2).flatten(1).mean(-1).clamp_min(1e-9)
            peak = pred.flatten(1).argmax(-1)
            hit = truth.flatten(1).gather(1, peak[:, None]).squeeze(1) > 0.5
            self.sums[name]["dice"] += float((numerator / denominator).sum())
            self.sums[name]["mse_skill"] += float((1 - mse / zero_mse).sum())
            self.sums[name]["peak_hit"] += float(hit.float().sum())
            self.counts[name] += int(present.sum())
        self.frames += len(target)

    def result(self, *, episodes: int, checkpoint_sha256: str,
               collection: str, elapsed_s: float) -> dict:
        return {
            "collection": collection, "checkpoint_sha256": checkpoint_sha256,
            "episodes": episodes, "observations": self.frames,
            "elapsed_s": elapsed_s,
            "classes": {
                name: {"present_observations": self.counts[name], **{
                    metric: self.sums[name][metric] / self.counts[name]
                    if self.counts[name] else None
                    for metric in self.sums[name]}}
                for name in HEATMAP_CLASSES},
        }


def evaluate(snapshot_path: str, checkpoint: str, output: str, *, episodes: int = 1000,
             workers: int = 6, chunk: int = 512, stage_root: str,
             device: str = "cuda", client=None) -> dict:
    """Evaluate the largest-fingerprint holdout and write its aggregate metrics."""
    if client is None:
        from google.cloud import storage
        client = storage.Client()
    snapshot = json.loads(Path(snapshot_path).read_text())
    if not 0 < episodes < len(snapshot["selected"]):
        raise ValueError("episodes must leave at least one collection episode for training")
    selected = snapshot["selected"][-episodes:]
    spec = EncoderSpec.from_checkpoint(checkpoint)
    stage = Path(stage_root)
    stage.mkdir(parents=True, exist_ok=True)
    for stale in stage.glob("*.baseline.npz"):
        stale.unlink()
    ready: queue.Queue[tuple[Path, int] | None] = queue.Queue(maxsize=20)
    errors: list[BaseException] = []
    metrics = Metrics()
    started = time.time()

    def consume() -> None:
        try:
            model = load_entity_encoder(checkpoint).to(device).eval()
            while True:
                item = ready.get()
                if item is None:
                    return
                path, _ = item
                with np.load(path, allow_pickle=False) as staged:
                    images = np.asarray(staged["images"], dtype=np.uint8)
                    targets = np.asarray(staged["targets"], dtype=np.float32)
                with torch.inference_mode():
                    for start in range(0, len(images), chunk):
                        image = torch.from_numpy(images[start:start + chunk]).to(device)
                        target = torch.from_numpy(targets[start:start + chunk]).to(device)
                        metrics.update(model.entity_logits(image), target)
                path.unlink()
        except BaseException as exc:
            errors.append(exc)

    consumer = threading.Thread(target=consume, name="encoder-baseline-gpu", daemon=True)
    consumer.start()
    context = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                   initializer=_init_worker)
    completed = 0
    try:
        by_batch = {}
        for row in selected:
            by_batch.setdefault(row["batch_index"], []).append(row)
        for batch_index, rows in sorted(by_batch.items()):
            batch = snapshot["batches"][batch_index]
            with tempfile.TemporaryDirectory(dir=stage) as temporary:
                temporary = Path(temporary)
                archive = temporary / "traces.tar.zst"
                _download_archive(client, batch, archive)
                sources = _extract_selected(archive, rows, temporary)
                archive.unlink()
                jobs = [(str(source), str(stage / f"{source.stem}.baseline.npz"),
                         spec.image_size) for source in sources]
                for path, frames in executor.map(_stage, jobs, chunksize=1):
                    if errors:
                        raise RuntimeError("baseline GPU consumer failed") from errors[0]
                    ready.put((Path(path), frames))
                    completed += 1
                    if completed % 25 == 0:
                        print(f"baseline: {completed}/{episodes} episodes staged", flush=True)
        ready.put(None)
        consumer.join()
        if errors:
            raise RuntimeError("baseline GPU consumer failed") from errors[0]
    finally:
        executor.shutdown(cancel_futures=True)
    result = metrics.result(episodes=episodes,
                            checkpoint_sha256=spec.checkpoint_sha256,
                            collection=snapshot["collection"],
                            elapsed_s=time.time() - started)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--stage-root", default="tmp/encoder-baseline-stage")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    evaluate(args.snapshot, args.checkpoint, args.output, episodes=args.episodes,
             workers=args.workers, chunk=args.chunk, stage_root=args.stage_root,
             device=args.device)


if __name__ == "__main__":
    main()
