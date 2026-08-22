"""Build and stream memory-mapped frame/target chunks from the lossless corpus."""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info


FRAME_HW = (224, 240)
GRID = 32
ENTITY_NAMES = ("player", "enemy", "projectile")
SIGMAS = (6.0, 6.0, 4.0)
SCHEMA_VERSION = 1


def targets_from_metadata(metadata: dict) -> np.ndarray:
    """Return the exact three Gaussian targets used by encoder training."""
    yy, xx = np.mgrid[:GRID, :GRID].astype(np.float32)
    targets = np.zeros((3, GRID, GRID), dtype=np.float32)
    for channel, (name, sigma) in enumerate(zip(ENTITY_NAMES, SIGMAS)):
        grid_sigma = sigma * GRID / FRAME_HW[0]
        for px, py in metadata[name]:
            gx = float(px) * GRID / FRAME_HW[1]
            gy = float(py) * GRID / FRAME_HW[0]
            blob = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) /
                          (2 * grid_sigma * grid_sigma))
            np.maximum(targets[channel], blob, out=targets[channel])
    return targets


def _expected_manifest(marker: dict) -> dict:
    splits = [name for name, episodes in marker["splits"].items() if episodes]
    if len(splits) != 1:
        raise ValueError(f"source shard must contain one split: {marker['file']}")
    return {"schema_version": SCHEMA_VERSION, "source_file": marker["file"],
            "source_sha256": marker["sha256"], "split": splits[0],
            "frames": marker["frames"], "frame_shape": [marker["frames"], 224, 240, 3],
            "frame_dtype": "uint8", "target_shape": [marker["frames"], 3, 32, 32],
            "target_dtype": "float16"}


def _is_complete(path: Path, expected: dict) -> bool:
    try:
        actual = json.loads((path / "manifest.json").read_text())
        return all(actual.get(key) == value for key, value in expected.items()) and \
            (path / "frames.npy").is_file() and (path / "targets.npy").is_file() and \
            (path / "keys.json").is_file()
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def build_chunk(source_root: str | Path, output_root: str | Path,
                marker_name: str) -> dict:
    source_root, output_root = Path(source_root), Path(output_root)
    marker = json.loads((source_root / marker_name).read_text())
    expected = _expected_manifest(marker)
    ordinal = int(marker["ordinal"])
    destination = output_root / f"chunk-{ordinal:05d}"
    if _is_complete(destination, expected):
        return {**expected, "status": "skipped", "chunk": destination.name}

    temporary = output_root / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    count = int(marker["frames"])
    frames = np.lib.format.open_memmap(temporary / "frames.npy", mode="w+",
                                       dtype=np.uint8, shape=(count, 224, 240, 3))
    targets = np.lib.format.open_memmap(temporary / "targets.npy", mode="w+",
                                        dtype=np.float16, shape=(count, 3, 32, 32))
    keys: list[str] = []
    pending: dict[str, dict[str, bytes]] = {}
    written = 0
    started = time.time()
    with tarfile.open(source_root / marker["file"]) as archive:
        for member in archive:
            if not member.isfile():
                continue
            key, suffix = member.name.rsplit(".", 1)
            row = pending.setdefault(key, {})
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot read {member.name}")
            row[suffix] = extracted.read()
            if "png" not in row or "json" not in row:
                continue
            metadata = json.loads(row["json"])
            image = np.asarray(Image.open(io.BytesIO(row["png"])).convert("RGB"))
            if image.shape != (224, 240, 3):
                raise ValueError(f"unexpected frame shape {image.shape} for {key}")
            frames[written] = image
            targets[written] = targets_from_metadata(metadata).astype(np.float16)
            keys.append(metadata.get("key", key))
            written += 1
            del pending[key]
    if written != count or pending:
        raise RuntimeError(f"{marker['file']}: expected {count}, wrote {written}")
    frames.flush(); targets.flush()
    del frames, targets
    (temporary / "keys.json").write_text(json.dumps(keys, separators=(",", ":")) + "\n")
    manifest = {**expected, "elapsed_seconds": time.time() - started}
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if destination.exists():
        raise RuntimeError(f"mismatched existing chunk: {destination}")
    os.replace(temporary, destination)
    return {**manifest, "status": "built", "chunk": destination.name}


def build_cache(source_root: str | Path, output_root: str | Path, *, jobs: int = 2,
                limit: int | None = None) -> None:
    source_root, output_root = Path(source_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    markers = sorted(path.name for path in source_root.glob("shard-*.json"))
    if limit is not None:
        markers = markers[:limit]
    if not markers:
        raise RuntimeError(f"no source markers under {source_root}")
    completed = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(build_chunk, source_root, output_root, marker): marker
                   for marker in markers}
        for future in as_completed(futures):
            result = future.result(); completed += 1
            print(json.dumps({"completed_chunks": completed, "total_chunks": len(markers),
                              **result}, sort_keys=True), flush=True)


class IndexedChunkDataset(IterableDataset):
    """Stream shuffled rows from memory-mapped chunks without decoding work."""

    def __init__(self, root: str | Path, split: str, *, seed: int = 0):
        self.root, self.split, self.seed = Path(root), split, seed

    def _chunks(self) -> list[Path]:
        chunks = []
        for manifest_path in sorted(self.root.glob("chunk-*/manifest.json")):
            if json.loads(manifest_path.read_text())["split"] == self.split:
                chunks.append(manifest_path.parent)
        if not chunks:
            raise RuntimeError(f"no {self.split} indexed chunks under {self.root}")
        return chunks

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        chunks = self._chunks()[worker_id::worker_count]
        if not chunks:
            raise RuntimeError("more loader workers than indexed chunks")
        rng = random.Random(self.seed + worker_id)
        epoch = 0
        while True:
            rng.shuffle(chunks)
            for path in chunks:
                frames = np.load(path / "frames.npy", mmap_mode="r")
                targets = np.load(path / "targets.npy", mmap_mode="r")
                keys = json.loads((path / "keys.json").read_text())
                indices = list(range(len(frames))); rng.shuffle(indices)
                for index in indices:
                    yield frames[index], targets[index], keys[index]
            epoch += 1
            rng.seed(self.seed + worker_id + epoch * 1009)


def collate_indexed(rows):
    frames = torch.from_numpy(np.stack([row[0] for row in rows]))
    targets = torch.from_numpy(np.stack([row[1] for row in rows]))
    return frames, targets, [row[2] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="tmp/0012-vq-codebook/corpus-1k-all")
    parser.add_argument("--output", default="tmp/0012-vq-codebook/indexed-1k-all")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    build_cache(args.source, args.output, jobs=args.jobs, limit=args.limit)


if __name__ == "__main__":
    main()
