"""Reproducible corpus and training pipeline for experiment 0012.

The corpus command replays the frozen ``l1-full-10k-v1`` snapshot, samples at
most 100 native RGB observations per episode, and writes sequential PNG tar
shards plus a JSONL manifest.  Shards are committed atomically and the command
is resumable at shard boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from datahouse.full_level import _download_archive, _extract_selected, sha256_file
from env.entity import scan
from util.replay import make_env, rewind_state, step_env


EXPERIMENT_SEED = 0
SELECTED_EPISODES = 1_000
EPISODES_PER_SHARD = 10
SPLIT_COUNTS = {"train": 800, "validation": 100, "test": 100}


def _seed(*parts: object) -> int:
    payload = ":".join(map(str, ("contra-0012", EXPERIMENT_SEED, *parts)))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def split_rows(rows: list[dict]) -> list[dict]:
    """Assign the frozen episodes to an exact 8k/1k/1k salted-hash split."""
    if len(rows) != 10_000:
        raise ValueError(f"expected 10000 frozen episodes, got {len(rows)}")
    ordered = sorted(rows, key=lambda row: (_seed("split", row["fingerprint"]),
                                             row["fingerprint"]))
    ordered = ordered[:SELECTED_EPISODES]
    result = []
    start = 0
    for split, count in SPLIT_COUNTS.items():
        for row in ordered[start:start + count]:
            result.append({**row, "split": split})
        start += count
    return result


def _visible_points(points: np.ndarray, shape: tuple[int, int]) -> list[list[int]]:
    height, width = shape
    xoff, yoff = (256 - width) // 2, (240 - height) // 2
    result = []
    for x, y in np.asarray(points).reshape(-1, 2):
        x, y = int(x) - xoff, int(y) - yoff
        if 0 <= x < width and 0 <= y < height:
            result.append([x, y])
    return result


def replay_samples(source: Path, fingerprint: str, env) -> list[tuple[np.ndarray, dict]]:
    """Replay one trace and return its deterministic native-frame samples."""
    with np.load(source, allow_pickle=False) as trace:
        actions = np.asarray(trace["actions"], dtype=np.uint8)
        state = bytes(np.asarray(trace["initial_state"], dtype=np.uint8))
        skip = int(trace["skip"]) if "skip" in trace else 4
    rewind_state(env, state)
    output = []

    def capture(index: int) -> None:
        frame = env.em.get_screen().copy()
        entities = scan(env.unwrapped.get_ram())
        projectile = np.concatenate((entities.player_bullets,
                                     entities.enemy_bullets), axis=0)
        meta = {
            "trace_fingerprint": fingerprint,
            "action_index": index,
            "player": _visible_points(entities.player.reshape(1, 2), frame.shape[:2]),
            "enemy": _visible_points(entities.enemies, frame.shape[:2]),
            "projectile": _visible_points(projectile, frame.shape[:2]),
        }
        output.append((frame, meta))

    capture(0)
    for index, action in enumerate(actions, 1):
        step_env(env, action, skip)
        capture(index)
    if len(output) != len(actions) + 1:
        raise RuntimeError(f"sample alignment failed for {fingerprint}")
    return output


def _png_bytes(frame: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, "PNG", optimize=False, compress_level=1)
    return buffer.getvalue()


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _committed_shards(root: Path) -> set[int]:
    committed = set()
    for marker in root.glob("shard-*.json"):
        try:
            row = json.loads(marker.read_text())
            tar_path = root / row["file"]
            if tar_path.is_file() and tar_path.stat().st_size == row["bytes"]:
                committed.add(int(row["ordinal"]))
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return committed


def build_corpus(snapshot_path: str, output_dir: str, *, client=None,
                 limit_shards: int | None = None) -> None:
    """Build or resume the frozen native-frame corpus."""
    if client is None:
        from google.cloud import storage
        client = storage.Client()
    snapshot_path = str(snapshot_path)
    snapshot = json.loads(Path(snapshot_path).read_text())
    if snapshot.get("snapshot_sha256") != \
            "14cf84631f15dade8fbf85be8a627b4faef16ffa8b639acb3000768f3220bf85":
        raise ValueError("experiment 0012 requires the frozen l1-full-10k-v1 snapshot")
    rows = split_rows(snapshot["selected"])
    split_order = {"train": 0, "validation": 1, "test": 2}
    rows.sort(key=lambda row: (split_order[row["split"]],
                               _seed("split", row["fingerprint"])))
    shards = [rows[i:i + EPISODES_PER_SHARD]
              for i in range(0, len(rows), EPISODES_PER_SHARD)]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    committed = _committed_shards(root)
    pending = [(ordinal, group) for ordinal, group in enumerate(shards)
               if ordinal not in committed]
    if limit_shards is not None:
        pending = pending[:limit_shards]
    env = make_env()
    try:
        for ordinal, group in pending:
            started = time.time()
            destination = root / f"shard-{ordinal:05d}.tar"
            temporary_tar = destination.with_suffix(".tar.tmp")
            records, frames = [], 0
            by_batch: dict[int, list[dict]] = {}
            for row in group:
                by_batch.setdefault(int(row["batch_index"]), []).append(row)
            with tarfile.open(temporary_tar, "w") as archive:
                for batch_index, batch_rows in sorted(by_batch.items()):
                    batch = snapshot["batches"][batch_index]
                    with tempfile.TemporaryDirectory(dir=root) as scratch_name:
                        scratch = Path(scratch_name)
                        compressed = scratch / "traces.tar.zst"
                        _download_archive(client, batch, compressed)
                        sources = _extract_selected(compressed, batch_rows, scratch)
                        for source in sources:
                            row = next(r for r in batch_rows
                                       if r["fingerprint"] == source.stem)
                            for sample_number, (frame, meta) in enumerate(
                                    replay_samples(source, source.stem, env)):
                                key = f"{source.stem}-{sample_number:03d}"
                                png = _png_bytes(frame)
                                meta.update(split=row["split"], key=key,
                                            frame_sha256=hashlib.sha256(
                                                frame.tobytes()).hexdigest())
                                _add_bytes(archive, f"{key}.png", png)
                                _add_bytes(archive, f"{key}.json",
                                           json.dumps(meta, sort_keys=True).encode())
                                records.append(meta)
                                frames += 1
            os.replace(temporary_tar, destination)
            marker = {
                "schema_version": 1, "ordinal": ordinal, "file": destination.name,
                "sha256": sha256_file(destination), "bytes": destination.stat().st_size,
                "episodes": len(group), "frames": frames,
                "splits": {split: sum(r["split"] == split for r in group)
                           for split in SPLIT_COUNTS},
                "snapshot_sha256": snapshot["snapshot_sha256"],
                "elapsed_seconds": time.time() - started,
            }
            marker_tmp = root / f"shard-{ordinal:05d}.json.tmp"
            marker_tmp.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
            os.replace(marker_tmp, root / f"shard-{ordinal:05d}.json")
            print(f"committed shard {ordinal:05d}: {frames} frames, "
                  f"{destination.stat().st_size / 2**20:.1f} MiB, "
                  f"{marker['elapsed_seconds']:.1f}s", flush=True)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    corpus = sub.add_parser("corpus")
    corpus.add_argument("--snapshot", default=
                        "game_trace/datahouse/collections/l1-full-10k-v1.json")
    corpus.add_argument("--output", default="tmp/0012-vq-codebook/corpus")
    corpus.add_argument("--limit-shards", type=int)
    args = parser.parse_args()
    if args.command == "corpus":
        build_corpus(args.snapshot, args.output, limit_shards=args.limit_shards)


if __name__ == "__main__":
    main()
