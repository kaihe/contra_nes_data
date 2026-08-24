"""Build canonical Level-1 boss token shards from weapon-specific MC win traces.

This is the datahouse producer: it never writes videos or a policy-local cache.
For each output shard it first replays tasks into compressed, resized RGB staging
files on temporary storage.  A separate CUDA-only phase consumes those files,
publishes and catalogs the token tar atomically, then removes that shard's staging
directory.  Emulator and CUDA lifetimes therefore never overlap.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from datahouse.encoder import EncoderSpec, load_encoder
from datahouse.catalog import Shard, connect, register_shard
from task_maker.base import load_task
from task_maker.boss_release import (frame_balanced_shards, import_traces,
                                     sha256_file, task_fingerprint)
from task_maker.export_hf import materialize


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, array)
    return output.getvalue()


def _add(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _action_indices(actions: np.ndarray) -> np.ndarray:
    """Convert the data-owned baseline action vectors to stable 21-way indices."""
    import yaml

    with open("src/agent/baseline.yaml") as fh:
        vocabulary = np.asarray(list(yaml.safe_load(fh)["actions"].values()), dtype=np.uint8)
    weights = (1 << np.arange(9, dtype=np.int64))
    lookup = {int(vector.astype(np.int64) @ weights): index
              for index, vector in enumerate(vocabulary)}
    keys = np.asarray(actions, dtype=np.int64) @ weights
    out = np.asarray([lookup.get(int(key), -1) for key in keys], dtype=np.int64)
    if np.any(out < 0):
        raise ValueError("trace contains an action outside baseline.yaml")
    return out


def _resize(frames: np.ndarray, size: int) -> np.ndarray:
    return np.asarray([cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
                       for frame in frames], dtype=np.uint8)


def _encode(encoder, images: np.ndarray, *, device: str, chunk: int) -> np.ndarray:
    batches = []
    with torch.no_grad():
        for start in range(0, len(images), chunk):
            batches.append(encoder.encode(torch.from_numpy(images[start:start + chunk]).to(device))
                           .float().cpu())
    return torch.cat(batches).numpy().astype(np.float16)


def _stage_episode(path: str, destination: Path, *, image_size: int) -> Path:
    """Replay one task into an atomic compressed RGB staging file."""
    if destination.exists():
        with np.load(destination, allow_pickle=False) as staged:
            if {"images", "actions", "meta"} <= set(staged.files):
                return destination
        raise ValueError(f"invalid existing staging file: {destination}")
    segment = load_task(path)
    rendered = materialize(segment)
    images = _resize(
        np.concatenate([rendered["goal_img"][None], rendered["frames"]]), image_size)
    actions = _action_indices(rendered["actions"])
    meta = {"uid": segment.uid, "family": "boss",
            "interaction": 4, "length": len(rendered["frames"]),
            "action_len": len(actions), "trace_fingerprint": task_fingerprint(path),
            "source_task": segment.meta["source_task"],
            "raw_trace": segment.meta["raw_trace"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with open(temporary, "wb") as output:
        np.savez_compressed(output, images=images, actions=actions,
                            meta=np.asarray(json.dumps(meta, sort_keys=True)))
    os.replace(temporary, destination)
    return destination


def _stage_job(args: tuple[str, str, int]) -> str:
    """Process-pool entry point for independent emulator replay."""
    path, destination, image_size = args
    cv2.setNumThreads(1)
    return str(_stage_episode(path, Path(destination), image_size=image_size))


def _stage_shard(paths: list[str], stage_dir: Path, *, image_size: int,
                 workers: int = 1) -> list[Path]:
    """Materialize all images for one shard without importing or touching CUDA."""
    if workers < 1:
        raise ValueError("stage workers must be positive")
    jobs = [(path, str(stage_dir / f"{Path(path).stem}.npz"), image_size)
            for path in paths]
    if workers == 1:
        results = map(_stage_job, jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_stage_job, jobs, chunksize=1)
    staged = []
    try:
        for number, result in enumerate(results, 1):
            staged.append(Path(result))
            if number % 100 == 0:
                print(f"  stage {stage_dir.name}: {number}/{len(paths)} episodes", flush=True)
    finally:
        if workers != 1:
            executor.shutdown(cancel_futures=True)
    return staged


def _write_token_shard(staged_paths: list[Path], destination: str, *, encoder,
                       device: str, chunk: int) -> dict:
    """Consume staged pixels in cross-episode GPU batches; no emulator is opened."""
    records, frames = [], 0
    temporary = destination + ".tmp"
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with tarfile.open(temporary, "w") as tar:
        pending, pending_images = [], 0

        def flush() -> None:
            nonlocal pending, pending_images, frames
            if not pending:
                return
            tokens = _encode(encoder,
                             np.concatenate([row["images"] for row in pending]),
                             device=device, chunk=chunk)
            cursor = 0
            for row in pending:
                count = len(row["images"])
                episode_tokens = tokens[cursor:cursor + count]
                cursor += count
                actions, meta = row["actions"], row["meta"]
                if len(episode_tokens) != int(meta["length"]) + 1:
                    raise RuntimeError(f"token/frame count mismatch: {meta['uid']}")
                uid = str(meta["uid"])
                _add(tar, f"{uid}.tokens.npy", _npy_bytes(episode_tokens))
                _add(tar, f"{uid}.actions.npy", _npy_bytes(actions))
                _add(tar, f"{uid}.json", json.dumps(meta, sort_keys=True).encode())
                records.append(meta)
                frames += int(meta["length"])
            pending, pending_images = [], 0

        for number, path in enumerate(staged_paths, 1):
            with np.load(path, allow_pickle=False) as staged:
                images = np.asarray(staged["images"], dtype=np.uint8)
                actions = np.asarray(staged["actions"], dtype=np.int64)
                meta = json.loads(str(staged["meta"]))
            pending.append({"images": images, "actions": actions, "meta": meta})
            pending_images += len(images)
            if pending_images >= chunk:
                flush()
            if number % 100 == 0:
                print(f"  encode {Path(destination).name}: "
                      f"{number}/{len(staged_paths)} episodes", flush=True)
        flush()
    os.replace(temporary, destination)
    return {"file": destination,
            "sha256": sha256_file(destination), "bytes": os.path.getsize(destination),
            "episodes": len(records), "frames": frames,
            "episodes_detail": [
                (record["trace_fingerprint"], record["uid"], record["raw_trace"],
                 record["action_len"])
                for record in records],
            "uids": [record["uid"] for record in records]}


def build_house(*, traces: list[str], house_dir: str, task_dir: str,
                encoder_path: str, weapon: str = "spread",
                target_frames: int = 60_000,
                device: str = "cuda", chunk: int = 256,
                stage_root: str = "tmp/datahouse-stage",
                stage_workers: int = 1) -> None:
    """Append an ordered raw snapshot to the split-free token datahouse.

    Cataloged fingerprints are skipped, making the operation resumable. The datahouse
    records only game taxonomy and provenance; consumers own train/validation policy.
    """
    if not traces:
        raise ValueError("at least one trace is required")
    if weapon not in {"spread", "laser", "regular"}:
        raise ValueError(f"unsupported boss weapon: {weapon}")
    house = Path(house_dir)
    token_dir = house / "level1" / "boss" / weapon
    catalog = connect(house / "catalog.sqlite")
    paths = import_traces(traces, batch_id=f"boss-{weapon}-house-v1",
                          out_root=task_dir, verify_goal=False)
    if len(paths) != len(traces):
        raise ValueError("raw snapshot has duplicate state/action fingerprints")
    cataloged = {str(row[0]) for row in catalog.execute(
        "SELECT fingerprint FROM shard_episodes")}
    remaining = [path for path in paths if task_fingerprint(path) not in cataloged]
    if not remaining:
        catalog.close()
        return

    spec = EncoderSpec.from_checkpoint(encoder_path)
    encoder = None
    start_ordinal = int(catalog.execute(
        "SELECT COALESCE(MAX(ordinal),-1)+1 FROM shards WHERE level=1 AND task='boss' "
        "AND weapon=? AND encoder_sha256=?",
        (weapon, spec.checkpoint_sha256)).fetchone()[0])
    groups = frame_balanced_shards(remaining, target_frames)
    for offset, group in enumerate(groups):
        ordinal = start_ordinal + offset
        name = f"token-{ordinal:05d}"
        stage_dir = Path(stage_root) / name
        staged = _stage_shard(group, stage_dir, image_size=spec.image_size,
                              workers=stage_workers)
        if encoder is None:
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but torch cannot access it")
            encoder = load_encoder(encoder_path).to(device).eval()
        row = _write_token_shard(staged, str(token_dir / f"{name}.tar"),
                                 encoder=encoder, device=device, chunk=chunk)
        register_shard(catalog, Shard(
            path=os.path.relpath(row["file"], house), sha256=row["sha256"],
            level=1, task="boss", weapon=weapon,
            encoder_sha256=spec.checkpoint_sha256, ordinal=ordinal,
            episodes=row["episodes"], frames=row["frames"]), row["episodes_detail"])
        shutil.rmtree(stage_dir)
    catalog.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--weapon", choices=("spread", "laser", "regular"),
                        default="spread")
    parser.add_argument("--house", default="game_trace/datahouse")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--stage-root", default="tmp/datahouse-stage")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--stage-workers", type=int, default=1)
    args = parser.parse_args(argv)
    all_traces = sorted(glob.glob(args.traces))
    if args.episodes < 1 or len(all_traces) < args.episodes:
        raise SystemExit(f"requested {args.episodes} episodes, found {len(all_traces)}")
    build_house(traces=all_traces[:args.episodes], house_dir=args.house,
                task_dir=args.tasks, encoder_path=args.encoder, weapon=args.weapon,
                device=args.device, chunk=args.chunk, stage_root=args.stage_root,
                stage_workers=args.stage_workers)


if __name__ == "__main__":
    main()
