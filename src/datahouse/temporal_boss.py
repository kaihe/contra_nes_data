"""Re-encode the canonical Laser boss store with the accepted 0019 encoder.

The source datahouse supplies the immutable episode order and shard boundaries. Raw
traces come from generation-pinned committed GCS batches, are converted with the same
``boss-laser-house-v1`` identity recipe, and are replayed at native 224x240. The output
is a separate datahouse because a catalog intentionally admits each episode once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from datahouse.boss_spread import _action_indices, _add, _npy_bytes
from datahouse.catalog import Shard, connect, register_shard
from datahouse.encoder import load_temporal_encoder
from datahouse.full_level import sha256_file
from task_maker.boss_release import import_traces, task_fingerprint
from task_maker.base import load_task
from task_maker.export_hf import materialize


GCS_PREFIX = ("contra-mc-tracehouse/schema-v1/level1/boss/batches/"
              "level1-boss-laser-canonical-40k/")
BATCH_ID = "boss-laser-house-v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch_canonical_traces(destination: Path, *, client=None) -> list[Path]:
    """Download and verify the 40 committed canonical Laser trace batches."""
    if client is None:
        from google.cloud import storage
        client = storage.Client()
    bucket = client.bucket("contra_nes_trace")
    markers = sorted(blob for blob in client.list_blobs(bucket, prefix=GCS_PREFIX)
                     if blob.name.endswith("/COMMITTED.json"))
    if len(markers) != 40:
        raise RuntimeError(f"expected 40 canonical Laser batches, found {len(markers)}")
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for number, marker_blob in enumerate(markers, 1):
        marker = json.loads(marker_blob.download_as_bytes())
        base = marker_blob.name.rsplit("/", 1)[0]
        manifest_payload = bucket.blob(
            f"{base}/manifest.json",
            generation=marker["object_generations"]["manifest.json"]
        ).download_as_bytes()
        if _sha256(manifest_payload) != marker["manifest_sha256"]:
            raise RuntimeError(f"manifest hash mismatch: {base}")
        manifest = json.loads(manifest_payload)
        rows = {row["member"]: row for row in manifest["traces"]}
        pending = [row for row in rows.values()
                   if not (destination / row["legacy_source_file"]).exists()]
        if pending:
            with tempfile.TemporaryDirectory(prefix="laser-batch-") as temporary:
                archive = Path(temporary) / "traces.tar.zst"
                blob = bucket.blob(
                    f"{base}/traces.tar.zst",
                    generation=marker["object_generations"]["traces.tar.zst"])
                blob.download_to_filename(archive)
                if sha256_file(archive) != marker["archive_sha256"]:
                    raise RuntimeError(f"archive hash mismatch: {base}")
                tar_path = Path(temporary) / "traces.tar"
                subprocess.run(["zstd", "-q", "-d", str(archive), "-o", str(tar_path)],
                               check=True)
                with tarfile.open(tar_path) as source:
                    for member in source:
                        row = rows.get(member.name)
                        if row is None:
                            continue
                        payload = source.extractfile(member).read()
                        if _sha256(payload) != row["sha256"]:
                            raise RuntimeError(f"trace hash mismatch: {member.name}")
                        target = destination / row["legacy_source_file"]
                        temporary_target = target.with_suffix(".tmp")
                        temporary_target.write_bytes(payload)
                        os.replace(temporary_target, target)
        for row in rows.values():
            target = destination / row["legacy_source_file"]
            if not target.exists() or sha256_file(target) != row["sha256"]:
                raise RuntimeError(f"missing or stale canonical trace: {target}")
            paths.append(target)
        print(f"fetch: {number}/{len(markers)} batches, {len(paths)} traces", flush=True)
    if len(paths) != 40_000 or len(set(paths)) != len(paths):
        raise RuntimeError(f"canonical trace identity mismatch: {len(paths)} rows")
    return sorted(paths)


def source_groups(source_house: Path) -> list[list[str]]:
    """Return old Laser raw-trace names grouped by exact shard ordinal/order."""
    db = sqlite3.connect(source_house / "catalog.sqlite")
    rows = db.execute(
        "SELECT path FROM shards WHERE level=1 AND task='boss' AND weapon='laser' "
        "ORDER BY ordinal").fetchall()
    db.close()
    groups = []
    for (relative,) in rows:
        names = []
        with tarfile.open(source_house / relative) as archive:
            for member in archive:
                if member.name.endswith(".json"):
                    names.append(json.load(archive.extractfile(member))["raw_trace"])
        groups.append(names)
    if len(groups) != 70 or sum(map(len, groups)) != 40_000:
        raise RuntimeError("source house is not the canonical 70-shard Laser store")
    return groups


def _stage_episode(job: tuple[str, str]) -> str:
    path, destination = job
    destination = Path(destination)
    if destination.exists():
        return str(destination)
    segment = load_task(path)
    rendered = materialize(segment)
    images = np.asarray([rendered["goal_img"], *rendered["frames"]], dtype=np.uint8)
    if images.shape[1:] != (224, 240, 3):
        raise ValueError(f"unexpected native frame shape: {images.shape}")
    meta = {"uid": segment.uid, "family": "boss", "interaction": 4,
            "length": len(rendered["frames"]), "action_len": len(segment.actions),
            "trace_fingerprint": task_fingerprint(path),
            "source_task": segment.meta["source_task"],
            "raw_trace": segment.meta["raw_trace"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, images=images,
                            actions=_action_indices(rendered["actions"]),
                            meta=np.asarray(json.dumps(meta, sort_keys=True)))
    os.replace(temporary, destination)
    return str(destination)


def _stage_group(paths: list[Path], destination: Path, workers: int) -> list[Path]:
    jobs = [(str(path), str(destination / f"{path.stem}.npz"))
            for path in paths]
    destination.mkdir(parents=True, exist_ok=True)
    if workers == 1:
        return [Path(_stage_episode(job)) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        staged = []
        for number, result in enumerate(executor.map(_stage_episode, jobs, chunksize=1), 1):
            staged.append(Path(result))
            if number % 100 == 0:
                print(f"  stage: {number}/{len(jobs)} episodes", flush=True)
    return staged


def _encode_episode(encoder, images: np.ndarray, *, device: str,
                    chunk: int) -> np.ndarray:
    # row 0 is a standalone visual goal; row 1 is the first decision frame.
    previous = images.copy()
    if len(images) > 2:
        previous[2:] = images[1:-1]
    batches = []
    with torch.inference_mode():
        for start in range(0, len(images), chunk):
            current = torch.from_numpy(images[start:start + chunk]).to(device)
            prior = torch.from_numpy(previous[start:start + chunk]).to(device)
            batches.append(encoder.encode_pair(current, prior).float().cpu())
    return torch.cat(batches).numpy().astype(np.float16)


def _write_shard(staged: list[Path], destination: Path, *, encoder,
                 device: str, chunk: int) -> dict:
    records, frames = [], 0
    temporary = destination.with_suffix(".tar.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(temporary, "w") as archive:
        for number, path in enumerate(staged, 1):
            with np.load(path, allow_pickle=False) as row:
                images = np.asarray(row["images"], dtype=np.uint8)
                actions = np.asarray(row["actions"], dtype=np.int64)
                meta = json.loads(str(row["meta"]))
            tokens = _encode_episode(encoder, images, device=device, chunk=chunk)
            if len(tokens) != int(meta["length"]) + 1:
                raise RuntimeError(f"token/frame count mismatch: {meta['uid']}")
            uid = meta["uid"]
            _add(archive, f"{uid}.tokens.npy", _npy_bytes(tokens))
            _add(archive, f"{uid}.actions.npy", _npy_bytes(actions))
            _add(archive, f"{uid}.json", json.dumps(meta, sort_keys=True).encode())
            records.append(meta)
            frames += int(meta["length"])
            if number % 100 == 0:
                print(f"  encode: {number}/{len(staged)} episodes", flush=True)
    os.replace(temporary, destination)
    return {"sha256": sha256_file(destination), "bytes": destination.stat().st_size,
            "episodes": len(records), "frames": frames,
            "details": [(row["trace_fingerprint"], row["uid"], row["raw_trace"],
                         row["action_len"]) for row in records]}


def build_temporal_house(*, source_house: Path, house: Path, encoder_path: Path,
                         raw_dir: Path, task_dir: Path, stage_root: Path,
                         workers: int = 8, device: str = "cuda",
                         chunk: int = 256) -> None:
    traces = fetch_canonical_traces(raw_dir)
    tasks = import_traces([str(path) for path in traces], batch_id=BATCH_ID,
                          out_root=str(task_dir), verify_goal=False)
    by_raw = {Path(np.load(path, allow_pickle=True)["raw_trace"].item()).name: Path(path)
              for path in tasks}
    groups = source_groups(source_house)
    expected = {name for group in groups for name in group}
    if set(by_raw) != expected:
        raise RuntimeError(f"task/source mismatch: tasks={len(by_raw)}, expected={len(expected)}")

    digest = sha256_file(encoder_path)
    bundle = house / "encoder" / digest
    bundle.mkdir(parents=True, exist_ok=True)
    for source in (encoder_path, encoder_path.parent / "spec.json"):
        target = bundle / source.name
        if not target.exists():
            shutil.copy2(source, target)
    catalog = connect(house / "catalog.sqlite")
    done = {int(row[0]) for row in catalog.execute(
        "SELECT ordinal FROM shards WHERE level=1 AND task='boss' AND weapon='laser' "
        "AND encoder_sha256=?", (digest,))}
    encoder = load_temporal_encoder(encoder_path).to(device).eval()
    for ordinal, raw_names in enumerate(groups):
        if ordinal in done:
            continue
        selected = [by_raw[name] for name in raw_names]
        stage = stage_root / f"token-{ordinal:05d}"
        staged = _stage_group(selected, stage, workers)
        destination = house / "level1" / "boss" / "laser" / f"token-{ordinal:05d}.tar"
        row = _write_shard(staged, destination, encoder=encoder,
                           device=device, chunk=chunk)
        register_shard(catalog, Shard(
            path=os.path.relpath(destination, house), sha256=row["sha256"],
            level=1, task="boss", weapon="laser", encoder_sha256=digest,
            ordinal=ordinal, episodes=row["episodes"], frames=row["frames"]),
            row["details"])
        shutil.rmtree(stage)
        print(f"published shard {ordinal + 1}/{len(groups)}", flush=True)
    catalog.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-house", type=Path, default=Path("game_trace/datahouse"))
    parser.add_argument("--house", type=Path,
                        default=Path("game_trace/datahouse-frame-difference-0019"))
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("tmp/laser-canonical-40k"))
    parser.add_argument("--tasks", type=Path, default=Path("tmp/laser-temporal-tasks"))
    parser.add_argument("--stage-root", type=Path, default=Path("tmp/laser-temporal-stage"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=256)
    args = parser.parse_args()
    build_temporal_house(source_house=args.source_house, house=args.house,
                         encoder_path=args.encoder, raw_dir=args.raw_dir,
                         task_dir=args.tasks, stage_root=args.stage_root,
                         workers=args.workers, device=args.device, chunk=args.chunk)


if __name__ == "__main__":
    main()
