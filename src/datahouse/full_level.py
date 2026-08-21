"""Build split-free Level-1 full token shards from committed GCS trace batches."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import queue
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from datahouse.catalog import (Shard, connect, register_boundaries,
                               register_collection, register_shard)
from datahouse.encoder import EncoderSpec, load_encoder
from env.utility import boss_scene
from util.replay import make_env, rewind_state, step_env


COLLECTION = "l1-full-10k-v1"
INSTRUCTIONS = {
    "start_to_boss": "reach the Level 1 boss",
    "boss_fight": "defeat the Level 1 boss",
    "full": "complete Level 1",
}
_WORKER_ENV = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, array)
    return output.getvalue()


def _add(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _action_indices(actions: np.ndarray) -> np.ndarray:
    with open("src/agent/baseline.yaml", encoding="utf-8") as source:
        vocabulary = np.asarray(list(yaml.safe_load(source)["actions"].values()),
                                dtype=np.uint8)
    weights = 1 << np.arange(9, dtype=np.int64)
    lookup = {int(vector.astype(np.int64) @ weights): index
              for index, vector in enumerate(vocabulary)}
    result = np.asarray([lookup.get(int(key), -1)
                         for key in np.asarray(actions, dtype=np.int64) @ weights],
                        dtype=np.int64)
    if np.any(result < 0):
        raise ValueError("trace contains an action outside baseline.yaml")
    return result


def _encode(encoder, images: np.ndarray, *, device: str, chunk: int) -> np.ndarray:
    batches = []
    with torch.inference_mode():
        for start in range(0, len(images), chunk):
            batch = torch.from_numpy(images[start:start + chunk]).to(device)
            batches.append(encoder.encode(batch).float().cpu())
    return torch.cat(batches).numpy().astype(np.float16)


def _json_sha(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def freeze_snapshot(gcs_root: str, output: str | Path, *, episodes: int = 10_000,
                    client=None) -> dict:
    """Freeze committed generations and the smallest unique trace fingerprints."""
    if not gcs_root.startswith("gs://"):
        raise ValueError("gcs_root must begin with gs://")
    if client is None:
        from google.cloud import storage
        client = storage.Client()
    bucket_name, _, prefix = gcs_root[5:].partition("/")
    bucket = client.bucket(bucket_name)
    prefix = prefix.rstrip("/") + "/batches/"
    markers = sorted(client.list_blobs(bucket, prefix=prefix), key=lambda blob: blob.name)
    markers = [blob for blob in markers if blob.name.endswith("/COMMITTED.json")]
    candidates, batches = {}, []
    for number, marker_blob in enumerate(markers, 1):
        marker = json.loads(marker_blob.download_as_bytes())
        base = marker_blob.name.rsplit("/", 1)[0]
        manifest_blob = bucket.blob(f"{base}/manifest.json",
                                    generation=marker["object_generations"]["manifest.json"])
        payload = manifest_blob.download_as_bytes()
        if hashlib.sha256(payload).hexdigest() != marker["manifest_sha256"]:
            raise RuntimeError(f"manifest hash mismatch: gs://{bucket_name}/{base}")
        manifest = json.loads(payload)
        archive_uri = f"gs://{bucket_name}/{base}/traces.tar.zst"
        batch = {
            "archive_uri": archive_uri,
            "archive_generation": int(marker["object_generations"]["traces.tar.zst"]),
            "archive_sha256": marker["archive_sha256"],
            "manifest_uri": f"gs://{bucket_name}/{base}/manifest.json",
            "manifest_generation": int(marker["object_generations"]["manifest.json"]),
            "manifest_sha256": marker["manifest_sha256"],
        }
        batches.append(batch)
        for row in manifest["traces"]:
            fingerprint = row["fingerprint"]
            candidate = dict(row)
            candidate.update(batch_index=len(batches) - 1)
            candidates.setdefault(fingerprint, candidate)
        if number % 25 == 0:
            print(f"snapshot: {number}/{len(markers)} committed batches, "
                  f"{len(candidates)} unique traces", flush=True)
    selected = [candidates[fingerprint]
                for fingerprint in sorted(candidates)[:episodes]]
    if len(selected) != episodes:
        raise RuntimeError(f"need {episodes} unique traces, found {len(selected)}")
    snapshot = {
        "schema_version": 1, "collection": COLLECTION,
        "created_at": time.time(), "gcs_root": gcs_root,
        "eligible_unique_traces": len(candidates), "requested_episodes": episodes,
        "selection": "lexicographically-smallest-fingerprint",
        "batches": batches, "selected": selected,
    }
    snapshot["snapshot_sha256"] = _json_sha(snapshot)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return snapshot


def _download_archive(client, batch: dict, destination: Path) -> None:
    bucket_name, _, name = batch["archive_uri"][5:].partition("/")
    blob = client.bucket(bucket_name).blob(name, generation=batch["archive_generation"])
    blob.download_to_filename(destination)
    if sha256_file(str(destination)) != batch["archive_sha256"]:
        raise RuntimeError(f"archive hash mismatch: {batch['archive_uri']}")


def _extract_selected(archive: Path, rows: list[dict], destination: Path) -> list[Path]:
    tar_path = archive.with_suffix("")
    subprocess.run(["zstd", "-q", "-d", "-f", str(archive), "-o", str(tar_path)],
                   check=True)
    wanted = {row["member"]: row for row in rows}
    extracted = []
    with tarfile.open(tar_path) as tar:
        for member in tar:
            row = wanted.get(member.name)
            if row is None:
                continue
            payload = tar.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise RuntimeError(f"member hash mismatch: {member.name}")
            target = destination / f"{row['fingerprint']}.npz"
            target.write_bytes(payload)
            extracted.append(target)
    tar_path.unlink()
    if len(extracted) != len(rows):
        raise RuntimeError(f"archive supplied {len(extracted)}/{len(rows)} selected traces")
    return extracted


def stage_trace(source: Path, destination: Path, *, image_size: int,
                source_row: dict, source_uri: str, env=None) -> dict:
    """Replay a full trace into resized observations and find its boss boundary."""
    with np.load(source, allow_pickle=False) as trace:
        actions_raw = np.asarray(trace["actions"], dtype=np.uint8)
        initial_state = bytes(np.asarray(trace["initial_state"], dtype=np.uint8))
        skip = int(trace["skip"]) if "skip" in trace else 4
    own_env = env is None
    env = env or make_env()
    rewind_state(env, initial_state)
    frames, boundary = [], None

    def capture(index: int) -> None:
        nonlocal boundary
        screen = env.em.get_screen().copy()
        frames.append(cv2.resize(screen, (image_size, image_size),
                                 interpolation=cv2.INTER_AREA))
        if boundary is None and boss_scene(env.unwrapped.get_ram()):
            boundary = index

    capture(0)
    for index, action in enumerate(actions_raw, 1):
        step_env(env, action, skip)
        capture(index)
    if own_env:
        env.close()
    if boundary is None:
        raise ValueError(f"trace never enters boss scene: {source}")
    actions = _action_indices(actions_raw)
    uid = source_row["fingerprint"][:24]
    meta = {
        "uid": uid, "family": "full", "level": 1,
        "trace_fingerprint": source_row["fingerprint"],
        "source_gcs_uri": source_uri, "source_member": source_row["member"],
        "source_sha256": source_row["sha256"],
        "action_steps": len(actions), "observation_steps": len(frames),
        "boss_observation_index": boundary,
        "boss_weapon": source_row.get("boss_weapon", "unknown"),
        "boss_rapid": source_row.get("boss_rapid"),
        "initial_state_file": source_row.get("initial_state_file"),
        "initial_state_sha256": source_row.get("initial_state_sha256"),
        "instructions": INSTRUCTIONS,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as output:
        # The stage is short-lived and bounded by target_frames. Compression makes
        # emulator workers contend for CPU and is slower than writing these arrays.
        np.savez(output, images=np.asarray(frames, dtype=np.uint8), actions=actions,
                 meta=np.asarray(json.dumps(meta, sort_keys=True)))
    os.replace(temporary, destination)
    return meta


def _stage_worker_init() -> None:
    global _WORKER_ENV
    cv2.setNumThreads(1)
    _WORKER_ENV = make_env()


def _stage_worker(job: tuple[str, str, int, dict, str]) -> tuple[str, dict]:
    source, destination, image_size, row, source_uri = job
    meta = stage_trace(Path(source), Path(destination), image_size=image_size,
                       source_row=row, source_uri=source_uri, env=_WORKER_ENV)
    return destination, meta


def _publish(staged: list[Path], destination: Path, *, encoder, device: str,
             chunk: int, encoder_sha256: str) -> dict:
    records, observations = [], 0
    temporary = destination.with_suffix(".tar.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(temporary, "w") as archive:
        for path in staged:
            with np.load(path, allow_pickle=False) as row:
                images = np.asarray(row["images"], dtype=np.uint8)
                actions = np.asarray(row["actions"], dtype=np.int64)
                meta = json.loads(str(row["meta"]))
            tokens = _encode(encoder, images, device=device, chunk=chunk)
            if len(tokens) != len(actions) + 1:
                raise RuntimeError(f"alignment failure: {meta['uid']}")
            meta["encoder_sha256"] = encoder_sha256
            _add(archive, f"{meta['uid']}.tokens.npy", _npy_bytes(tokens))
            _add(archive, f"{meta['uid']}.actions.npy", _npy_bytes(actions))
            _add(archive, f"{meta['uid']}.json",
                 json.dumps(meta, sort_keys=True).encode())
            records.append(meta)
            observations += len(tokens)
    os.replace(temporary, destination)
    return {"sha256": sha256_file(str(destination)), "records": records,
            "observations": observations, "bytes": destination.stat().st_size}


def build(snapshot_path: str, *, house_dir: str, encoder_path: str,
          stage_root: str, target_frames: int = 60_000, chunk: int = 256,
          device: str = "cuda", client=None, limit: int | None = None,
          stage_workers: int = 1) -> None:
    """Resume construction of the frozen snapshot, publishing complete shards."""
    if client is None:
        from google.cloud import storage
        client = storage.Client()
    snapshot = json.loads(Path(snapshot_path).read_text())
    all_selected = snapshot["selected"]
    selected = all_selected[:limit]
    house = Path(house_dir)
    house.mkdir(parents=True, exist_ok=True)
    spec = EncoderSpec.from_checkpoint(encoder_path)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch cannot access it")
    if stage_workers < 1:
        raise ValueError("stage_workers must be positive")
    db = connect(house / "catalog.sqlite")
    existing = {row[0] for row in db.execute("SELECT fingerprint FROM shard_episodes")}
    selected = [row for row in selected if row["fingerprint"] not in existing]
    ordinal = int(db.execute(
        "SELECT COALESCE(MAX(ordinal),-1)+1 FROM shards WHERE level=1 AND task='full' "
        "AND weapon='mixed' AND encoder_sha256=?", (spec.checkpoint_sha256,)).fetchone()[0])
    db.close()
    stage = Path(stage_root)
    stage.mkdir(parents=True, exist_ok=True)
    # Staging is a disposable journal between replay and publication. Anything
    # left by an interrupted producer has no catalog visibility and is replayed.
    for stale in stage.glob("*.stage.npz"):
        stale.unlink()

    ready: queue.Queue[tuple[Path, dict] | None] = queue.Queue(maxsize=200)
    consumer_errors: list[BaseException] = []

    def consume_ready() -> None:
        next_ordinal = ordinal
        pending: list[Path] = []
        pending_frames = 0
        encoder = None
        writer = connect(house / "catalog.sqlite")

        def publish_pending() -> None:
            nonlocal next_ordinal, pending, pending_frames, encoder
            if not pending:
                return
            if encoder is None:
                encoder = load_encoder(encoder_path).to(device).eval()
            destination = house / "level1/full/mixed" / f"token-{next_ordinal:05d}.tar"
            result = _publish(pending, destination, encoder=encoder, device=device,
                              chunk=chunk, encoder_sha256=spec.checkpoint_sha256)
            records = result["records"]
            register_shard(writer, Shard(
                path=os.path.relpath(destination, house), sha256=result["sha256"],
                level=1, task="full", weapon="mixed",
                encoder_sha256=spec.checkpoint_sha256, ordinal=next_ordinal,
                episodes=len(records), frames=result["observations"]),
                [(r["trace_fingerprint"], r["uid"], r["source_gcs_uri"],
                  r["action_steps"]) for r in records])
            register_boundaries(writer, [
                (r["trace_fingerprint"], r["observation_steps"],
                 r["boss_observation_index"], r["source_gcs_uri"], r["source_sha256"])
                for r in records])
            for path in pending:
                path.unlink()
            print(f"published {destination.name}: {len(records)} episodes, "
                  f"{result['observations']} observations, "
                  f"{result['bytes']/2**30:.2f} GiB", flush=True)
            next_ordinal += 1
            pending, pending_frames = [], 0

        try:
            while True:
                item = ready.get()
                if item is None:
                    publish_pending()
                    return
                path, meta = item
                pending.append(path)
                pending_frames += meta["observation_steps"]
                if pending_frames >= target_frames:
                    publish_pending()
        except BaseException as exc:
            consumer_errors.append(exc)
        finally:
            writer.close()

    consumer = threading.Thread(target=consume_ready, name="cuda-token-consumer",
                                daemon=True)
    consumer.start()

    executor = None
    if stage_workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=stage_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_stage_worker_init)
    env = make_env() if executor is None else None
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
                jobs = []
                for source in sources:
                    row = next(item for item in rows
                               if item["fingerprint"] == source.stem)
                    staged_path = stage / f"{source.stem}.stage.npz"
                    jobs.append((str(source), str(staged_path), spec.image_size,
                                 row, batch["archive_uri"]))
                if executor is None:
                    results = [(_job[1], stage_trace(
                        Path(_job[0]), Path(_job[1]), image_size=_job[2],
                        source_row=_job[3], source_uri=_job[4], env=env))
                               for _job in jobs]
                else:
                    results = executor.map(_stage_worker, jobs, chunksize=1)
                for staged_name, meta in results:
                    staged_path = Path(staged_name)
                    if consumer_errors:
                        raise RuntimeError("CUDA token consumer failed") from consumer_errors[0]
                    ready.put((staged_path, meta))
        ready.put(None)
        consumer.join()
        if consumer_errors:
            raise RuntimeError("CUDA token consumer failed") from consumer_errors[0]
    finally:
        if env is not None:
            env.close()
        if executor is not None:
            executor.shutdown(cancel_futures=True)

    if limit is None:
        db = connect(house / "catalog.sqlite")
        fingerprints = [row["fingerprint"] for row in all_selected]
        present = int(db.execute(
            "SELECT COUNT(*) FROM shard_episodes WHERE fingerprint IN "
            f"({','.join('?' for _ in fingerprints)})", fingerprints).fetchone()[0])
        if present != len(fingerprints):
            db.close()
            raise RuntimeError(f"publication gate: catalog has {present}/{len(fingerprints)} episodes")
        manifest_sha = sha256_file(snapshot_path)
        if db.execute("SELECT 1 FROM collections WHERE name=?", (snapshot["collection"],)).fetchone() is None:
            register_collection(
                db, name=snapshot["collection"], level=1, task="full",
                encoder_sha256=spec.checkpoint_sha256,
                manifest_path=os.path.relpath(snapshot_path, house),
                manifest_sha256=manifest_sha, fingerprints=fingerprints)
        db.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--gcs-root", required=True)
    freeze.add_argument("--output", default=f"game_trace/datahouse/collections/{COLLECTION}.json")
    freeze.add_argument("--episodes", type=int, default=10_000)
    produce = sub.add_parser("build")
    produce.add_argument("--snapshot", default=f"game_trace/datahouse/collections/{COLLECTION}.json")
    produce.add_argument("--house", default="game_trace/datahouse")
    produce.add_argument("--encoder", required=True)
    produce.add_argument("--stage-root", default="tmp/l1-full-10k-stage")
    produce.add_argument("--target-frames", type=int, default=60_000)
    produce.add_argument("--chunk", type=int, default=256)
    produce.add_argument("--device", default="cuda")
    produce.add_argument("--limit", type=int)
    produce.add_argument("--stage-workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze_snapshot(args.gcs_root, args.output, episodes=args.episodes)
    else:
        build(args.snapshot, house_dir=args.house, encoder_path=args.encoder,
              stage_root=args.stage_root, target_frames=args.target_frames,
              chunk=args.chunk, device=args.device, limit=args.limit,
              stage_workers=args.stage_workers)


if __name__ == "__main__":
    main()
