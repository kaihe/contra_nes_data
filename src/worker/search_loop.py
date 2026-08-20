"""Continuously search, batch, and upload winning MC traces.

Every winning trace is first written beneath a durable local spool. A batch is
sealed only at 100 wins (or by the explicit ``--flush`` command), archived, and
queued to a background rclone uploader. ``COMMITTED.json`` is uploaded last and
is the remote visibility boundary. Restarting the process resumes sealed batches
and continues filling the one open batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import tarfile
import threading
import time
import uuid

import numpy as np

from agent.mc_search import _run_one_search


BATCH_SIZE = 100
SCHEMA_VERSION = 1


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(data, key, default=None):
    if key not in data:
        return default
    value = data[key]
    return value.item() if value.ndim == 0 else value.tolist()


def trace_record(path: Path) -> dict:
    """Return manifest metadata and the content identity for one trace."""
    with np.load(path, allow_pickle=False) as data:
        identity = hashlib.sha256()
        identity.update(np.asarray(data["initial_state"], dtype=np.uint8).tobytes())
        identity.update(np.asarray(data["actions"], dtype=np.uint8).tobytes())
        for key in ("skip", "level", "goal"):
            identity.update(str(_scalar(data, key, "")).encode("utf-8"))
            identity.update(b"\0")
        record = {
            "fingerprint": identity.hexdigest(),
            "member": f"traces/{path.name}",
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for key in (
            "level", "goal", "outcome", "trace_steps", "search_steps",
            "sampled_actions", "search_wall_s", "initial_state_file",
            "initial_state_sha256", "prior_sha256", "reward_config",
            "boss_weapon", "boss_rapid", "boss_entry_step",
            "rollouts", "rollout_len", "max_time", "max_rewind",
            "max_actions", "workers",
        ):
            value = _scalar(data, key)
            if value is not None:
                record[key] = value
    return record


class RcloneUploader:
    """Upload and verify one sealed batch using an existing rclone remote."""

    def __init__(self, remote_root: str):
        self.remote_root = remote_root.rstrip("/")

    def _run(self, *args: str) -> str:
        result = subprocess.run(
            ["rclone", *args], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return result.stdout

    def upload(self, batch_dir: Path, worker_id: str, batch_id: str) -> None:
        remote = f"{self.remote_root}/batches/{worker_id}/{batch_id}"
        archive = batch_dir / "traces.tar.zst"
        manifest = batch_dir / "manifest.json"
        for source in (archive, manifest):
            target = f"{remote}/{source.name}"
            self._run("copyto", str(source), target)
            listing = json.loads(self._run("lsjson", target, "--hash"))
            item = listing[0] if isinstance(listing, list) else listing
            if int(item["Size"]) != source.stat().st_size:
                raise RuntimeError(f"remote size mismatch for {target}")
            hashes = {key.lower(): value.lower()
                      for key, value in item.get("Hashes", {}).items()}
            if hashes.get("md5") != _md5(source):
                raise RuntimeError(f"remote MD5 mismatch for {target}")

        marker = {
            "schema_version": SCHEMA_VERSION,
            "worker_id": worker_id,
            "batch_id": batch_id,
            "trace_count": len(json.loads(manifest.read_text())["traces"]),
            "archive_sha256": _sha256(archive),
            "manifest_sha256": _sha256(manifest),
            "committed_at": time.time(),
        }
        _atomic_json(batch_dir / "COMMITTED.json", marker)
        self._run("copyto", str(batch_dir / "COMMITTED.json"),
                  f"{remote}/COMMITTED.json")


class WorkerLoop:
    """Durable producer loop with one open batch and resumable sealed batches."""

    def __init__(self, spool_dir: Path, uploader, search_one, *,
                 worker_id: str | None = None, batch_size: int = BATCH_SIZE):
        self.spool_dir = Path(spool_dir)
        self.open_root = self.spool_dir / "open"
        self.sealed_root = self.spool_dir / "sealed"
        self.worker_id = worker_id or self._load_worker_id()
        self.uploader = uploader
        self.search_one = search_one
        self.batch_size = batch_size
        self.stop = threading.Event()
        self.upload_queue: queue.Queue[Path | None] = queue.Queue()
        self.upload_thread: threading.Thread | None = None

    def _load_worker_id(self) -> str:
        identity_path = self.spool_dir / "worker.json"
        if identity_path.exists():
            return json.loads(identity_path.read_text())["worker_id"]
        worker_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:12]}"
        _atomic_json(identity_path, {"worker_id": worker_id})
        return worker_id

    def _open_batch(self) -> Path:
        batches = sorted(path for path in self.open_root.glob("*") if path.is_dir())
        if len(batches) > 1:
            raise RuntimeError(f"multiple open batches in {self.open_root}")
        if batches:
            return batches[0]
        batch = self.open_root / uuid.uuid4().hex
        (batch / "traces").mkdir(parents=True)
        _atomic_json(batch / "batch.json", {
            "schema_version": SCHEMA_VERSION,
            "worker_id": self.worker_id,
            "batch_id": batch.name,
            "state": "open",
            "created_at": time.time(),
        })
        return batch

    @staticmethod
    def _trace_count(batch: Path) -> int:
        return sum(1 for _ in (batch / "traces").glob("*.npz"))

    def _seal(self, batch: Path) -> Path:
        traces = sorted((batch / "traces").glob("*.npz"))
        if not traces:
            raise ValueError("cannot seal an empty batch")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "worker_id": self.worker_id,
            "batch_id": batch.name,
            "trace_count": len(traces),
            "created_at": time.time(),
            "traces": [trace_record(path) for path in traces],
        }
        _atomic_json(batch / "manifest.json", manifest)

        tar_path = batch / "traces.tar"
        with tarfile.open(tar_path, "w") as archive:
            archive.add(batch / "manifest.json", arcname="manifest.json")
            for trace in traces:
                archive.add(trace, arcname=f"traces/{trace.name}")
        compressed = batch / "traces.tar.zst"
        subprocess.run(
            ["zstd", "-q", "-f", str(tar_path), "-o", str(compressed)], check=True)
        tar_path.unlink()

        state = json.loads((batch / "batch.json").read_text())
        state.update(state="sealed", sealed_at=time.time(), trace_count=len(traces))
        _atomic_json(batch / "batch.json", state)
        self.sealed_root.mkdir(parents=True, exist_ok=True)
        destination = self.sealed_root / batch.name
        os.replace(batch, destination)
        return destination

    def _upload_forever(self) -> None:
        while True:
            batch = self.upload_queue.get()
            if batch is None:
                self.upload_queue.task_done()
                return
            delay = 5
            while not (batch / "ACKNOWLEDGED.json").exists():
                try:
                    self.uploader.upload(batch, self.worker_id, batch.name)
                    _atomic_json(batch / "uploaded.json", {"uploaded_at": time.time()})
                    break
                except Exception as exc:  # durable spool makes indefinite retry safe
                    _atomic_json(batch / "upload_error.json", {
                        "error": str(exc), "failed_at": time.time(), "retry_s": delay,
                    })
                    if self.stop.wait(delay):
                        break
                    delay = min(delay * 2, 300)
            self.upload_queue.task_done()

    def _start_uploader(self) -> None:
        if self.upload_thread is not None:
            return
        self.sealed_root.mkdir(parents=True, exist_ok=True)
        self.upload_thread = threading.Thread(target=self._upload_forever, daemon=True)
        self.upload_thread.start()
        for batch in sorted(self.sealed_root.glob("*")):
            if batch.is_dir() and not (batch / "ACKNOWLEDGED.json").exists():
                self.upload_queue.put(batch)

    def flush(self) -> bool:
        """Seal the partial open batch, returning whether anything was queued."""
        batch = self._open_batch()
        if not self._trace_count(batch):
            return False
        self.upload_queue.put(self._seal(batch))
        return True

    def run(self, *, max_wins: int | None = None) -> int:
        """Search until signalled; ``max_wins`` supports bounded operator runs."""
        self._start_uploader()
        wins = 0
        while not self.stop.is_set() and (max_wins is None or wins < max_wins):
            batch = self._open_batch()
            temporary = batch / "traces" / f"{uuid.uuid4().hex}.partial.npz"
            result = self.search_one(temporary)
            if result is None:
                temporary.unlink(missing_ok=True)
                continue
            final = temporary.with_name(temporary.name.replace(".partial", ""))
            os.replace(temporary, final)
            wins += 1
            count = self._trace_count(batch)
            print(f"win {wins}: batch={batch.name} traces={count}/{self.batch_size}", flush=True)
            if count >= self.batch_size:
                self.upload_queue.put(self._seal(batch))
        return wins

    def close(self) -> None:
        self.stop.set()
        if self.upload_thread is not None:
            self.upload_queue.put(None)
            self.upload_thread.join(timeout=30)


def _parse_args():
    parser = argparse.ArgumentParser(description="Persistent MC search and Drive upload worker")
    parser.add_argument("--drive-remote", required=True,
                        help="rclone path through level/goal, e.g. gdrive:Contra MC Tracehouse/schema-v1/level1/full")
    parser.add_argument("--spool-dir", default="game_trace/worker_spool")
    parser.add_argument("--worker-id")
    parser.add_argument("--flush", action="store_true",
                        help="seal and upload a partial batch without searching")
    parser.add_argument("--max-wins", type=int, help="stop after this many wins; partial batch remains open")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--rollout-len", type=int, default=48)
    parser.add_argument("--settle-margin", type=int, default=16)
    parser.add_argument("--max-time", type=int, default=600)
    parser.add_argument("--max-rewind", type=int, default=30)
    parser.add_argument("--max-actions", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--goal", choices=("boss_entry", "level_up", "game_clear"),
                        default="level_up")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    def search_one(path: Path):
        return _run_one_search(
            level=args.level, rollouts=args.rollouts, rollout_len=args.rollout_len,
            max_time=args.max_time, max_rewind=args.max_rewind,
            max_actions=args.max_actions, goal=args.goal, workers=args.workers,
            settle_margin=args.settle_margin, verbose=False, trace_path=str(path),
        )

    loop = WorkerLoop(
        Path(args.spool_dir), RcloneUploader(args.drive_remote), search_one,
        worker_id=args.worker_id,
    )
    signal.signal(signal.SIGINT, lambda *_: loop.stop.set())
    signal.signal(signal.SIGTERM, lambda *_: loop.stop.set())
    try:
        if args.flush:
            loop._start_uploader()
            loop.flush()
            loop.upload_queue.join()
        else:
            loop.run(max_wins=args.max_wins)
    finally:
        loop.close()


if __name__ == "__main__":
    main()
