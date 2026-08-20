"""Import existing MC traces through the worker's durable GCS batch protocol."""

from __future__ import annotations

import argparse
import errno
import glob
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import stable_retro as retro

from env.constant import ADDR_WEAPON, WEAPON_NAMES
from env.utility import boss_scene
from util.replay import GAME, INTTYPE, SKIP, rewind_state, step_env
from worker.search_loop import (
    GCSUploader, SourceExhausted, WorkerLoop, _atomic_json, _sha256,
)


def recover_boss_loadout(path: Path, env) -> dict:
    """Replay a legacy trace and recover its first boss-scene loadout."""
    with np.load(path, allow_pickle=False) as data:
        if ("boss_weapon" in data and "boss_rapid" in data
                and "boss_entry_step" in data and int(data["boss_entry_step"]) >= 0):
            return {
                "boss_weapon": str(data["boss_weapon"]),
                "boss_rapid": bool(data["boss_rapid"]),
                "boss_entry_step": int(data["boss_entry_step"]),
                "boss_metadata_source": "trace",
            }
        # Older boss-search traces recorded the start-state loadout under the
        # generic names. Those searches begin in the boss scene, so this is the
        # same information without an expensive replay.
        if "weapon" in data and "rapid" in data:
            return {
                "boss_weapon": str(data["weapon"]),
                "boss_rapid": bool(data["rapid"]),
                "boss_entry_step": 0,
                "boss_metadata_source": "legacy_trace",
            }
        initial = bytes(data["initial_state"])
        actions = np.asarray(data["actions"], dtype=np.uint8)
        skip = int(data["skip"]) if "skip" in data else SKIP

    rewind_state(env, initial)
    ram = env.unwrapped.get_ram().copy()
    if boss_scene(ram):
        raw = int(ram[ADDR_WEAPON])
        return _loadout(raw, 0, "replay_initial")
    for step, action in enumerate(actions):
        pre = env.unwrapped.get_ram().copy()
        step_env(env, action, skip)
        cur = env.unwrapped.get_ram().copy()
        if boss_scene(cur) and not boss_scene(pre):
            return _loadout(int(cur[ADDR_WEAPON]), step, "replay")
    return {
        "boss_weapon": "",
        "boss_rapid": False,
        "boss_entry_step": -1,
        "boss_metadata_source": "replay_not_reached",
    }


def _loadout(raw: int, step: int, source: str) -> dict:
    gun = raw & 0x0F
    return {
        "boss_weapon": WEAPON_NAMES.get(gun, f"Unknown{gun}"),
        "boss_rapid": bool(raw & 0x10),
        "boss_entry_step": step,
        "boss_metadata_source": source,
    }


def make_replay_env():
    """Create one RAM-only emulator reused by the entire legacy import."""
    env = retro.make(
        game=GAME, state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.RAM, render_mode=None, inttype=INTTYPE,
    )
    env.reset()
    return env


class LegacyTraceImporter:
    """Finite search-compatible source with replay enrichment and restart journal."""

    def __init__(self, patterns: list[str], spool_dir: Path, *, source_paths=(),
                 enrich=None, static_metadata: dict | None = None):
        self.spool_dir = Path(spool_dir)
        self.journal = self.spool_dir / "legacy_import.jsonl"
        self.imported = self._load_imported()
        sources = {Path(item).resolve() for pattern in patterns
                   for item in glob.iglob(pattern, recursive=True)}
        sources.update(Path(item).resolve() for item in source_paths)
        self.sources = iter(sorted(path for path in sources if path.is_file()))
        self.enrich = enrich
        self.static_metadata = dict(static_metadata or {})
        self.env = None
        self.pending: tuple[Path, str] | None = None

    def _load_imported(self) -> set[str]:
        imported = set()
        if self.journal.exists():
            with self.journal.open() as fh:
                imported.update(json.loads(line)["sha256"] for line in fh if line.strip())
        for trace in (self.spool_dir / "open").glob("*/traces/*.npz"):
            if not trace.name.endswith(".partial.npz"):
                imported.add(_sha256(trace))
        for manifest in (self.spool_dir / "sealed").glob("*/manifest.json"):
            imported.update(row["sha256"] for row in json.loads(manifest.read_text())["traces"])
        return imported

    def __call__(self, destination: Path):
        for source in self.sources:
            digest = _sha256(source)
            if digest in self.imported:
                continue
            metadata = self._metadata(source)
            final_name = destination.name.replace(".partial", "")
            sidecar = destination.parents[1] / "metadata" / f"{final_name}.json"
            _atomic_json(sidecar, {
                **metadata,
                "legacy_source_file": source.name,
                "legacy_source_sha256": digest,
            })
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
            except OSError as exc:
                if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES):
                    raise
                shutil.copy2(source, destination)
            self.pending = source, digest
            return str(destination)
        raise SourceExhausted

    def _metadata(self, source: Path) -> dict:
        if self.enrich is not None:
            recovered = self.enrich(source)
        else:
            if self.env is None:
                self.env = make_replay_env()
            recovered = recover_boss_loadout(source, self.env)
        return {**recovered, **self.static_metadata}

    def saved(self, destination: Path) -> None:
        source, digest = self.pending
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "source": str(source), "sha256": digest,
                "spooled": str(destination), "imported_at": time.time(),
            }, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.imported.add(digest)
        self.pending = None

    def close(self) -> None:
        if self.env is not None:
            self.env.close()


def _parse_args():
    parser = argparse.ArgumentParser(description="Upload existing MC traces to GCS")
    parser.add_argument("--gcs-root", required=True)
    parser.add_argument("--spool-dir", default="game_trace/legacy_upload_spool")
    parser.add_argument("--worker-id")
    parser.add_argument("--trace-glob", action="append", default=[],
                        help="quoted NPZ glob; may be repeated")
    parser.add_argument("--trace-list", action="append", default=[],
                        help="UTF-8 file containing one source NPZ path per line")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="traces per archive (default: 100)")
    parser.add_argument("--trace-scope", choices=("full_level", "boss_fight"),
                        help="explicit collection class stored in each manifest row")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.trace_glob and not args.trace_list:
        raise SystemExit("at least one --trace-glob or --trace-list is required")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    listed = []
    for path in args.trace_list:
        with open(path, encoding="utf-8") as fh:
            listed.extend(line.strip() for line in fh
                          if line.strip() and not line.lstrip().startswith("#"))
    static_metadata = ({"trace_scope": args.trace_scope}
                       if args.trace_scope else {})
    importer = LegacyTraceImporter(
        args.trace_glob, Path(args.spool_dir), source_paths=listed,
        static_metadata=static_metadata,
    )
    loop = WorkerLoop(
        Path(args.spool_dir), GCSUploader(args.gcs_root), importer,
        worker_id=args.worker_id, batch_size=args.batch_size,
    )
    signal.signal(signal.SIGINT, lambda *_: loop.stop.set())
    signal.signal(signal.SIGTERM, lambda *_: loop.stop.set())
    try:
        loop.run()
        if not loop.stop.is_set():
            loop.flush()
            loop.upload_queue.join()
    finally:
        importer.close()
        loop.close()


if __name__ == "__main__":
    main()
