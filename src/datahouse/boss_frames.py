"""Publish Level-1 boss episodes as native-resolution frame shards.

The datahouse producer for `doc/0012-design-boss-spread-frame-shards.md`. Frames are
stored exactly as the emulator renders them -- 224x240 RGB, no resize, no goal row --
beside the 21-way action index for each frame.

Episodes are addressed by the fingerprints the token producer already cataloged, so a
pixel run and a token run provably see the same episodes. No encoder is loaded, so
unlike `boss_spread` there is no GPU phase to keep clear of the emulator: each worker
replays one episode and returns its finished members, which the parent streams into
the shard tar.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from datahouse.catalog import (FrameShard, connect, frame_shard_fingerprints,
                               register_frame_shard, token_prefix_fingerprints)
from datahouse.full_level import sha256_file

SCHEMA_VERSION = 1
FORMAT = "png-mkv-v1"
FRAME_HW = (224, 240)
SHARD_EPISODES = 250


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> dict:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    offset = archive.offset + 512
    archive.addfile(info, io.BytesIO(payload))
    return {"name": name, "offset": offset, "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, array)
    return output.getvalue()


def _verify_video(payload: bytes, digests: list[str]) -> None:
    """Decode the published video and re-check its first and last frame.

    Cheap enough to run on every episode, and it catches the failure that matters:
    an encoder that silently converts colour space would produce a playable video
    holding different pixels.
    """
    import av

    count = 0
    with av.open(io.BytesIO(payload)) as container:
        for frame in container.decode(video=0):
            if count in (0, len(digests) - 1):
                actual = hashlib.sha256(
                    frame.to_ndarray(format="rgb24").tobytes()).hexdigest()
                if actual != digests[count]:
                    raise RuntimeError(f"decoded frame {count} does not match replay")
            count += 1
    if count != len(digests):
        raise RuntimeError(f"video holds {count} frames, expected {len(digests)}")


def _episode_members(task_path: str, fingerprint: str) -> tuple[bytes, bytes, bytes]:
    """Replay one task and return its (video, actions, metadata) payloads."""
    from task_maker.base import load_task
    from task_maker.boss_release import task_fingerprint
    from task_maker.export_hf import _encode_video, materialize

    from datahouse.boss_spread import _action_indices

    actual = task_fingerprint(task_path)
    if actual != fingerprint:
        raise RuntimeError(f"{task_path} has fingerprint {actual}, catalog says {fingerprint}")
    segment = load_task(task_path)
    rendered = materialize(segment)
    frames = [np.ascontiguousarray(frame) for frame in rendered["frames"]]
    if not frames:
        raise RuntimeError(f"empty episode: {task_path}")
    unexpected = [f.shape for f in frames if f.shape != (*FRAME_HW, 3)]
    if unexpected:
        raise RuntimeError(f"frame shape {unexpected[0]} is not {(*FRAME_HW, 3)}")
    actions = _action_indices(rendered["actions"])
    if len(actions) != len(frames):
        raise RuntimeError(f"{len(frames)} frames but {len(actions)} actions")

    video, extension = _encode_video(frames, "png")
    if extension != "mkv":
        raise RuntimeError(f"PNG video produced unexpected extension {extension}")
    digests = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
    _verify_video(video, digests)
    metadata = {"schema_version": SCHEMA_VERSION, "format": FORMAT,
                "uid": segment.uid, "fingerprint": fingerprint,
                "frames": len(frames), "frame_height": FRAME_HW[0],
                "frame_width": FRAME_HW[1], "actions": len(actions),
                "source_task": segment.meta["source_task"],
                "raw_trace": segment.meta["raw_trace"],
                "frame_sha256": digests}
    return (video, _npy_bytes(actions),
            json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode())


def _episode_job(args: tuple[str, str]) -> tuple[str, bytes, bytes, bytes]:
    task_path, fingerprint = args
    video, actions, metadata = _episode_members(task_path, fingerprint)
    return fingerprint, video, actions, metadata


def write_frame_shard(jobs: list[tuple[str, str]], destination: Path, *,
                      workers: int = 1) -> dict:
    """Replay a shard's episodes and publish one tar atomically."""
    if workers < 1:
        raise ValueError("workers must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".tar.tmp-{os.getpid()}")
    episodes, total_frames = [], 0
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = (map(_episode_job, jobs) if executor is None
                   else executor.map(_episode_job, jobs, chunksize=1))
        with tarfile.open(temporary, "w") as archive:
            for number, (fingerprint, video, actions, metadata) in enumerate(results, 1):
                row = json.loads(metadata)
                uid = row["uid"]
                members = [_add_bytes(archive, f"{uid}.obs.mkv", video),
                           _add_bytes(archive, f"{uid}.actions.npy", actions),
                           _add_bytes(archive, f"{uid}.json", metadata)]
                episodes.append({"uid": uid, "fingerprint": fingerprint,
                                 "frames": row["frames"], "members": members})
                total_frames += int(row["frames"])
                if number % 50 == 0:
                    print(f"  {destination.name}: {number}/{len(jobs)} episodes",
                          flush=True)
            manifest = {"schema_version": SCHEMA_VERSION, "format": FORMAT,
                        "frame_height": FRAME_HW[0], "frame_width": FRAME_HW[1],
                        "episodes": episodes, "frames": total_frames,
                        "decode_window": 512}
            _add_bytes(archive, "manifest.json",
                       json.dumps(manifest, separators=(",", ":"),
                                  sort_keys=True).encode())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    os.replace(temporary, destination)
    return {"file": destination, "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size, "frames": total_frames,
            "fingerprints": [row["fingerprint"] for row in episodes]}


def build_frames(*, house_dir: str, task_dir: str, weapon: str = "spread",
                 shard_count: int = 13, level: int = 1, task: str = "boss",
                 shard_episodes: int = SHARD_EPISODES, workers: int = 1,
                 limit: int | None = None) -> None:
    """Publish the frame release for a whole-shard prefix of the token release.

    Already-published fingerprints are skipped, so an interrupted run resumes with no
    manual cleanup and a shard is either absent or complete and cataloged.
    """
    house = Path(house_dir)
    output = house / f"level{level}" / task / weapon / "frames"
    catalog = connect(house / "catalog.sqlite")
    try:
        wanted = token_prefix_fingerprints(catalog, level=level, task=task,
                                           weapon=weapon, shard_count=shard_count)
        done = frame_shard_fingerprints(catalog, level=level, task=task,
                                        weapon=weapon, format=FORMAT)
        remaining = [f for f in wanted if f not in done]
        if not remaining:
            print(f"nothing to do: {len(wanted)} episodes already published")
            return
        uids = {str(row[0]): str(row[1]) for row in catalog.execute(
            "SELECT fingerprint, uid FROM episodes")}
        missing = [f for f in remaining if f not in uids]
        if missing:
            raise RuntimeError(f"catalog has no uid for {len(missing)} episodes")
        start = int(catalog.execute(
            "SELECT COALESCE(MAX(ordinal),-1)+1 FROM frame_shards "
            "WHERE level=? AND task=? AND weapon=? AND format=?",
            (level, task, weapon, FORMAT)).fetchone()[0])
        groups = [remaining[i:i + shard_episodes]
                  for i in range(0, len(remaining), shard_episodes)]
        if limit is not None:
            groups = groups[:limit]
        print(f"publishing {sum(len(g) for g in groups)} episodes "
              f"in {len(groups)} shards from ordinal {start}", flush=True)
        for offset, group in enumerate(groups):
            ordinal = start + offset
            jobs = [(str(Path(task_dir) / f"{uids[f]}.npz"), f) for f in group]
            unreadable = [path for path, _ in jobs if not os.path.isfile(path)]
            if unreadable:
                raise RuntimeError(f"missing task file: {unreadable[0]}")
            row = write_frame_shard(jobs, output / f"frames-{ordinal:05d}.tar",
                                    workers=workers)
            register_frame_shard(catalog, FrameShard(
                path=os.path.relpath(row["file"], house), sha256=row["sha256"],
                level=level, task=task, weapon=weapon, format=FORMAT,
                frame_height=FRAME_HW[0], frame_width=FRAME_HW[1], ordinal=ordinal,
                episodes=len(group), frames=row["frames"]), row["fingerprints"])
            print(json.dumps({"shard": ordinal, "episodes": len(group),
                              "frames": row["frames"], "bytes": row["bytes"]},
                             sort_keys=True), flush=True)
    finally:
        catalog.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--house", default="game_trace/datahouse")
    parser.add_argument("--tasks", required=True,
                        help="directory holding <uid>.npz task files")
    parser.add_argument("--weapon", default="spread",
                        choices=("spread", "laser", "regular"))
    parser.add_argument("--shard-count", type=int, default=13,
                        help="token-shard prefix to mirror (13 = D10k Spread)")
    parser.add_argument("--shard-episodes", type=int, default=SHARD_EPISODES)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, help="stop after this many shards")
    args = parser.parse_args(argv)
    build_frames(house_dir=args.house, task_dir=args.tasks, weapon=args.weapon,
                 shard_count=args.shard_count, shard_episodes=args.shard_episodes,
                 workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
