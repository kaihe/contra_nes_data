"""Repack the frozen PNG corpus into indexed all-intra episode shards."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import numpy as np
from PIL import Image

from datahouse.full_level import sha256_file
from task_maker.export_hf import _encode_video

SCHEMA_VERSION = 1
FRAME_HW = (224, 240)
ENTITY_NAMES = ("player", "enemy", "projectile")


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> dict:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    offset = archive.offset + 512
    archive.addfile(info, io.BytesIO(payload))
    return {"name": name, "offset": offset, "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def _video_bytes(pngs: list[bytes]) -> bytes:
    frames = [np.asarray(Image.open(io.BytesIO(png)).convert("RGB")) for png in pngs]
    unexpected = [frame.shape for frame in frames if frame.shape != (*FRAME_HW, 3)]
    if unexpected:
        raise ValueError(f"unexpected frame shape {unexpected[0]}")
    payload, extension = _encode_video(frames, "png")
    if extension != "mkv":
        raise RuntimeError(f"PNG video produced unexpected extension {extension}")
    return payload


def _coordinate_bytes(rows: list[dict]) -> bytes:
    arrays = {}
    for name in ENTITY_NAMES:
        coordinates: list[tuple[int, int]] = []
        offsets = [0]
        for row in rows:
            coordinates.extend((int(x), int(y)) for x, y in row[name])
            offsets.append(len(coordinates))
        arrays[f"{name}_xy"] = np.asarray(coordinates, dtype=np.int16).reshape(-1, 2)
        arrays[f"{name}_offsets"] = np.asarray(offsets, dtype=np.int32)
    destination = io.BytesIO()
    np.savez(destination, **arrays)
    return destination.getvalue()


def _verify_video(payload: bytes, rows: list[dict]) -> None:
    decoded = []
    with av.open(io.BytesIO(payload)) as container:
        for frame in container.decode(video=0):
            if len(decoded) in (0, len(rows) - 1):
                decoded.append(frame.to_ndarray(format="rgb24"))
            else:
                decoded.append(None)
    if len(decoded) != len(rows):
        raise RuntimeError(f"video has {len(decoded)} frames, expected {len(rows)}")
    for index in (0, len(rows) - 1):
        digest = hashlib.sha256(decoded[index].tobytes()).hexdigest()
        if digest != rows[index]["frame_sha256"]:
            raise RuntimeError(f"decoded frame {index} does not match source")


def _write_episode(archive: tarfile.TarFile, uid: str, pngs: list[bytes],
                   rows: list[dict], source: dict) -> dict:
    if not rows or len(rows) != len(pngs):
        raise RuntimeError(f"incomplete episode {uid}")
    video = _video_bytes(pngs)
    _verify_video(video, rows)
    coordinates = _coordinate_bytes(rows)
    metadata = {"schema_version": SCHEMA_VERSION, "uid": uid,
                "frames": len(rows), "split": rows[0]["split"],
                "source_file": source["file"], "source_sha256": source["sha256"],
                "frame_sha256": [row["frame_sha256"] for row in rows]}
    members = [_add_bytes(archive, f"{uid}.obs.mkv", video),
               _add_bytes(archive, f"{uid}.entities.npz", coordinates),
               _add_bytes(archive, f"{uid}.json",
                          json.dumps(metadata, separators=(",", ":"),
                                     sort_keys=True).encode())]
    return {"uid": uid, "frames": len(rows), "split": rows[0]["split"],
            "members": members}


def build_shard(source_root: str | Path, output_root: str | Path,
                marker_name: str) -> dict:
    source_root, output_root = Path(source_root), Path(output_root)
    source = json.loads((source_root / marker_name).read_text())
    ordinal = int(source["ordinal"])
    destination = output_root / f"shard-{ordinal:05d}.tar"
    marker_path = output_root / f"shard-{ordinal:05d}.json"
    if destination.is_file() and marker_path.is_file():
        existing = json.loads(marker_path.read_text())
        if existing.get("source_sha256") == source["sha256"] and \
                existing.get("frames") == source["frames"]:
            return {**existing, "status": "skipped"}
        raise RuntimeError(f"mismatched existing shard {destination}")
    temporary = destination.with_suffix(f".tar.tmp-{os.getpid()}")
    episodes, total_frames, started = [], 0, time.time()
    current_uid, pngs, rows, pending = None, [], [], {}
    with tarfile.open(source_root / source["file"]) as source_tar, \
            tarfile.open(temporary, "w") as output_tar:
        for member in source_tar:
            if not member.isfile():
                continue
            key, suffix = member.name.rsplit(".", 1)
            extracted = source_tar.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot read {member.name}")
            pending.setdefault(key, {})[suffix] = extracted.read()
            pair = pending[key]
            if "png" not in pair or "json" not in pair:
                continue
            row = json.loads(pair["json"])
            uid = row["trace_fingerprint"]
            if current_uid is not None and uid != current_uid:
                episode = _write_episode(output_tar, current_uid, pngs, rows, source)
                episodes.append(episode); total_frames += len(rows)
                pngs, rows = [], []
            current_uid = uid
            pngs.append(pair["png"]); rows.append(row); del pending[key]
        if current_uid is not None:
            episode = _write_episode(output_tar, current_uid, pngs, rows, source)
            episodes.append(episode); total_frames += len(rows)
        if pending:
            raise RuntimeError(f"unpaired source members: {list(pending)[:3]}")
        manifest = {"schema_version": SCHEMA_VERSION, "ordinal": ordinal,
                    "source_file": source["file"], "source_sha256": source["sha256"],
                    "snapshot_sha256": source["snapshot_sha256"],
                    "episodes": episodes, "frames": total_frames,
                    "decode_window": 512}
        _add_bytes(output_tar, "manifest.json",
                   json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode())
    if total_frames != source["frames"] or len(episodes) != source["episodes"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"source count mismatch for {source['file']}")
    os.replace(temporary, destination)
    marker = {"schema_version": SCHEMA_VERSION, "ordinal": ordinal,
              "file": destination.name, "sha256": sha256_file(destination),
              "bytes": destination.stat().st_size, "frames": total_frames,
              "episodes": len(episodes), "source_file": source["file"],
              "source_sha256": source["sha256"], "splits": source["splits"],
              "snapshot_sha256": source["snapshot_sha256"],
              "decode_window": 512, "elapsed_seconds": time.time() - started}
    temporary_marker = marker_path.with_suffix(f".json.tmp-{os.getpid()}")
    temporary_marker.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_marker, marker_path)
    return {**marker, "status": "built"}


def build_dataset(source: str | Path, output: str | Path, *, jobs: int = 2,
                  limit: int | None = None) -> None:
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    markers = sorted(path.name for path in source.glob("shard-*.json"))
    if limit is not None:
        markers = markers[:limit]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(build_shard, source, output, marker) for marker in markers]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(json.dumps({"completed_shards": completed,
                              "total_shards": len(markers), **result}, sort_keys=True),
                  flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="tmp/0012-vq-codebook/corpus-1k-all")
    parser.add_argument("--output", default="tmp/0010-one-token-compressed-1k")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    build_dataset(args.source, args.output, jobs=args.jobs, limit=args.limit)


if __name__ == "__main__":
    main()
