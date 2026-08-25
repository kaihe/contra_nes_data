"""Publish frozen ``view_backbone + reduce`` features from boss frame shards.

The output is a versioned sibling of the raw-frame and one-token releases.  Each
episode stores float16 ``(T, 256, 4, 4)`` features, the original unshifted action
indices, and provenance sufficient to reproduce the frozen producer exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import av
import cv2
import numpy as np
import torch

from datahouse.boss_frames import FORMAT as SOURCE_FORMAT, _add_bytes
from datahouse.catalog import (FeatureShard, connect, feature_shard_fingerprints,
                               register_feature_shard)
from datahouse.encoder import EncoderSpec, load_encoder
from datahouse.full_level import sha256_file

REPRESENTATION = "reduced-view-v1"
BOUNDARY = "view_backbone+reduce"
DTYPE = "float16"
INPUT_LAYOUT = "uint8_rgb_hwc"
INTERPOLATION = "INTER_AREA"
VERIFY_ATOL = 1 / 128


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def _decode_video(payload: bytes) -> np.ndarray:
    frames = []
    with av.open(io.BytesIO(payload)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError("frame-shard episode has no decoded frames")
    return np.asarray(frames, dtype=np.uint8)


def encode_reduced(encoder, frames: np.ndarray, *, device: str,
                   chunk: int) -> np.ndarray:
    """Apply the checkpoint's exact resize and frozen boundary."""
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must be uint8 THWC RGB")
    height, width = encoder.input_hw
    resized = np.asarray([
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        for frame in frames
    ], dtype=np.uint8)
    batches = []
    with torch.inference_mode():
        for start in range(0, len(resized), chunk):
            images = torch.from_numpy(resized[start:start + chunk]).to(device)
            x = images.permute(0, 3, 1, 2).float().div(255.0)
            value = encoder.reduce(encoder.view_backbone.forward_features(x))
            batches.append(value.to(torch.float16).cpu().numpy())
    return np.concatenate(batches)


def _source_members(source: Path) -> tuple[dict, list[dict]]:
    with tarfile.open(source, "r") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
    if manifest.get("format") != SOURCE_FORMAT:
        raise ValueError(f"{source} is not a {SOURCE_FORMAT} frame shard")
    return manifest, list(manifest["episodes"])


def write_feature_shard(source: Path, destination: Path, *, encoder,
                        encoder_sha256: str, device: str, chunk: int) -> dict:
    """Convert one immutable frame tar to one atomic reduced-feature tar."""
    source_manifest, source_episodes = _source_members(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".tar.tmp-{os.getpid()}")
    output_episodes, frame_count = [], 0
    try:
        with tarfile.open(source, "r") as source_tar, tarfile.open(temporary, "w") as out:
            for number, episode in enumerate(source_episodes, 1):
                uid = str(episode["uid"])
                meta = json.load(source_tar.extractfile(f"{uid}.json"))
                video = source_tar.extractfile(f"{uid}.obs.mkv").read()
                action_payload = source_tar.extractfile(f"{uid}.actions.npy").read()
                actions = np.load(io.BytesIO(action_payload), allow_pickle=False)
                frames = _decode_video(video)
                if len(frames) != len(actions) or len(frames) != int(meta["frames"]):
                    raise RuntimeError(f"source count mismatch for {uid}")
                features = encode_reduced(encoder, frames, device=device, chunk=chunk)
                expected = (len(frames), 256, 4, 4)
                if features.shape != expected or features.dtype != np.float16:
                    raise RuntimeError(
                        f"feature contract mismatch for {uid}: {features.shape} {features.dtype}")
                feature_payload = _npy_bytes(features)
                output_meta = {
                    "schema_version": 1,
                    "representation": REPRESENTATION,
                    "boundary": BOUNDARY,
                    "uid": uid,
                    "fingerprint": str(episode["fingerprint"]),
                    "frames": len(frames),
                    "actions": len(actions),
                    "action_alignment": "frames[i]_is_post_action_actions[i];target_actions[i+1]",
                    "feature_shape": [256, 4, 4],
                    "feature_dtype": DTYPE,
                    "source_frame_format": SOURCE_FORMAT,
                }
                meta_payload = json.dumps(
                    output_meta, separators=(",", ":"), sort_keys=True).encode()
                members = [
                    _add_bytes(out, f"{uid}.features.npy", feature_payload),
                    _add_bytes(out, f"{uid}.actions.npy", action_payload),
                    _add_bytes(out, f"{uid}.json", meta_payload),
                ]
                output_episodes.append({
                    "uid": uid, "fingerprint": str(episode["fingerprint"]),
                    "frames": len(frames), "members": members,
                })
                frame_count += len(frames)
                if number % 50 == 0:
                    print(f"  {destination.name}: {number}/{len(source_episodes)} episodes",
                          flush=True)
            manifest = {
                "schema_version": 1,
                "representation": REPRESENTATION,
                "boundary": BOUNDARY,
                "encoder_checkpoint_sha256": encoder_sha256,
                "feature_order": "TCHW",
                "feature_shape": [256, 4, 4],
                "feature_dtype": DTYPE,
                "input_layout": INPUT_LAYOUT,
                "input_source_height": int(source_manifest["frame_height"]),
                "input_source_width": int(source_manifest["frame_width"]),
                "input_height": int(encoder.input_hw[0]),
                "input_width": int(encoder.input_hw[1]),
                "interpolation": INTERPOLATION,
                "normalization": "float32_rgb/255",
                "source_frame_format": SOURCE_FORMAT,
                "source_frame_shard": os.path.basename(source),
                "source_frame_shard_sha256": sha256_file(source),
                "action_alignment": "frames[i]_is_post_action_actions[i];target_actions[i+1]",
                "episodes": output_episodes,
                "frames": frame_count,
            }
            _add_bytes(out, "manifest.json", json.dumps(
                manifest, separators=(",", ":"), sort_keys=True).encode())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, destination)
    return {
        "file": destination,
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "episodes": len(output_episodes),
        "frames": frame_count,
        "fingerprints": [row["fingerprint"] for row in output_episodes],
    }


def write_release_spec(db, *, house: Path, output: Path, level: int, task: str,
                       weapon: str, encoder_sha256: str, encoder) -> dict:
    """Write the policy-facing selector and complete ordered membership."""
    shards = [dict(row) for row in db.execute(
        "SELECT path,sha256,ordinal,episodes,frames FROM feature_shards "
        "WHERE level=? AND task=? AND weapon=? AND representation=? "
        "AND encoder_sha256=? ORDER BY ordinal",
        (level, task, weapon, REPRESENTATION, encoder_sha256)).fetchall()]
    fingerprints = [str(row[0]) for row in db.execute(
        "SELECT fse.fingerprint FROM feature_shard_episodes fse "
        "JOIN feature_shards s ON s.id=fse.shard_id "
        "WHERE s.level=? AND s.task=? AND s.weapon=? AND s.representation=? "
        "AND s.encoder_sha256=? ORDER BY s.ordinal,fse.ordinal",
        (level, task, weapon, REPRESENTATION, encoder_sha256)).fetchall()]
    membership_payload = "".join(f"{value}\n" for value in fingerprints).encode()
    spec = {
        "schema_version": 1,
        "catalog_selector": {
            "level": level, "task": task, "weapon": weapon,
            "representation": REPRESENTATION,
            "encoder_sha256": encoder_sha256,
        },
        "boundary": BOUNDARY,
        "excluded_modules": ["proj", "token_ln"],
        "feature_order": "TCHW",
        "feature_shape": [256, 4, 4],
        "feature_dtype": DTYPE,
        "input_layout": INPUT_LAYOUT,
        "input_source_shape": [224, 240, 3],
        "input_encoder_shape": [int(encoder.input_hw[0]), int(encoder.input_hw[1]), 3],
        "interpolation": INTERPOLATION,
        "normalization": "float32_rgb/255",
        "source_frame_format": SOURCE_FORMAT,
        "action_alignment": "frames[i]_is_post_action_actions[i];target_actions[i+1]",
        "shards": shards,
        "shard_count": len(shards),
        "episodes": len(fingerprints),
        "frames": sum(int(row["frames"]) for row in shards),
        "bytes": sum((house / str(row["path"])).stat().st_size for row in shards),
        "membership_sha256": hashlib.sha256(membership_payload).hexdigest(),
        "fingerprints": fingerprints,
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "spec.json"
    temporary = destination.with_suffix(f".json.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    return spec


def build_features(*, house_dir: str, encoder_path: str, weapon: str = "laser",
                   level: int = 1, task: str = "boss", device: str = "cuda",
                   chunk: int = 64, limit: int | None = None) -> None:
    """Publish every available source frame shard, resumably and in source order."""
    if chunk < 1:
        raise ValueError("chunk must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch cannot access it")
    house = Path(house_dir)
    spec = EncoderSpec.from_checkpoint(encoder_path)
    encoder = load_encoder(encoder_path).to(device).eval()
    catalog = connect(house / "catalog.sqlite")
    try:
        sources = catalog.execute(
            "SELECT id,path,sha256,ordinal,episodes,frames FROM frame_shards "
            "WHERE level=? AND task=? AND weapon=? AND format=? ORDER BY ordinal",
            (level, task, weapon, SOURCE_FORMAT)).fetchall()
        if not sources:
            raise RuntimeError("no matching source frame shards")
        done = feature_shard_fingerprints(
            catalog, level=level, task=task, weapon=weapon,
            representation=REPRESENTATION, encoder_sha256=spec.checkpoint_sha256)
        pending = []
        for source in sources:
            fingerprints = [str(row[0]) for row in catalog.execute(
                "SELECT fingerprint FROM frame_shard_episodes WHERE shard_id=? "
                "ORDER BY ordinal", (int(source["id"]),)).fetchall()]
            overlap = [fingerprint in done for fingerprint in fingerprints]
            if any(overlap) and not all(overlap):
                raise RuntimeError(f"partial feature membership for frame shard {source['ordinal']}")
            if not all(overlap):
                pending.append((source, fingerprints))
        if limit is not None:
            pending = pending[:limit]
        output = house / f"level{level}" / task / weapon / "features" / REPRESENTATION / spec.checkpoint_sha256
        if not pending:
            print(f"nothing to encode: {len(done)} episodes already published")
        else:
            print(f"publishing {sum(len(f) for _, f in pending)} episodes from "
                  f"{len(pending)} frame shards", flush=True)
        for source, fingerprints in pending:
            ordinal = int(source["ordinal"])
            source_path = house / str(source["path"])
            if sha256_file(source_path) != str(source["sha256"]):
                raise RuntimeError(f"source frame shard hash mismatch: {source_path}")
            destination = output / f"reduced-{ordinal:05d}.tar"
            row = write_feature_shard(
                source_path, destination, encoder=encoder,
                encoder_sha256=spec.checkpoint_sha256, device=device, chunk=chunk)
            if row["fingerprints"] != fingerprints:
                raise RuntimeError(f"source membership/order mismatch at ordinal {ordinal}")
            register_feature_shard(catalog, FeatureShard(
                path=os.path.relpath(row["file"], house), sha256=row["sha256"],
                level=level, task=task, weapon=weapon,
                representation=REPRESENTATION,
                encoder_sha256=spec.checkpoint_sha256, boundary=BOUNDARY,
                dtype=DTYPE, channels=256, feature_height=4, feature_width=4,
                ordinal=ordinal, episodes=row["episodes"], frames=row["frames"]),
                row["fingerprints"])
            print(json.dumps({"shard": ordinal, "episodes": row["episodes"],
                              "frames": row["frames"], "bytes": row["bytes"]},
                             sort_keys=True), flush=True)
        release = write_release_spec(
            catalog, house=house, output=output, level=level, task=task,
            weapon=weapon, encoder_sha256=spec.checkpoint_sha256, encoder=encoder)
        print(json.dumps({key: release[key] for key in
                          ("shard_count", "episodes", "frames", "bytes",
                           "membership_sha256")}, sort_keys=True), flush=True)
    finally:
        catalog.close()


def verify_sample(*, house_dir: str, encoder_path: str, device: str = "cuda",
                  samples: int = 9) -> dict:
    """Compare stored samples with fresh decode, resize, and live trunk forwards."""
    house = Path(house_dir)
    spec = EncoderSpec.from_checkpoint(encoder_path)
    encoder = load_encoder(encoder_path).to(device).eval()
    db = connect(house / "catalog.sqlite")
    try:
        rows = db.execute(
            "SELECT fs.path AS feature_path, fr.path AS frame_path, fse.fingerprint,"
            "e.uid FROM feature_shards fs "
            "JOIN feature_shard_episodes fse ON fse.shard_id=fs.id "
            "JOIN episodes e ON e.fingerprint=fse.fingerprint "
            "JOIN frame_shard_episodes rse ON rse.fingerprint=fse.fingerprint "
            "JOIN frame_shards fr ON fr.id=rse.shard_id "
            "WHERE fs.representation=? AND fs.encoder_sha256=? "
            "AND fr.format=? ORDER BY fs.ordinal,fse.ordinal",
            (REPRESENTATION, spec.checkpoint_sha256, SOURCE_FORMAT)).fetchall()
    finally:
        db.close()
    if not rows:
        raise RuntimeError("no published reduced features to verify")
    indices = np.linspace(0, len(rows) - 1, min(samples, len(rows)), dtype=int)
    maximum, checked = 0.0, 0
    for index in indices:
        row = rows[int(index)]
        uid = str(row["uid"])
        with tarfile.open(house / str(row["frame_path"])) as source:
            frames = _decode_video(source.extractfile(f"{uid}.obs.mkv").read())
        with tarfile.open(house / str(row["feature_path"])) as features:
            stored = np.load(io.BytesIO(
                features.extractfile(f"{uid}.features.npy").read()),
                allow_pickle=False)
        positions = sorted(set((0, len(frames) // 2, len(frames) - 1)))
        live = encode_reduced(encoder, frames[positions], device=device,
                              chunk=len(positions))
        error = float(np.max(np.abs(live.astype(np.float32) -
                                    stored[positions].astype(np.float32))))
        maximum = max(maximum, error)
        checked += len(positions)
    passed = maximum <= VERIFY_ATOL
    result = {"episodes_sampled": len(indices), "frames_checked": checked,
              "max_abs_error_float16": maximum,
              "absolute_tolerance": VERIFY_ATOL,
              "within_float16_tolerance": passed}
    print(json.dumps(result, sort_keys=True))
    if not passed:
        raise RuntimeError(f"stored features differ from live forward: {maximum}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--house", default="game_trace/datahouse")
    build.add_argument("--encoder", required=True)
    build.add_argument("--weapon", default="laser")
    build.add_argument("--device", default="cuda")
    build.add_argument("--chunk", type=int, default=64)
    build.add_argument("--limit", type=int)
    verify = sub.add_parser("verify")
    verify.add_argument("--house", default="game_trace/datahouse")
    verify.add_argument("--encoder", required=True)
    verify.add_argument("--device", default="cuda")
    verify.add_argument("--samples", type=int, default=9)
    args = parser.parse_args(argv)
    if args.command == "build":
        build_features(house_dir=args.house, encoder_path=args.encoder,
                       weapon=args.weapon, device=args.device, chunk=args.chunk,
                       limit=args.limit)
    else:
        verify_sample(house_dir=args.house, encoder_path=args.encoder,
                      device=args.device, samples=args.samples)


if __name__ == "__main__":
    main()
