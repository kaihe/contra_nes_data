"""Build a committed per-level action-prior artifact from a fixed trace snapshot.

Workers load this YAML from the git checkout. They never scan human recordings
or rebuild the prior from live spool output. Refresh it from committed GCS
batches:

    python -m util.build_action_prior --level 2 --out src/agent/priors/level2.yaml \\
        --gcs-root gs://contra_nes_trace/contra-mc-tracehouse/schema-v1/level2/full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import yaml

from agent.sampler import _combo_index, action_table_sha256
from agent.sampler import ActionSampler


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("GCS root must start with gs://")
    bucket_name, _, prefix = uri[5:].partition("/")
    if not bucket_name:
        raise ValueError("GCS root must include a bucket name")
    return bucket_name, prefix.strip("/")


def collect_committed_traces(gcs_root: str, destination: Path, *, client=None) -> tuple[list[str], dict]:
    """Download unique committed traces; ``COMMITTED.json`` is the visibility gate."""
    if client is None:
        from google.cloud import storage
        client = storage.Client()
    bucket_name, prefix = _parse_gs(gcs_root)
    bucket = client.bucket(bucket_name)
    listing_prefix = f"{prefix}/batches/" if prefix else "batches/"
    markers = sorted(
        (blob for blob in client.list_blobs(bucket, prefix=listing_prefix)
         if blob.name.endswith("/COMMITTED.json")),
        key=lambda blob: blob.name,
    )
    if not markers:
        raise ValueError(f"no committed batches under {gcs_root}")

    destination.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    paths: list[str] = []
    batches: list[str] = []
    duplicates = 0
    for marker_blob in markers:
        marker = json.loads(marker_blob.download_as_bytes())
        base = marker_blob.name.rsplit("/", 1)[0]
        relative = base[len(listing_prefix):] if base.startswith(listing_prefix) else base
        batches.append(relative)
        manifest_blob = bucket.blob(
            f"{base}/manifest.json",
            generation=marker["object_generations"]["manifest.json"],
        )
        manifest_bytes = manifest_blob.download_as_bytes()
        if _sha256_bytes(manifest_bytes) != marker["manifest_sha256"]:
            raise RuntimeError(f"manifest hash mismatch: gs://{bucket_name}/{base}")
        rows = []
        for row in json.loads(manifest_bytes)["traces"]:
            fingerprint = row["fingerprint"]
            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)
            rows.append(row)
        if not rows:
            continue
        archive = destination / f"{relative.replace('/', '_')}.tar.zst"
        archive.parent.mkdir(parents=True, exist_ok=True)
        bucket.blob(
            f"{base}/traces.tar.zst",
            generation=marker["object_generations"]["traces.tar.zst"],
        ).download_to_filename(str(archive))
        if _sha256_file(archive) != marker["archive_sha256"]:
            raise RuntimeError(f"archive hash mismatch: gs://{bucket_name}/{base}")
        tar_path = archive.with_suffix("")
        subprocess.run(["zstd", "-q", "-d", "-f", str(archive), "-o", str(tar_path)],
                       check=True)
        wanted = {row["member"]: row for row in rows}
        extracted = 0
        with tarfile.open(tar_path) as tar:
            for member in tar:
                row = wanted.get(member.name)
                if row is None:
                    continue
                payload = tar.extractfile(member).read()
                if _sha256_bytes(payload) != row["sha256"]:
                    raise RuntimeError(f"member hash mismatch: {member.name}")
                target = destination / f"{row['fingerprint']}.npz"
                target.write_bytes(payload)
                paths.append(str(target))
                extracted += 1
        tar_path.unlink()
        archive.unlink()
        if extracted != len(rows):
            raise RuntimeError(
                f"archive supplied {extracted}/{len(rows)} traces: gs://{bucket_name}/{base}")
    if not paths:
        raise ValueError(f"committed batches under {gcs_root} contained no traces")
    meta = {
        "source_gcs_root": gcs_root,
        "source_batch_count": len(batches),
        "source_batches": batches,
        "source_duplicate_traces": duplicates,
    }
    return sorted(paths), meta


def build_artifact(level: int, paths: list[str], *, source: dict | None = None) -> dict:
    """Return exact transition counts and provenance for sorted winning traces."""
    if not paths:
        raise ValueError("at least one seed trace is required")
    _, actions, names, _ = ActionSampler._level_config(level)
    combo_to_idx = {_combo_index(action): i for i, action in enumerate(actions)}
    counts = np.zeros((len(actions), len(actions)), dtype=np.int64)
    trace_hashes, action_steps, candidate_pairs, skipped = [], 0, 0, 0
    for name in sorted(paths):
        payload = Path(name).read_bytes()
        trace_hashes.append(hashlib.sha256(payload).hexdigest())
        with np.load(name, allow_pickle=True) as trace:
            recorded = np.asarray(trace["actions"], dtype=np.uint8)
        if recorded.ndim != 2 or recorded.shape[1] != 9:
            raise ValueError(f"unexpected action shape {recorded.shape}: {name}")
        indices = [combo_to_idx.get(_combo_index(action)) for action in recorded]
        for previous, current in zip(indices, indices[1:]):
            candidate_pairs += 1
            if previous is None or current is None:
                skipped += 1
            else:
                counts[previous, current] += 1
        action_steps += len(recorded)
    source_digest = hashlib.sha256("\n".join(sorted(trace_hashes)).encode()).hexdigest()
    artifact = {
        "format_version": 1,
        "level": level,
        "mode": "bigram",
        "action_names": list(names),
        "action_table_sha256": action_table_sha256(actions),
        "seed_trace_count": len(paths),
        "seed_action_steps": action_steps,
        "candidate_pairs": candidate_pairs,
        "included_pairs": int(counts.sum()),
        "skipped_pairs": skipped,
        "source_set_sha256": source_digest,
    }
    if source:
        artifact.update(source)
    artifact["transition_counts"] = counts.tolist()
    return artifact


def write_artifact(artifact: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(artifact, sort_keys=False))
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, required=True, choices=range(1, 9))
    parser.add_argument("--out", required=True)
    parser.add_argument("--gcs-root",
                        help="committed GCS prefix; unique fingerprints become the seed set")
    parser.add_argument("--scratch", default=None,
                        help="directory for GCS downloads (default: <out>.scratch)")
    parser.add_argument("traces", nargs="*")
    args = parser.parse_args()
    if bool(args.gcs_root) == bool(args.traces):
        raise SystemExit("provide either --gcs-root or local trace paths, not both")
    source = None
    if args.gcs_root:
        scratch = Path(args.scratch or (args.out + ".scratch"))
        paths, source = collect_committed_traces(args.gcs_root, scratch)
    else:
        paths = args.traces
    write_artifact(build_artifact(args.level, paths, source=source), Path(args.out))


if __name__ == "__main__":
    main()
