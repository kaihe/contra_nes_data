"""Count visible tracehouse traces from committed GCS batch markers.

``COMMITTED.json`` is the tracehouse visibility boundary and already records an
exact ``trace_count``. Counting those small markers concurrently avoids the
slow N+1 pattern of downloading every manifest and never touches archives.

Examples::

    python -m util.gcs_trace_count
    python -m util.gcs_trace_count --gcs-root gs://bucket/root/schema-v1 --json
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import re


DEFAULT_ROOT = "gs://contra_nes_trace/contra-mc-tracehouse/schema-v1"
LEVEL_RE = re.compile(r"level([1-8])$")


def parse_gs(uri: str) -> tuple[str, str]:
    """Return ``(bucket, prefix)`` for a non-empty ``gs://`` URI."""
    if not uri.startswith("gs://"):
        raise ValueError("GCS root must start with gs://")
    bucket, separator, prefix = uri[5:].partition("/")
    if not bucket:
        raise ValueError("GCS root must include a bucket name")
    return bucket, prefix.strip("/") if separator else ""


def _location(name: str) -> tuple[int, str, str]:
    """Extract level, scope, and worker from a tracehouse marker name."""
    parts = name.split("/")
    level_index = next(
        (index for index, part in enumerate(parts) if LEVEL_RE.fullmatch(part)),
        None,
    )
    if level_index is None or level_index + 1 >= len(parts):
        raise ValueError(f"commit marker has no level/scope path: {name}")
    level = int(LEVEL_RE.fullmatch(parts[level_index]).group(1))
    scope = parts[level_index + 1]
    try:
        batches_index = parts.index("batches", level_index + 2)
        worker = parts[batches_index + 1]
    except (ValueError, IndexError):
        raise ValueError(f"commit marker has no batches/<worker> path: {name}")
    return level, scope, worker


def _read_marker(blob) -> tuple[str, int]:
    marker = json.loads(blob.download_as_bytes(timeout=30))
    count = marker.get("trace_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"invalid trace_count in gs://{blob.bucket.name}/{blob.name}")
    return blob.name, count


def count_committed(gcs_root: str = DEFAULT_ROOT, *, client=None,
                    max_workers: int = 32) -> dict:
    """Return exact committed counts grouped by level, scope, and worker."""
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    bucket_name, prefix = parse_gs(gcs_root)
    if client is None:
        from google.cloud import storage
        client = storage.Client()
    bucket = client.bucket(bucket_name)
    listing_prefix = prefix + "/" if prefix else ""
    markers = sorted(
        (blob for blob in client.list_blobs(bucket, prefix=listing_prefix)
         if blob.name.endswith("/COMMITTED.json")),
        key=lambda blob: blob.name,
    )

    levels: dict[int, dict] = {}
    worker_counts = Counter()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rows = pool.map(_read_marker, markers)
        for name, trace_count in rows:
            level, scope, worker = _location(name)
            entry = levels.setdefault(level, {
                "batches": 0, "traces": 0, "scopes": Counter(),
                "workers": Counter(),
            })
            entry["batches"] += 1
            entry["traces"] += trace_count
            entry["scopes"][scope] += trace_count
            entry["workers"][worker] += trace_count
            worker_counts[worker] += trace_count

    # A schema-root query is expected to answer "each level", including levels
    # with no committed output. A level-scoped query must not invent zeros for
    # prefixes it did not inspect.
    if not any(LEVEL_RE.fullmatch(part) for part in prefix.split("/")):
        for level in range(1, 9):
            levels.setdefault(level, {
                "batches": 0, "traces": 0, "scopes": Counter(),
                "workers": Counter(),
            })

    rendered = {
        str(level): {
            "batches": row["batches"],
            "traces": row["traces"],
            "scopes": dict(sorted(row["scopes"].items())),
            "workers": dict(sorted(row["workers"].items())),
        }
        for level, row in sorted(levels.items())
    }
    return {
        "gcs_root": gcs_root.rstrip("/"),
        "committed_batches": len(markers),
        "committed_traces": sum(row["traces"] for row in levels.values()),
        "levels": rendered,
        "workers": dict(sorted(worker_counts.items())),
    }


def format_table(report: dict) -> str:
    """Render the compact operator-facing level/scope table."""
    lines = ["level  batches  full  boss  other  total"]
    for level, row in report["levels"].items():
        scopes = row["scopes"]
        full = scopes.get("full", 0)
        boss = scopes.get("boss", 0)
        other = row["traces"] - full - boss
        lines.append(
            f"{int(level):>5}  {row['batches']:>7}  {full:>4}  {boss:>4}  "
            f"{other:>5}  {row['traces']:>5}"
        )
    lines.append(
        f"total batches={report['committed_batches']} "
        f"traces={report['committed_traces']}"
    )
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gcs-root", default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=32,
                        help="concurrent marker downloads (default: 32)")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        report = count_committed(args.gcs_root, max_workers=args.workers)
    except Exception as exc:
        try:
            from google.auth.exceptions import DefaultCredentialsError
        except ImportError:
            raise exc
        if isinstance(exc, DefaultCredentialsError):
            parser.error(
                "Google Application Default Credentials are unavailable; "
                "export GOOGLE_APPLICATION_CREDENTIALS or run "
                "'gcloud auth application-default login'"
            )
        raise
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_table(report))


if __name__ == "__main__":
    main()
