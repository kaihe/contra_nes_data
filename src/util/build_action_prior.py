"""Build a committed per-level action-prior artifact from a fixed trace snapshot."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import yaml

from agent.sampler import _combo_index, action_table_sha256
from agent.sampler import ActionSampler


def build_artifact(level: int, paths: list[str]) -> dict:
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
    return {
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
        "transition_counts": counts.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, required=True, choices=range(1, 9))
    parser.add_argument("--out", required=True)
    parser.add_argument("traces", nargs="+")
    args = parser.parse_args()
    artifact = build_artifact(args.level, args.traces)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(artifact, sort_keys=False))
    os.replace(temporary, destination)


if __name__ == "__main__":
    main()
