"""Zero-copy logical views over full-episode token-shard members."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np


VIEWS = ("start_to_boss", "boss_fight", "full")


def view_bounds(view: str, action_steps: int,
                boss_observation_index: int) -> tuple[slice, slice]:
    """Return aligned observation/action slices for exactly one named view."""
    if view not in VIEWS:
        raise ValueError(f"view must be exactly one of {VIEWS}; received {view!r}")
    if not 0 <= boss_observation_index <= action_steps:
        raise ValueError("boss_observation_index is outside the episode")
    if view == "start_to_boss":
        return slice(0, boss_observation_index + 1), slice(0, boss_observation_index)
    if view == "boss_fight":
        return slice(boss_observation_index, action_steps + 1), slice(
            boss_observation_index, action_steps)
    return slice(0, action_steps + 1), slice(0, action_steps)


def read_episode(shard: str | Path, uid: str, *, view: str) -> dict:
    """Read one stored episode and project its selected view in memory."""
    with tarfile.open(shard) as archive:
        meta = json.load(archive.extractfile(f"{uid}.json"))
        tokens = np.load(io.BytesIO(archive.extractfile(
            f"{uid}.tokens.npy").read()), allow_pickle=False)
        actions = np.load(io.BytesIO(archive.extractfile(
            f"{uid}.actions.npy").read()), allow_pickle=False)
    observations, targets = view_bounds(
        view, len(actions), int(meta["boss_observation_index"]))
    result = {"tokens": tokens[observations], "actions": actions[targets],
              "instruction": meta["instructions"][view], "meta": meta}
    if len(result["tokens"]) != len(result["actions"]) + 1:
        raise RuntimeError(f"unaligned episode view: {uid}/{view}")
    return result
