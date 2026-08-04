"""Full-fight candidate diversity and release planning."""

import gzip
import hashlib
import io
import json
import os
import tarfile

import numpy as np
import pytest
import yaml

from task_maker.boss_release import (
    DiversityFeatures,
    _action_ngrams,
    _js_distance,
    build_release,
    diversity_distance,
    frame_balanced_shards,
    import_traces,
    load_full_bank,
)
from task_maker.base import load_task


def _task(path, *, length, weapon):
    np.savez_compressed(
        path,
        actions=np.zeros((length, 9), dtype=np.uint8),
        initial_state=np.arange(8, dtype=np.uint8),
        label="boss_level1", level=0, skip=3,
        start_step=0, end_step=length - 1,
        src_trace=path.name, split="train",
        weapon=weapon,
    )


def _one_json_tar(path, uid):
    payload = json.dumps({"uid": uid}).encode()
    with tarfile.open(path, "w") as tar:
        info = tarfile.TarInfo(uid + ".json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))


def test_full_bank_rejects_partial_and_verifies_state_checksum(tmp_path):
    full = b"full-state"
    partial = b"partial-state"
    for name, state in (("full.state", full), ("partial.state", partial)):
        with gzip.open(tmp_path / name, "wb") as fh:
            fh.write(state)
    manifest = {
        "states": [
            {
                "file": "full.state", "stage": "full", "split": "train",
                "offset_frac": 0.0, "state_sha256": hashlib.sha256(full).hexdigest(),
            },
            {
                "file": "partial.state", "stage": "partial", "split": "train",
                "offset_frac": 0.5,
                "state_sha256": hashlib.sha256(partial).hexdigest(),
            },
        ],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest))

    bank = load_full_bank(str(path))

    assert list(bank) == [hashlib.sha256(full).hexdigest()]

    manifest["states"][0]["state_sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_full_bank(str(path))


def test_action_and_combined_distance_identify_same_and_different_traces():
    noop = np.zeros((8, 9), dtype=np.uint8)
    firing = noop.copy()
    firing[::2, 0] = 1
    state = np.zeros((20, 8), dtype=np.float32)
    a = DiversityFeatures("a", "a", 8, "Regular", _action_ngrams(noop), state)
    same = DiversityFeatures("same", "a", 8, "Regular", _action_ngrams(noop), state)
    other = DiversityFeatures(
        "other", "b", 8, "Regular", _action_ngrams(firing), state + 0.5)

    assert _js_distance(a.ngrams, same.ngrams) == 0.0
    assert diversity_distance(a, same) == 0.0
    assert diversity_distance(a, other) > 0.2


def test_frame_balanced_shards_use_frame_count_and_spread_weapons(tmp_path):
    paths = []
    for index, (length, weapon) in enumerate([
        (80, "Regular"), (70, "Regular"),
        (60, "Spread"), (50, "Spread"),
    ]):
        path = tmp_path / f"task{index}.npz"
        _task(path, length=length, weapon=weapon)
        paths.append(str(path))

    shards = frame_balanced_shards(paths, target_frames=150)
    lengths = [sum(len(np.load(p)["actions"]) for p in shard) for shard in shards]
    weapons = [set(str(np.load(p)["weapon"]) for p in shard) for shard in shards]

    assert len(shards) == 2
    assert sorted(p for shard in shards for p in shard) == sorted(paths)
    assert max(lengths) - min(lengths) <= 20
    assert all(group == {"Regular", "Spread"} for group in weapons)


def test_import_real_full_source_replays_to_verified_train_task(tmp_path):
    with open("src/agent/states/boss_level1/manifest.yaml") as fh:
        manifest = yaml.safe_load(fh)
    entry = manifest["states"][0]
    source_path = (
        "game_trace/tasks/boss/boss_level1/" + entry["source_task"] + ".npz"
    )
    source = load_task(source_path)
    raw = tmp_path / "raw.npz"
    np.savez_compressed(
        raw,
        actions=source.actions,
        initial_state=np.frombuffer(source.initial_state, dtype=np.uint8),
        skip=np.array(source.skip, dtype=np.int32),
        outcome="win",
    )

    paths = import_traces(
        [str(raw)], batch_id="test", out_root=str(tmp_path / "tasks"))

    assert len(paths) == 1
    imported = load_task(paths[0])
    assert imported.split == "train"
    assert imported.meta["stage"] == "full"
    assert imported.meta["source_task"] == source.uid
    assert imported.meta["initial_state_sha256"] == entry["state_sha256"]


def test_release_copies_validation_and_records_accepted_hashes(tmp_path, monkeypatch):
    baseline_task = tmp_path / "baseline.npz"
    _task(baseline_task, length=5, weapon="Regular")
    out = tmp_path / "release"
    candidate_dir = out / "tasks" / "boss_level1"
    candidate_dir.mkdir(parents=True)
    candidate_task = candidate_dir / "candidate.npz"
    _task(candidate_task, length=4, weapon="Spread")
    baseline_tar = tmp_path / "baseline.tar"
    validation_tar = tmp_path / "validation.tar"
    _one_json_tar(baseline_tar, "baseline")
    _one_json_tar(validation_tar, "validation")

    monkeypatch.setattr(
        "task_maker.boss_release.import_traces",
        lambda *args, **kwargs: [str(candidate_task)],
    )
    row = {
        "uid": "candidate", "path": str(candidate_task), "weapon": "Spread",
        "length": 4, "fingerprint": "fingerprint", "nearest_uid": "baseline",
        "nearest_distance": 0.25, "accepted": True, "reason": "accepted",
    }
    monkeypatch.setattr(
        "task_maker.boss_release.select_diverse",
        lambda *args, **kwargs: ([str(candidate_task)], [row]),
    )

    def fake_shard(paths, dst, *, codec):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        payload = "\n".join(sorted(os.path.basename(p) for p in paths)).encode()
        with open(dst, "wb") as fh:
            fh.write(payload)
        return len(payload), len(paths), sum(len(load_task(p).actions) for p in paths)

    monkeypatch.setattr("task_maker.boss_release._atomic_shard", fake_shard)

    manifest = build_release(
        trace_paths=["unused"], batch_id="test-v1", out_dir=str(out),
        baseline_train=str(baseline_tar), validation=str(validation_tar),
        task_pattern=str(tmp_path / "*.npz"), target_frames=100,
    )

    copied_validation = out / "hf" / "boss-val-00000.tar"
    assert copied_validation.read_bytes() == validation_tar.read_bytes()
    assert manifest["baseline_train_episodes"] == 1
    assert manifest["accepted_generated_episodes"] == 1
    assert manifest["accepted_generated_tasks"][0]["sha256"] == \
        hashlib.sha256(candidate_task.read_bytes()).hexdigest()
    assert manifest["train_shards"][0]["episodes"] == 2
