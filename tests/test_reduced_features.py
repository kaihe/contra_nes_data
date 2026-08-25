"""Reduced frozen-feature representation contract (issue #15)."""

import io
import json
import tarfile

import numpy as np
import pytest

from datahouse.boss_frames import FORMAT, _add_bytes
from datahouse.catalog import (FeatureShard, Shard, connect,
                               feature_shard_fingerprints,
                               register_feature_shard, register_shard)
from datahouse.reduced_features import (BOUNDARY, DTYPE, REPRESENTATION,
                                        write_feature_shard)


def _npy(array):
    output = io.BytesIO()
    np.save(output, array, allow_pickle=False)
    return output.getvalue()


def _catalog(tmp_path):
    db = connect(tmp_path / "catalog.sqlite")
    episodes = [("a" * 64, "uid-a", "trace-a", 3),
                ("b" * 64, "uid-b", "trace-b", 4)]
    register_shard(db, Shard(
        path="tokens.tar", sha256="1" * 64, level=1, task="boss",
        weapon="laser", encoder_sha256="2" * 64, ordinal=0,
        episodes=2, frames=7), episodes)
    return db


def _feature(ordinal=0):
    return FeatureShard(
        path=f"features-{ordinal}.tar", sha256=f"{ordinal + 3:064x}",
        level=1, task="boss", weapon="laser", representation=REPRESENTATION,
        encoder_sha256="e" * 64, boundary=BOUNDARY, dtype=DTYPE,
        channels=256, feature_height=4, feature_width=4, ordinal=ordinal,
        episodes=2, frames=7)


def test_feature_catalog_is_separate_and_versioned(tmp_path):
    db = _catalog(tmp_path)
    fingerprints = ["a" * 64, "b" * 64]
    register_feature_shard(db, _feature(), fingerprints)
    assert feature_shard_fingerprints(
        db, level=1, task="boss", weapon="laser",
        representation=REPRESENTATION, encoder_sha256="e" * 64) == set(fingerprints)
    assert db.execute("SELECT COUNT(*) FROM shards").fetchone()[0] == 1


def test_feature_catalog_rejects_duplicate_membership(tmp_path):
    db = _catalog(tmp_path)
    fingerprints = ["a" * 64, "b" * 64]
    register_feature_shard(db, _feature(), fingerprints)
    with pytest.raises(ValueError, match="already in feature shard"):
        register_feature_shard(db, _feature(1), fingerprints)


def test_writer_preserves_actions_and_records_boundary(tmp_path, monkeypatch):
    import datahouse.reduced_features as reduced

    uid, fingerprint = "episode", "a" * 64
    source = tmp_path / "frames.tar"
    actions = np.asarray([2, 4, 6], dtype=np.int64)
    metadata = {"uid": uid, "fingerprint": fingerprint, "frames": 3}
    episode = {"uid": uid, "fingerprint": fingerprint, "frames": 3}
    manifest = {"format": FORMAT, "frame_height": 224, "frame_width": 240,
                "episodes": [episode], "frames": 3}
    with tarfile.open(source, "w") as archive:
        _add_bytes(archive, f"{uid}.obs.mkv", b"video")
        _add_bytes(archive, f"{uid}.actions.npy", _npy(actions))
        _add_bytes(archive, f"{uid}.json", json.dumps(metadata).encode())
        _add_bytes(archive, "manifest.json", json.dumps(manifest).encode())

    frames = np.zeros((3, 224, 240, 3), dtype=np.uint8)
    values = np.arange(3 * 256 * 4 * 4, dtype=np.float16).reshape(3, 256, 4, 4)
    monkeypatch.setattr(reduced, "_decode_video", lambda payload: frames)
    monkeypatch.setattr(reduced, "encode_reduced", lambda *a, **kw: values)

    class Encoder:
        input_hw = (256, 256)

    destination = tmp_path / "features.tar"
    row = write_feature_shard(source, destination, encoder=Encoder(),
                              encoder_sha256="e" * 64, device="cpu", chunk=2)
    assert row["frames"] == 3 and row["fingerprints"] == [fingerprint]
    with tarfile.open(destination) as archive:
        stored_actions = np.load(io.BytesIO(
            archive.extractfile(f"{uid}.actions.npy").read()), allow_pickle=False)
        stored_features = np.load(io.BytesIO(
            archive.extractfile(f"{uid}.features.npy").read()), allow_pickle=False)
        output_manifest = json.load(archive.extractfile("manifest.json"))
    np.testing.assert_array_equal(stored_actions, actions)
    np.testing.assert_array_equal(stored_features, values)
    assert output_manifest["boundary"] == BOUNDARY
    assert output_manifest["feature_shape"] == [256, 4, 4]
    assert output_manifest["interpolation"] == "INTER_AREA"
    assert "actions[i+1]" in output_manifest["action_alignment"]
