"""Frame-shard catalog contract and tar layout (doc/0021)."""

import io
import json
import sqlite3
import tarfile

import numpy as np
import pytest

from datahouse.boss_frames import FORMAT, FRAME_HW, _add_bytes, write_frame_shard
from datahouse.catalog import (FrameShard, Shard, connect, frame_shard_fingerprints,
                               register_frame_shard, register_shard,
                               token_prefix_fingerprints)


def _house(tmp_path, shards=3, per_shard=2):
    """A catalog holding token shards only, as the token producer leaves it."""
    db = connect(tmp_path / "catalog.sqlite")
    for ordinal in range(shards):
        rows = [(f"fp{ordinal}{i:02d}" + "0" * 58, f"uid-{ordinal}-{i}",
                 f"trace-{ordinal}-{i}.npz", 10 + i) for i in range(per_shard)]
        register_shard(db, Shard(
            path=f"level1/boss/spread/token-{ordinal:05d}.tar",
            sha256=f"{ordinal:064x}", level=1, task="boss", weapon="spread",
            encoder_sha256="e" * 64, ordinal=ordinal, episodes=per_shard,
            frames=sum(r[3] for r in rows)), rows)
    return db


def _frame_shard(ordinal, episodes, **kw):
    return FrameShard(path=f"level1/boss/spread/frames/frames-{ordinal:05d}.tar",
                      sha256=f"{ordinal + 100:064x}", level=1, task="boss",
                      weapon="spread", format=FORMAT, frame_height=FRAME_HW[0],
                      frame_width=FRAME_HW[1], ordinal=ordinal,
                      episodes=len(episodes), frames=10 * len(episodes), **kw)


def test_token_prefix_is_consumer_order(tmp_path):
    """The prefix must match how contra_nes_policy selects shard_counts."""
    db = _house(tmp_path, shards=3, per_shard=2)
    assert len(token_prefix_fingerprints(db, level=1, task="boss", weapon="spread",
                                         shard_count=2)) == 4
    everything = token_prefix_fingerprints(db, level=1, task="boss", weapon="spread",
                                           shard_count=3)
    assert everything[:4] == token_prefix_fingerprints(
        db, level=1, task="boss", weapon="spread", shard_count=2)
    with pytest.raises(ValueError, match="asked for 9"):
        token_prefix_fingerprints(db, level=1, task="boss", weapon="spread",
                                  shard_count=9)


def test_frame_shard_registers_against_existing_episodes(tmp_path):
    db = _house(tmp_path)
    wanted = token_prefix_fingerprints(db, level=1, task="boss", weapon="spread",
                                       shard_count=1)
    register_frame_shard(db, _frame_shard(0, wanted), wanted)
    assert frame_shard_fingerprints(db, level=1, task="boss", weapon="spread",
                                    format=FORMAT) == set(wanted)


def test_frame_shard_rejects_uncataloged_episode(tmp_path):
    """A frame release that drifted from the token release must fail loudly."""
    db = _house(tmp_path)
    stranger = ["f" * 64]
    with pytest.raises(ValueError, match="not cataloged"):
        register_frame_shard(db, _frame_shard(0, stranger), stranger)


def test_frame_shard_rejects_republished_episode(tmp_path):
    db = _house(tmp_path)
    wanted = token_prefix_fingerprints(db, level=1, task="boss", weapon="spread",
                                       shard_count=1)
    register_frame_shard(db, _frame_shard(0, wanted), wanted)
    with pytest.raises(ValueError, match="already in frame shard"):
        register_frame_shard(db, _frame_shard(1, wanted), wanted)


def test_token_shards_are_untouched_by_frame_registration(tmp_path):
    """The whole point of a separate table: policy's query must not see frame rows.

    contra_nes_policy selects on (level, task, weapon) without filtering by encoder,
    then raises StaleCache if the result mixes encoders.
    """
    db = _house(tmp_path)
    wanted = token_prefix_fingerprints(db, level=1, task="boss", weapon="spread",
                                       shard_count=1)
    before = db.execute("SELECT COUNT(*), COUNT(DISTINCT encoder_sha256) FROM shards "
                        "WHERE level=1 AND task='boss' AND weapon='spread'").fetchone()
    register_frame_shard(db, _frame_shard(0, wanted), wanted)
    after = db.execute("SELECT COUNT(*), COUNT(DISTINCT encoder_sha256) FROM shards "
                       "WHERE level=1 AND task='boss' AND weapon='spread'").fetchone()
    assert tuple(before) == tuple(after) == (3, 1)


def test_episode_identity_is_a_join(tmp_path):
    """Frame and token releases must be provably the same episode set."""
    db = _house(tmp_path, shards=3, per_shard=2)
    wanted = token_prefix_fingerprints(db, level=1, task="boss", weapon="spread",
                                       shard_count=2)
    register_frame_shard(db, _frame_shard(0, wanted), wanted)
    overlap = db.execute(
        "SELECT COUNT(*) FROM frame_shard_episodes f JOIN shard_episodes s "
        "USING (fingerprint) JOIN shards ON shards.id = s.shard_id "
        "WHERE shards.weapon='spread' AND shards.ordinal < 2").fetchone()[0]
    assert overlap == len(wanted) == 4


def test_shard_tar_offsets_address_every_member(tmp_path):
    """A reader must be able to seek to a member instead of scanning the tar."""
    destination = tmp_path / "frames-00000.tar"
    payloads = {"a.obs.mkv": b"video-bytes", "a.actions.npy": b"\x01\x02",
                "a.json": b"{}"}
    with tarfile.open(destination, "w") as archive:
        members = [_add_bytes(archive, name, payload)
                   for name, payload in payloads.items()]
    blob = destination.read_bytes()
    for member in members:
        assert blob[member["offset"]:member["offset"] + member["size"]] == \
            payloads[member["name"]]


def test_write_frame_shard_is_atomic_on_failure(tmp_path, monkeypatch):
    """A crashed shard leaves no partial tar for a resumed run to trust."""
    import datahouse.boss_frames as boss_frames

    def explode(args):
        raise RuntimeError("replay failed")

    monkeypatch.setattr(boss_frames, "_episode_job", explode)
    destination = tmp_path / "frames-00000.tar"
    with pytest.raises(RuntimeError, match="replay failed"):
        write_frame_shard([("task.npz", "f" * 64)], destination)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp-*"))


def test_written_shard_carries_a_manifest(tmp_path, monkeypatch):
    import datahouse.boss_frames as boss_frames

    def fake(args):
        _, fingerprint = args
        meta = {"schema_version": 1, "format": FORMAT, "uid": f"uid-{fingerprint[:4]}",
                "fingerprint": fingerprint, "frames": 7, "actions": 7,
                "frame_height": FRAME_HW[0], "frame_width": FRAME_HW[1]}
        return (fingerprint, b"video", np.save(io.BytesIO(), np.arange(7)) or b"acts",
                json.dumps(meta, sort_keys=True).encode())

    monkeypatch.setattr(boss_frames, "_episode_job", fake)
    destination = tmp_path / "frames-00000.tar"
    row = write_frame_shard([("t0.npz", "a" * 64), ("t1.npz", "b" * 64)], destination)
    assert row["frames"] == 14 and len(row["fingerprints"]) == 2
    with tarfile.open(destination) as archive:
        manifest = json.loads(archive.extractfile("manifest.json").read())
    assert manifest["format"] == FORMAT
    assert [e["fingerprint"] for e in manifest["episodes"]] == ["a" * 64, "b" * 64]
    assert all(len(e["members"]) == 3 for e in manifest["episodes"])
