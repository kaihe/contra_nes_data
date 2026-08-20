"""Boss HP, metadata and train-only reverse-curriculum generation."""

import glob

import numpy as np
import pytest
import yaml

from agent import reward
from agent.boss_search import (batch_requests, build_state_bank, capture_start,
                               train_sources)
from agent.mc_search import SearchEffort, save_trace
from env.constant import (ADDR_ENEMY_HP, ADDR_ENEMY_TYPE, ADDR_LEVEL,
                          ADDR_WEAPON)
from env.utility import boss_enemy_present, boss_hp
from task_maker.export_hf import _META_PASSTHROUGH
from task_maker.kill_boss import KillBossMaker


def test_boss_hp_sums_only_active_objective_slots():
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_LEVEL] = 0
    ram[ADDR_ENEMY_HP:ADDR_ENEMY_HP + 16] = 0xff
    ram[ADDR_ENEMY_TYPE] = 0x10       # level-1 boss objective
    ram[ADDR_ENEMY_HP] = 16
    ram[ADDR_ENEMY_TYPE + 1] = 0x11   # another level-1 objective
    ram[ADDR_ENEMY_HP + 1] = 7
    ram[ADDR_ENEMY_TYPE + 2] = 0x05   # regular soldier, not counted
    ram[ADDR_ENEMY_HP + 2] = 9
    ram[ADDR_ENEMY_TYPE + 3] = 0x10   # inactive sentinel, not counted
    ram[ADDR_ENEMY_HP + 3] = 0xf0

    assert boss_enemy_present(ram)
    assert boss_hp(ram) == 23
    assert KillBossMaker.boss_hp(ram) == 23


def test_boss_metadata_is_exported():
    assert {"weapon", "rapid", "boss_hp_start", "offset_frac"} <= \
        set(_META_PASSTHROUGH)


def test_boss_search_reward_uses_hp_and_disables_forward_scroll(monkeypatch):
    class ZeroEvent:
        def trigger(self, pre, cur):
            return 0

    class PushRight:
        tag = "push_right"

        def trigger(self, pre, cur):
            return 100

    monkeypatch.setattr(reward, "enemy_hp_deltas", lambda pre, cur: (0.0, 5.0))
    monkeypatch.setattr(reward, "advance_style", lambda level: "forward")
    monkeypatch.setattr(reward, "boss_scene", lambda ram: True)
    monkeypatch.setattr(reward, "MARCH_EVENTS", {"forward": [PushRight()]})
    for name in ("EV_SPREAD_PICK", "EV_SPREAD_LOSE", "EV_RAPID_FIRE",
                 "EV_LEVELUP", "EV_DIE"):
        monkeypatch.setattr(reward, name, ZeroEvent())

    ram = np.zeros(0x800, dtype=np.uint8)
    parts = reward.reward_components(ram, ram, reward.DEFAULT_REWARD_WEIGHTS)

    assert parts["boss_hp"] == pytest.approx(5.0)
    assert parts["push_right"] == 0.0


def _minimal_task(path, *, split, label="boss_level1", **meta):
    np.savez_compressed(
        path, actions=np.zeros((1, 9), dtype=np.uint8),
        initial_state=np.zeros(4, dtype=np.uint8), label=label, level=0,
        skip=3, start_step=0, end_step=0, src_trace=path.name, split=split,
        **meta,
    )


def test_train_sources_rejects_validation_and_other_labels(tmp_path):
    _minimal_task(tmp_path / "train.npz", split="train")
    _minimal_task(tmp_path / "val.npz", split="val")
    _minimal_task(tmp_path / "kill.npz", split="train", label="kill_sniper")
    _minimal_task(tmp_path / "derived.npz", split="train", source_task="train")

    assert train_sources(str(tmp_path / "*.npz")) == [str(tmp_path / "train.npz")]


def test_batch_schedule_covers_every_full_source_and_shards_are_disjoint():
    paths = ["c.npz", "a.npz", "b.npz"]
    whole = batch_requests(paths, full_per_source=1, partial_runs=7, seed=9)
    shards = [batch_requests(paths, full_per_source=1, partial_runs=7, seed=9,
                             num_shards=4, shard_index=i) for i in range(4)]

    assert len(whole) == 10
    assert [r.source_path for r in whole[:3]] == sorted(paths)
    assert all(r.full for r in whole[:3])
    assert not any(r.full for r in whole[3:])
    assert {r.request_id for r in whole} == {
        r.request_id for shard in shards for r in shard
    }
    assert sum(len(shard) for shard in shards) == len(whole)


def test_build_state_bank_writes_one_full_state_per_weapon(tmp_path, monkeypatch):
    paths = []
    weapons = ["Regular", "Spread"]
    for index, weapon in enumerate(weapons):
        path = tmp_path / f"source{index}.npz"
        _minimal_task(path, split="train", weapon=weapon)
        paths.append(str(path))

    def fake_capture(path, *, full, rng):
        from agent.boss_search import BossStart
        from task_maker.base import load_task
        source = load_task(path)
        weapon = str(source.meta["weapon"])
        return BossStart(path, source, f"{weapon}-{full}".encode(),
                         0 if full else 1, 0.0 if full else 0.5,
                         64 if full else 32, weapon, weapon == "Spread", 200)

    monkeypatch.setattr("agent.boss_search.capture_start", fake_capture)
    out = tmp_path / "bank"
    entries = build_state_bank(paths, str(out), seed=11)

    assert len(entries) == 2
    assert {e["name"] for e in entries} == {
        "full_regular", "full_spread"
    }
    assert all((out / e["file"]).exists() for e in entries)
    manifest = yaml.safe_load((out / "manifest.yaml").read_text())
    assert manifest["seed"] == 11
    assert len(manifest["states"]) == 2
    assert all(e["stage"] == "full" for e in manifest["states"])


def test_save_trace_metadata_round_trip_and_reserved_keys(tmp_path):
    path = tmp_path / "trace.npz"
    save_trace(b"state", [np.zeros(9, dtype=np.uint8)], str(path),
               effort=SearchEffort(search_steps=1),
               metadata={"src_trace": "root.npz", "offset_frac": 0.5})
    with np.load(path, allow_pickle=True) as d:
        assert str(d["src_trace"]) == "root.npz"
        assert float(d["offset_frac"]) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="reserved key"):
        save_trace(b"state", [], str(path), metadata={"actions": "bad"})


@pytest.mark.parametrize("full", [True, False])
def test_capture_real_train_start_has_provenance_and_hp(full):
    # Emulator-backed regression: pins save-state/decision alignment using the
    # real local dataset while remaining a sub-second single-source replay.
    path = next(
        p for p in glob.glob("game_trace/tasks/boss/boss_level1/*.npz")
        if str(np.load(p, allow_pickle=True)["split"]) == "train"
    )
    start = capture_start(path, full=full, rng=np.random.default_rng(7))

    assert start.source.split == "train"
    assert start.boss_hp_start > 0
    assert start.weapon in {"Regular", "MachineGun", "Flamethrower", "Spread", "Laser"}
    if full:
        assert start.offset == 0
        assert start.offset_frac == 0.0
    else:
        assert 0 < start.offset < len(start.source.actions)
        assert 0.0 < start.offset_frac < 1.0
