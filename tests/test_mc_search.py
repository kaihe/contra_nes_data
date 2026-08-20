import gzip
import hashlib
import os

import numpy as np
import pytest
import yaml

from agent import mc_search
from agent.sampler import ActionSampler


def test_level2_uses_tuned_prior_and_costs():
    sampler = ActionSampler.for_level(2)

    assert not np.allclose(sampler.prior_pmf, sampler.uniform_pmf)
    assert sampler.reward_config.reward_weights["F"] == pytest.approx(-0.02)
    assert sampler.reward_config.reward_weights["J"] == pytest.approx(-0.02)


def test_winning_trace_name_matches_dataset_glob(tmp_path, monkeypatch):
    class Finished:
        done = True

    actions = [np.zeros(9, dtype=np.uint8)]
    effort = mc_search.SearchEffort(search_steps=1)
    monkeypatch.setattr(mc_search, "TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(mc_search, "make_search_env", lambda level, obs_type: FakeEnv())
    monkeypatch.setattr(mc_search, "search_and_play",
                        lambda *args, **kwargs: (actions, Finished(), [0.0], effort))

    path = mc_search._run_one_search(
        level=1, rollouts=1, rollout_len=2, max_time=1, max_rewind=1,
        max_actions=2, goal="level_up", workers=1, settle_margin=0,
    )

    assert os.path.basename(path).startswith("win_level1_")
    assert os.path.exists(path)


def test_search_can_start_from_supplied_savestate(tmp_path, monkeypatch):
    class Finished:
        done = True

    actions = [np.zeros(9, dtype=np.uint8)]
    effort = mc_search.SearchEffort(search_steps=1)
    env = FakeEnv()
    monkeypatch.setattr(mc_search, "make_search_env", lambda level, obs_type: env)
    monkeypatch.setattr(mc_search, "search_and_play",
                        lambda *args, **kwargs: (actions, Finished(), [0.0], effort))
    trace_path = tmp_path / "boss.npz"

    result = mc_search._run_one_search(
        level=1, rollouts=1, rollout_len=2, max_time=1, max_rewind=1,
        max_actions=2, goal="level_up", workers=1, settle_margin=0,
        initial_emu_state=b"boss-state", trace_path=str(trace_path),
        trace_metadata={"src_trace": "root.npz"},
    )

    assert result == str(trace_path)
    assert env.em.state == b"boss-state"
    with np.load(trace_path, allow_pickle=True) as d:
        assert bytes(d["initial_state"]) == b"boss-state"
        assert str(d["src_trace"]) == "root.npz"
        assert str(d["prior_sha256"]) == ActionSampler.for_level(1).prior_sha256


def test_load_initial_state_checks_manifest_and_returns_lineage(tmp_path):
    state_path = tmp_path / "partial_spread.state"
    state = b"boss-state"
    with gzip.open(state_path, "wb") as fh:
        fh.write(state)
    digest = hashlib.sha256(state).hexdigest()
    manifest = {
        "seed": 7,
        "states": [{
            "file": state_path.name, "state_sha256": digest,
            "source_task": "train-task", "boss_hp_start": 32, "skip": 3,
        }],
    }
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(manifest))

    loaded, metadata = mc_search.load_initial_state(str(state_path))

    assert loaded == state
    assert metadata["source_task"] == "train-task"
    assert metadata["boss_hp_start"] == 32
    assert metadata["state_bank_seed"] == 7
    assert metadata["source_skip"] == 3
    assert metadata["initial_state_sha256"] == digest


class FakeEmulator:
    def __init__(self):
        self.state = b"state"

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state


class FakeData:
    def update_ram(self):
        pass


class FakeEnv:
    def __init__(self):
        self.em = FakeEmulator()
        self.data = FakeData()

    def close(self):
        pass


def test_boss_entry_goal_fires_only_on_scene_edge(monkeypatch):
    search = object.__new__(mc_search._Search)
    search.goal = "boss_entry"
    monkeypatch.setattr(mc_search, "boss_scene", lambda ram: bool(ram[0]))

    assert search._reached_goal(np.array([0]), np.array([1]), 1)
    assert not search._reached_goal(np.array([1]), np.array([1]), 1)
    assert not search._reached_goal(np.array([0]), np.array([0]), 1)
