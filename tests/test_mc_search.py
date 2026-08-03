import os

import numpy as np
import pytest

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
