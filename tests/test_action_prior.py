import hashlib

import numpy as np
import pytest
import yaml

from agent.sampler import ActionSampler, load_prior_artifact
from util.build_action_prior import build_artifact


def _trace(path, actions):
    np.savez_compressed(path, actions=np.asarray(actions, dtype=np.uint8))


def test_artifact_round_trip_and_source_order_are_stable(tmp_path):
    _, actions, names, _ = ActionSampler._level_config(1)
    first, second = tmp_path / "b.npz", tmp_path / "a.npz"
    _trace(first, [actions[0], actions[1], actions[1]])
    _trace(second, [actions[1], actions[0]])
    artifact = build_artifact(1, [str(first), str(second)])
    assert artifact == build_artifact(1, [str(second), str(first)])
    assert artifact["included_pairs"] == 3
    path = tmp_path / "prior.yaml"
    path.write_text(yaml.safe_dump(artifact, sort_keys=False))

    pmf, digest = load_prior_artifact(str(path), actions, names)

    assert pmf.shape == (15, 15)
    assert np.allclose(pmf.sum(axis=1), 1)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_artifact_rejects_action_table_mismatch(tmp_path):
    _, actions, names, _ = ActionSampler._level_config(1)
    trace = tmp_path / "one.npz"
    _trace(trace, [actions[0], actions[1]])
    artifact = build_artifact(1, [str(trace)])
    artifact["action_table_sha256"] = "wrong"
    path = tmp_path / "prior.yaml"
    path.write_text(yaml.safe_dump(artifact))

    with pytest.raises(ValueError, match="action table digest"):
        load_prior_artifact(str(path), actions, names)


def test_level1_uses_committed_prior_without_trace_glob(monkeypatch):
    monkeypatch.setattr("agent.sampler.glob.glob",
                        lambda pattern: pytest.fail(f"unexpected trace scan: {pattern}"))
    sampler = ActionSampler.for_level(1)
    assert sampler.prior_sha256
    assert not np.allclose(sampler.prior_pmf, sampler.uniform_pmf)
