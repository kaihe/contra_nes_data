import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import pytest
import yaml

from agent.sampler import ActionSampler, load_prior_artifact
from util.build_action_prior import build_artifact, collect_committed_traces


def _trace(path, actions):
    np.savez_compressed(path, actions=np.asarray(actions, dtype=np.uint8))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Blob:
    def __init__(self, name, data, generation=1):
        self.name = name
        self._data = data
        self.generation = generation

    def download_as_bytes(self):
        return self._data

    def download_to_filename(self, filename):
        Path(filename).write_bytes(self._data)


class _Bucket:
    def __init__(self, blobs):
        self._blobs = blobs

    def blob(self, name, generation=None):
        blob = self._blobs[name]
        if generation is not None:
            assert blob.generation == generation
        return blob


class _Client:
    def __init__(self, bucket):
        self._bucket = bucket

    def bucket(self, name):
        assert name == "contra_nes_trace"
        return self._bucket

    def list_blobs(self, bucket, prefix=""):
        return [blob for name, blob in self._bucket._blobs.items()
                if name.startswith(prefix)]


def _committed_batch(tmp_path, worker, batch_id, traces):
    prefix = f"root/batches/{worker}/{batch_id}"
    members = []
    tar_path = tmp_path / f"{batch_id}.tar"
    with tarfile.open(tar_path, "w") as archive:
        for name, actions in traces:
            npz = tmp_path / name
            _trace(npz, actions)
            archive.add(npz, arcname=f"traces/{name}")
            members.append({
                "fingerprint": hashlib.sha256(name.encode()).hexdigest(),
                "member": f"traces/{name}",
                "sha256": _sha256(npz),
            })
    archive = tmp_path / f"{batch_id}.tar.zst"
    subprocess.run(["zstd", "-q", "-f", str(tar_path), "-o", str(archive)], check=True)
    manifest = {"traces": members}
    manifest_bytes = json.dumps(manifest).encode()
    marker = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "archive_sha256": _sha256(archive),
        "object_generations": {"manifest.json": 3, "traces.tar.zst": 4},
    }
    return {
        f"{prefix}/COMMITTED.json": _Blob(
            f"{prefix}/COMMITTED.json", json.dumps(marker).encode(), 5),
        f"{prefix}/manifest.json": _Blob(
            f"{prefix}/manifest.json", manifest_bytes, 3),
        f"{prefix}/traces.tar.zst": _Blob(
            f"{prefix}/traces.tar.zst", archive.read_bytes(), 4),
    }


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


def test_artifact_applies_recorded_uniform_smoothing(tmp_path):
    _, actions, names, _ = ActionSampler._level_config(1)
    trace = tmp_path / "one.npz"
    _trace(trace, [actions[0], actions[0]])
    artifact = build_artifact(1, [str(trace)])
    artifact["smooth"] = 0.1
    path = tmp_path / "prior.yaml"
    path.write_text(yaml.safe_dump(artifact, sort_keys=False))

    pmf, _ = load_prior_artifact(str(path), actions, names)

    assert np.allclose(pmf.sum(axis=1), 1)
    assert np.all(pmf > 0)
    assert pmf[0, 0] > pmf[0, 1]


def test_level1_uses_committed_prior_without_trace_glob(monkeypatch):
    monkeypatch.setattr("agent.sampler.glob.glob",
                        lambda pattern: pytest.fail(f"unexpected trace scan: {pattern}"))
    sampler = ActionSampler.for_level(1)
    assert sampler.prior_sha256
    assert not np.allclose(sampler.prior_pmf, sampler.uniform_pmf)


def test_level2_uses_committed_prior_without_trace_glob(monkeypatch):
    monkeypatch.setattr("agent.sampler.glob.glob",
                        lambda pattern: pytest.fail(f"unexpected trace scan: {pattern}"))
    sampler = ActionSampler.for_level(2)
    assert sampler.prior_sha256
    assert not np.allclose(sampler.prior_pmf, sampler.uniform_pmf)


def test_level3_uses_smoothed_committed_prior_without_trace_glob(monkeypatch):
    monkeypatch.setattr("agent.sampler.glob.glob",
                        lambda pattern: pytest.fail(f"unexpected trace scan: {pattern}"))
    sampler = ActionSampler.for_level(3)
    assert sampler.prior_sha256
    assert sampler.level == 3
    assert np.all(sampler.prior_pmf > 0)
    assert not np.allclose(sampler.prior_pmf, sampler.uniform_pmf)


def test_level4_uses_committed_prior_without_trace_glob(monkeypatch):
    monkeypatch.setattr("agent.sampler.glob.glob",
                        lambda pattern: pytest.fail(f"unexpected trace scan: {pattern}"))
    sampler = ActionSampler.for_level(4)
    assert sampler.prior_sha256
    assert sampler.level == 4
    assert not np.allclose(sampler.prior_pmf, sampler.uniform_pmf)


def test_gcs_collector_keeps_unique_committed_fingerprints(tmp_path):
    _, actions, _, _ = ActionSampler._level_config(2)
    first = _committed_batch(tmp_path, "cloud-a", "batch-a", [
        ("one.npz", [actions[0], actions[1]]),
        ("two.npz", [actions[1], actions[2]]),
    ])
    second = _committed_batch(tmp_path, "cloud-b", "batch-b", [
        ("one.npz", [actions[0], actions[1]]),
        ("three.npz", [actions[2], actions[0]]),
    ])
    client = _Client(_Bucket({**first, **second}))
    paths, meta = collect_committed_traces(
        "gs://contra_nes_trace/root", tmp_path / "out", client=client)
    assert meta["source_batch_count"] == 2
    assert meta["source_duplicate_traces"] == 1
    assert len(paths) == 3
    artifact = build_artifact(2, paths, source=meta)
    assert artifact["seed_trace_count"] == 3
    assert artifact["source_gcs_root"] == "gs://contra_nes_trace/root"
