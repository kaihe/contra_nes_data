import hashlib
import json
from pathlib import Path
import base64

import numpy as np

from agent.mc_search import SearchEffort, save_trace
from worker.legacy_import import LegacyTraceImporter, recover_boss_loadout
from worker.search_loop import GCSUploader, WorkerLoop


class RecordingUploader:
    def __init__(self):
        self.uploads = []

    def upload(self, batch_dir, worker_id, batch_id):
        self.uploads.append((batch_dir, worker_id, batch_id))


def make_search():
    sequence = 0

    def search(path: Path):
        nonlocal sequence
        action = np.zeros(9, dtype=np.uint8)
        action[sequence % 9] = 1
        save_trace(
            b"state", [action], str(path), effort=SearchEffort(
                search_steps=1, boss_weapon="Spread", boss_rapid=True,
                boss_entry_step=0,
            ), goal="level_up",
        )
        sequence += 1
        return str(path)

    return search


def test_worker_seals_exact_batch_and_uploads_in_background(tmp_path):
    uploader = RecordingUploader()
    loop = WorkerLoop(tmp_path, uploader, make_search(), worker_id="worker-a", batch_size=3)
    try:
        assert loop.run(max_wins=3) == 3
        loop.upload_queue.join()
    finally:
        loop.close()

    assert len(uploader.uploads) == 1
    batch = uploader.uploads[0][0]
    manifest = json.loads((batch / "manifest.json").read_text())
    assert manifest["trace_count"] == 3
    assert len(manifest["traces"]) == 3
    assert manifest["traces"][0]["boss_weapon"] == "Spread"
    assert manifest["traces"][0]["boss_rapid"] is True
    assert (batch / "traces.tar.zst").is_file()


def test_restart_continues_partial_batch_and_explicit_flush_seals_it(tmp_path):
    first = WorkerLoop(tmp_path, RecordingUploader(), make_search(),
                       worker_id="worker-a", batch_size=100)
    try:
        assert first.run(max_wins=2) == 2
    finally:
        first.close()

    uploader = RecordingUploader()
    resumed = WorkerLoop(tmp_path, uploader, make_search(),
                         worker_id="worker-a", batch_size=100)
    resumed._start_uploader()
    try:
        assert resumed._trace_count(resumed._open_batch()) == 2
        assert resumed.flush()
        resumed.upload_queue.join()
    finally:
        resumed.close()

    manifest = json.loads((uploader.uploads[0][0] / "manifest.json").read_text())
    assert manifest["trace_count"] == 2


def test_gcs_uploader_verifies_payloads_and_commits_last(tmp_path):
    batch = tmp_path / "batch-a"
    batch.mkdir()
    (batch / "traces.tar.zst").write_bytes(b"archive")
    (batch / "manifest.json").write_text(json.dumps({"traces": [{}, {}]}))

    class FakeBlob:
        def __init__(self, name):
            self.name = name
            self.uploads = []
            self.generation = len(bucket.blobs) + 10

        def upload_from_filename(self, filename, **kwargs):
            source = Path(filename)
            self.uploads.append((source, kwargs))
            self.size = source.stat().st_size
            digest = hashlib.md5(source.read_bytes()).digest()
            self.md5_hash = base64.b64encode(digest).decode("ascii")

        def reload(self):
            pass

    class FakeBucket:
        name = "trace-bucket"

        def __init__(self):
            self.blobs = []

        def blob(self, name):
            blob = FakeBlob(name)
            self.blobs.append(blob)
            return blob

    class FakeClient:
        def bucket(self, name):
            assert name == "trace-bucket"
            return bucket

    bucket = FakeBucket()
    uploader = GCSUploader("gs://trace-bucket/root", client=FakeClient())
    uploader.upload(batch, "worker-a", "batch-a")

    assert bucket.blobs[-1].name.endswith("/COMMITTED.json")
    assert all(blob.uploads[0][1]["if_generation_match"] == 0
               for blob in bucket.blobs)
    marker = json.loads((batch / "COMMITTED.json").read_text())
    assert marker["trace_count"] == 2
    assert set(marker["object_generations"]) == {"traces.tar.zst", "manifest.json"}


def test_legacy_import_preserves_npz_and_enriches_manifest(tmp_path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    for index in range(2):
        action = np.zeros(9, dtype=np.uint8)
        action[index] = 1
        save_trace(b"state", [action],
                   str(source_dir / f"legacy-{index}.npz"))
    before = {path.name: path.read_bytes() for path in source_dir.glob("*.npz")}
    spool = tmp_path / "spool"
    importer = LegacyTraceImporter(
        [str(source_dir / "*.npz")], spool,
        enrich=lambda _: {"boss_weapon": "Laser", "boss_rapid": True,
                          "boss_entry_step": 42, "boss_metadata_source": "test"},
    )
    uploader = RecordingUploader()
    loop = WorkerLoop(spool, uploader, importer, worker_id="legacy", batch_size=2)
    try:
        assert loop.run() == 2
        loop.upload_queue.join()
    finally:
        importer.close()
        loop.close()

    assert {path.name: path.read_bytes() for path in source_dir.glob("*.npz")} == before
    manifest = json.loads((uploader.uploads[0][0] / "manifest.json").read_text())
    assert manifest["traces"][0]["boss_weapon"] == "Laser"
    assert manifest["traces"][0]["boss_rapid"] is True
    assert manifest["traces"][0]["legacy_source_sha256"]

    resumed = LegacyTraceImporter([str(source_dir / "*.npz")], spool, enrich=lambda _: {})
    resumed_loop = WorkerLoop(spool, RecordingUploader(), resumed,
                              worker_id="legacy", batch_size=2)
    try:
        assert resumed_loop.run() == 0
    finally:
        resumed.close()
        resumed_loop.close()


def test_legacy_import_accepts_exact_sources_and_static_scope(tmp_path):
    selected = tmp_path / "selected.npz"
    excluded = tmp_path / "excluded.npz"
    for path in (selected, excluded):
        save_trace(b"state", [np.zeros(9, dtype=np.uint8)], str(path))

    spool = tmp_path / "spool"
    importer = LegacyTraceImporter(
        [], spool, source_paths=[selected],
        enrich=lambda _: {"boss_weapon": "Spread"},
        static_metadata={"trace_scope": "boss_fight"},
    )
    uploader = RecordingUploader()
    loop = WorkerLoop(spool, uploader, importer, worker_id="boss", batch_size=1)
    try:
        assert loop.run() == 1
        loop.upload_queue.join()
    finally:
        importer.close()
        loop.close()

    manifest = json.loads((uploader.uploads[0][0] / "manifest.json").read_text())
    assert len(manifest["traces"]) == 1
    assert manifest["traces"][0]["trace_scope"] == "boss_fight"
    assert manifest["traces"][0]["legacy_source_file"] == selected.name


def test_legacy_replay_recovers_spread_rapid_at_boss_edge(tmp_path, monkeypatch):
    trace = tmp_path / "legacy.npz"
    np.savez_compressed(trace, initial_state=np.frombuffer(b"state", dtype=np.uint8),
                        actions=np.zeros((1, 9), dtype=np.uint8), skip=np.array(1))

    class FakeEnv:
        def __init__(self):
            self.unwrapped = self
            self.ram = np.zeros(0xAB, dtype=np.uint8)

        def get_ram(self):
            return self.ram

    env = FakeEnv()
    monkeypatch.setattr("worker.legacy_import.rewind_state", lambda *_: None)
    monkeypatch.setattr("worker.legacy_import.boss_scene", lambda ram: bool(ram[0]))

    def step(*_):
        env.ram[0] = 1
        env.ram[0xAA] = 0x13

    monkeypatch.setattr("worker.legacy_import.step_env", step)

    metadata = recover_boss_loadout(trace, env)

    assert metadata == {
        "boss_weapon": "Spread", "boss_rapid": True,
        "boss_entry_step": 0, "boss_metadata_source": "replay",
    }


def test_legacy_boss_trace_maps_existing_loadout_without_replay(tmp_path):
    trace = tmp_path / "boss.npz"
    np.savez_compressed(
        trace, initial_state=np.frombuffer(b"state", dtype=np.uint8),
        actions=np.zeros((1, 9), dtype=np.uint8), weapon=np.array("Laser"),
        rapid=np.array(True),
    )

    assert recover_boss_loadout(trace, env=None) == {
        "boss_weapon": "Laser", "boss_rapid": True,
        "boss_entry_step": 0, "boss_metadata_source": "legacy_trace",
    }
