import hashlib
import json
from pathlib import Path
import base64

import numpy as np

from agent.mc_search import SearchEffort, save_trace
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
