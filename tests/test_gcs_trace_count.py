import json
import threading

import pytest

from util import gcs_trace_count
from util.gcs_trace_count import count_committed, format_table, parse_gs


class Blob:
    def __init__(self, bucket, name, marker, barrier=None):
        self.bucket = bucket
        self.name = name
        self.marker = marker
        self.barrier = barrier

    def download_as_bytes(self, timeout=None):
        assert timeout == 30
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        return json.dumps(self.marker).encode()


class Bucket:
    def __init__(self, name):
        self.name = name


class Client:
    def __init__(self, names_and_markers, barrier=None):
        self._bucket = Bucket("traces")
        self.blobs = [Blob(self._bucket, name, marker, barrier)
                      for name, marker in names_and_markers]

    def bucket(self, name):
        assert name == "traces"
        return self._bucket

    def list_blobs(self, bucket, prefix=""):
        assert bucket is self._bucket
        return [blob for blob in self.blobs if blob.name.startswith(prefix)]


def marker(level, scope, worker, batch, count):
    name = f"root/schema-v1/level{level}/{scope}/batches/{worker}/{batch}/COMMITTED.json"
    return name, {"trace_count": count}


def test_schema_root_counts_committed_markers_by_level_scope_and_worker():
    rows = [
        marker(1, "full", "cloud1", "a", 100),
        marker(1, "boss", "archive", "b", 384),
        marker(2, "full", "cloud2", "c", 31),
        ("root/schema-v1/level2/full/batches/cloud2/c/manifest.json", {}),
    ]

    report = count_committed("gs://traces/root/schema-v1", client=Client(rows))

    assert report["committed_batches"] == 3
    assert report["committed_traces"] == 515
    assert report["levels"]["1"]["scopes"] == {"boss": 384, "full": 100}
    assert report["levels"]["2"]["workers"] == {"cloud2": 31}
    assert report["levels"]["8"]["traces"] == 0
    assert report["workers"] == {"archive": 384, "cloud1": 100, "cloud2": 31}


def test_marker_downloads_run_concurrently():
    barrier = threading.Barrier(2)
    client = Client([
        marker(3, "full", "a", "one", 10),
        marker(3, "full", "b", "two", 20),
    ], barrier=barrier)

    report = count_committed(
        "gs://traces/root/schema-v1/level3/full", client=client, max_workers=2)

    assert report["committed_traces"] == 30


@pytest.mark.parametrize("value", [None, -1, 1.5, True, "100"])
def test_invalid_trace_count_fails_instead_of_estimating(value):
    client = Client([marker(4, "full", "worker", "bad", value)])

    with pytest.raises(ValueError, match="invalid trace_count"):
        count_committed("gs://traces/root/schema-v1", client=client)


def test_table_separates_full_boss_and_other_scopes():
    client = Client([
        marker(1, "full", "a", "one", 10),
        marker(1, "boss", "a", "two", 20),
        marker(1, "kill", "a", "three", 3),
    ])

    table = format_table(count_committed("gs://traces/root/schema-v1", client=client))

    assert "full  boss  other  total" in table
    assert "10    20      3     33" in table


def test_parse_gs_rejects_non_gcs_and_missing_bucket():
    with pytest.raises(ValueError):
        parse_gs("https://example.com")
    with pytest.raises(ValueError):
        parse_gs("gs://")


def test_cli_explains_missing_application_default_credentials(monkeypatch,
                                                               capsys):
    from google.auth.exceptions import DefaultCredentialsError

    def fail(*args, **kwargs):
        raise DefaultCredentialsError("missing")

    monkeypatch.setattr(gcs_trace_count, "count_committed", fail)

    with pytest.raises(SystemExit) as error:
        gcs_trace_count.main([])

    assert error.value.code == 2
    assert "export GOOGLE_APPLICATION_CREDENTIALS" in capsys.readouterr().err
