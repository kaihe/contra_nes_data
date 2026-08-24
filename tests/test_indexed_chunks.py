import io
import json
import tarfile

import numpy as np
import torch
from PIL import Image

from datahouse.indexed_chunks import (IndexedChunkDataset, build_chunk,
                                      collate_indexed, targets_from_metadata)
from datahouse.frame_training import entity_targets, prepare_targets


def _source_corpus(root):
    image = np.zeros((224, 240, 3), dtype=np.uint8)
    image[10, 20] = (1, 2, 3)
    metadata = {"key": "trace-000", "player": [[120, 100]], "enemy": [],
                "projectile": [[10, 20]]}
    archive_path = root / "shard-00000.tar"
    with tarfile.open(archive_path, "w") as archive:
        png = io.BytesIO(); Image.fromarray(image).save(png, format="PNG")
        payloads = {"trace-000.png": png.getvalue(),
                    "trace-000.json": json.dumps(metadata).encode()}
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name); info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    marker = {"file": archive_path.name, "frames": 1, "ordinal": 0,
              "sha256": "test-sha", "splits": {"train": 1, "validation": 0,
                                                  "test": 0}}
    (root / "shard-00000.json").write_text(json.dumps(marker))
    return image, metadata


def test_numpy_targets_match_online_torch_targets():
    metadata = {"player": [[120, 100]], "enemy": [[8, 9]],
                "projectile": [[10, 20], [30, 40]]}
    expected, _ = entity_targets([metadata], device=torch.device("cpu"))
    actual = targets_from_metadata(metadata)
    np.testing.assert_allclose(actual, expected[0].numpy(), rtol=2e-5, atol=2e-6)


def test_build_and_stream_indexed_chunk(tmp_path):
    source = tmp_path / "source"; output = tmp_path / "indexed"
    source.mkdir(); output.mkdir()
    image, metadata = _source_corpus(source)
    result = build_chunk(source, output, "shard-00000.json")
    assert result["status"] == "built"
    assert build_chunk(source, output, "shard-00000.json")["status"] == "skipped"
    frames = np.load(output / "chunk-00000" / "frames.npy", mmap_mode="r")
    targets = np.load(output / "chunk-00000" / "targets.npy", mmap_mode="r")
    np.testing.assert_array_equal(frames[0], image)
    np.testing.assert_allclose(targets[0], targets_from_metadata(metadata), atol=5e-4)

    row = next(iter(IndexedChunkDataset(output, "train")))
    raw, target, keys = collate_indexed([row])
    assert raw.shape == (1, 224, 240, 3)
    assert target.shape == (1, 3, 32, 32)
    assert keys == ["trace-000"]
    moved, weights = prepare_targets(target, device=torch.device("cpu"))
    assert moved.dtype == torch.float32
    assert weights.shape == (1, 1, 224, 240)
