import hashlib
import io
import json
import tarfile

import numpy as np
import pytest
from PIL import Image

from datahouse.compressed_episodes import build_shard
from datahouse.compressed_loader import (CompressedEpisodeDataset,
                                         CompressedFramePairDataset,
                                         is_compressed_corpus)
from datahouse.full_level import sha256_file
from datahouse.frame_training import FrameTarDataset, frame_loader


def _png(image):
    destination = io.BytesIO()
    Image.fromarray(image).save(destination, format="PNG")
    return destination.getvalue()


def _frame(value):
    image = np.zeros((224, 240, 3), np.uint8)
    image[:] = value
    image[value % 224, value % 240] = (255, 255, 255)
    return image


def _source_shard(root, ordinal, episodes):
    """Write one per-frame PNG tar the way experiment 0012 froze the corpus."""
    tar_path = root / f"shard-{ordinal:05d}.tar"
    frames = splits = 0
    counts = {}
    with tarfile.open(tar_path, "w") as archive:
        for uid, split, length in episodes:
            counts[split] = counts.get(split, 0) + 1
            for index in range(length):
                image = _frame(frames + index)
                key = f"{uid}-{index:03d}"
                row = {"key": key, "trace_fingerprint": uid, "split": split,
                       "frame_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                       "player": [[index, 2]],
                       "enemy": [[1, index], [2, 3]] if index % 2 else [],
                       "projectile": []}
                for suffix, payload in (("png", _png(image)),
                                        ("json", json.dumps(row).encode())):
                    info = tarfile.TarInfo(f"{key}.{suffix}")
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
            frames += length
    marker = {"ordinal": ordinal, "file": tar_path.name, "sha256": sha256_file(tar_path),
              "frames": frames, "episodes": len(episodes), "splits": counts,
              "snapshot_sha256": "snapshot"}
    (root / f"shard-{ordinal:05d}.json").write_text(json.dumps(marker))
    return splits


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus")
    source, output = root / "source", root / "output"
    source.mkdir(); output.mkdir()
    _source_shard(source, 0, [("episode-a", "train", 7), ("episode-b", "validation", 5)])
    _source_shard(source, 1, [("episode-c", "train", 4)])
    for ordinal in (0, 1):
        build_shard(source, output, f"shard-{ordinal:05d}.json")
    return source, output


def test_is_compressed_corpus_distinguishes_layouts(corpus):
    source, output = corpus
    assert is_compressed_corpus(output)
    assert not is_compressed_corpus(source)


def test_window_frames_and_coordinates_match_the_source_corpus(corpus):
    source, output = corpus
    expected = {}
    with tarfile.open(source / "shard-00000.tar") as archive:
        for member in archive:
            key, suffix = member.name.rsplit(".", 1)
            payload = archive.extractfile(member).read()
            if suffix == "json":
                expected.setdefault(key, {}).update(json.loads(payload))
            else:
                expected.setdefault(key, {})["png"] = payload
    dataset = CompressedEpisodeDataset(output, "validation", window=2)
    rows = list(dataset)
    assert [row[1]["key"] for row in rows] == [f"episode-b-{i:03d}" for i in range(5)]
    for image, meta in rows:
        source_row = expected[meta["key"]]
        source_image = np.asarray(Image.open(io.BytesIO(source_row["png"])).convert("RGB"))
        assert np.array_equal(image, source_image)
        assert meta["player"] == [tuple(xy) for xy in source_row["player"]]
        assert meta["enemy"] == [tuple(xy) for xy in source_row["enemy"]]
        assert meta["projectile"] == []
        assert meta["split"] == "validation"
        assert meta["trace_fingerprint"] == "episode-b"


def test_evaluation_split_makes_one_ordered_pass(corpus):
    _, output = corpus
    dataset = CompressedEpisodeDataset(output, "train", shuffle=False, loop=False,
                                       window=3)
    keys = [meta["key"] for _, meta in dataset]
    assert keys == [f"episode-a-{i:03d}" for i in range(7)] + \
                   [f"episode-c-{i:03d}" for i in range(4)]


def test_training_split_shuffles_within_windows_and_repeats(corpus):
    _, output = corpus
    dataset = CompressedEpisodeDataset(output, "train", window=4)
    iterator = iter(dataset)
    keys = [next(iterator)[1]["key"] for _ in range(33)]
    assert len(set(keys)) == 11                      # 11 train frames, then a new epoch
    assert keys[:11] != sorted(keys[:11])            # window-local shuffle
    assert sorted(keys[:11]) == sorted(set(keys))    # every frame once per epoch


def test_temporal_pairs_stay_consecutive_across_decode_windows(corpus):
    _, output = corpus
    dataset = CompressedFramePairDataset(output, "validation", window=2)
    rows = list(dataset)
    assert len(rows) == 5
    for index, (previous, current, meta) in enumerate(rows):
        assert meta["key"] == f"episode-b-{index:03d}"
        if index == 0:
            assert np.array_equal(previous, current)
        else:
            assert np.array_equal(previous, rows[index - 1][1])


def test_frame_loader_dispatches_to_the_episode_reader(corpus):
    _, output = corpus
    loader = frame_loader(output, "validation", batch=2, workers=0)
    images, metadata, keys = next(iter(loader))
    assert tuple(images.shape) == (2, 224, 240, 3)
    assert keys == [meta["key"] for meta in metadata] == ["episode-b-000", "episode-b-001"]


def test_per_frame_reader_refuses_episode_shards(corpus):
    _, output = corpus
    with pytest.raises(RuntimeError, match="episode shards"):
        FrameTarDataset(output, "train")._shards()
