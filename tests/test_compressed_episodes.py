import hashlib
import io
import json
import tarfile

import av
import numpy as np
from PIL import Image

from datahouse.compressed_episodes import build_shard
from datahouse.full_level import sha256_file


def _png(image):
    destination = io.BytesIO()
    Image.fromarray(image).save(destination, format="PNG")
    return destination.getvalue()


def test_build_shard_writes_lossless_episode_video_and_coordinates(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir(); output.mkdir()
    tar_path = source / "shard-00000.tar"
    images = [np.full((224, 240, 3), value, np.uint8) for value in (3, 17)]
    with tarfile.open(tar_path, "w") as archive:
        for index, image in enumerate(images):
            key = f"episode-000-{index:03d}"
            row = {"key": key, "trace_fingerprint": "episode-000", "split": "train",
                   "frame_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                   "player": [[index, 2]], "enemy": [], "projectile": []}
            for suffix, payload in (("png", _png(image)),
                                    ("json", json.dumps(row).encode())):
                info = tarfile.TarInfo(f"{key}.{suffix}"); info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
    marker = {"ordinal": 0, "file": tar_path.name, "sha256": sha256_file(tar_path),
              "frames": 2, "episodes": 1, "splits": {"train": 1},
              "snapshot_sha256": "snapshot"}
    (source / "shard-00000.json").write_text(json.dumps(marker))
    result = build_shard(source, output, "shard-00000.json")
    assert result["status"] == "built"
    with tarfile.open(output / "shard-00000.tar") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        video = archive.extractfile("episode-000.obs.mkv").read()
        coordinate_bytes = archive.extractfile("episode-000.entities.npz").read()
    coordinates = np.load(io.BytesIO(coordinate_bytes))
    with av.open(io.BytesIO(video)) as container:
        decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    assert all(np.array_equal(actual, expected) for actual, expected in zip(decoded, images))
    assert manifest["decode_window"] == 512
    assert coordinates["player_xy"].tolist() == [[0, 2], [1, 2]]
    assert coordinates["player_offsets"].tolist() == [0, 1, 2]
    assert build_shard(source, output, "shard-00000.json")["status"] == "skipped"
