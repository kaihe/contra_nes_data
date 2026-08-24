"""Stream frames from the indexed all-intra episode shards built by 0009."""

from __future__ import annotations

import io
import json
import random
import tarfile
from pathlib import Path

import av
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from datahouse.compressed_episodes import ENTITY_NAMES

DECODE_WINDOW = 512


def is_compressed_corpus(root: str | Path) -> bool:
    """True when ``root`` holds episode shards rather than per-frame PNG tars."""
    for marker in sorted(Path(root).glob("shard-*.json")):
        return "decode_window" in json.loads(marker.read_text())
    return False


def _coordinates(payload: bytes, frames: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    arrays = np.load(io.BytesIO(payload))
    table = {}
    for name in ENTITY_NAMES:
        offsets = arrays[f"{name}_offsets"]
        if len(offsets) != frames + 1:
            raise RuntimeError(f"{name} offsets cover {len(offsets) - 1} frames, "
                               f"expected {frames}")
        table[name] = (arrays[f"{name}_xy"], offsets)
    return table


class CompressedEpisodeDataset(IterableDataset):
    """Decode contiguous frame windows out of all-intra episode videos.

    One episode's video is read by tar offset and decoded in ``window``-frame
    passes; each window is shuffled in RAM before its frames are emitted, so the
    window is an I/O unit and never a model context. Training loops forever over
    shuffled shards and episodes. Every other split makes one pass in fixed shard,
    episode, and frame order, which is what makes evaluation reproducible and lets
    a validation loop end on its own.
    """

    def __init__(self, root: str | Path, split: str, *, seed: int = 0,
                 window: int = DECODE_WINDOW, shuffle: bool | None = None,
                 loop: bool | None = None):
        self.root, self.split, self.seed = Path(root), split, seed
        self.window = window
        self.shuffle = split == "train" if shuffle is None else shuffle
        self.loop = self.shuffle if loop is None else loop

    def _shards(self) -> list[Path]:
        shards = []
        for marker_path in sorted(self.root.glob("shard-*.json")):
            marker = json.loads(marker_path.read_text())
            if "decode_window" not in marker:
                raise RuntimeError(f"{self.root} holds per-frame tars, not episode "
                                   "shards; use FrameTarDataset")
            if marker["splits"].get(self.split, 0):
                shards.append(self.root / marker["file"])
        if not shards:
            raise RuntimeError(f"no {self.split} episode shards under {self.root}")
        return shards

    @staticmethod
    def _manifest(path: Path) -> dict:
        with tarfile.open(path) as archive:
            member = archive.extractfile("manifest.json")
            if member is None:
                raise RuntimeError(f"{path} has no manifest.json")
            return json.load(member)

    @staticmethod
    def _member(handle, episode: dict, suffix: str) -> bytes:
        for member in episode["members"]:
            if member["name"].endswith(suffix):
                handle.seek(member["offset"])
                payload = handle.read(member["size"])
                if len(payload) != member["size"]:
                    raise RuntimeError(f"short read of {member['name']}")
                return payload
        raise RuntimeError(f"episode {episode['uid']} has no {suffix} member")

    def _rows(self, episode: dict, table: dict, start: int, count: int) -> list[dict]:
        rows = []
        for index in range(start, start + count):
            row = {"key": f"{episode['uid']}-{index:03d}",
                   "trace_fingerprint": episode["uid"], "split": episode["split"]}
            for name, (xy, offsets) in table.items():
                row[name] = [(int(x), int(y))
                             for x, y in xy[offsets[index]:offsets[index + 1]]]
            rows.append(row)
        return rows

    def _episode(self, handle, episode: dict, rng: random.Random):
        table = _coordinates(self._member(handle, episode, ".entities.npz"),
                             episode["frames"])
        video = io.BytesIO(self._member(handle, episode, ".obs.mkv"))
        decoded = 0
        buffer: list[np.ndarray] = []
        with av.open(video) as container:
            for frame in container.decode(video=0):
                buffer.append(frame.to_ndarray(format="rgb24"))
                if len(buffer) == self.window:
                    yield from self._flush(episode, table, decoded, buffer, rng)
                    decoded += len(buffer); buffer = []
            if buffer:
                yield from self._flush(episode, table, decoded, buffer, rng)
                decoded += len(buffer)
        if decoded != episode["frames"]:
            raise RuntimeError(f"episode {episode['uid']} decoded {decoded} frames, "
                               f"manifest declares {episode['frames']}")

    def _flush(self, episode: dict, table: dict, start: int,
               buffer: list[np.ndarray], rng: random.Random):
        rows = self._rows(episode, table, start, len(buffer))
        order = list(range(len(buffer)))
        if self.shuffle:
            rng.shuffle(order)
        for index in order:
            yield buffer[index], rows[index]

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        shards = self._shards()[worker_id::worker_count]
        if not shards:
            raise RuntimeError("more loader workers than episode shards")
        rng = random.Random(self.seed + worker_id)
        epoch = 0
        while True:
            order = list(shards)
            if self.shuffle:
                rng.shuffle(order)
            for path in order:
                episodes = [episode for episode in self._manifest(path)["episodes"]
                            if episode["split"] == self.split]
                if self.shuffle:
                    rng.shuffle(episodes)
                with open(path, "rb") as handle:
                    for episode in episodes:
                        yield from self._episode(handle, episode, rng)
            if not self.loop:
                return
            epoch += 1
            rng.seed(self.seed + worker_id + epoch * 1009)
