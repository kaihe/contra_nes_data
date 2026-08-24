import io
import json
import tarfile

import numpy as np
import torch

from datahouse.boss_spread import _stage_shard, _write_token_shard


class FakeEncoder:
    def encode(self, images):
        # One deterministic two-wide token per image, preserving batch order.
        values = images[:, 0, 0, 0].float()
        return torch.stack([values, values + 1], dim=1)


def test_stage_workers_must_be_positive(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="stage workers"):
        _stage_shard([], tmp_path, image_size=128, workers=0)


def test_staged_images_are_batched_and_split_back_into_episodes(tmp_path):
    staged = []
    for index, length in enumerate((3, 2)):
        path = tmp_path / f"episode-{index}.npz"
        images = np.full((length + 1, 4, 4, 3), index + 1, dtype=np.uint8)
        actions = np.arange(length, dtype=np.int64)
        meta = {"uid": f"ep-{index}", "length": length, "action_len": length,
                "trace_fingerprint": f"fp-{index}", "raw_trace": f"raw-{index}"}
        np.savez_compressed(path, images=images, actions=actions,
                            meta=np.asarray(json.dumps(meta)))
        staged.append(path)

    output = tmp_path / "token.tar"
    result = _write_token_shard(staged, str(output), encoder=FakeEncoder(),
                                device="cpu", chunk=4)

    assert result["episodes"] == 2
    assert result["frames"] == 5
    with tarfile.open(output) as tar:
        first = np.load(io.BytesIO(tar.extractfile("ep-0.tokens.npy").read()))
        second = np.load(io.BytesIO(tar.extractfile("ep-1.tokens.npy").read()))
    assert first.shape == (4, 2)
    assert second.shape == (3, 2)
    assert np.all(first[:, 0] == 1)
    assert np.all(second[:, 0] == 2)
