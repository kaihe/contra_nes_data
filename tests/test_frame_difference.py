import json

import numpy as np
import torch

from datahouse.encoder import load_temporal_encoder
from datahouse.frame_difference import FrameDifferenceAutoencoder, export_encoder
from datahouse.one_token_baseline import OneTokenAutoencoder
from datahouse.temporal_boss import _encode_episode


CONFIG = {
    "depth": 2, "proj_ch": 4, "hiddim": 512, "aux_size": 32,
    "head_depth": 2, "image_size": 256, "minres": 4,
}


def test_zero_delta_matches_identically_initialized_rgb_model():
    torch.manual_seed(0)
    reference = OneTokenAutoencoder(CONFIG)
    torch.manual_seed(0)
    candidate = FrameDifferenceAutoencoder(CONFIG)
    images = torch.randint(0, 256, (2, 3, 224, 240)).float().div(255)
    with torch.no_grad():
        expected = reference(images)[2]
        actual = candidate(images, images)[2]
    assert torch.equal(actual, expected)
    first = candidate.encoder.view_backbone.convs[0]
    assert first.in_channels == 6
    assert torch.count_nonzero(first.weight[:, 3:]) == 0


def test_exported_temporal_encoder_matches_training_model(tmp_path, monkeypatch):
    monkeypatch.setattr("datahouse.frame_difference.PARAMETERS", 2_447_628)
    torch.manual_seed(0)
    model = FrameDifferenceAutoencoder(CONFIG)
    source = tmp_path / "source.pt"
    torch.save({"config": CONFIG}, source)
    training = tmp_path / "latest.pt"
    torch.save({"model": model.state_dict(),
                "config": {"architecture_source": str(source)}}, training)
    output = tmp_path / "bundle" / "encoder.pt"
    spec = export_encoder(training, output)
    exported = load_temporal_encoder(output)
    images = torch.randint(0, 256, (3, 224, 240, 3), dtype=torch.uint8)
    current = images.permute(0, 3, 1, 2).float().div(255)
    previous = torch.cat((current[:1], current[:-1]), dim=0)
    with torch.no_grad():
        expected = model.encode_pair(current, previous)
        actual = exported.encode_sequence(images)
    assert torch.equal(actual, expected)
    assert json.loads((output.parent / "spec.json").read_text()) == spec


def test_boss_sequence_zeros_goal_and_first_frame_delta():
    class Recorder:
        def __init__(self):
            self.pairs = []

        def encode_pair(self, current, previous):
            self.pairs.append((current.cpu().clone(), previous.cpu().clone()))
            return torch.zeros((len(current), 512), device=current.device)

    images = np.arange(5, dtype=np.uint8)[:, None, None, None]
    images = np.broadcast_to(images, (5, 2, 3, 3)).copy()
    recorder = Recorder()
    tokens = _encode_episode(recorder, images, device="cpu", chunk=2)
    current = torch.cat([pair[0] for pair in recorder.pairs])
    previous = torch.cat([pair[1] for pair in recorder.pairs])
    assert tokens.shape == (5, 512)
    assert torch.equal(previous[0], current[0])       # goal
    assert torch.equal(previous[1], current[1])       # first decision frame
    assert torch.equal(previous[2:], current[1:-1])   # later decision frames
