import torch

from datahouse.frame_difference import FrameDifferenceAutoencoder
from datahouse.one_token_baseline import OneTokenAutoencoder


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
