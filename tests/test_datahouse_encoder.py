import torch

from datahouse.encoder import FrameEncoder


def test_frame_encoder_rejects_non_rgb_uint8_input():
    encoder = FrameEncoder({"image_size": 8, "hiddim": 4, "depth": 1,
                            "minres": 4, "proj_ch": 2})
    with torch.no_grad():
        out = encoder.encode(torch.zeros((2, 8, 8, 3), dtype=torch.uint8))
    assert out.shape == (2, 4)
    try:
        encoder.encode(torch.zeros((2, 8, 8, 3), dtype=torch.float32))
    except ValueError as exc:
        assert "uint8" in str(exc)
    else:
        raise AssertionError("float input must be rejected")
