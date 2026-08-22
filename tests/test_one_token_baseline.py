import torch

from datahouse.one_token_baseline import (BaselineMetrics, OneTokenAutoencoder,
                                          OneTokenDecoder)


def test_one_token_decoder_outputs_native_frame():
    with torch.no_grad():
        output = OneTokenDecoder()(torch.randn(2, 512))
    assert output.shape == (2, 3, 224, 240)
    assert 0 <= float(output.min()) <= float(output.max()) <= 1


def test_one_token_autoencoder_uses_native_frame_without_resize():
    config = {"image_size": 256, "minres": 4, "depth": 4, "proj_ch": 8,
              "hiddim": 512, "aux_size": 32, "entity_classes": 4,
              "head_depth": 4}
    model = OneTokenAutoencoder(config)
    assert model.encoder.input_hw == (224, 240)
    assert model.encoder.view_backbone.output_hw == (3, 3)
    assert model.encoder.proj[0].in_features == 3 * 3 * 8
    with torch.no_grad():
        reconstruction, entities, token = model(torch.rand(1, 3, 224, 240))
    assert token.shape == (1, 512)
    assert reconstruction.shape == (1, 3, 224, 240)
    assert entities.shape == (1, 3, 32, 32)


def test_perfect_baseline_metrics():
    images = torch.zeros(2, 3, 224, 240)
    targets = torch.zeros(2, 3, 32, 32)
    presence = [False, True]
    targets[1, 2, 1, 1] = 1
    probability = targets.clone()
    metrics = BaselineMetrics()
    metrics.update(images, images, probability, targets,
                   torch.ones(2, 1, 224, 240), presence)
    result = metrics.result()
    assert result["exact_rgb_pixel_accuracy"] == 1
    assert result["unweighted_mse"] == 0
    assert result["projectile_presence_ap"] == 1
    assert result["projectile_empty_fpr_0.5"] == 0
