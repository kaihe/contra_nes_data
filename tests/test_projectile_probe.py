import numpy as np
import torch

from datahouse.projectile_probe import DirectImageCNN, TokenProbe, _metric_rows


def test_probe_shapes_and_perfect_metrics():
    assert DirectImageCNN()(torch.zeros(2, 256, 256, 3, dtype=torch.uint8)).shape == (2, 32, 32)
    assert TokenProbe(8)(torch.zeros(2, 8)).shape == (2, 32, 32)
    target = torch.zeros(2, 32, 32); target[:, 4, 5] = 1
    logits = torch.where(target > 0, torch.tensor(20.0), torch.tensor(-20.0))
    rows = _metric_rows(logits, target)
    assert np.all(rows["present"])
    assert np.allclose(rows["dice"], 1)
    assert np.allclose(rows["mse_skill"], 1)
    assert np.allclose(rows["peak_hit"], 1)
