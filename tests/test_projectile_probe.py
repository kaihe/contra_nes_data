import json
import numpy as np
import torch

from datahouse.projectile_probe import (DirectImageCNN, TokenProbe, _average_precision,
                                        _metric_rows, summarize_results)


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


def test_presence_average_precision():
    present = np.asarray([True, False, True, False])
    scores = np.asarray([0.9, 0.8, 0.7, 0.1])
    assert np.isclose(_average_precision(present, scores), (1 + 2 / 3) / 2)


def test_summary_uses_paired_episode_bootstrap(tmp_path):
    for arm, offset, seeds in (("published_control", -0.1, (0,)),
                               ("token_probe", 0.0, (0, 1, 2)),
                               ("direct_image", 0.2, (0,))):
        for seed in seeds:
            metrics = {}
            for weapon in ("Spread", "Laser"):
                metrics[weapon] = {
                    "dice": 0.5 + offset, "mse_skill": 0.4 + offset,
                    "peak_hit": 0.6 + offset,
                    "episode_dice": {"1": 0.4 + offset, "2": 0.6 + offset},
                }
            payload = {"arm": arm, "seed": seed, "elapsed_s": 10 + seed,
                       "metrics": metrics}
            (tmp_path / f"{arm}-seed{seed}.json").write_text(json.dumps(payload))
    summary = summarize_results(tmp_path, bootstrap_samples=100, bootstrap_seed=1)
    assert summary["matched_seeds"] == [0]
    for weapon in ("Spread", "Laser"):
        gap = summary["weapons"][weapon]["direct_minus_token_episode_dice"]
        assert np.isclose(gap["mean"], 0.2)
        assert np.allclose(gap["ci95"], (0.2, 0.2))
