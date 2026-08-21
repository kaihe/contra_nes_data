import torch

from datahouse.encoder_baseline import Metrics


def test_perfect_entity_heatmaps_score_one():
    target = torch.zeros(2, 4, 4, 4)
    target[:, :, 1, 2] = 1
    logits = torch.where(target > 0, torch.tensor(20.0), torch.tensor(-20.0))
    metrics = Metrics()
    metrics.update(logits, target)
    result = metrics.result(episodes=1, checkpoint_sha256="x",
                            collection="c", elapsed_s=1)
    for row in result["classes"].values():
        assert row["dice"] == 1.0
        assert row["mse_skill"] == 1.0
        assert row["peak_hit"] == 1.0
