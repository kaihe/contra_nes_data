from datahouse.vq_codebook import split_rows
from datahouse.vq_train import (ContinuousAutoencoder, entity_targets,
                                learning_rate, warmup_loss)
import torch


def test_split_is_exact_disjoint_and_input_order_independent():
    rows = [{"fingerprint": f"{i:064x}"} for i in range(10_000)]
    a = split_rows(rows)
    b = split_rows(list(reversed(rows)))
    assert [row["fingerprint"] for row in a] == [row["fingerprint"] for row in b]
    assert len(a) == 1_000
    assert sum(row["split"] == "train" for row in a) == 800
    assert sum(row["split"] == "validation" for row in a) == 100
    assert sum(row["split"] == "test" for row in a) == 100


def test_continuous_autoencoder_has_four_latents_and_native_reconstruction():
    model = ContinuousAutoencoder()
    images = torch.rand(2, 3, 224, 240)
    with torch.no_grad():
        latent = model.encode(images)
        reconstruction, entities = model(images)
    assert latent.shape == (2, 256, 2, 2)
    assert reconstruction.shape == images.shape
    assert entities.shape == (2, 3, 32, 32)


def test_entity_targets_and_warmup_loss_are_finite():
    metadata = [{"player": [[120, 100]], "enemy": [], "projectile": [[10, 20]]}]
    targets, weights = entity_targets(metadata, device=torch.device("cpu"))
    assert targets.shape == (1, 3, 32, 32)
    assert weights.shape == (1, 1, 224, 240)
    assert 1 < float(weights.max()) <= 16
    images = torch.rand(1, 3, 224, 240)
    total, parts = warmup_loss(images.clone(), images, torch.zeros_like(targets),
                               targets, weights)
    assert torch.isfinite(total)
    assert float(parts["pixel"]) == 0


def test_learning_rate_warms_up_then_decays():
    assert learning_rate(0, base=3e-4, warmup=2000, total=20_000) < \
        learning_rate(1999, base=3e-4, warmup=2000, total=20_000)
    assert learning_rate(2000, base=3e-4, warmup=2000, total=20_000) == 3e-4
    assert learning_rate(20_000, base=3e-4, warmup=2000, total=20_000) == 0
