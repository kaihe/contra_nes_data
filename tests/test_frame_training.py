import torch

from datahouse.frame_training import (entity_targets, frame_loss, learning_rate,
                                      wsd_learning_rate)


def test_entity_targets_and_frame_loss_are_finite():
    metadata = [{"player": [[120, 100]], "enemy": [], "projectile": [[10, 20]]}]
    targets, weights = entity_targets(metadata, device=torch.device("cpu"))
    assert targets.shape == (1, 3, 32, 32)
    assert weights.shape == (1, 1, 224, 240)
    assert 1 < float(weights.max()) <= 16
    images = torch.rand(1, 3, 224, 240)
    total, parts = frame_loss(images.clone(), images, torch.zeros_like(targets),
                              targets, weights)
    assert torch.isfinite(total)
    assert float(parts["pixel"]) == 0


def test_learning_rate_warms_up_then_decays():
    assert learning_rate(0, base=3e-4, warmup=2000, total=20_000) < \
        learning_rate(1999, base=3e-4, warmup=2000, total=20_000)
    assert learning_rate(2000, base=3e-4, warmup=2000, total=20_000) == 3e-4
    assert learning_rate(20_000, base=3e-4, warmup=2000, total=20_000) == 0


def test_wsd_holds_a_stable_phase_then_anneals_to_zero():
    schedule = dict(base=3e-4, warmup=200, decay_start=1600, total=2000)
    assert wsd_learning_rate(0, **schedule) < wsd_learning_rate(199, **schedule)
    assert wsd_learning_rate(199, **schedule) == 3e-4          # warmup ends at base
    assert wsd_learning_rate(800, **schedule) == 3e-4          # stable phase is flat
    assert wsd_learning_rate(1599, **schedule) == 3e-4
    assert wsd_learning_rate(1800, **schedule) < 3e-4          # decay branch
    assert wsd_learning_rate(2000, **schedule) == 0


def test_wsd_stable_phase_is_independent_of_the_declared_total():
    """The property the ladder rests on: one stable run serves every budget."""
    short = wsd_learning_rate(1000, base=3e-4, warmup=200, decay_start=4000, total=5000)
    long = wsd_learning_rate(1000, base=3e-4, warmup=200, decay_start=16000, total=20000)
    assert short == long == 3e-4
    cosine_short = learning_rate(1000, base=3e-4, warmup=200, total=5000)
    cosine_long = learning_rate(1000, base=3e-4, warmup=200, total=20000)
    assert cosine_short != cosine_long                          # cosine cannot
