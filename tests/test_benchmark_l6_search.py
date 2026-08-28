from util.benchmark_l6_search import ARMS, HIGH_COMPUTE_SCOUT


def test_level6_screen_starts_from_level5_winner():
    assert next(iter(ARMS)) == "l5_winner"
    assert ARMS["l5_winner"].rollouts == 4
    assert ARMS["l5_winner"].rollout_len == 24
    assert ARMS["l5_winner"].settle_margin == 8
    assert ARMS["l5_winner"].max_rewind == 8


def test_high_compute_scout_starts_from_old_contra_defaults():
    baseline = HIGH_COMPUTE_SCOUT[next(iter(HIGH_COMPUTE_SCOUT))]

    assert baseline.rollouts == 64
    assert baseline.rollout_len == 48
    assert baseline.settle_margin == 16
    assert baseline.max_rewind == 60
