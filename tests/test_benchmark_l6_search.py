from util.benchmark_l6_search import ARMS


def test_level6_screen_starts_from_level5_winner():
    assert next(iter(ARMS)) == "l5_winner"
    assert ARMS["l5_winner"].rollouts == 4
    assert ARMS["l5_winner"].rollout_len == 24
    assert ARMS["l5_winner"].settle_margin == 8
    assert ARMS["l5_winner"].max_rewind == 8
