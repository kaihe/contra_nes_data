from util.benchmark_l6_search import ARMS


def test_level6_screen_starts_from_full_action_scout_winner():
    assert next(iter(ARMS)) == "scout_winner"
    assert ARMS["scout_winner"].rollouts == 64
    assert ARMS["scout_winner"].rollout_len == 48
    assert ARMS["scout_winner"].settle_margin == 16
    assert ARMS["scout_winner"].max_rewind == 30


def test_level6_targeted_grid_matches_documented_arms():
    assert set(ARMS) == {
        "scout_winner", "rewind_15", "rewind_45", "rollouts_32",
        "rollouts_96", "old_contra_baseline",
    }
