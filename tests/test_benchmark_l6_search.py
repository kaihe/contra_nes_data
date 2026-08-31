from util.benchmark_l6_search import ARMS


def test_level6_screen_starts_from_repaired_old_contra_baseline():
    assert next(iter(ARMS)) == "old_contra_baseline"
    assert ARMS["old_contra_baseline"].rollouts == 64
    assert ARMS["old_contra_baseline"].rollout_len == 48
    assert ARMS["old_contra_baseline"].settle_margin == 16
    assert ARMS["old_contra_baseline"].max_rewind == 60


def test_level6_targeted_grid_matches_documented_arms():
    assert set(ARMS) == {
        "old_contra_baseline", "rewind_30", "rewind_15", "rewind_45", "rollouts_32",
        "rollouts_96",
    }
