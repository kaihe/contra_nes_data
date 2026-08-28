from util.benchmark_l5_search import ARMS, round_order, summarize


def test_round_order_keeps_l1_baseline_first_and_is_complete():
    order = round_order(attempt=2, seed=20260828)

    assert order[0] == "l1_fast"
    assert set(order) == set(ARMS)


def test_summary_counts_valid_throughput_and_duplicates():
    rows = [
        {"arm": "l1_fast", "attempt_wall_s": 20, "search_win": True,
         "replay_valid": True, "fingerprint": "same"},
        {"arm": "l1_fast", "attempt_wall_s": 40, "search_win": True,
         "replay_valid": True, "fingerprint": "same"},
        {"arm": "l1_fast", "attempt_wall_s": 60, "search_win": False},
    ]

    result = summarize(rows)["l1_fast"]

    assert result["attempts"] == 3
    assert result["search_wins"] == 2
    assert result["replay_valid"] == 2
    assert result["wins_per_hour"] == 60
    assert result["mean_wall_s_per_valid_win"] == 60
    assert result["exact_duplicates"] == 1
