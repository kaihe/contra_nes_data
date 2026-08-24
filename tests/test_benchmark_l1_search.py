from util.benchmark_l1_search import summarize


def test_summary_counts_failures_and_spread_throughput():
    rows = [
        {"arm": "classic", "attempt_wall_s": 10, "search_win": True,
         "replay_valid": True, "spread_equipped": True, "fingerprint": "a"},
        {"arm": "classic", "attempt_wall_s": 20, "search_win": False},
        {"arm": "fast_spread", "attempt_wall_s": 5, "search_win": True,
         "replay_valid": True, "spread_equipped": False, "fingerprint": "b"},
    ]

    result = summarize(rows)

    assert result["classic"]["attempts"] == 2
    assert result["classic"]["spread_wins_per_hour"] == 120
    assert result["fast_spread"]["boss_entries_per_hour"] == 720
    assert result["fast_spread"]["spread_equipped"] == 0
