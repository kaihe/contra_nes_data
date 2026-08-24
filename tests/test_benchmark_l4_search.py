from util.benchmark_l4_search import round_order, summarize


def test_l4_summary_counts_failures_and_throughput():
    rows = [
        {"arm": "l2_production", "attempt_wall_s": 30, "search_win": True,
         "replay_valid": True, "fingerprint": "a"},
        {"arm": "l2_production", "attempt_wall_s": 90, "search_win": False},
    ]

    result = summarize(rows)

    assert result["l2_production"]["attempts"] == 2
    assert result["l2_production"]["search_wins"] == 1
    assert result["l2_production"]["replay_valid"] == 1
    assert result["l2_production"]["wins_per_hour"] == 30
    assert result["l2_production"]["exact_duplicates"] == 0


def test_scale_up_round_keeps_baseline_first():
    names = ["baseline", "wider", "narrower", "deeper", "shallower"]
    order = round_order(names, attempt=0, seed=20260824)
    assert order[0] == "baseline"
    assert sorted(order) == sorted(names)


def test_l1_shape_stage_exists():
    from util.benchmark_l4_search import STAGES
    assert STAGES["l1-shape"]["l1_fast"].rollouts == 16
    assert STAGES["l1-shape"]["l1_fast"].rollout_len == 24
    assert list(STAGES["l1-shape"])[0] == "few_long"


def test_confirm_stage_is_48_step_rollout_sweep():
    from util.benchmark_l4_search import STAGES
    arms = STAGES["confirm"]
    assert list(arms) == ["few_long", "narrower", "baseline"]
    assert all(arm.rollout_len == 48 and arm.settle_margin == 8
               and arm.max_rewind == 30 for arm in arms.values())


def test_settle_rewind_holds_16x48():
    from util.benchmark_l4_search import STAGES
    arms = STAGES["settle-rewind"]
    assert list(arms)[0] == "few_long"
    assert all(arm.rollouts == 16 and arm.rollout_len == 48 for arm in arms.values())
    assert arms["settle_4"].settle_margin == 4
    assert arms["rewind_45"].max_rewind == 45
