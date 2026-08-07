import json

from util.benchmark_mc_search import (
    BASELINE,
    SearchConfig,
    confirmation_configs,
    round_schedule,
    screening_configs,
    summarize,
)


def _rows(config, *, attempts=12, wall=10.0, valid=True):
    return [{
        "stage": "screen",
        "config_id": config.uid,
        "config": {
            "rollouts": config.rollouts,
            "rollout_len": config.rollout_len,
            "settle_margin": config.settle_margin,
            "max_rewind": config.max_rewind,
        },
        "attempt": attempt,
        "win": valid,
        "replay_valid": valid,
        "attempt_wall_s": wall,
        "sampled_actions": 100,
        "trace_steps": 10,
        "fingerprint": f"{config.uid}-{attempt}",
    } for attempt in range(attempts)]


def test_screening_grid_has_27_unique_valid_cells():
    configs = screening_configs()

    assert len(configs) == 27
    assert len({config.uid for config in configs}) == 27
    assert BASELINE in configs
    assert all(config.settle_margin < config.rollout_len for config in configs)


def test_round_schedule_interleaves_one_of_every_cell():
    configs = screening_configs()[:4]
    schedule = list(round_schedule(configs, attempts=3, seed=7))

    assert len(schedule) == 12
    for attempt in range(3):
        group = schedule[attempt * 4:(attempt + 1) * 4]
        assert {config for config, _ in group} == set(configs)
        assert {index for _, index in group} == {attempt}


def test_summary_charges_failed_attempt_wall_time_to_throughput():
    config = screening_configs()[0]
    rows = _rows(config, attempts=2, wall=10.0)
    rows[1].update(win=False, replay_valid=False, fingerprint=None)

    summary = summarize(rows, "screen")[0]

    assert summary["wins"] == 1
    assert summary["replay_valid_wins"] == 1
    assert summary["total_attempt_wall_s"] == 20.0
    assert summary["mean_wall_s_per_valid_win"] == 20.0
    assert summary["wins_per_hour"] == 180.0


def test_confirmation_selects_four_fastest_perfect_cells_plus_baseline():
    candidates = [c for c in screening_configs() if c != BASELINE][:5]
    rows = []
    for index, config in enumerate(candidates):
        rows.extend(_rows(config, wall=float(index + 1)))
    rows.extend(_rows(BASELINE, wall=0.5))
    screen = summarize(rows, "screen")

    selected = confirmation_configs(screen, attempts=12)

    assert selected == candidates[:4] + [BASELINE]
