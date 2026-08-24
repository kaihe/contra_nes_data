"""Resumable Level 4 full-clear throughput benchmark.

Stage ``baseline`` is the production Level 2 search shape (64/48/8/30). Stage
``scale-up`` holds settle margin 8 and rewind 30 and varies rollouts and
lookahead. Working files stay under ``tmp/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from agent.mc_search import _run_one_search, get_level
from env.event import event_by_tag
from util.replay import make_env, rewind_state, step_env


EV_LEVEL_TRANSITION = event_by_tag("in_level_transition")


@dataclass(frozen=True)
class Arm:
    rollouts: int
    rollout_len: int
    settle_margin: int
    max_rewind: int


STAGES = {
    "baseline": {
        "l2_production": Arm(64, 48, 8, 30),
    },
    "scale-up": {
        "baseline": Arm(64, 48, 8, 30),
        "wider": Arm(96, 48, 8, 30),
        "narrower": Arm(32, 48, 8, 30),
        "deeper": Arm(64, 64, 8, 30),
        "shallower": Arm(64, 32, 8, 30),
    },
    "l1-shape": {
        "few_long": Arm(16, 48, 8, 30),
        "l1_fast": Arm(16, 24, 8, 15),
        "l1_more": Arm(32, 24, 8, 15),
        "few_mid": Arm(16, 32, 8, 15),
    },
    "confirm": {
        "few_long": Arm(16, 48, 8, 30),
        "narrower": Arm(32, 48, 8, 30),
        "baseline": Arm(64, 48, 8, 30),
    },
    "settle-rewind": {
        "few_long": Arm(16, 48, 8, 30),
        "settle_4": Arm(16, 48, 4, 30),
        "settle_16": Arm(16, 48, 16, 30),
        "rewind_15": Arm(16, 48, 8, 15),
        "rewind_45": Arm(16, 48, 8, 45),
    },
}


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        output.write(json.dumps(row, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def replay_level_up(path: Path, *, start_level: int = 4) -> dict:
    """Replay a search NPZ and confirm the Level 4 → 5 transition edge."""
    with np.load(path, allow_pickle=True) as data:
        initial = bytes(data["initial_state"])
        actions = np.asarray(data["actions"], dtype=np.uint8)
        skip = int(data["skip"])
        sampled_actions = int(data["sampled_actions"])
        search_wall_s = float(data["search_wall_s"])
    fingerprint = hashlib.sha256(initial + actions.tobytes()).hexdigest()
    env = make_env()
    rewind_state(env, initial)
    reached = False
    try:
        for action in actions:
            pre = env.unwrapped.get_ram().copy()
            step_env(env, action, skip)
            cur = env.unwrapped.get_ram().copy()
            if EV_LEVEL_TRANSITION.trigger(pre, cur) or get_level(cur) != start_level:
                reached = True
                break
    finally:
        env.close()
    return {
        "replay_valid": reached,
        "trace_steps": len(actions),
        "sampled_actions": sampled_actions,
        "search_wall_s": search_wall_s,
        "fingerprint": fingerprint,
    }


def round_order(arm_names: list[str], attempt: int, seed: int) -> list[str]:
    """Baseline first each round; shuffle the remaining arms."""
    first, rest = arm_names[0], list(arm_names[1:])
    random.Random(seed + attempt).shuffle(rest)
    return [first, *rest]


def summarize(rows: list[dict], arms: dict[str, Arm] | None = None) -> dict:
    names = list(arms) if arms is not None else []
    for row in rows:
        if row["arm"] not in names:
            names.append(row["arm"])
    result = {}
    for arm in names:
        group = [row for row in rows if row["arm"] == arm]
        wall = sum(float(row["attempt_wall_s"]) for row in group)
        valid = [row for row in group if row.get("replay_valid")]
        fingerprints = [row["fingerprint"] for row in valid if row.get("fingerprint")]
        result[arm] = {
            "attempts": len(group),
            "search_wins": sum(bool(row["search_win"]) for row in group),
            "replay_valid": len(valid),
            "attempt_wall_s": wall,
            "wins_per_hour": 3600 * len(valid) / wall if wall else 0.0,
            "mean_wall_s_per_valid_win": (
                wall / len(valid) if valid else None),
            "exact_duplicates": len(fingerprints) - len(set(fingerprints)),
        }
    return result


def write_summary(path: Path, rows: list[dict], arms: dict[str, Arm] | None = None) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summarize(rows, arms), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(*, out: Path, stage: str, attempts: int, seed: int, workers: int,
        max_time: int, max_actions: int) -> None:
    arms = STAGES[stage]
    rows = read_rows(out / "results.jsonl")
    done = {(row["arm"], int(row["attempt"])) for row in rows}
    sequence = 0
    names = list(arms)
    for attempt in range(attempts):
        order = round_order(names, attempt, seed)
        for arm_name in order:
            sequence += 1
            if (arm_name, attempt) in done:
                continue
            arm = arms[arm_name]
            trace = out / "traces" / arm_name / f"attempt-{attempt:03d}.npz"
            trace.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{sequence}/{attempts * len(arms)}] {arm_name} "
                  f"attempt={attempt}", flush=True)
            started = time.perf_counter()
            won = _run_one_search(
                level=4, rollouts=arm.rollouts, rollout_len=arm.rollout_len,
                settle_margin=arm.settle_margin, max_rewind=arm.max_rewind,
                max_time=max_time, max_actions=max_actions, goal="level_up",
                workers=workers, verbose=False,
                instance_id=attempt * len(arms) + names.index(arm_name),
                trace_path=str(trace),
                trace_metadata={"experiment": "l4-search-efficiency",
                                "experiment_arm": arm_name,
                                "experiment_attempt": attempt,
                                "experiment_seed": seed},
            )
            row = {
                "arm": arm_name, "config": asdict(arm), "attempt": attempt,
                "stage": stage, "seed": seed, "workers": workers,
                "max_time": max_time, "max_actions": max_actions,
                "level": 4, "goal": "level_up",
                "attempt_wall_s": time.perf_counter() - started,
                "search_win": won is not None,
                "trace_path": str(trace) if won else None,
            }
            if won:
                row.update(replay_level_up(trace))
            append_row(out / "results.jsonl", row)
            rows.append(row)
            write_summary(out / "summary.json", rows, arms)
            print(json.dumps(row, sort_keys=True), flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), default="baseline")
    parser.add_argument("--out", default="tmp/level4-search-efficiency-baseline")
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-time", type=int, default=600)
    parser.add_argument("--max-actions", type=int, default=6000)
    args = parser.parse_args(argv)
    if min(args.attempts, args.workers, args.max_time, args.max_actions) < 1:
        raise SystemExit("attempt and resource limits must be positive")
    run(out=Path(args.out), stage=args.stage, attempts=args.attempts, seed=args.seed,
        workers=args.workers, max_time=args.max_time, max_actions=args.max_actions)


if __name__ == "__main__":
    main()
