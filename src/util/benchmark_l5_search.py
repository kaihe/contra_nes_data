"""Resumable Level 5 full-clear parameter screen around the Level 1 winner."""

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


# Keep the Level 1 winner first in every round; shuffle only the challengers.
ARMS = {
    "l1_fast": Arm(16, 24, 8, 15),
    "narrower": Arm(8, 24, 8, 15),
    "wider": Arm(32, 24, 8, 15),
    "shallower": Arm(16, 16, 8, 15),
    "deeper": Arm(16, 32, 8, 15),
}

STAGES = {
    "screen": ARMS,
    "breadth-lookahead": {
        "current_winner": Arm(8, 24, 8, 15),
        "rollouts_4": Arm(4, 24, 8, 15),
        "rollouts_6": Arm(6, 24, 8, 15),
        "rollouts_12": Arm(12, 24, 8, 15),
        "length_20": Arm(8, 20, 8, 15),
        "length_28": Arm(8, 28, 8, 15),
    },
}


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        output.write(json.dumps(row, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def replay_level_up(path: Path) -> dict:
    """Replay a candidate and verify its Level 5 to Level 6 transition."""
    with np.load(path, allow_pickle=True) as data:
        initial = bytes(data["initial_state"])
        actions = np.asarray(data["actions"], dtype=np.uint8)
        skip = int(data["skip"])
        sampled = int(data["sampled_actions"])
        search_wall = float(data["search_wall_s"])
    fingerprint = hashlib.sha256(initial + actions.tobytes()).hexdigest()
    env = make_env()
    rewind_state(env, initial)
    reached = False
    try:
        for action in actions:
            before = env.unwrapped.get_ram().copy()
            step_env(env, action, skip)
            after = env.unwrapped.get_ram().copy()
            if EV_LEVEL_TRANSITION.trigger(before, after) or get_level(after) != 5:
                reached = True
                break
    finally:
        env.close()
    return {
        "replay_valid": reached,
        "fingerprint": fingerprint,
        "trace_steps": len(actions),
        "sampled_actions": sampled,
        "search_wall_s": search_wall,
    }


def summarize(rows: list[dict]) -> dict:
    result = {}
    for name in ARMS:
        group = [row for row in rows if row["arm"] == name]
        valid = [row for row in group if row.get("replay_valid")]
        wall = sum(float(row["attempt_wall_s"]) for row in group)
        fingerprints = [row["fingerprint"] for row in valid]
        result[name] = {
            "attempts": len(group),
            "search_wins": sum(bool(row["search_win"]) for row in group),
            "replay_valid": len(valid),
            "attempt_wall_s": wall,
            "wins_per_hour": 3600 * len(valid) / wall if wall else 0.0,
            "mean_wall_s_per_valid_win": wall / len(valid) if valid else None,
            "exact_duplicates": len(fingerprints) - len(set(fingerprints)),
        }
    return result


def write_summary(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summarize(rows), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def round_order(attempt: int, seed: int) -> list[str]:
    baseline = next(iter(ARMS))
    challengers = list(ARMS)[1:]
    random.Random(seed + attempt).shuffle(challengers)
    return [baseline, *challengers]


def run(*, out: Path, attempts: int, seed: int, workers: int,
        max_time: int, max_actions: int) -> None:
    rows = read_rows(out / "results.jsonl")
    done = {(row["arm"], int(row["attempt"])) for row in rows}
    total = attempts * len(ARMS)
    sequence = 0
    for attempt in range(attempts):
        for arm_name in round_order(attempt, seed):
            sequence += 1
            if (arm_name, attempt) in done:
                continue
            arm = ARMS[arm_name]
            trace = out / "traces" / arm_name / f"attempt-{attempt:03d}.npz"
            trace.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{sequence}/{total}] {arm_name} attempt={attempt}", flush=True)
            started = time.perf_counter()
            won = _run_one_search(
                level=5, rollouts=arm.rollouts, rollout_len=arm.rollout_len,
                settle_margin=arm.settle_margin, max_rewind=arm.max_rewind,
                max_time=max_time, max_actions=max_actions, goal="level_up",
                workers=workers, verbose=False,
                instance_id=attempt * len(ARMS) + list(ARMS).index(arm_name),
                trace_path=str(trace),
                trace_metadata={"experiment": "l5-search-efficiency",
                                "experiment_arm": arm_name,
                                "experiment_attempt": attempt,
                                "experiment_seed": seed},
            )
            row = {
                "arm": arm_name, "config": asdict(arm), "attempt": attempt,
                "seed": seed, "workers": workers, "max_time": max_time,
                "max_actions": max_actions,
                "attempt_wall_s": time.perf_counter() - started,
                "search_win": won is not None,
                "trace_path": str(trace) if won else None,
            }
            if won:
                row.update(replay_level_up(trace))
            append_row(out / "results.jsonl", row)
            rows.append(row)
            write_summary(out / "summary.json", rows)
            print(json.dumps(row, sort_keys=True), flush=True)


def main(argv=None) -> None:
    global ARMS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), default="screen")
    parser.add_argument("--out", default="tmp/level5-search-efficiency-screen")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-time", type=int, default=600)
    parser.add_argument("--max-actions", type=int, default=6000)
    args = parser.parse_args(argv)
    if min(args.attempts, args.workers, args.max_time, args.max_actions) < 1:
        raise SystemExit("attempt and resource limits must be positive")
    ARMS = STAGES[args.stage]
    run(out=Path(args.out), attempts=args.attempts, seed=args.seed,
        workers=args.workers, max_time=args.max_time, max_actions=args.max_actions)


if __name__ == "__main__":
    main()
