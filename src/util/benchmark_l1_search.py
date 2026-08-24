"""Compare classic and fast Spread-derived search shapes to Level-1 boss entry.

The experiment is fixed-attempt and resumable. Successful searches remain under
``tmp/`` until the experiment is accepted; every attempt, including failures, is
recorded in JSONL so throughput cannot hide search failures.
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

from agent.mc_search import _run_one_search
from env.constant import ADDR_WEAPON
from env.utility import boss_scene
from util.replay import make_env, rewind_state, step_env


@dataclass(frozen=True)
class Arm:
    rollouts: int
    rollout_len: int
    settle_margin: int
    max_rewind: int


ARMS = {
    "classic": Arm(64, 48, 16, 30),
    "fast_spread": Arm(16, 24, 8, 15),
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


def replay_boss_entry(path: Path) -> dict:
    """Verify the boss-scene edge and report the equipped weapon at that edge."""
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
    weapon_byte = -1
    try:
        for action in actions:
            pre = env.unwrapped.get_ram().copy()
            step_env(env, action, skip)
            cur = env.unwrapped.get_ram().copy()
            if boss_scene(cur) and not boss_scene(pre):
                reached = True
                weapon_byte = int(cur[ADDR_WEAPON])
                break
    finally:
        env.close()
    return {
        "replay_valid": reached,
        "weapon": weapon_byte & 0x0f if reached else None,
        "rapid": bool(weapon_byte & 0x10) if reached else None,
        "spread_equipped": bool(reached and weapon_byte & 0x0f == 3),
        "trace_steps": len(actions),
        "sampled_actions": sampled_actions,
        "search_wall_s": search_wall_s,
        "fingerprint": fingerprint,
    }


def summarize(rows: list[dict]) -> dict:
    result = {}
    for arm in ARMS:
        group = [row for row in rows if row["arm"] == arm]
        wall = sum(float(row["attempt_wall_s"]) for row in group)
        spread = sum(bool(row.get("spread_equipped")) for row in group)
        valid = sum(bool(row.get("replay_valid")) for row in group)
        fingerprints = [row["fingerprint"] for row in group if row.get("fingerprint")]
        result[arm] = {
            "attempts": len(group),
            "search_wins": sum(bool(row["search_win"]) for row in group),
            "replay_valid": valid,
            "spread_equipped": spread,
            "attempt_wall_s": wall,
            "spread_wins_per_hour": 3600 * spread / wall if wall else 0.0,
            "boss_entries_per_hour": 3600 * valid / wall if wall else 0.0,
            "exact_duplicates": len(fingerprints) - len(set(fingerprints)),
        }
    return result


def write_summary(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summarize(rows), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(*, out: Path, attempts: int, seed: int, workers: int,
        max_time: int, max_actions: int) -> None:
    rows = read_rows(out / "results.jsonl")
    done = {(row["arm"], int(row["attempt"])) for row in rows}
    sequence = 0
    for attempt in range(attempts):
        order = list(ARMS)
        random.Random(seed + attempt).shuffle(order)
        for arm_name in order:
            sequence += 1
            if (arm_name, attempt) in done:
                continue
            arm = ARMS[arm_name]
            trace = out / "traces" / arm_name / f"attempt-{attempt:03d}.npz"
            trace.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{sequence}/{attempts * len(ARMS)}] {arm_name} "
                  f"attempt={attempt}", flush=True)
            started = time.perf_counter()
            won = _run_one_search(
                level=1, rollouts=arm.rollouts, rollout_len=arm.rollout_len,
                settle_margin=arm.settle_margin, max_rewind=arm.max_rewind,
                max_time=max_time, max_actions=max_actions, goal="boss_entry",
                workers=workers, verbose=False,
                instance_id=attempt * len(ARMS) + sorted(ARMS).index(arm_name),
                trace_path=str(trace),
                trace_metadata={"experiment": "l1-search-efficiency",
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
                row.update(replay_boss_entry(trace))
            append_row(out / "results.jsonl", row)
            rows.append(row)
            write_summary(out / "summary.json", rows)
            print(json.dumps(row, sort_keys=True), flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tmp/level1-spread-efficiency")
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--max-time", type=int, default=300)
    parser.add_argument("--max-actions", type=int, default=3000)
    args = parser.parse_args(argv)
    if min(args.attempts, args.workers, args.max_time, args.max_actions) < 1:
        raise SystemExit("attempt and resource limits must be positive")
    run(out=Path(args.out), attempts=args.attempts, seed=args.seed,
        workers=args.workers, max_time=args.max_time, max_actions=args.max_actions)


if __name__ == "__main__":
    main()
