"""Resumable fixed-attempt throughput benchmark for the Level-1 Spread boss.

The benchmark writes its resumable working set to ``tmp/``. Every replay-valid,
fingerprint-unique win is also copied atomically into the production trace tree,
so useful searches are retained without duplicating existing data. Screening
interleaves a shuffled round of all 27 parameter cells before starting the next
round, then confirmation compares the four fastest perfect screening cells with
the current baseline. Search wall time includes emulator and worker-pool setup
plus trace serialization; replay verification is measured separately and acts
only as a validity gate.

Usage::

    python -m util.benchmark_mc_search --stage all

The JSONL log is append-only and keyed by stage/config/attempt, so rerunning the
same command safely resumes after the last completed attempt.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from agent.mc_search import _run_one_search, load_initial_state
from task_maker.base import Segment
from task_maker.kill_boss import KillBossMaker


DEFAULT_STATE = "src/agent/states/boss_level1/full_spread.state"
DEFAULT_OUT = "tmp/boss-spread-grid"


@dataclass(frozen=True, order=True)
class SearchConfig:
    rollouts: int
    rollout_len: int
    settle_margin: int
    max_rewind: int

    @property
    def uid(self) -> str:
        return (f"r{self.rollouts:02d}-l{self.rollout_len:02d}-"
                f"s{self.settle_margin:02d}-w{self.max_rewind:02d}")


BASELINE = SearchConfig(64, 48, 16, 30)


def screening_configs() -> list[SearchConfig]:
    """The approved 3 x 3 x 3 algorithm grid."""
    return [
        SearchConfig(rollouts, rollout_len, settle_margin, max_rewind)
        for rollouts in (16, 32, 64)
        for rollout_len, settle_margin in ((24, 8), (36, 12), (48, 16))
        for max_rewind in (15, 30, 45)
    ]


def round_schedule(configs: list[SearchConfig], attempts: int, seed: int):
    """Yield shuffled complete rounds, preserving one attempt per cell per round."""
    rng = random.Random(seed)
    for attempt in range(attempts):
        order = list(configs)
        rng.shuffle(order)
        for config in order:
            yield config, attempt


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def summarize(rows: list[dict], stage: str) -> list[dict]:
    """Aggregate completed attempt rows for one stage, fastest valid cells first."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row["stage"] == stage:
            grouped.setdefault(row["config_id"], []).append(row)
    summaries = []
    for config_id, group in grouped.items():
        valid = [r for r in group if r["win"] and r["replay_valid"]]
        fingerprints = [r["fingerprint"] for r in valid]
        total_wall = sum(float(r["attempt_wall_s"]) for r in group)
        win_walls = [float(r["attempt_wall_s"]) for r in valid]
        summaries.append({
            "config_id": config_id,
            "config": group[0]["config"],
            "attempts": len(group),
            "wins": sum(bool(r["win"]) for r in group),
            "replay_valid_wins": len(valid),
            "replay_failures": sum(bool(r["win"]) and not r["replay_valid"]
                                   for r in group),
            "exact_duplicates": len(fingerprints) - len(set(fingerprints)),
            "total_attempt_wall_s": total_wall,
            "wins_per_hour": 3600.0 * len(valid) / total_wall if total_wall else 0.0,
            "mean_wall_s_per_valid_win": total_wall / len(valid) if valid else None,
            "p90_success_wall_s": float(np.quantile(win_walls, 0.9))
            if win_walls else None,
            "mean_sampled_actions": float(np.mean([
                r["sampled_actions"] for r in valid])) if valid else None,
            "mean_trace_steps": float(np.mean([
                r["trace_steps"] for r in valid])) if valid else None,
        })
    return sorted(summaries, key=lambda x: (-x["wins_per_hour"], x["config_id"]))


def confirmation_configs(screen: list[dict], attempts: int) -> list[SearchConfig]:
    """Choose four perfect non-baseline screening cells plus the baseline."""
    eligible = [
        row for row in screen
        if row["attempts"] == attempts
        and row["replay_valid_wins"] == attempts
        and row["exact_duplicates"] == 0
        and row["config_id"] != BASELINE.uid
    ]
    if len(eligible) < 4:
        raise RuntimeError(
            f"only {len(eligible)} non-baseline cells passed {attempts}/{attempts}; "
            "confirmation requires four")
    selected = [SearchConfig(**row["config"]) for row in eligible[:4]]
    return selected + [BASELINE]


def replay_trace(path: Path) -> tuple[bool, str, int, int, float]:
    """Replay a saved raw trace and return validity plus compact trace metrics."""
    with np.load(path, allow_pickle=True) as data:
        actions = np.asarray(data["actions"], dtype=np.uint8)
        initial_state = bytes(data["initial_state"])
        skip = int(data["skip"])
        sampled = int(data["sampled_actions"])
        search_wall = float(data["search_wall_s"])
    fingerprint = hashlib.sha256(initial_state + actions.tobytes()).hexdigest()
    segment = Segment(
        initial_state=initial_state, actions=actions, label="boss_level1",
        level=0, start_step=0, end_step=len(actions) - 1, skip=skip,
        src_trace=path.name, uid=path.stem, split="train",
    )
    return (KillBossMaker().verify_segment(segment), fingerprint,
            len(actions), sampled, search_wall)


def production_index(directory: Path) -> dict[str, Path]:
    """Index full-fight production traces by exact state+action fingerprint."""
    index = {}
    if not directory.exists():
        return index
    for path in sorted(directory.glob("win_boss_level1_full_*.npz")):
        with np.load(path, allow_pickle=True) as data:
            initial_state = bytes(data["initial_state"])
            actions = np.asarray(data["actions"], dtype=np.uint8)
        fingerprint = hashlib.sha256(initial_state + actions.tobytes()).hexdigest()
        index.setdefault(fingerprint, path)
    return index


def promote_trace(path: Path, *, stage: str, config: SearchConfig, attempt: int,
                  fingerprint: str, directory: Path, weapon: str = "spread",
                  known: dict[str, Path]) -> tuple[str, str]:
    """Atomically retain a valid grid win, without copying an exact duplicate."""
    if fingerprint in known:
        return "existing", str(known[fingerprint])
    directory.mkdir(parents=True, exist_ok=True)
    name = (f"win_boss_level1_full_{weapon}_grid-{stage}-{config.uid}-"
            f"a{attempt:03d}-{fingerprint[:16]}.npz")
    destination = directory / name
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(path, temporary)
    os.replace(temporary, destination)
    known[fingerprint] = destination
    return "promoted", str(destination)


def run_attempt(*, config: SearchConfig, stage: str, attempt: int,
                sequence: int, state: bytes, metadata: dict, out: Path,
                workers: int, max_time: int, max_actions: int,
                production_dir: Path, known: dict[str, Path],
                weapon: str = "spread") -> dict:
    trace_path = out / "traces" / stage / config.uid / f"attempt-{attempt:03d}.npz"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    won = _run_one_search(
        level=1, rollouts=config.rollouts, rollout_len=config.rollout_len,
        settle_margin=config.settle_margin, max_time=max_time,
        max_rewind=config.max_rewind, max_actions=max_actions,
        goal="level_up", workers=workers, verbose=False, instance_id=sequence,
        initial_emu_state=state, trace_path=str(trace_path),
        trace_metadata={
            **metadata, "benchmark_stage": stage,
            "benchmark_config": config.uid, "benchmark_attempt": attempt,
        },
    )
    attempt_wall = time.perf_counter() - started
    replay_valid = False
    fingerprint = None
    trace_steps = sampled_actions = None
    search_wall = None
    verify_wall = 0.0
    promotion_status = None
    production_trace_path = None
    if won:
        verify_started = time.perf_counter()
        replay_valid, fingerprint, trace_steps, sampled_actions, search_wall = \
            replay_trace(trace_path)
        verify_wall = time.perf_counter() - verify_started
        if replay_valid:
            promotion_status, production_trace_path = promote_trace(
                trace_path, stage=stage, config=config, attempt=attempt,
                fingerprint=fingerprint, directory=production_dir,
                weapon=weapon, known=known,
            )
    return {
        "stage": stage,
        "config_id": config.uid,
        "config": asdict(config),
        "attempt": attempt,
        "sequence": sequence,
        "workers": workers,
        "max_time": max_time,
        "max_actions": max_actions,
        "win": bool(won),
        "replay_valid": replay_valid,
        "attempt_wall_s": attempt_wall,
        "verify_wall_s": verify_wall,
        "search_wall_s": search_wall,
        "trace_steps": trace_steps,
        "sampled_actions": sampled_actions,
        "fingerprint": fingerprint,
        "trace_path": str(trace_path) if won else None,
        "promotion_status": promotion_status,
        "production_trace_path": production_trace_path,
    }


def write_summary(path: Path, rows: list[dict], screen_attempts: int,
                  confirm_attempts: int) -> None:
    payload = {
        "screen": summarize(rows, "screen"),
        "confirm": summarize(rows, "confirm"),
        "screen_attempts_per_config": screen_attempts,
        "confirm_attempts_per_config": confirm_attempts,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def run_stage(*, stage: str, configs: list[SearchConfig], attempts: int,
              seed: int, state: bytes, metadata: dict, out: Path,
              workers: int, max_time: int, max_actions: int,
              production_dir: Path, known: dict[str, Path],
              rows: list[dict], weapon: str = "spread") -> list[dict]:
    done = {(r["stage"], r["config_id"], int(r["attempt"])) for r in rows}
    schedule = list(round_schedule(configs, attempts, seed))
    for sequence, (config, attempt) in enumerate(schedule):
        key = (stage, config.uid, attempt)
        if key in done:
            continue
        print(f"[{stage} {sequence + 1}/{len(schedule)}] "
              f"{config.uid} attempt={attempt}", flush=True)
        row = run_attempt(
            config=config, stage=stage, attempt=attempt, sequence=sequence,
            state=state, metadata=metadata, out=out, workers=workers,
            max_time=max_time, max_actions=max_actions,
            production_dir=production_dir, known=known, weapon=weapon,
        )
        append_row(out / "results.jsonl", row)
        rows.append(row)
        write_summary(out / "summary.json", rows, 12, 100)
        outcome = "VALID WIN" if row["replay_valid"] else (
            "REPLAY FAIL" if row["win"] else "NO WIN")
        print(f"  {outcome} wall={row['attempt_wall_s']:.2f}s", flush=True)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "confirm", "all"),
                        default="all")
    parser.add_argument("--state", default=DEFAULT_STATE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--production-dir",
                        default="game_trace/mc_trace/boss_level1")
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--screen-attempts", type=int, default=12)
    parser.add_argument("--confirm-attempts", type=int, default=100)
    parser.add_argument("--max-time", type=int, default=90)
    parser.add_argument("--max-actions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260807)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if min(args.workers, args.screen_attempts, args.confirm_attempts,
           args.max_time, args.max_actions) < 1:
        raise SystemExit("worker, attempt and limit arguments must be positive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state, metadata = load_initial_state(args.state)
    rows = read_rows(out / "results.jsonl")
    production_dir = Path(args.production_dir)
    known = production_index(production_dir)
    print(f"indexed {len(known)} existing full-fight production fingerprints",
          flush=True)

    if args.stage in ("screen", "all"):
        rows = run_stage(
            stage="screen", configs=screening_configs(),
            attempts=args.screen_attempts, seed=args.seed, state=state,
            metadata=metadata, out=out, workers=args.workers,
            max_time=args.max_time, max_actions=args.max_actions,
            production_dir=production_dir, known=known, rows=rows,
        )
    if args.stage in ("confirm", "all"):
        screen = summarize(rows, "screen")
        configs = confirmation_configs(screen, args.screen_attempts)
        rows = run_stage(
            stage="confirm", configs=configs, attempts=args.confirm_attempts,
            seed=args.seed + 1, state=state, metadata=metadata, out=out,
            workers=args.workers, max_time=args.max_time,
            max_actions=args.max_actions, production_dir=production_dir,
            known=known, rows=rows,
        )
    write_summary(out / "summary.json", rows, args.screen_attempts,
                  args.confirm_attempts)


if __name__ == "__main__":
    main()
