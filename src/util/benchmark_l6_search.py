"""Resumable Level 6 parameter sweep over the full 21-action space."""

import argparse
from pathlib import Path

import util.benchmark_l5_search as benchmark
from util.benchmark_l5_search import Arm


ARMS = {
    "scout_winner": Arm(64, 48, 16, 30),
    "rewind_15": Arm(64, 48, 16, 15),
    "rewind_45": Arm(64, 48, 16, 45),
    "rollouts_32": Arm(32, 48, 16, 30),
    "rollouts_96": Arm(96, 48, 16, 30),
    "old_contra_baseline": Arm(64, 48, 16, 60),
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tmp/level6-search-efficiency-targeted")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-time", type=int, default=600)
    parser.add_argument("--max-actions", type=int, default=6000)
    args = parser.parse_args(argv)
    if min(args.attempts, args.workers, args.max_time, args.max_actions) < 1:
        raise SystemExit("attempt and resource limits must be positive")
    benchmark.ARMS = ARMS
    benchmark.run(
        out=Path(args.out), attempts=args.attempts, seed=args.seed,
        workers=args.workers, max_time=args.max_time,
        max_actions=args.max_actions, level=6,
    )


if __name__ == "__main__":
    main()
