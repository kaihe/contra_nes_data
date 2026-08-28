"""Resumable Level 6 parameter screen around the Level 5 winner."""

import argparse
from pathlib import Path

import util.benchmark_l5_search as benchmark
from util.benchmark_l5_search import Arm


ARMS = {
    "l5_winner": Arm(4, 24, 8, 8),
    "rollouts_2": Arm(2, 24, 8, 8),
    "rollouts_6": Arm(6, 24, 8, 8),
    "rollouts_8": Arm(8, 24, 8, 8),
    "length_20": Arm(4, 20, 8, 8),
    "length_28": Arm(4, 28, 8, 8),
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tmp/level6-search-efficiency-screen")
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
