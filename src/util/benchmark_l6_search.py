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

HIGH_COMPUTE_SCOUT = {
    "old_contra_baseline": Arm(64, 48, 16, 60),
    "rollouts_16": Arm(16, 48, 16, 60),
    "rollouts_32": Arm(32, 48, 16, 60),
    "rollouts_96": Arm(96, 48, 16, 60),
    "length_64": Arm(64, 64, 16, 60),
    "rewind_30": Arm(64, 48, 16, 30),
    "settle_8": Arm(64, 48, 8, 60),
}

STAGES = {
    "low-compute": ARMS,
    "high-compute-scout": HIGH_COMPUTE_SCOUT,
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES),
                        default="high-compute-scout")
    parser.add_argument("--out", default="tmp/level6-search-efficiency-screen")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-time", type=int, default=600)
    parser.add_argument("--max-actions", type=int, default=6000)
    args = parser.parse_args(argv)
    if min(args.attempts, args.workers, args.max_time, args.max_actions) < 1:
        raise SystemExit("attempt and resource limits must be positive")
    benchmark.ARMS = STAGES[args.stage]
    benchmark.run(
        out=Path(args.out), attempts=args.attempts, seed=args.seed,
        workers=args.workers, max_time=args.max_time,
        max_actions=args.max_actions, level=6,
    )


if __name__ == "__main__":
    main()
