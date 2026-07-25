"""Fan a task maker across every mc_trace whose filename matches a pattern.

Selects source traces under ``--trace-root`` by a regular expression on their
filename, then runs each requested task maker over that set — writing into
``<out-root>/<subdir>`` and rebuilding that maker's manifest. One command turns a
level's worth of traces into task datasets.

    # every Level-1 win trace → both kill and item tasks
    python -m util.make_tasks --pattern 'win_level1_' --makers kill_enemy pick_item

    # just kill tasks, first 10 matching traces
    python -m util.make_tasks --pattern 'win_level1_' --makers kill_enemy --limit 10

The pattern is a Python regex matched with ``re.search`` against each trace's
basename, so ``win_level1_`` (or ``win_level1_.*``) selects all Level-1 wins.
"""

import argparse
import os
import re

from task_maker.kill_boss import KillBossMaker
from task_maker.kill_enemy import KillEnemyMaker
from task_maker.pick_item import PickItemMaker
from task_maker.traverse import TraverseMaker

# maker name -> TaskMaker subclass (each carries its own output `kind` subdir)
MAKERS = {
    "kill_enemy": KillEnemyMaker,
    "pick_item":  PickItemMaker,
    "kill_boss":  KillBossMaker,
    "traverse":   TraverseMaker,
}


def find_traces(root: str, pattern: str) -> list[str]:
    """All ``*.npz`` under ``root`` whose basename matches ``pattern`` (re.search)."""
    rx = re.compile(pattern)
    hits = []
    for dirpath, _dirs, names in os.walk(root):
        for n in names:
            if n.endswith(".npz") and rx.search(n):
                hits.append(os.path.join(dirpath, n))
    return sorted(hits)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", default="win_level1_",
                   help="regex matched against trace filenames (re.search)")
    p.add_argument("--makers", nargs="+", choices=list(MAKERS), default=list(MAKERS),
                   help="which task makers to run")
    p.add_argument("--trace-root", default="game_trace/mc_trace",
                   help="root directory searched for source traces")
    p.add_argument("--out-root", default="game_trace/tasks",
                   help="root directory for task output (<out-root>/<subdir> per maker)")
    p.add_argument("--limit", type=int, default=None, help="max traces to process")
    args = p.parse_args()

    files = find_traces(args.trace_root, args.pattern)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no traces under {args.trace_root} match /{args.pattern}/")
    print(f"{len(files)} traces match /{args.pattern}/ under {args.trace_root}")

    for name in args.makers:
        maker = MAKERS[name]()
        out = os.path.join(args.out_root, maker.kind)
        print(f"\n=== {name} → {out} ===")
        maker.run(files, out)


if __name__ == "__main__":
    main()
