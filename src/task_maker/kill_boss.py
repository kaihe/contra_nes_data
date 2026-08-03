"""Extract per-level boss-fight tasks (``boss_level<N>``) by backward relabeling.

Anchored on the boss-stage event: for each trace we find where the boss fight
*begins* (the rising edge into the boss scene) and where the level is *cleared*
(level byte increments, or the post-boss transition routine starts), and emit one
self-contained episode (:class:`task_maker.base.Segment`) covering the whole
fight — the emulator save-state at the boss reveal + the actions that clear the
level.

CLI
---
    python -m task_maker.kill_boss \
        --traces "game_trace/mc_trace/level1/*.npz" \
        --out game_trace/tasks/boss --limit 20 --verify
"""

import os
import glob

import numpy as np

from env.constant import ADDR_LEVEL, ADDR_LEVEL_ROUTINE, ADDR_WEAPON, WEAPON_NAMES
from env.entity import player_x
from env.utility import boss_hp as read_boss_hp, boss_scene
from task_maker.base import (
    Segment,
    TaskMaker,
    build_manifest,
    iter_segment,
    iter_steps,
    load_task,
    load_trace,
    write_segment,
)

_TRANSITION_ROUTINES = (0x08, 0x09)   # post-boss end-of-level sequence


def _routine(ram) -> int:
    return int(ram[ADDR_LEVEL_ROUTINE])


def _level(ram) -> int:
    return int(ram[ADDR_LEVEL])


def _boss_started(pre, cur) -> bool:
    """Rising edge into the boss scene (same signal as the in_boss_stage event)."""
    return boss_scene(cur) and not boss_scene(pre)


def _level_cleared(pre, cur) -> bool:
    """Level byte incremented, or the post-boss transition routine just started.

    Traces frequently stop at the transition before the level byte increments,
    so both are accepted as "boss beaten / level cleared".
    """
    if _level(cur) > _level(pre):
        return True
    return _routine(pre) not in _TRANSITION_ROUTINES and \
        _routine(cur) in _TRANSITION_ROUTINES


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_boss(trace_path: str) -> Segment | None:
    """Replay a trace and return its boss fight, or None if it has none.

    Takes the first boss-stage rising edge and the first level-clear after it.
    """
    ctx = load_trace(trace_path)
    stem = os.path.splitext(ctx.src)[0]

    start_step = start_snap = start_level = start_x = None
    start_weapon = start_rapid = None
    max_boss_hp = 0
    seg = None

    for step, prev, cur, snap in iter_steps(ctx):
        if start_step is None:
            if _boss_started(prev, cur):
                start_step, start_snap, start_level = step, snap, _level(prev)
                start_x = player_x(prev)
                weapon_byte = int(prev[ADDR_WEAPON])
                start_weapon = WEAPON_NAMES.get(weapon_byte & 0x0f,
                                                f"Unknown{weapon_byte & 0x0f}")
                start_rapid = bool(weapon_byte & 0x10)
                max_boss_hp = max(read_boss_hp(prev), read_boss_hp(cur))
        elif seg is None:
            max_boss_hp = max(max_boss_hp, read_boss_hp(cur))
            if _level_cleared(prev, cur):
                seg = Segment(
                    initial_state=start_snap,
                    actions=np.asarray(ctx.actions[start_step:step + 1], dtype=np.uint8),
                    label=f"boss_level{start_level + 1}",
                    level=start_level,
                    start_step=start_step,
                    end_step=step,
                    skip=ctx.skip,
                    src_trace=ctx.src,
                    uid=stem,
                    # ROCKET goal: mark every boss component on the reveal frame
                    meta={"start_x": start_x, "end_x": player_x(cur),
                          "goal_when": "boss", "goal_kind": "boss",
                          "weapon": start_weapon, "rapid": start_rapid,
                          # Reveal states precede component activation by 3-4
                          # decisions, so use the maximum objective HP exposed
                          # during the reveal rather than the opening-frame zero.
                          "boss_hp_start": max_boss_hp, "offset_frac": 0.0},
                )
                # keep iterating so iter_steps runs to completion and closes the emu
    return seg


def add_boss_metadata(seg: Segment) -> Segment:
    """Annotate an existing boss task without changing its episode payload.

    This is the fast metadata-backfill path: it replays only the extracted fight
    rather than its full source-level trace. Generated partial tasks already
    carry start HP/offset metadata and retain those values.
    """
    max_hp = 0
    opening = None
    for pre, cur in iter_segment(seg):
        if opening is None:
            opening = pre
        max_hp = max(max_hp, read_boss_hp(pre), read_boss_hp(cur))
    if opening is None:
        raise ValueError(f"empty boss task: {seg.uid}")
    weapon_byte = int(opening[ADDR_WEAPON])
    seg.meta["weapon"] = WEAPON_NAMES.get(weapon_byte & 0x0f,
                                          f"Unknown{weapon_byte & 0x0f}")
    seg.meta["rapid"] = bool(weapon_byte & 0x10)
    if "source_task" not in seg.meta:  # original full-fight baseline
        seg.meta["boss_hp_start"] = max_hp
        seg.meta["offset_frac"] = 0.0
    return seg


def backfill_metadata(paths, out_root: str) -> int:
    """Add boss metadata to task files while preserving actions/state/timing."""
    for i, path in enumerate(paths, 1):
        write_segment(add_boss_metadata(load_task(path)), out_root)
        if i % 50 == 0 or i == len(paths):
            print(f"  metadata {i}/{len(paths)}", flush=True)
    build_manifest(out_root)
    return len(paths)


# ── Maker ─────────────────────────────────────────────────────────────────────

class KillBossMaker(TaskMaker):
    kind = "boss"
    task_noun = "boss task"

    def __init__(self, *, verify: bool = False):
        self.verify = verify
        self.backfill_glob = None

    def extract(self, trace_path: str) -> list[Segment]:
        seg = extract_boss(trace_path)
        return [seg] if seg is not None else []

    def goal_reached(self, seg: Segment, pre, cur) -> bool:
        """Goal predicate: the level clears (boss beaten) on this step."""
        return _level_cleared(pre, cur)

    @staticmethod
    def boss_hp(ram) -> int:
        """Return interpreted live boss HP without exposing RAM addresses."""
        return read_boss_hp(ram)

    def reject(self, seg: Segment, kept) -> str | None:
        if self.verify and not self.verify_segment(seg):
            return "no-clear"
        return None

    def add_arguments(self, p) -> None:
        p.add_argument("--verify", action="store_true",
                       help="replay each segment and drop any that doesn't clear the level")
        p.add_argument("--backfill-metadata", metavar="TASK_GLOB",
                       help="annotate existing boss tasks instead of extracting traces")

    def configure(self, args) -> None:
        self.verify = args.verify
        self.backfill_glob = args.backfill_metadata

    def before_run(self, args) -> bool:
        if not self.backfill_glob:
            return False
        paths = sorted(glob.glob(self.backfill_glob))
        if args.limit:
            paths = paths[:args.limit]
        if not paths:
            raise SystemExit(f"no boss tasks matched: {self.backfill_glob}")
        n = backfill_metadata(paths, args.out)
        print(f"backfilled metadata for {n} boss tasks")
        return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    KillBossMaker().main()


if __name__ == "__main__":
    main()
