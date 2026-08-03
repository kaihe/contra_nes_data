"""Shared task-segment format and replay/IO machinery for task makers.

Every task type (kill an enemy, beat a boss, reach a landmark, …) emits the same
:class:`Segment`: a self-contained ``bc_data``-shaped episode — the emulator
save-state at the moment the task becomes active, plus the actions that complete
it. Task-specific fields live in ``meta`` so the on-disk format stays uniform.
"""

import argparse
import csv
import glob
import hashlib
import inspect
import os
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from env.entity import (ADDR_ENEMY_ROUTINE, ADDR_ENEMY_TYPE, ADDR_ENEMY_X,
                        ADDR_ENEMY_Y, ADDR_PLAYER_X, ADDR_PLAYER_Y,
                        ENEMY_TYPE_BULLET, N_SLOTS, on_screen, overlay_pointers)
from util.replay import SKIP, make_env, rewind_state


# ── Train / val split ─────────────────────────────────────────────────────────

VAL_PCT = 10          # target share of *source traces* held out


def split_of(src_trace: str, val_pct: int = VAL_PCT) -> str:
    """Which half a task belongs to, decided by its **source trace**.

    The split is a property of the dataset, owned here, because it has to be a
    property of the *trace*: one trace yields 10–24 overlapping tasks, so any
    split drawn over episodes scatters siblings across both halves and a val task
    ends up sharing frames with a train task cut from the same replay.

    Hashing the name (rather than permuting the file list) makes the assignment
    stable under regeneration: adding, removing or reordering traces never
    reassigns the ones already there.
    """
    h = int(hashlib.sha1(src_trace.encode()).hexdigest(), 16)
    return "val" if h % 100 < val_pct else "train"


# ── Common segment ────────────────────────────────────────────────────────────

@dataclass
class Segment:
    initial_state: bytes        # emu save-state at the start of the window
    actions: np.ndarray         # (M, 9) actions from start to completion (inclusive)
    label: str                  # task label / class (e.g. "kill_soldier", "boss_level1")
    level: int                  # 0-indexed
    start_step: int             # decision-step where the task became active
    end_step: int               # decision-step where it completed
    skip: int                   # NES frames per action
    src_trace: str              # originating trace filename
    uid: str                    # unique filename stem within the label directory
    meta: dict = field(default_factory=dict)   # task-specific extras (enemy_type, slot, …)
    split: str = ""             # "train" / "val"; derived from src_trace when blank

    def __post_init__(self):
        if not self.split:
            self.split = split_of(self.src_trace)

    @property
    def window_len(self) -> int:
        return self.end_step - self.start_step + 1


# ── Trace loading / stepping ──────────────────────────────────────────────────

@dataclass
class TraceCtx:
    actions: np.ndarray
    skip: int
    src: str
    initial_state: bytes


def load_trace(path: str) -> TraceCtx:
    d = np.load(path, allow_pickle=True)
    return TraceCtx(
        actions=d["actions"],
        skip=int(d["skip"]) if "skip" in d else SKIP,
        src=os.path.basename(path),
        initial_state=bytes(d["initial_state"]),
    )


def iter_steps(ctx: TraceCtx):
    """Replay a trace, yielding ``(step, prev_ram, cur_ram, snap)`` per decision step.

    ``snap`` is the emu save-state captured at the START of the step (it pairs
    with ``prev_ram``). The emulator is always closed when iteration ends.
    """
    env = make_env()
    rewind_state(env, ctx.initial_state)
    try:
        prev = env.unwrapped.get_ram().copy()
        for step, act in enumerate(ctx.actions):
            snap = env.em.get_state()
            a = np.asarray(act, dtype=np.uint8)
            for _ in range(ctx.skip):
                env.step(a)
            cur = env.unwrapped.get_ram()
            yield step, prev, cur, snap
            prev = cur.copy()
    finally:
        env.close()


def iter_segment(seg: Segment):
    """Replay a segment from its own ``initial_state``, yielding ``(prev, cur)`` per step.

    Used by task self-consistency checks (does the segment reproduce its goal?).
    """
    env = make_env()
    rewind_state(env, seg.initial_state)
    try:
        prev = env.unwrapped.get_ram().copy()
        for act in seg.actions:
            a = np.asarray(act, dtype=np.uint8)
            for _ in range(seg.skip):
                env.step(a)
            cur = env.unwrapped.get_ram()
            yield prev, cur
            prev = cur.copy()
    finally:
        env.close()


# ── Goal frame (ROCKET-style visual prompt) ───────────────────────────────────

def _boss_parts(ram) -> list[tuple[int, int]]:
    """On-screen PPU positions of every live boss component (non-bullet enemy)."""
    return [(int(ram[ADDR_ENEMY_X + s]), int(ram[ADDR_ENEMY_Y + s]))
            for s in range(N_SLOTS)
            if int(ram[ADDR_ENEMY_ROUTINE + s]) != 0
            and int(ram[ADDR_ENEMY_TYPE + s]) != ENEMY_TYPE_BULLET
            and on_screen(int(ram[ADDR_ENEMY_X + s]), int(ram[ADDR_ENEMY_Y + s]))]


def render_goal(seg: "Segment", *, sigma: float = 12.0, max_seek: int = 8,
                boss_reveal_seek: int = 24):
    """Render a task's goal frame: the reference image with colored Gaussian blob(s)
    on the goal entity(-ies) — the ROCKET visual prompt.

    Driven by ``seg.meta`` — ``goal_when``:

    * ``"first"`` — on-view pointing (kill / item): mark the target enemy at
      ``goal_slot`` on the opening frame (spawn/birth can lag the save-state, so we
      seek forward until the slot is live).
    * ``"last"`` — goal image (traverse): mark the player's ending position.
    * ``"boss"`` — mark **every** boss component: seek through the fortress reveal and
      keep the frame that shows the most parts, blobbing them all.

    ``goal_kind`` picks the blob colour. Returns ``(rgb_uint8, heatmap_float32,
    points)`` where ``points`` is the list of marked PPU coordinates.
    """
    when = seg.meta.get("goal_when", "first")
    kind = seg.meta.get("goal_kind", "target")
    env = make_env()
    rewind_state(env, seg.initial_state)
    try:
        if when == "last":
            for act in seg.actions:
                a = np.asarray(act, dtype=np.uint8)
                for _ in range(seg.skip):
                    env.step(a)
            ram = env.unwrapped.get_ram()
            points = [(int(ram[ADDR_PLAYER_X]), int(ram[ADDR_PLAYER_Y]))]
            frame = env.unwrapped.get_screen()
        elif when == "boss":
            # the fortress reveals over the first ~seconds; keep the frame that shows
            # the most parts (the fully-revealed boss)
            best_n, frame, points = -1, None, []
            for k in range(min(boss_reveal_seek, len(seg.actions))):
                a = np.asarray(seg.actions[k], dtype=np.uint8)
                for _ in range(seg.skip):
                    env.step(a)
                pts = _boss_parts(env.unwrapped.get_ram())
                if len(pts) > best_n:
                    best_n, points = len(pts), pts
                    frame = env.unwrapped.get_screen().copy()
        else:  # "first"
            slot = int(seg.meta["goal_slot"])
            ram = env.unwrapped.get_ram()
            # A freshly loaded save-state hasn't rendered a frame yet, so always
            # advance at least one step; keep going until the target slot is live.
            seek = 0
            while seek < max_seek and seek < len(seg.actions) and \
                    (seek == 0 or int(ram[ADDR_ENEMY_ROUTINE + slot]) == 0):
                a = np.asarray(seg.actions[seek], dtype=np.uint8)
                for _ in range(seg.skip):
                    env.step(a)
                ram = env.unwrapped.get_ram()
                seek += 1
            points = [(int(ram[ADDR_ENEMY_X + slot]), int(ram[ADDR_ENEMY_Y + slot]))]
            frame = env.unwrapped.get_screen()
    finally:
        env.close()
    img, hm = overlay_pointers(frame, points, kind=kind, sigma=sigma)
    return img, hm, points


# ── Writing ───────────────────────────────────────────────────────────────────

_MANIFEST_FIELDS = ["file", "label", "split", "start_step", "end_step", "window_len",
                    "start_x", "end_x", "target_entity", "source_entity"]


def write_segment(seg: Segment, out_root: str) -> str:
    """Write one segment as a bc_data-shaped .npz under ``out_root/<label>/``."""
    d = os.path.join(out_root, seg.label)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, seg.uid + ".npz")
    np.savez_compressed(
        path,
        actions=seg.actions,
        initial_state=np.frombuffer(seg.initial_state, dtype=np.uint8),
        label=seg.label,
        level=seg.level,
        skip=seg.skip,
        start_step=seg.start_step,
        end_step=seg.end_step,
        src_trace=seg.src_trace,
        split=seg.split,
        **seg.meta,
    )
    return path


_SEG_FIELDS = {"actions", "initial_state", "label", "level", "skip",
               "start_step", "end_step", "src_trace", "split"}


def load_task(path: str) -> Segment:
    """Load a task ``.npz`` written by :func:`write_segment` back into a Segment
    (its extra scalars become ``meta``). Inverse of the write path."""
    d = np.load(path, allow_pickle=True)
    meta = {}
    for k in d.files:
        if k in _SEG_FIELDS:
            continue
        v = d[k]
        meta[k] = v.item() if getattr(v, "ndim", 1) == 0 else v
    return Segment(
        initial_state=bytes(d["initial_state"]),
        actions=d["actions"],
        label=str(d["label"]),
        level=int(d["level"]),
        start_step=int(d["start_step"]),
        end_step=int(d["end_step"]),
        skip=int(d["skip"]),
        src_trace=str(d["src_trace"]),
        uid=os.path.splitext(os.path.basename(path))[0],
        meta=meta,
        # tasks written before the split existed fall back to the derived value
        split=str(d["split"]) if "split" in d.files else "",
    )


def build_manifest(out_root: str) -> tuple[str, int]:
    """(Re)build ``manifest.csv`` by scanning every task ``.npz`` under ``out_root``.

    The manifest is a *derived index* — the ``.npz`` files are the source of
    truth — so this is idempotent and never loses tasks from earlier runs: any
    run just adds ``.npz`` files, and rebuilding reflects everything on disk. It
    carries a small summary per task (see ``_MANIFEST_FIELDS``), including the
    train/val ``split`` inherited from the task's source trace; the full metadata
    lives in each ``.npz``. Task-specific columns are blank where they don't apply
    (``target_entity`` for kills, ``source_entity`` for items). Only these scalars
    are read — the big ``actions`` / ``initial_state`` arrays are never decompressed.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(out_root, "*", "*.npz"))):
        d = np.load(path, allow_pickle=True)

        def val(key):
            if key not in d.files:
                return ""
            v = d[key]
            return v.item() if getattr(v, "ndim", 1) == 0 else v.tolist()

        start, end = int(d["start_step"]), int(d["end_step"])
        rows.append({
            "file": path,   # path relative to the cwd, e.g. game_trace/tasks/kill/red_turret/xxx.npz
            "label": str(d["label"]),
            "split": val("split") or split_of(str(d["src_trace"])),
            "start_step": start,
            "end_step": end,
            "window_len": end - start + 1,
            "start_x": val("start_x"),
            "end_x": val("end_x"),
            "target_entity": val("target_entity"),
            "source_entity": val("source_entity"),
        })

    path = os.path.join(out_root, "manifest.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path, len(rows)


# ── Common task-maker driver ──────────────────────────────────────────────────

class TaskMaker:
    """Base for every task maker: owns the file loop, per-segment gating, manifest
    rebuild and the CLI, so a concrete maker only supplies :meth:`extract` (and,
    optionally, :meth:`reject` plus a few CLI knobs).

    Subclass contract
    -----------------
    * set ``kind`` (output subdir under ``game_trace/tasks``) and, if the maker has
      parameters, initialise them as attributes in ``__init__``;
    * implement :meth:`extract` → the segments found in one trace;
    * override :meth:`reject` to drop segments (the returned string is the reason,
      shown in the run summary); :meth:`add_arguments` / :meth:`configure` to expose
      and apply extra CLI flags.

    Both the CLI (``Maker().main()``) and ``util.make_tasks`` (``Maker().run(...)``)
    go through the same code path here.
    """

    kind: str = ""
    task_noun: str = "task"
    default_traces: str = "game_trace/mc_trace/level1/*.npz"

    @property
    def default_out(self) -> str:
        return f"game_trace/tasks/{self.kind}"

    # -- to override -------------------------------------------------------
    def extract(self, trace_path: str) -> list["Segment"]:
        """Return every task segment found in one trace (may be empty)."""
        raise NotImplementedError

    def reject(self, seg: "Segment", kept: Counter) -> str | None:
        """Reason to drop ``seg`` (shown in the summary), or ``None`` to keep it.
        ``kept`` is the running per-label count, for balancing caps."""
        return None

    def goal_reached(self, seg: "Segment", pre: np.ndarray, cur: np.ndarray) -> bool:
        """Whether this task's goal is satisfied on the ``pre``→``cur`` step (event
        predicates like a kill / pickup) or holds at ``cur`` (state predicates like
        reaching a position). Abstract — each maker defines completion from
        ``seg.meta``. One predicate drives three things: dataset verification
        (:meth:`verify_segment`), eval success detection, and outcome reward."""
        raise NotImplementedError

    def verify_segment(self, seg: "Segment") -> bool:
        """True if :meth:`goal_reached` fires while the segment is replayed from its
        own ``initial_state`` — the self-consistency check shared by every maker."""
        return any(self.goal_reached(seg, pre, cur) for pre, cur in iter_segment(seg))

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        """Register maker-specific CLI flags."""

    def configure(self, args: argparse.Namespace) -> None:
        """Apply parsed CLI flags onto ``self`` before the run."""

    def before_run(self, args: argparse.Namespace) -> bool:
        """Optional pre-run hook. Return ``True`` to short-circuit (e.g. an
        ``--inspect`` mode that renders instead of generating a dataset)."""
        return False

    # -- shared machinery --------------------------------------------------
    def run(self, files, out_root: str | None = None) -> int:
        """Extract tasks from ``files`` into ``out_root`` and rebuild the manifest.

        Returns the number of segments kept. Used by both the CLI and make_tasks.
        """
        out_root = out_root or self.default_out
        os.makedirs(out_root, exist_ok=True)
        kept: Counter = Counter()
        by_split: Counter = Counter()   # keyed by (label, split)
        rejected: Counter = Counter()   # keyed by reason
        empty = 0

        for i, f in enumerate(files, 1):
            segs = self.extract(f)
            empty += not segs
            for seg in segs:
                reason = self.reject(seg, kept)
                if reason:
                    rejected[reason] += 1
                    continue
                write_segment(seg, out_root)
                kept[seg.label] += 1
                by_split[(seg.label, seg.split)] += 1
            print(f"  [{i}/{len(files)}] {os.path.basename(f)}: "
                  f"{len(segs)} segments  (kept {sum(kept.values())})")

        _, total = build_manifest(out_root)
        print(f"\n  wrote {sum(kept.values())} {self.task_noun}s this run; "
              f"{total} total on disk → {out_root}/manifest.csv")
        if empty:
            print(f"    ({empty} traces produced no segments)")
        print(f"    {'label':<24} {'total':>6} {'train':>6} {'val':>6}")
        for label, n in kept.most_common():
            tr, va = by_split[(label, "train")], by_split[(label, "val")]
            flag = "   <- empty on one side" if not tr or not va else ""
            print(f"    {label:<24} {n:>6} {tr:>6} {va:>6}{flag}")
        if rejected:
            print("  rejected: "
                  + ", ".join(f"{r}={c}" for r, c in rejected.most_common()))
        return sum(kept.values())

    def main(self, argv=None) -> None:
        """Standard CLI: ``--traces / --out / --limit`` plus maker-specific flags."""
        desc = inspect.getmodule(type(self)).__doc__
        p = argparse.ArgumentParser(description=desc,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
        p.add_argument("--traces", default=self.default_traces,
                       help="glob of source trace .npz files")
        p.add_argument("--out", default=self.default_out, help="output dataset root")
        p.add_argument("--limit", type=int, default=None,
                       help="max number of source traces to process")
        self.add_arguments(p)
        args = p.parse_args(argv)
        self.configure(args)
        if self.before_run(args):
            return

        files = sorted(glob.glob(args.traces))
        if args.limit:
            files = files[:args.limit]
        if not files:
            raise SystemExit(f"no traces matched: {args.traces}")
        self.run(files, args.out)
