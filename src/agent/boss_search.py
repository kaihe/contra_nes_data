"""Generate additive level-1 boss tasks from full and partial fight states.

The source unit is an existing replayable boss task. Sources are selected
uniformly, validation sources are rejected, and a source is replayed once to
capture either its reveal state or a uniformly sampled state with live boss HP.
Monte-Carlo search then starts from that exact emulator state. Each win is saved
both as a raw search trace and as a normal ``boss_level1`` task carrying the
original trace provenance.

Example pilot::

    python -m agent.boss_search --runs 4 --workers 8 --max-time 300
"""

import argparse
import glob
import gzip
import hashlib
import os
import time
from dataclasses import dataclass

import numpy as np
import yaml

from agent.mc_search import _run_one_search
from env.constant import ADDR_WEAPON, WEAPON_NAMES
from env.entity import player_x
from env.utility import boss_hp
from task_maker.base import (Segment, build_manifest, iter_segment, load_task,
                             write_segment)
from task_maker.kill_boss import KillBossMaker
from util.replay import SKIP, make_env, rewind_state, step_env

DEFAULT_SOURCES = "game_trace/tasks/boss/boss_level1/*.npz"
DEFAULT_TRACE_OUT = "game_trace/mc_trace/boss_level1"
DEFAULT_TASK_OUT = "game_trace/tasks/boss"
DEFAULT_STATE_BANK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "states", "boss_level1")


@dataclass(frozen=True)
class BossStart:
    """A sampled search start plus its immutable source lineage."""

    source_path: str
    source: Segment
    initial_state: bytes
    offset: int
    offset_frac: float
    boss_hp_start: int
    weapon: str
    rapid: bool
    start_x: int


@dataclass(frozen=True)
class BatchRequest:
    """One deterministic request in a resumable generation batch."""

    request_id: int
    source_path: str
    full: bool
    start_seed: int


def train_sources(pattern: str = DEFAULT_SOURCES) -> list[str]:
    """Return boss task paths that explicitly belong to the train split.

    Missing split metadata is rejected rather than re-derived here. Generation
    is a leak-sensitive operation and must consume the dataset's recorded split.
    """
    paths = []
    rejected = 0
    for path in sorted(glob.glob(pattern)):
        with np.load(path, allow_pickle=True) as d:
            split = str(d["split"]) if "split" in d.files else ""
            label = str(d["label"]) if "label" in d.files else ""
            derived = "source_task" in d.files
        if split == "train" and label == "boss_level1" and not derived:
            paths.append(path)
        else:
            rejected += 1
    if not paths:
        raise ValueError(f"no explicit train boss_level1 tasks match {pattern!r}")
    print(f"sources: {len(paths)} train boss tasks ({rejected} non-train/other rejected)")
    return paths


def _weapon_meta(ram) -> tuple[str, bool]:
    raw = int(ram[ADDR_WEAPON])
    gun = raw & 0x0f
    return WEAPON_NAMES.get(gun, f"Unknown{gun}"), bool(raw & 0x10)


def build_state_bank(paths: list[str], out_dir: str = DEFAULT_STATE_BANK,
                     *, seed: int = 0) -> list[dict]:
    """Save a compact, weapon-balanced bank of full and partial boss states.

    One representative train source per observed weapon is chosen by median
    fight length. Each contributes its reveal state and one deterministic
    post-reveal partial state. Files use the same gzip-compressed stable-retro
    state format as ``states/spread_gun/Level<N>.state``; ``manifest.yaml`` owns
    the source lineage and interpreted metadata.
    """
    by_weapon = {}
    for path in paths:
        seg = load_task(path)
        weapon = str(seg.meta.get("weapon", ""))
        if not weapon:
            raise ValueError(f"boss task lacks weapon metadata: {path}")
        by_weapon.setdefault(weapon, []).append((len(seg.actions), path))
    os.makedirs(out_dir, exist_ok=True)
    entries = []

    for weapon_index, weapon in enumerate(sorted(by_weapon)):
        ranked = sorted(by_weapon[weapon])
        _, source_path = ranked[len(ranked) // 2]
        starts = [
            ("full", capture_start(source_path, full=True,
                                   rng=np.random.default_rng(seed + weapon_index))),
            ("partial", capture_start(
                source_path, full=False,
                rng=np.random.default_rng(
                    np.random.SeedSequence([seed, weapon_index, 1]).generate_state(1)[0]
                ),
            )),
        ]
        slug = weapon.lower().replace(" ", "_")
        for stage, start in starts:
            filename = f"{stage}_{slug}.state"
            path = os.path.join(out_dir, filename)
            with open(path, "wb") as raw:
                with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as fh:
                    fh.write(start.initial_state)
            entries.append({
                "name": f"{stage}_{slug}",
                "file": filename,
                "stage": stage,
                "source_task": start.source.uid,
                "src_trace": start.source.src_trace,
                "split": "train",
                "source_offset": start.offset,
                "offset_frac": float(start.offset_frac),
                "weapon": start.weapon,
                "rapid": bool(start.rapid),
                "boss_hp_start": int(start.boss_hp_start),
                "skip": int(start.source.skip),
                "state_sha256": hashlib.sha256(start.initial_state).hexdigest(),
            })
    manifest = {
        "level": 1,
        "seed": int(seed),
        "selection": "median-length train source per weapon; full + one partial",
        "states": entries,
    }
    with open(os.path.join(out_dir, "manifest.yaml"), "w") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False)
    print(f"wrote {len(entries)} boss states -> {out_dir}")
    return entries


def capture_start(source_path: str, *, full: bool,
                  rng: np.random.Generator, min_remaining: int = 8) -> BossStart:
    """Replay one source and capture a reveal or live-HP start state.

    Partial starts are uniform over decision-boundary states at or after the
    fully revealed objective-HP peak, with at least ``min_remaining`` source
    decisions left. The tail guard excludes states where the boss is already
    defeated but the level-transition edge has not fired yet. Full starts keep
    offset zero; their HP metadata is the maximum objective HP exposed during
    the reveal because the saved reveal state precedes component activation by
    a few decisions.
    """
    source = load_task(source_path)
    if source.split != "train":
        raise ValueError(f"refusing non-train boss source: {source_path}")
    if source.label != "boss_level1":
        raise ValueError(f"expected boss_level1 source, got {source.label!r}")
    if source.skip != SKIP:
        raise ValueError(
            f"source skip {source.skip} != search skip {SKIP}: {source_path}")

    env = make_env()
    rewind_state(env, source.initial_state)
    initial = bytes(source.initial_state)
    initial_ram = env.unwrapped.get_ram().copy()
    candidates = []
    max_hp = boss_hp(initial_ram)
    try:
        # State at offset i is the state before source.actions[i]. The final
        # post-action state is not eligible because no source decision remains.
        for offset, action in enumerate(source.actions):
            ram = env.unwrapped.get_ram().copy()
            hp = boss_hp(ram)
            max_hp = max(max_hp, hp)
            if hp > 0 and len(source.actions) - offset >= min_remaining:
                weapon, rapid = _weapon_meta(ram)
                candidates.append((bytes(env.em.get_state()), offset, hp,
                                   weapon, rapid, player_x(ram)))
            step_env(env, np.asarray(action, dtype=np.uint8))
            max_hp = max(max_hp, boss_hp(env.unwrapped.get_ram()))
    finally:
        env.close()

    if full:
        weapon, rapid = _weapon_meta(initial_ram)
        return BossStart(
            source_path=source_path, source=source, initial_state=initial,
            offset=0, offset_frac=0.0, boss_hp_start=max_hp,
            weapon=weapon, rapid=rapid, start_x=player_x(initial_ram),
        )
    # Do not treat the staged component spawn (16 -> 48 -> ~64 HP) as a partial
    # fight. The first source state at the observed maximum marks full reveal.
    if not candidates:
        raise ValueError(
            f"source has no post-reveal boss state with {min_remaining} decisions left: "
            f"{source_path}")
    peak_hp = max(c[2] for c in candidates)
    reveal_offset = next(c[1] for c in candidates if c[2] == peak_hp)
    candidates = [c for c in candidates if c[1] >= reveal_offset]
    chosen = candidates[int(rng.integers(len(candidates)))]
    state, offset, hp, weapon, rapid, start_x = chosen
    return BossStart(
        source_path=source_path, source=source, initial_state=state,
        offset=offset, offset_frac=offset / len(source.actions),
        boss_hp_start=hp, weapon=weapon, rapid=rapid, start_x=start_x,
    )


def _uid(start: BossStart, batch: str, attempt: int) -> str:
    return (f"{start.source.uid}__bosssearch_{batch}_"
            f"o{start.offset:04d}_i{attempt:04d}")


def batch_requests(paths: list[str], *, full_per_source: int,
                   partial_runs: int, seed: int, num_shards: int = 1,
                   shard_index: int = 0) -> list[BatchRequest]:
    """Build an exact, deterministic full/partial schedule for one shard.

    Every source appears exactly ``full_per_source`` times in the full bucket.
    Partial sources are sampled trace-first with replacement. Global request IDs
    are partitioned modulo ``num_shards``, so independently running shards are
    disjoint and their union is the unsharded schedule.
    """
    if full_per_source < 0 or partial_runs < 0:
        raise ValueError("batch request counts cannot be negative")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if not paths:
        raise ValueError("at least one train source is required")

    ordered = sorted(paths)
    all_requests = []
    request_id = 0
    for _ in range(full_per_source):
        for source_path in ordered:
            start_seed = int(np.random.SeedSequence([seed, request_id, 0]).generate_state(1)[0])
            all_requests.append(BatchRequest(request_id, source_path, True, start_seed))
            request_id += 1

    source_rng = np.random.default_rng(seed)
    for _ in range(partial_runs):
        source_path = ordered[int(source_rng.integers(len(ordered)))]
        start_seed = int(np.random.SeedSequence([seed, request_id, 1]).generate_state(1)[0])
        all_requests.append(BatchRequest(request_id, source_path, False, start_seed))
        request_id += 1
    return [r for r in all_requests if r.request_id % num_shards == shard_index]


def generate_batch(requests: list[BatchRequest], *, batch_id: str,
                   trace_out: str = DEFAULT_TRACE_OUT,
                   task_out: str = DEFAULT_TASK_OUT, rollouts: int = 64,
                   rollout_len: int = 48, settle_margin: int = 16,
                   max_time: int = 600, max_rewind: int = 30,
                   max_actions: int = 1000, workers: int | None = None,
                   attempts_per_request: int = 3, min_remaining: int = 8,
                   rebuild_manifest: bool = True) -> tuple[int, int, list[int]]:
    """Run a deterministic, resumable batch.

    Returns ``(written, skipped, failed_request_ids)``. A task whose stable UID
    already exists is skipped, so the same command safely resumes an interrupted
    shard. Each request keeps the same sampled start across retries.
    """
    if attempts_per_request < 1:
        raise ValueError("attempts_per_request must be positive")
    workers = workers or os.cpu_count() or 1
    written = skipped = 0
    failed = []
    total = len(requests)

    for position, request in enumerate(requests, 1):
        start = capture_start(
            request.source_path, full=request.full,
            rng=np.random.default_rng(request.start_seed),
            min_remaining=min_remaining,
        )
        uid = (f"{start.source.uid}__bosssearch_{batch_id}_"
               f"r{request.request_id:05d}_o{start.offset:04d}")
        task_path = os.path.join(task_out, start.source.label, uid + ".npz")
        trace_path = os.path.join(trace_out, uid + ".npz")
        if os.path.exists(task_path) and os.path.exists(trace_path):
            skipped += 1
            print(f"[{position}/{total}] skip existing r{request.request_id:05d}",
                  flush=True)
            continue

        metadata = {
            "batch_id": batch_id,
            "batch_request_id": request.request_id,
            "source_task": start.source.uid,
            "src_trace": start.source.src_trace,
            "split": "train",
            "source_offset": start.offset,
            "offset_frac": start.offset_frac,
            "boss_hp_start": start.boss_hp_start,
            "weapon": start.weapon,
            "rapid": start.rapid,
        }
        print(f"[{position}/{total}] r{request.request_id:05d} "
              f"{'full' if request.full else 'partial'} {start.source.uid} "
              f"offset={start.offset} hp={start.boss_hp_start}", flush=True)
        won = None
        for retry in range(attempts_per_request):
            instance_id = request.request_id * attempts_per_request + retry
            won = _run_one_search(
                level=start.source.level + 1,
                rollouts=rollouts, rollout_len=rollout_len,
                max_time=max_time, max_rewind=max_rewind,
                max_actions=max_actions, goal="level_up", workers=workers,
                settle_margin=settle_margin, verbose=False,
                instance_id=instance_id,
                initial_emu_state=start.initial_state, trace_path=trace_path,
                trace_metadata=metadata,
            )
            if won is not None:
                break
        if won is None:
            failed.append(request.request_id)
            print(f"  FAILED after {attempts_per_request} attempts", flush=True)
            continue
        seg = task_from_trace(won, start, uid)
        seg.meta["batch_id"] = batch_id
        seg.meta["batch_request_id"] = request.request_id
        write_segment(seg, task_out)
        written += 1
        print(f"  WIN {len(seg.actions)} decisions -> {task_path}", flush=True)

    if rebuild_manifest:
        build_manifest(task_out)
    print(f"batch {batch_id}: written={written} skipped={skipped} failed={len(failed)}")
    return written, skipped, failed


def task_from_trace(trace_path: str, start: BossStart, uid: str) -> Segment:
    """Build and replay-verify a normal boss task from a winning search trace."""
    with np.load(trace_path, allow_pickle=True) as d:
        actions = np.asarray(d["actions"], dtype=np.uint8)
        skip = int(d["skip"])
    start_step = start.source.start_step + start.offset
    seg = Segment(
        initial_state=start.initial_state,
        actions=actions,
        label=start.source.label,
        level=start.source.level,
        start_step=start_step,
        end_step=start_step + len(actions) - 1,
        skip=skip,
        src_trace=start.source.src_trace,
        uid=uid,
        split="train",
        meta={
            "start_x": start.start_x,
            "goal_when": "boss",
            "goal_kind": "boss",
            "weapon": start.weapon,
            "rapid": start.rapid,
            "boss_hp_start": start.boss_hp_start,
            "offset_frac": start.offset_frac,
            "source_task": start.source.uid,
            "source_offset": start.offset,
        },
    )
    reached = False
    end_ram = None
    for pre, cur in iter_segment(seg):
        end_ram = cur
        reached = reached or KillBossMaker().goal_reached(seg, pre, cur)
    if not reached:
        raise RuntimeError(f"winning search trace does not replay to boss clear: {trace_path}")
    seg.meta["end_x"] = player_x(end_ram)
    return seg


def generate(paths: list[str], runs: int, *, full_fraction: float = 0.3,
             seed: int = 0, trace_out: str = DEFAULT_TRACE_OUT,
             task_out: str = DEFAULT_TASK_OUT, rollouts: int = 64,
             rollout_len: int = 48, settle_margin: int = 16,
             max_time: int = 600, max_rewind: int = 30,
             max_actions: int = 1000, workers: int | None = None,
             max_attempts: int | None = None, min_remaining: int = 8) -> list[str]:
    """Collect ``runs`` replay-verified boss wins and return task paths."""
    if not 0.0 <= full_fraction <= 1.0:
        raise ValueError("full_fraction must be in [0, 1]")
    if runs < 1:
        raise ValueError("runs must be positive")
    if not paths:
        raise ValueError("at least one train source is required")
    if min_remaining < 1:
        raise ValueError("min_remaining must be positive")
    workers = workers or os.cpu_count() or 1
    max_attempts = max_attempts or runs * 3
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    rng = np.random.default_rng(seed)
    batch = time.strftime("%Y%m%d%H%M%S")
    written = []

    for attempt in range(max_attempts):
        if len(written) >= runs:
            break
        source_path = paths[int(rng.integers(len(paths)))]
        full = bool(rng.random() < full_fraction)
        start = capture_start(source_path, full=full, rng=rng,
                              min_remaining=min_remaining)
        uid = _uid(start, batch, attempt)
        trace_path = os.path.join(trace_out, uid + ".npz")
        metadata = {
            "source_task": start.source.uid,
            "src_trace": start.source.src_trace,
            "split": "train",
            "source_offset": start.offset,
            "offset_frac": start.offset_frac,
            "boss_hp_start": start.boss_hp_start,
            "weapon": start.weapon,
            "rapid": start.rapid,
        }
        print(f"attempt {attempt + 1}/{max_attempts}: "
              f"{'full' if full else 'partial'} {start.source.uid} "
              f"offset={start.offset} hp={start.boss_hp_start}", flush=True)
        won = _run_one_search(
            level=start.source.level + 1,
            rollouts=rollouts, rollout_len=rollout_len,
            max_time=max_time, max_rewind=max_rewind,
            max_actions=max_actions, goal="level_up", workers=workers,
            settle_margin=settle_margin, verbose=False, instance_id=attempt,
            initial_emu_state=start.initial_state, trace_path=trace_path,
            trace_metadata=metadata,
        )
        if won is None:
            continue
        seg = task_from_trace(won, start, uid)
        written.append(write_segment(seg, task_out))
        print(f"  WIN {len(seg.actions)} decisions -> {written[-1]}", flush=True)

    build_manifest(task_out)
    print(f"generated {len(written)}/{runs} boss tasks in "
          f"{min(max_attempts, attempt + 1)} attempts")
    return written


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", default=DEFAULT_SOURCES,
                   help="glob of existing boss task .npz files")
    p.add_argument("--runs", type=int, default=1, help="winning tasks to collect")
    p.add_argument("--full-fraction", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trace-out", default=DEFAULT_TRACE_OUT)
    p.add_argument("--task-out", default=DEFAULT_TASK_OUT)
    p.add_argument("--rollouts", type=int, default=64)
    p.add_argument("--rollout-len", type=int, default=48)
    p.add_argument("--settle-margin", type=int, default=16)
    p.add_argument("--max-time", type=int, default=600)
    p.add_argument("--max-rewind", type=int, default=30)
    p.add_argument("--max-actions", type=int, default=1000)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--min-remaining", type=int, default=8,
                   help="minimum source decisions left for a partial start")
    p.add_argument("--full-per-source", type=int, default=None,
                   help="exact full-start wins per source; enables batch mode")
    p.add_argument("--partial-runs", type=int, default=0,
                   help="exact partial-start wins in batch mode")
    p.add_argument("--batch-id", default="issue2-k1")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--attempts-per-request", type=int, default=3)
    p.add_argument("--no-manifest", action="store_true",
                   help="skip manifest rebuild (use for concurrently running shards)")
    p.add_argument("--build-state-bank", nargs="?", const=DEFAULT_STATE_BANK,
                   metavar="DIR", help="write curated boss states and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    paths = train_sources(args.sources)
    if args.build_state_bank:
        build_state_bank(paths, args.build_state_bank, seed=args.seed)
        return
    if args.full_per_source is not None:
        requests = batch_requests(
            paths, full_per_source=args.full_per_source,
            partial_runs=args.partial_runs, seed=args.seed,
            num_shards=args.num_shards, shard_index=args.shard_index,
        )
        _, _, failed = generate_batch(
            requests, batch_id=args.batch_id, trace_out=args.trace_out,
            task_out=args.task_out, rollouts=args.rollouts,
            rollout_len=args.rollout_len, settle_margin=args.settle_margin,
            max_time=args.max_time, max_rewind=args.max_rewind,
            max_actions=args.max_actions, workers=args.workers,
            attempts_per_request=args.attempts_per_request,
            min_remaining=args.min_remaining,
            rebuild_manifest=not args.no_manifest,
        )
        if failed:
            raise SystemExit(f"failed batch request IDs: {failed}")
        return
    generate(
        paths, args.runs, full_fraction=args.full_fraction, seed=args.seed,
        trace_out=args.trace_out, task_out=args.task_out,
        rollouts=args.rollouts, rollout_len=args.rollout_len,
        settle_margin=args.settle_margin, max_time=args.max_time,
        max_rewind=args.max_rewind, max_actions=args.max_actions,
        workers=args.workers, max_attempts=args.max_attempts,
        min_remaining=args.min_remaining,
    )


if __name__ == "__main__":
    main()
