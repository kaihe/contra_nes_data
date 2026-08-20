r"""Build verified, diversity-audited full-boss WebDataset releases.

The input is a set of raw ``mc_search --initial-state`` traces.  A raw trace is
matched to the checksummed full-fight state bank, replayed into a normal
``boss_level1`` task, compared with the published train set, and then combined
with that published baseline in deterministic frame-balanced shards.  The
published validation tar is copied byte-for-byte; it is never regenerated.
``--train-mode generated_only`` instead uses the published train set solely as
a diversity reference and produces nested shard-prefix scales over generated
tasks only.

Example::

    python -m task_maker.boss_release --batch-id boss-full-v1 \
        --traces 'game_trace/mc_trace/boss_level1/win_boss_level1_full_*.npz' \
        --out game_trace/releases/boss-full-v1
"""

import argparse
import glob
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import tarfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import yaml

from agent.boss_search import BossStart, task_from_trace
from env.entity import ADDR_PLAYER_X, ADDR_PLAYER_Y, scan, terrain_state
from env.utility import boss_hp
from task_maker.base import Segment, iter_segment, load_task, write_segment
from task_maker.export_hf import write_shard


DEFAULT_BANK = "src/agent/states/boss_level1/manifest.yaml"
DEFAULT_TASK_GLOB = "game_trace/tasks/boss/boss_level1/*.npz"
DEFAULT_BASELINE_TRAIN = "game_trace/hf/boss-train-00000.tar"
DEFAULT_VALIDATION = "game_trace/hf/boss-val-00000.tar"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_full_bank(path: str = DEFAULT_BANK) -> dict[str, dict]:
    """Return full-fight bank entries keyed by decompressed state SHA-256."""
    with open(path) as fh:
        manifest = yaml.safe_load(fh) or {}
    entries = {}
    root = os.path.dirname(path)
    for entry in manifest.get("states", []):
        if entry.get("stage") != "full":
            continue
        state_path = os.path.join(root, entry["file"])
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"state-bank file is missing: {state_path}")
        with gzip.open(state_path, "rb") as fh:
            actual_sha = hashlib.sha256(fh.read()).hexdigest()
        if actual_sha != str(entry["state_sha256"]):
            raise ValueError(f"state-bank checksum mismatch: {state_path}")
        if entry.get("split") != "train" or float(entry.get("offset_frac", -1)) != 0.0:
            raise ValueError(f"not a full train start: {entry}")
        entries[str(entry["state_sha256"])] = dict(entry)
    if not entries:
        raise ValueError(f"no full-fight entries in {path}")
    return entries


def _task_index(pattern: str = DEFAULT_TASK_GLOB) -> dict[str, str]:
    return {os.path.splitext(os.path.basename(p))[0]: p
            for p in sorted(glob.glob(pattern))}


def import_traces(trace_paths: list[str], *, batch_id: str, out_root: str,
                  bank_path: str = DEFAULT_BANK,
                  source_pattern: str = DEFAULT_TASK_GLOB) -> list[str]:
    """Convert raw full-state wins into replay-verified train-only boss tasks."""
    bank = load_full_bank(bank_path)
    sources = _task_index(source_pattern)
    written = []
    seen = set()
    for trace_path in sorted(trace_paths):
        with np.load(trace_path, allow_pickle=True) as d:
            actions = np.asarray(d["actions"], dtype=np.uint8)
            initial = bytes(d["initial_state"])
            skip = int(d["skip"])
            outcome = str(d["outcome"]) if "outcome" in d.files else ""
        if outcome and outcome != "win":
            raise ValueError(f"candidate is not a win: {trace_path}")
        state_sha = hashlib.sha256(initial).hexdigest()
        if state_sha not in bank:
            raise ValueError(f"trace does not start from the full-fight bank: {trace_path}")
        entry = bank[state_sha]
        source_uid = str(entry["source_task"])
        if source_uid not in sources:
            raise FileNotFoundError(f"source task {source_uid!r} is unavailable")
        source_path = sources[source_uid]
        source = load_task(source_path)
        if source.split != "train" or source.src_trace != str(entry["src_trace"]):
            raise ValueError(f"state-bank lineage disagrees with {source_path}")
        if skip != source.skip:
            raise ValueError(f"trace skip {skip} != source skip {source.skip}: {trace_path}")

        fingerprint = hashlib.sha256(initial + actions.tobytes()).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        uid = f"{source.uid}__bossfull_{batch_id}_{fingerprint[:16]}"
        start = BossStart(
            source_path=source_path, source=source, initial_state=initial,
            offset=0, offset_frac=0.0,
            boss_hp_start=int(entry["boss_hp_start"]),
            weapon=str(entry["weapon"]), rapid=bool(entry["rapid"]),
            start_x=int(source.meta["start_x"]),
        )
        seg = task_from_trace(trace_path, start, uid)
        seg.meta.update({
            "batch_id": batch_id,
            "stage": "full",
            "initial_state_name": str(entry["name"]),
            "initial_state_sha256": state_sha,
            "raw_trace": os.path.basename(trace_path),
            "trace_fingerprint": fingerprint,
        })
        written.append(write_segment(seg, out_root))
    return written


def task_fingerprint(path: str) -> str:
    """Return the state/action identity used to keep release partitions disjoint."""
    seg = load_task(path)
    return hashlib.sha256(seg.initial_state + seg.actions.tobytes()).hexdigest()


def split_holdout_paths(candidate_paths: list[str], validation_count: int) -> tuple[list[str], list[str]]:
    """Deterministically reserve fingerprint-ranked candidates for validation.

    The raw boss starts come from training lineage, but a release-level holdout
    needs a stable, action-level partition so generated demonstrations cannot
    appear in both its train and validation shards.
    """
    if validation_count < 0:
        raise ValueError("validation_count must be non-negative")
    keyed = sorted((task_fingerprint(path), path) for path in candidate_paths)
    fingerprints = [fingerprint for fingerprint, _ in keyed]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("candidate tasks contain exact state/action duplicates")
    if validation_count >= len(keyed):
        raise ValueError(
            f"validation_count {validation_count} leaves no training candidates")
    validation = [path for _, path in keyed[:validation_count]]
    train = [path for _, path in keyed[validation_count:]]
    return train, validation


def _action_tokens(actions: np.ndarray) -> np.ndarray:
    weights = (1 << np.arange(9, dtype=np.uint16))
    return np.asarray(actions, dtype=np.uint16).dot(weights)


def _action_ngrams(actions: np.ndarray) -> Counter:
    tokens = _action_tokens(actions).tolist()
    out = Counter(("u", int(a)) for a in tokens)
    out.update(("b", int(a), int(b)) for a, b in zip(tokens, tokens[1:]))
    return out


def _js_distance(a: Counter, b: Counter) -> float:
    keys = sorted(set(a) | set(b))
    if not keys:
        return 0.0
    sa, sb = float(sum(a.values())), float(sum(b.values()))
    pa = np.array([a[k] / sa for k in keys], dtype=np.float64)
    pb = np.array([b[k] / sb for k in keys], dtype=np.float64)
    mid = (pa + pb) / 2.0

    def kl(p):
        nz = p > 0
        return float(np.sum(p[nz] * np.log2(p[nz] / mid[nz])))

    return math.sqrt(max(0.0, (kl(pa) + kl(pb)) / 2.0))


def _state_signature(seg: Segment, bins: int = 20) -> np.ndarray:
    """Progress-align compact replay state to a fixed shape.

    Alignment uses cumulative positive boss-HP decrements rather than time, so
    strategies of different duration are compared at equivalent fight progress.
    The time coordinate retained in each bin still distinguishes fast and slow
    execution of the same spatial strategy.
    """
    states = []
    progress = []
    damage = 0
    for index, (pre, cur) in enumerate(iter_segment(seg)):
        damage += max(0, boss_hp(pre) - boss_hp(cur))
        entities = scan(cur)
        if len(entities.enemy_bullets):
            delta = entities.enemy_bullets.astype(np.float32) - \
                entities.player.astype(np.float32)
            nearest = float(np.sqrt(np.sum(delta * delta, axis=1)).min()) / 350.0
        else:
            nearest = 1.0
        terrain = terrain_state(cur)
        states.append([
            index / max(1, len(seg.actions) - 1),
            int(cur[ADDR_PLAYER_X]) / 255.0,
            int(cur[ADDR_PLAYER_Y]) / 239.0,
            float(terrain == "air"),
            float(terrain == "water"),
            min(1.0, len(entities.player_bullets) / 4.0),
            min(1.0, len(entities.enemy_bullets) / 16.0),
            min(1.0, nearest),
        ])
        progress.append(damage)
    if not states:
        raise ValueError(f"empty task: {seg.uid}")
    total = max(progress[-1], 1)
    progress = np.asarray(progress, dtype=np.float32) / total
    targets = np.linspace(0.0, 1.0, bins, dtype=np.float32)
    indices = np.searchsorted(progress, targets, side="left").clip(0, len(states) - 1)
    return np.asarray(states, dtype=np.float32)[indices]


@dataclass
class DiversityFeatures:
    uid: str
    fingerprint: str
    length: int
    weapon: str
    ngrams: Counter
    state: np.ndarray


def features_for(path: str) -> DiversityFeatures:
    seg = load_task(path)
    fingerprint = hashlib.sha256(seg.initial_state + seg.actions.tobytes()).hexdigest()
    return DiversityFeatures(
        uid=seg.uid, fingerprint=fingerprint, length=len(seg.actions),
        weapon=str(seg.meta.get("weapon", "")),
        ngrams=_action_ngrams(seg.actions), state=_state_signature(seg),
    )


def diversity_distance(a: DiversityFeatures, b: DiversityFeatures) -> float:
    """Combined state/action/duration distance in approximately ``[0, 1]``."""
    state = float(np.sqrt(np.mean((a.state - b.state) ** 2)))
    action = _js_distance(a.ngrams, b.ngrams)
    duration = min(1.0, abs(math.log((a.length + 1) / (b.length + 1))) / math.log(4.0))
    return 0.45 * state + 0.35 * action + 0.20 * duration


def select_diverse(candidate_paths: list[str], reference_paths: list[str], *,
                   min_distance: float = 0.0) -> tuple[list[str], list[dict]]:
    """Greedily accept candidates and report their nearest prior neighbour.

    Exact state+action duplicates are always rejected. ``min_distance`` can add
    a near-duplicate gate; zero keeps every non-exact candidate while still
    producing the evidence needed to choose a threshold after inspection.
    """
    if not 0.0 <= min_distance <= 1.0:
        raise ValueError("min_distance must be in [0, 1]")
    references = [(p, features_for(p)) for p in reference_paths]
    accepted = []
    accepted_features = []
    rows = []
    known = {f.fingerprint for _, f in references}
    for path in sorted(candidate_paths):
        feature = features_for(path)
        whole_pool = references + accepted_features
        # Compare like with like first. Weapon changes alter both the action
        # semantics and the starting RAM, which would inflate apparent novelty.
        pool = [(p, f) for p, f in whole_pool if f.weapon == feature.weapon]
        pool = pool or whole_pool
        nearest_path = None
        nearest = None
        if pool:
            nearest_path, nearest_feature = min(
                pool, key=lambda item: diversity_distance(feature, item[1]))
            nearest = diversity_distance(feature, nearest_feature)
        exact = feature.fingerprint in known
        keep = not exact and (nearest is None or nearest >= min_distance)
        reason = "accepted" if keep else ("exact_duplicate" if exact else "near_duplicate")
        rows.append({
            "uid": feature.uid,
            "path": path,
            "weapon": feature.weapon,
            "length": feature.length,
            "fingerprint": feature.fingerprint,
            "nearest_uid": None if nearest_path is None else
                os.path.splitext(os.path.basename(nearest_path))[0],
            "nearest_distance": nearest,
            "accepted": keep,
            "reason": reason,
        })
        if keep:
            accepted.append(path)
            accepted_features.append((path, feature))
            known.add(feature.fingerprint)
    return accepted, rows


def shard_uids(path: str) -> list[str]:
    """Read episode UIDs from an existing WebDataset shard without decoding video."""
    with tarfile.open(path) as tar:
        return sorted(m.name[:-5] for m in tar if m.name.endswith(".json"))


def task_paths_for_uids(uids: list[str], pattern: str = DEFAULT_TASK_GLOB) -> list[str]:
    index = _task_index(pattern)
    missing = sorted(set(uids) - set(index))
    if missing:
        raise FileNotFoundError(f"{len(missing)} published task files are missing: {missing[:3]}")
    return [index[uid] for uid in uids]


def frame_balanced_shards(paths: list[str], target_frames: int = 60_000) -> list[list[str]]:
    """Deterministically distribute weapon strata by total decision frames."""
    if target_frames < 1:
        raise ValueError("target_frames must be positive")
    items = []
    total = 0
    for path in paths:
        seg = load_task(path)
        length = len(seg.actions)
        total += length
        items.append((str(seg.meta.get("weapon", "")), length, seg.uid, path))
    if not items:
        return []
    count = max(1, math.ceil(total / target_frames))
    shards = [[] for _ in range(count)]
    frames = [0] * count
    by_weapon = {}
    for item in items:
        by_weapon.setdefault(item[0], []).append(item)
    for weapon in sorted(by_weapon):
        for _, length, _, path in sorted(by_weapon[weapon], key=lambda x: (-x[1], x[2])):
            index = min(range(count), key=lambda i: (frames[i], i))
            shards[index].append(path)
            frames[index] += length
    return [sorted(shard) for shard in shards]


def _atomic_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _atomic_shard(paths: list[str], dst: str, *, codec: str) -> tuple[int, int, int]:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    result = write_shard(paths, tmp, codec=codec)
    os.replace(tmp, dst)
    return result


def build_release(*, trace_paths: list[str], batch_id: str, out_dir: str,
                  bank_path: str = DEFAULT_BANK,
                  baseline_train: str = DEFAULT_BASELINE_TRAIN,
                  validation: str = DEFAULT_VALIDATION,
                  task_pattern: str = DEFAULT_TASK_GLOB,
                  target_frames: int = 60_000, min_distance: float = 0.0,
                  codec: str = "png",
                  train_mode: str = "baseline_plus_generated",
                  expected_candidates: int | None = None,
                  holdout_validation_count: int = 0) -> dict:
    """Run the full candidate-to-release pipeline and return its manifest."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", batch_id):
        raise ValueError("batch_id may contain only letters, digits, dot, underscore and dash")
    if train_mode not in {"baseline_plus_generated", "generated_only"}:
        raise ValueError(f"unknown train_mode: {train_mode!r}")
    if holdout_validation_count and train_mode != "generated_only":
        raise ValueError("a generated validation holdout requires generated_only mode")
    if expected_candidates is not None and len(trace_paths) != expected_candidates:
        raise ValueError(
            f"expected {expected_candidates} raw candidates, found {len(trace_paths)}")
    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_path):
        raise FileExistsError(f"release is immutable and already exists: {manifest_path}")
    task_root = os.path.join(out_dir, "tasks")
    candidate_paths = import_traces(
        trace_paths, batch_id=batch_id, out_root=task_root,
        bank_path=bank_path, source_pattern=task_pattern,
    )
    if holdout_validation_count and len(candidate_paths) != len(trace_paths):
        raise ValueError("raw candidates included exact duplicates before holdout split")
    train_sources, validation_sources = split_holdout_paths(
        candidate_paths, holdout_validation_count) if holdout_validation_count \
        else (candidate_paths, [])
    release_path_by_source = {}
    for path in candidate_paths:
        seg = load_task(path)
        seg.meta["source_split"] = seg.split
        seg.meta["release_partition"] = "validation" if path in validation_sources else "train"
        if path in validation_sources:
            seg.split = "val"
        release_path_by_source[path] = write_segment(seg, task_root)
    candidate_paths = [release_path_by_source[path] for path in candidate_paths]
    train_candidates = [release_path_by_source[path] for path in train_sources]
    validation_paths = [release_path_by_source[path] for path in validation_sources]
    baseline_paths = task_paths_for_uids(shard_uids(baseline_train), task_pattern)
    accepted, diversity = select_diverse(
        train_candidates, baseline_paths, min_distance=min_distance)
    row_by_path = {row["path"]: row for row in diversity}
    measured = [row["nearest_distance"] for row in diversity
                if row["nearest_distance"] is not None]
    by_weapon = {}
    for row in diversity:
        counts = by_weapon.setdefault(row["weapon"], {"candidates": 0, "accepted": 0})
        counts["candidates"] += 1
        counts["accepted"] += int(row["accepted"])
    diversity_summary = {
        "exact_duplicates": sum(row["reason"] == "exact_duplicate" for row in diversity),
        "near_duplicates": sum(row["reason"] == "near_duplicate" for row in diversity),
        "nearest_distance_p10": float(np.quantile(measured, 0.10)) if measured else None,
        "nearest_distance_median": float(np.median(measured)) if measured else None,
        "nearest_distance_p90": float(np.quantile(measured, 0.90)) if measured else None,
        "by_weapon": by_weapon,
    }
    for path in accepted:
        seg = load_task(path)
        row = row_by_path[path]
        seg.meta["diversity_nearest_uid"] = row["nearest_uid"] or ""
        seg.meta["diversity_nearest_distance"] = row["nearest_distance"] \
            if row["nearest_distance"] is not None else -1.0
        write_segment(seg, task_root)

    hf_dir = os.path.join(out_dir, "hf")
    val_out = os.path.join(hf_dir, "boss-val-00000.tar")
    if validation_paths:
        validation_bytes, validation_episodes, validation_frames = _atomic_shard(
            validation_paths, val_out, codec=codec)
        validation_sha = sha256_file(val_out)
        validation_info = {
            "file": os.path.relpath(val_out, out_dir),
            "sha256": validation_sha,
            "episodes": validation_episodes,
            "frames": validation_frames,
            "bytes": validation_bytes,
            "kind": "generated_holdout",
            "tasks": [
                {"uid": os.path.splitext(os.path.basename(path))[0],
                 "sha256": sha256_file(path),
                 "trace_fingerprint": task_fingerprint(path)}
                for path in validation_paths
            ],
        }
    else:
        _atomic_copy(validation, val_out)
        validation_sha = sha256_file(validation)
        if sha256_file(val_out) != validation_sha:
            raise RuntimeError("copied validation shard hash changed")
        validation_info = {
            "file": os.path.relpath(val_out, out_dir),
            "sha256": validation_sha,
            "episodes": len(shard_uids(validation)),
            "kind": "copied",
        }

    train_paths = (accepted if train_mode == "generated_only"
                   else baseline_paths + accepted)
    groups = frame_balanced_shards(train_paths, target_frames)
    shard_rows = []
    for index, group in enumerate(groups):
        path = os.path.join(hf_dir, f"boss-train-{index:05d}.tar")
        size, episodes, frames = _atomic_shard(group, path, codec=codec)
        weapon_counts = Counter(str(load_task(p).meta.get("weapon", "")) for p in group)
        shard_rows.append({
            "file": os.path.relpath(path, out_dir),
            "sha256": sha256_file(path),
            "episodes": episodes,
            "frames": frames,
            "bytes": size,
            "by_weapon": dict(sorted(weapon_counts.items())),
            "uids": [os.path.splitext(os.path.basename(p))[0] for p in group],
        })

    prefix_counts = []
    count = 1
    while count < len(shard_rows):
        prefix_counts.append(count)
        count *= 2
    if shard_rows and (not prefix_counts or prefix_counts[-1] != len(shard_rows)):
        prefix_counts.append(len(shard_rows))
    scaling_prefixes = []
    for count in prefix_counts:
        prefix = shard_rows[:count]
        scaling_prefixes.append({
            "shard_count": count,
            "files": [row["file"] for row in prefix],
            "episodes": sum(row["episodes"] for row in prefix),
            "frames": sum(row["frames"] for row in prefix),
            "by_weapon": dict(sorted(sum(
                (Counter(row["by_weapon"]) for row in prefix), Counter()).items())),
        })

    manifest = {
        "format_version": 1,
        "batch_id": batch_id,
        "train_mode": train_mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "codec": codec,
        "target_frames_per_train_shard": target_frames,
        "min_diversity_distance": min_distance,
        "baseline_reference_shard": baseline_train,
        "baseline_reference_episodes": len(baseline_paths),
        "baseline_train_episodes": (0 if train_mode == "generated_only"
                                    else len(baseline_paths)),
        "expected_candidate_files": expected_candidates,
        "raw_candidate_files": len(trace_paths),
        "candidate_episodes": len(candidate_paths),
        "holdout_validation_count": holdout_validation_count,
        "accepted_generated_episodes": len(accepted),
        "accepted_generated_tasks": [
            {
                "uid": os.path.splitext(os.path.basename(path))[0],
                "file": os.path.relpath(path, out_dir),
                "sha256": sha256_file(path),
                "trace_fingerprint": row_by_path[path]["fingerprint"],
            }
            for path in accepted
        ],
        "validation": validation_info,
        "train_shards": shard_rows,
        "train_scaling_prefixes": scaling_prefixes,
        "diversity_summary": diversity_summary,
        "diversity": diversity,
    }
    os.makedirs(out_dir, exist_ok=True)
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, manifest_path)
    print(f"release manifest: {manifest_path}")
    return manifest


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-id", required=True)
    p.add_argument("--traces", required=True, help="glob of raw full-state wins")
    p.add_argument("--out", required=True, help="versioned release directory")
    p.add_argument("--state-bank", default=DEFAULT_BANK)
    p.add_argument("--source-tasks", default=DEFAULT_TASK_GLOB)
    p.add_argument("--baseline-train", default=DEFAULT_BASELINE_TRAIN)
    p.add_argument("--validation", default=DEFAULT_VALIDATION)
    p.add_argument("--target-frames", type=int, default=60_000)
    p.add_argument("--min-distance", type=float, default=0.0)
    p.add_argument("--codec", choices=["png", "ffv1", "h264"], default="png")
    p.add_argument("--train-mode",
                   choices=["baseline_plus_generated", "generated_only"],
                   default="baseline_plus_generated")
    p.add_argument("--expected-candidates", type=int,
                   help="refuse a live/incomplete trace snapshot with another count")
    p.add_argument("--holdout-validation-count", type=int, default=0,
                   help="reserve this many generated candidates as a disjoint validation shard")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    traces = sorted(glob.glob(args.traces))
    if not traces:
        raise SystemExit(f"no traces match {args.traces!r}")
    build_release(
        trace_paths=traces, batch_id=args.batch_id, out_dir=args.out,
        bank_path=args.state_bank, baseline_train=args.baseline_train,
        validation=args.validation, task_pattern=args.source_tasks,
        target_frames=args.target_frames, min_distance=args.min_distance,
        codec=args.codec, train_mode=args.train_mode,
        expected_candidates=args.expected_candidates,
        holdout_validation_count=args.holdout_validation_count,
    )


if __name__ == "__main__":
    main()
