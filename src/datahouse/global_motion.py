"""Audit pixel-only global translation against replay-time RAM scroll truth."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from datahouse.full_level import _download_archive, _extract_selected, sha256_file
from env.utility import xscroll
from util.replay import make_env, rewind_state, step_env


DEFAULT_SNAPSHOT = Path("game_trace/datahouse/collections/l1-full-10k-v1.json")
DEFAULT_FINGERPRINT = "9c15be3fc41e7febaafd1dcc8ca77a468b0d1283a14b13dfec571355aa8fc6ce"


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Return deterministic BT.601-like integer luminance."""
    value = np.asarray(rgb, dtype=np.uint16)
    return ((77 * value[..., 0] + 150 * value[..., 1] +
             29 * value[..., 2]) >> 8).astype(np.uint8)


def overlap(previous: np.ndarray, current: np.ndarray, dx: int, dy: int):
    """Return overlapping views when previous is translated by (dx, dy)."""
    height, width = previous.shape[:2]
    if abs(dx) >= width or abs(dy) >= height:
        raise ValueError("translation leaves no valid overlap")
    px = slice(max(0, -dx), min(width, width - dx))
    py = slice(max(0, -dy), min(height, height - dy))
    cx = slice(max(0, dx), min(width, width + dx))
    cy = slice(max(0, dy), min(height, height + dy))
    return previous[py, px], current[cy, cx]


def trimmed_score(previous: np.ndarray, current: np.ndarray,
                  dx: int, dy: int, trim: float = .2) -> float:
    """Mean absolute error after dropping the largest ``trim`` fraction."""
    left, right = overlap(previous, current, dx, dy)
    errors = cv2.absdiff(left, right).reshape(-1)
    keep = max(1, int(len(errors) * (1.0 - trim)))
    if keep == len(errors):
        return float(errors.mean())
    return float(np.partition(errors, keep - 1)[:keep].mean())


def estimate_translation(previous_rgb: np.ndarray, current_rgb: np.ndarray,
                         *, max_shift: int = 16, coarse_factor: int = 4,
                         fine_radius: int = 3) -> dict:
    """Bounded exhaustive coarse-to-fine integer translation search."""
    previous = luminance(previous_rgb)
    current = luminance(current_rgb)
    height, width = previous.shape
    if current.shape != previous.shape:
        raise ValueError("frames must have equal dimensions")
    if height % coarse_factor or width % coarse_factor:
        raise ValueError("frame dimensions must be divisible by coarse_factor")
    size = (width // coarse_factor, height // coarse_factor)
    coarse_previous = cv2.resize(previous, size, interpolation=cv2.INTER_AREA)
    coarse_current = cv2.resize(current, size, interpolation=cv2.INTER_AREA)
    coarse_limit = max_shift // coarse_factor
    coarse = []
    for cy in range(-coarse_limit, coarse_limit + 1):
        for cx in range(-coarse_limit, coarse_limit + 1):
            score = trimmed_score(coarse_previous, coarse_current, cx, cy)
            coarse.append((score, cx * coarse_factor, cy * coarse_factor))
    _, center_x, center_y = min(coarse)

    candidates = []
    for dy in range(max(-max_shift, center_y - fine_radius),
                    min(max_shift, center_y + fine_radius) + 1):
        for dx in range(max(-max_shift, center_x - fine_radius),
                        min(max_shift, center_x + fine_radius) + 1):
            candidates.append((trimmed_score(previous, current, dx, dy), dx, dy))
    candidates.sort()
    best, second = candidates[:2]
    return {"dx": best[1], "dy": best[2], "score": best[0],
            "second_score": second[0], "confidence_gap": second[0] - best[0]}


def align_frame(previous: np.ndarray, dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    """Translate a frame, zero exposed borders, and return its validity mask."""
    aligned = np.zeros_like(previous)
    mask = np.zeros(previous.shape[:2], dtype=np.uint8)
    height, width = previous.shape[:2]
    px = slice(max(0, -dx), min(width, width - dx))
    py = slice(max(0, -dy), min(height, height - dy))
    cx = slice(max(0, dx), min(width, width + dx))
    cy = slice(max(0, dy), min(height, height + dy))
    aligned[cy, cx] = previous[py, px]
    mask[cy, cx] = 1
    return aligned, mask


def replay_frames_and_scrolls(source: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(source, allow_pickle=False) as trace:
        actions = np.asarray(trace["actions"], dtype=np.uint8)
        state = bytes(np.asarray(trace["initial_state"], dtype=np.uint8))
        skip = int(trace["skip"]) if "skip" in trace else 4
    env = make_env()
    try:
        rewind_state(env, state)
        frames = [env.em.get_screen().copy()]
        scrolls = [xscroll(env.unwrapped.get_ram())]
        for action in actions:
            step_env(env, action, skip)
            frames.append(env.em.get_screen().copy())
            scrolls.append(xscroll(env.unwrapped.get_ram()))
    finally:
        env.close()
    return np.asarray(frames, dtype=np.uint8), np.asarray(scrolls, dtype=np.int64)


def resolve_trace(snapshot_path: Path, fingerprint: str, cache: Path) -> tuple[Path, dict, dict]:
    snapshot = json.loads(snapshot_path.read_text())
    row = next(row for row in snapshot["selected"] if row["fingerprint"] == fingerprint)
    batch = snapshot["batches"][row["batch_index"]]
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{fingerprint}.npz"
    if target.exists() and sha256_file(target) == row["sha256"]:
        return target, row, batch
    from google.cloud import storage
    with tempfile.TemporaryDirectory(prefix="0017-") as temporary:
        archive = Path(temporary) / "traces.tar.zst"
        _download_archive(storage.Client(), batch, archive)
        _extract_selected(archive, [row], cache)
    return target, row, batch


def _metrics(rows: list[dict], selected: np.ndarray) -> dict:
    items = [row for row, include in zip(rows, selected) if include]
    if not items:
        return {"count": 0}
    dx_error = np.asarray([abs(row["estimated_dx"] - row["truth_dx"]) for row in items])
    dy_error = np.asarray([abs(row["estimated_dy"] - row["truth_dy"]) for row in items])
    exact = (dx_error == 0) & (dy_error == 0)
    within = (dx_error <= 1) & (dy_error <= 1)
    return {
        "count": len(items), "exact_accuracy": float(exact.mean()),
        "within_one_pixel_accuracy": float(within.mean()),
        "dx_mae": float(dx_error.mean()), "dy_mae": float(dy_error.mean()),
        "dx_max_error": int(dx_error.max()), "dy_max_error": int(dy_error.max()),
        "median_zero_residual": float(np.median([row["zero_score"] for row in items])),
        "median_aligned_residual": float(np.median([row["aligned_score"] for row in items])),
    }


def _audit_sheet(frames: np.ndarray, rows: list[dict], output: Path) -> None:
    worst = sorted(rows, key=lambda row: (
        abs(row["estimated_dx"] - row["truth_dx"]) + abs(row["estimated_dy"]),
        row["aligned_score"]), reverse=True)[:3]
    stationary = sorted((row for row in rows if row["truth_dx"] == 0),
                        key=lambda row: row["aligned_score"], reverse=True)[:2]
    onset = next((row for index, row in enumerate(rows[1:], 1)
                  if row["truth_dx"] and rows[index - 1]["truth_dx"] == 0), None)
    selected = worst + stationary + ([onset] if onset else [])
    selected += [rows[min(1050, len(rows) - 1)], rows[-1]]
    ranked, seen = [], set()
    for row in selected:
        if row["pair_index"] not in seen:
            ranked.append(row)
            seen.add(row["pair_index"])
    panels = []
    for row in ranked:
        index = row["pair_index"]
        aligned, _ = align_frame(frames[index], row["estimated_dx"], row["estimated_dy"])
        residual = cv2.absdiff(aligned, frames[index + 1])
        strip = np.concatenate([frames[index], frames[index + 1], residual], axis=1)
        cv2.putText(strip, f"pair {index} truth {row['truth_dx']},0 est "
                    f"{row['estimated_dx']},{row['estimated_dy']}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(strip)
    if panels:
        cv2.imwrite(str(output), cv2.cvtColor(np.concatenate(panels, axis=0),
                                             cv2.COLOR_RGB2BGR))


def evaluate(source: Path, output: Path, *, source_metadata: dict | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    frames, scrolls = replay_frames_and_scrolls(source)
    rows = []
    for index, (previous, current) in enumerate(zip(frames[:-1], frames[1:])):
        truth_dx = -int(scrolls[index + 1] - scrolls[index])
        started = time.perf_counter()
        estimate = estimate_translation(previous, current)
        runtime_ms = (time.perf_counter() - started) * 1000
        previous_y, current_y = luminance(previous), luminance(current)
        zero_score = trimmed_score(previous_y, current_y, 0, 0)
        discontinuity = abs(truth_dx) > 16
        rows.append({
            "pair_index": index, "truth_dx": truth_dx, "truth_dy": 0,
            "discontinuity": discontinuity,
            "estimated_dx": estimate["dx"], "estimated_dy": estimate["dy"],
            "zero_score": zero_score, "aligned_score": estimate["score"],
            "best_score": estimate["score"], "second_score": estimate["second_score"],
            "confidence_gap": estimate["confidence_gap"], "runtime_ms": runtime_ms,
        })
        if (index + 1) % 100 == 0:
            print(f"estimated {index + 1}/{len(frames) - 1} pairs", flush=True)
    with (output / "pairs.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    truth = np.asarray([row["truth_dx"] for row in rows])
    discontinuity = np.asarray([row["discontinuity"] for row in rows])
    stationary = (truth == 0) & ~discontinuity
    scrolling = (truth != 0) & ~discontinuity
    non_discontinuity = ~discontinuity
    estimates_nonzero = np.asarray([(row["estimated_dx"], row["estimated_dy"]) != (0, 0)
                                    for row in rows])
    gaps = np.asarray([row["confidence_gap"] for row in rows])
    runtimes = np.asarray([row["runtime_ms"] for row in rows])
    strata = {
        "all": _metrics(rows, np.ones(len(rows), dtype=bool)),
        "non_discontinuity": _metrics(rows, non_discontinuity),
        "stationary": _metrics(rows, stationary),
        "in_range_scrolling": _metrics(rows, scrolling),
        "discontinuity": _metrics(rows, discontinuity),
    }
    primary = strata["non_discontinuity"]
    zero_rows = [{**row, "estimated_dx": 0, "estimated_dy": 0,
                  "aligned_score": row["zero_score"]} for row in rows]
    zero_strata = {
        "all": _metrics(zero_rows, np.ones(len(rows), dtype=bool)),
        "non_discontinuity": _metrics(zero_rows, non_discontinuity),
        "stationary": _metrics(zero_rows, stationary),
        "in_range_scrolling": _metrics(zero_rows, scrolling),
        "discontinuity": _metrics(zero_rows, discontinuity),
    }
    summary = {
        "source": source_metadata or {"path": str(source)}, "pairs": len(rows),
        "estimators": {"zero": {"strata": zero_strata},
                       "robust": {"strata": strata}},
        "strata": strata,
        "stationary_nonzero_rate": float(estimates_nonzero[stationary].mean()) if stationary.any() else None,
        "confidence_gap": {"p10": float(np.quantile(gaps, .1)), "median": float(np.median(gaps)),
                           "p90": float(np.quantile(gaps, .9))},
        "runtime_ms": {"mean": float(runtimes.mean()), "median": float(np.median(runtimes)),
                       "p90": float(np.quantile(runtimes, .9))},
        "gates": {
            "exact_at_least_99_percent": primary["exact_accuracy"] >= .99,
            "all_non_discontinuity_within_one": primary["within_one_pixel_accuracy"] == 1.0,
            "scrolling_residual_reduced": strata["in_range_scrolling"].get("median_aligned_residual", np.inf) < strata["in_range_scrolling"].get("median_zero_residual", -np.inf),
            "stationary_residual_not_increased": strata["stationary"].get("median_aligned_residual", np.inf) <= strata["stationary"].get("median_zero_residual", -np.inf),
        },
    }
    summary["passed"] = all(summary["gates"].values())
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _audit_sheet(frames, rows, output / "audit-sheet.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--fingerprint", default=DEFAULT_FINGERPRINT)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tmp/0017-global-motion"))
    args = parser.parse_args()
    if args.trace:
        source, metadata = args.trace, {"path": str(args.trace)}
    else:
        source, row, batch = resolve_trace(args.snapshot, args.fingerprint,
                                           args.output / "source")
        metadata = {"fingerprint": args.fingerprint, "split": "validation",
                    "batch_index": row["batch_index"], "member": row["member"],
                    **batch}
    summary = evaluate(source, args.output, source_metadata=metadata)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
