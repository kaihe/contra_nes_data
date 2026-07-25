"""replay.py — replay a Contra action trace and print the events it triggers.

Replays an action sequence through the stable-retro emulator, and at every step
scans the reimplemented event set (:func:`env.event.all_events`) to report which
events fired.

Usage
-----
    # Single trace (prints per-step events + a tag summary):
    python -m util.replay game_trace/mc_trace/level1/win_level1_*.npz

    # Save a video of the replay too:
    python -m util.replay <trace.npz> --video out.mp4

    # From Python:
    result = replay_actions("trace.npz")          # -> {"outcome", "events", "video"}

Trace format
------------
    npz keys: actions (N, 9) uint8, initial_state (bytes), and optionally skip.
    The Contra-Nes retro integration is provided by the installed ``contra``
    package, which registers it on import.
"""

import os
import warnings
from collections import Counter, defaultdict

import numpy as np

warnings.filterwarnings("ignore", message=".*Gym.*")

import stable_retro as retro

from agent.action import DEFAULT as ACTION_SPACE
from env.event import EventType, all_events

GAME = "Contra-Nes"
SKIP = ACTION_SPACE.skip   # NES frames per decision (fallback if trace omits it)
FPS  = round(60 / SKIP)    # logical fps for saved video (= 60 NES fps / SKIP)

# Every event to scan each step (all levels; kill events self-guard by level).
EVENTS = all_events()


# ── Emulator helpers ─────────────────────────────────────────────────────────

def make_env():
    env = retro.make(
        game=GAME,
        state=retro.State.NONE,  # actual state comes from the trace's initial_state
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE,
        render_mode=None,
        inttype=retro.data.Integrations.EXPERIMENTAL_ONLY,  # ships Contra-Nes + rom
    )
    env.reset()
    return env


def rewind_state(env, emu_state: bytes) -> None:
    env.em.set_state(emu_state)
    env.data.update_ram()


def save_video(frames: np.ndarray, path: str) -> None:
    """Save a (N, H, W, 3) uint8 frame array to an MP4 file via ffmpeg."""
    import subprocess
    _, h, w, _ = frames.shape
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
            "-r", str(FPS), "-i", "-",
            "-c:v", "libopenh264", "-pix_fmt", "yuv420p",
            path,
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


# ── Event scanning ───────────────────────────────────────────────────────────

def scan_events(pre_ram: np.ndarray, curr_ram: np.ndarray, step: int) -> list[dict]:
    """Every event that fired this step, with its magnitude and event class."""
    out = []
    for ev in EVENTS:
        value = ev.trigger(pre_ram, curr_ram)
        if value:
            out.append({"step": step, "tag": ev.tag,
                        "value": value, "type": ev.type.value})
    return out


def _fmt_value(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.2f}"


# ── Replay ───────────────────────────────────────────────────────────────────

def replay_actions(source, *,
                   initial_state: bytes = None,
                   want_video:    bool  = False,
                   verbose:       bool  = True) -> dict:
    """Replay a Contra action sequence and collect the events it triggers.

    Parameters
    ----------
    source : str | os.PathLike | np.ndarray
        Path to a .npz trace, or an (N, 9) action array.
    initial_state : bytes
        Required when ``source`` is an array — emulator save-state to rewind to.
    want_video : bool
        If True, also return frames as "video" (N+1, H, W, 3) uint8.

    Returns
    -------
    dict with "outcome" (game_clear|level_up|lose), "events" (list of dicts),
    and "video" (ndarray or None).
    """
    skip = SKIP
    if isinstance(source, (str, os.PathLike)):
        ckpt = np.load(source, allow_pickle=True)
        actions = ckpt["actions"]
        initial_emu_state = bytes(ckpt["initial_state"])
        if "skip" in ckpt:
            skip = int(ckpt["skip"])   # honour the skip the trace was recorded at
        if verbose:
            print(f"  Load {source}  ({len(actions)} actions, skip={skip})")
    else:
        actions = np.asarray(source)
        if initial_state is None:
            raise ValueError("initial_state is required when source is an array")
        initial_emu_state = bytes(initial_state)
        if verbose:
            print(f"  Replaying {len(actions)} actions from array (skip={skip})")

    env = make_env()
    rewind_state(env, initial_emu_state)

    frames = [env.em.get_screen().copy()] if want_video else None

    events: list[dict] = []

    for step, act in enumerate(actions):
        pre_ram = env.unwrapped.get_ram().copy()
        act_arr = np.asarray(act, dtype=np.uint8)
        for i in range(skip):
            obs, _, _, _, _ = env.step(act_arr.copy())
            if i == 0 and frames is not None:
                frames.append(obs.copy())
        curr_ram = env.unwrapped.get_ram()
        events.extend(scan_events(pre_ram, curr_ram, step))

    env.close()

    # Outcome from the fired event tags. A trace often stops at the post-boss
    # transition (in_level_transition) before the level byte increments, so
    # treat that as level-cleared too.
    tags = {e["tag"] for e in events}
    if "clear_game" in tags:
        outcome = "game_clear"
    elif tags & {"clear_level", "in_level_transition"}:
        outcome = "level_up"
    else:
        outcome = "lose"

    if verbose:
        for e in events:
            print(f"  step {e['step']:4d}  {e['tag']:<24}  {_fmt_value(e['value'])}")

        # Tag counts grouped by event class, in EventType declaration order.
        by_type: dict[str, Counter] = defaultdict(Counter)
        for e in events:
            by_type[e["type"]][e["tag"]] += 1
        print(f"\n  {len(events)} events across {len(by_type)} classes:")
        for etype in (t.value for t in EventType):
            counts = by_type.get(etype)
            if not counts:
                continue
            print(f"  [{etype}]  {sum(counts.values())}")
            for tag, n in counts.most_common():
                print(f"    {tag:<24} ×{n}")
        print(f"\n  outcome: {outcome}")

    return {
        "outcome": outcome,
        "events":  events,
        "video":   np.stack(frames) if frames is not None else None,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Replay a Contra trace and print its events")
    parser.add_argument("npz",     help="Path to a trace .npz file")
    parser.add_argument("--video", default=None, help="Save an MP4 of the replay to this path")
    args = parser.parse_args()

    result = replay_actions(args.npz, want_video=bool(args.video), verbose=True)
    if args.video:
        save_video(result["video"], args.video)
        print(f"  MP4 saved → {args.video}")


if __name__ == "__main__":
    main()
