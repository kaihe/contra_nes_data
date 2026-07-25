"""GUI task-segment inspector — replay a task .npz and step through it action-by-action.

Shows, for every step: the game frame, the action that produced it (action-space
name + decoded buttons), and the events that fired. The right panel lists the other
.npz files in the same directory so you can page through a whole task folder with
Up / Down without relaunching.

Usage:
    python -m util.gui_task_inspect <path/to/segment.npz>

Works on task segments (task_maker output) and on raw traces alike.

A few no-op steps are appended after the real actions (visualization only, the
``.npz`` is never modified) so the final kill's death/explosion animation — which
plays out on frames *after* hp reaches 0 — is visible instead of being cut off.

Controls:
    Left / Right arrow  : step -/+ 1 frame  (hold to scrub)
    Up / Down arrow     : previous / next .npz file in the folder
    Space               : play / pause
    Q / Escape          : quit

Each file is replayed on first view (and cached); the emulator is closed before the
GUI draws, since stable-retro allows only one emulator instance per process.
"""

import glob
import os
import sys
import time
import warnings

import numpy as np
import pygame

warnings.filterwarnings("ignore", message=".*Gym.*")

from agent.action import DEFAULT as ACTION_SPACE
from env.constant import (
    ENEMY_TYPE_NAMES_BY_LEVEL,
    ENEMY_TYPE_NAMES_COMMON,
    ITEM_WEAPON_NAMES,
)
from env.event import all_events
from util.replay import make_env, rewind_state

SCALE         = 3     # upscale 256×240 NES frame
PANEL_HEIGHT  = 108   # bottom strip: action / events / timeline
STATE_PANEL_W = 340   # right panel: file browser
PAD_STEPS     = 8     # no-op steps appended after the real actions (visualization only)

NOOP = np.zeros(9, dtype=np.uint8)   # idle action for the visualization tail

_C_SECTION = (255, 215, 60)
_C_TBLHDR  = (100, 100, 100)
_C_DATA    = (200, 200, 200)
_C_SELECT  = (60, 255, 80)    # green — the currently loaded file
_C_EVENT   = (255, 140, 60)   # orange — fired events
_C_ACTION  = (120, 200, 255)  # blue — the action

EVENTS = all_events()

# vector (tuple) -> action-space short name (e.g. (…) -> "RF")
_ACTION_NAME = {tuple(int(x) for x in v): n
                for n, v in zip(ACTION_SPACE.names, ACTION_SPACE.actions)}

# button index (bit order [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A])
_BUTTONS = [(4, "Up"), (5, "Down"), (6, "Left"), (7, "Right"),
            (0, "Fire"), (8, "Jump")]


def _action_str(vec) -> str:
    parts = [name for i, name in _BUTTONS if vec[i]]
    combo = "+".join(parts) if parts else "idle"
    short = _ACTION_NAME.get(tuple(int(x) for x in vec), "?")
    return f"{combo}   [{short}]"


def _enemy_name(etype: int, level: int) -> str:
    table = ENEMY_TYPE_NAMES_COMMON if etype <= 0x0F else ENEMY_TYPE_NAMES_BY_LEVEL.get(level, {})
    return table.get(etype, f"0x{etype:02x}")


def _meta_level(v) -> int:
    """Coerce the npz ``level`` field to a 0-indexed int.

    Task segments store it 0-indexed already; raw mc_search traces store the
    retro state name instead (e.g. "Level1"), which is 1-indexed.
    """
    if isinstance(v, str):
        digits = "".join(ch for ch in v if ch.isdigit())
        return int(digits) - 1 if digits else 0
    return int(v)


def _target_desc(meta: dict, level: int) -> str:
    """One-line description of what the task is anchored on (for the bottom HUD)."""
    if "enemy_type" in meta:
        d = f"target: {_enemy_name(int(meta['enemy_type']), level)}"
        if "slot" in meta:
            d += f"  slot {int(meta['slot'])}"
        return d
    if "item_weapon" in meta:
        return f"item: {ITEM_WEAPON_NAMES.get(int(meta['item_weapon']), '?')}"
    return ""


# ── Replay ─────────────────────────────────────────────────────────────────────

def _replay(npz_path: str, pad: int = PAD_STEPS):
    """Replay a segment npz.

    Returns ``(frames, rams, actions, meta, n_real)`` where ``n_real`` is the
    number of real decision steps; any frames beyond that are the no-op
    visualization tail so the last kill's animation finishes on screen.
    """
    ckpt    = np.load(npz_path, allow_pickle=True)
    actions = ckpt["actions"]
    initial = bytes(ckpt["initial_state"])
    skip    = int(ckpt["skip"]) if "skip" in ckpt else ACTION_SPACE.skip

    meta = {k: (ckpt[k].item() if ckpt[k].ndim == 0 else ckpt[k])
            for k in ckpt.files if k not in ("actions", "initial_state")}
    n = len(actions)
    print(f"Replaying {n} steps (+{pad} no-op tail)  label={meta.get('label', '?')}  skip={skip}")

    env = make_env()
    rewind_state(env, initial)
    frames = [env.em.get_screen().copy()]
    rams   = [env.unwrapped.get_ram().copy()]

    # One entry per decision step, snapshotted at the END of the step (after all
    # `skip` frames) so rams[i-1] -> rams[i] is exactly the pre/cur pair events
    # (and the extractor) are defined on. The trailing no-op steps are appended
    # the same way but never written back — they only let the death animation run.
    for i, act in enumerate(list(actions) + [NOOP] * pad):
        if i % 200 == 0:
            print(f"  {i}/{n}\r", end="", flush=True)
        a = np.asarray(act, dtype=np.uint8)
        for _ in range(skip):
            env.step(a.copy())
        frames.append(env.em.get_screen().copy())
        rams.append(env.unwrapped.get_ram().copy())

    env.close()
    print(f"  {n}/{n}  done")
    return frames, rams, np.asarray(actions, dtype=np.uint8), meta, n


# ── Drawing ─────────────────────────────────────────────────────────────────────

def _draw_file_list(screen, font, files, cur_idx, x, w, max_y):
    """Right panel: the .npz files in the folder, current one highlighted."""
    line_h, pad = 16, 8
    screen.blit(font.render(f"Files  {cur_idx + 1}/{len(files)}", True, _C_SECTION),
                (x + pad, pad))
    folder = os.path.basename(os.path.dirname(files[cur_idx])) or "."
    screen.blit(font.render(f"{folder}/", True, _C_TBLHDR), (x + pad, pad + line_h))

    top = pad + line_h * 2 + 4
    rows = max(1, (max_y - top - pad) // line_h)
    start = 0
    if len(files) > rows:                              # scroll to keep cur_idx centred
        start = min(max(cur_idx - rows // 2, 0), len(files) - rows)
    max_chars = (w - 2 * pad) // 7

    y = top
    for i in range(start, min(start + rows, len(files))):
        name = os.path.basename(files[i])
        if len(name) > max_chars:
            name = name[:max_chars - 1] + "…"
        if i == cur_idx:
            pygame.draw.rect(screen, (40, 40, 40), pygame.Rect(x + 4, y - 1, w - 8, line_h))
            color, marker = _C_SELECT, "►"
        else:
            color, marker = _C_DATA, " "
        screen.blit(font.render(f"{marker}{name}", True, color), (x + pad, y))
        y += line_h

    if start > 0:
        screen.blit(font.render("▲ more", True, _C_TBLHDR), (x + w - 60, top - line_h + 2))
    if start + rows < len(files):
        screen.blit(font.render("▼ more", True, _C_TBLHDR), (x + w - 60, y + 2))


# ── Main GUI ───────────────────────────────────────────────────────────────────

def main(npz_path: str) -> None:
    npz_path = os.path.abspath(npz_path)
    folder = os.path.dirname(npz_path)
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if npz_path not in files:                          # tolerate an odd argument path
        files = sorted(set(files) | {npz_path})
    file_idx = files.index(npz_path)

    cache: dict[str, tuple] = {}

    def get_replay(path):
        if path not in cache:
            cache[path] = _replay(path)
        return cache[path]

    # Display geometry is constant across NES frames (always 224×240), so the
    # window size is fixed once.
    frames0 = get_replay(files[file_idx])[0]
    h, w = frames0[0].shape[:2]
    disp_w, disp_h = w * SCALE, h * SCALE
    total_w = disp_w + STATE_PANEL_W

    pygame.init()
    screen = pygame.display.set_mode((total_w, disp_h + PANEL_HEIGHT))
    clock = pygame.time.Clock()
    font_hud   = pygame.font.SysFont("monospace", 14)
    font_big   = pygame.font.SysFont("monospace", 18, bold=True)
    font_list  = pygame.font.SysFont("monospace", 12)

    # Per-file state, (re)populated by load().
    frames = rams = actions = meta = None
    n_real = total = fps = 0
    label = target_desc = ""
    level = start_step = 0
    target_slot = None
    current, paused, scrubbing = 0, True, False
    cached_idx, cached_surf = -1, None

    def load(idx):
        nonlocal frames, rams, actions, meta, n_real, total, fps
        nonlocal label, level, start_step, target_slot, target_desc
        nonlocal current, paused, cached_idx
        frames, rams, actions, meta, n_real = get_replay(files[idx])
        label = str(meta.get("label", "?"))
        level = _meta_level(meta.get("level", 0))
        start_step = int(meta.get("start_step", 0))
        target_slot = int(meta["slot"]) if "slot" in meta else None
        target_desc = _target_desc(meta, level)
        total = len(frames)
        fps = round(60 / (int(meta["skip"]) if "skip" in meta else ACTION_SPACE.skip))
        current, paused, cached_idx = 0, True, -1
        pygame.display.set_caption(
            f"Task Inspector — {os.path.basename(files[idx])}  [{label}]")

    def get_surf(idx):
        nonlocal cached_idx, cached_surf
        if idx != cached_idx:
            surf = pygame.surfarray.make_surface(frames[idx].swapaxes(0, 1))
            cached_surf = pygame.transform.scale(surf, (disp_w, disp_h))
            cached_idx = idx
        return cached_surf

    def bar_rect():
        return pygame.Rect(8, disp_h + PANEL_HEIGHT - 12, disp_w - 16, 7)

    def frame_from_x(mx):
        r = bar_rect()
        return int(max(0.0, min(1.0, (mx - r.x) / r.width)) * (total - 1))

    load(file_idx)
    held: dict[int, float] = {}
    HOLD_DELAY = 0.25

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and bar_rect().collidepoint(ev.pos):
                scrubbing, current, paused = True, frame_from_x(ev.pos[0]), True
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                scrubbing = False
            elif ev.type == pygame.MOUSEMOTION and scrubbing:
                current = frame_from_x(ev.pos[0])
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_RIGHT:
                    current, paused = min(current + 1, total - 1), True
                    held[pygame.K_RIGHT] = time.time()
                elif ev.key == pygame.K_LEFT:
                    current, paused = max(current - 1, 0), True
                    held[pygame.K_LEFT] = time.time()
                elif ev.key == pygame.K_DOWN and file_idx < len(files) - 1:
                    file_idx += 1
                    load(file_idx)
                elif ev.key == pygame.K_UP and file_idx > 0:
                    file_idx -= 1
                    load(file_idx)
            elif ev.type == pygame.KEYUP:
                held.pop(ev.key, None)

        now = time.time()
        for key, delta in ((pygame.K_RIGHT, 1), (pygame.K_LEFT, -1)):
            if key in held and now - held[key] > HOLD_DELAY:
                current = max(0, min(current + delta, total - 1))

        if not paused:
            current = min(current + 1, total - 1)
            if current == total - 1:
                paused = True

        # video + right-hand file browser
        screen.blit(get_surf(current), (0, 0))
        pygame.draw.rect(screen, (15, 15, 15), pygame.Rect(disp_w, 0, STATE_PANEL_W, disp_h))
        pygame.draw.line(screen, (50, 50, 50), (disp_w, 0), (disp_w, disp_h))
        _draw_file_list(screen, font_list, files, file_idx, disp_w, STATE_PANEL_W, disp_h)

        # bottom panel
        pygame.draw.rect(screen, (20, 20, 20), pygame.Rect(0, disp_h, total_w, PANEL_HEIGHT))

        # the action that produced this frame (actions[current-1]); step 0 = initial.
        # frames past n_real are the no-op visualization tail.
        is_pad = current > n_real
        if current == 0:
            act_line, fired = "action:  (initial state)", []
        elif is_pad:
            act_line = "action:  (no-op · animation tail)"
            fired = [e.tag for e in EVENTS if e.trigger(rams[current - 1], rams[current])]
        else:
            act_line = f"action:  {_action_str(actions[current - 1])}"
            fired = [e.tag for e in EVENTS if e.trigger(rams[current - 1], rams[current])]

        gstep = start_step + min(max(current - 1, 0), n_real - 1)
        step_lbl = (f"step {current}/{total - 1}"
                    + (f"  [pad +{current - n_real}]" if is_pad
                       else f"   (trace step {gstep})"))
        screen.blit(font_hud.render(
            f"{step_lbl}   {'PAUSED' if paused else 'PLAYING'}", True, (150, 150, 150)),
            (8, disp_h + 8))
        screen.blit(font_big.render(act_line, True, _C_ACTION), (8, disp_h + 28))
        ev_str = ("events:  " + ", ".join(fired)) if fired else ""
        screen.blit(font_hud.render(ev_str, True, _C_EVENT), (8, disp_h + 54))
        if target_desc:
            screen.blit(font_hud.render(target_desc, True, _C_SELECT),
                        (disp_w - 320, disp_h + 8))

        # timeline: grey track, dim tail for the pad region, white tick at the
        # real end (kill), red playhead
        r = bar_rect()
        pygame.draw.rect(screen, (50, 50, 50), r)
        if total > 1 and n_real < total - 1:
            ex = int(n_real / (total - 1) * r.width) + r.x
            pygame.draw.rect(screen, (35, 35, 35), pygame.Rect(ex, r.y, r.right - ex, r.height))
            pygame.draw.line(screen, (200, 200, 200), (ex, r.y - 2), (ex, r.y + 9), 1)
        cx = (int(current / (total - 1) * r.width) + r.x) if total > 1 else r.x
        pygame.draw.line(screen, (255, 80, 80), (cx, r.y - 2), (cx, r.y + 9), 2)

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m util.gui_task_inspect <segment.npz>")
        sys.exit(1)
    main(sys.argv[1])
