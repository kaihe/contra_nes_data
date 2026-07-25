"""env/ground.py — the interface for a level's ground/terrain.

Contra's engine keeps a live background-collision map of the two currently-loaded
screens in ``BG_COLLISION_DATA`` ($0680, 128 bytes): a 16×16-px grid where every
cell is one of four codes (empty / floor / water / solid). :func:`get_bg_collision`
is a faithful port of the game's own routine (bank7.asm), so "is there ground at
screen (x, y)?" is a RAM read + the same scroll arithmetic — no statistics.

Because the map only covers ~2 screens at a time, the whole-level terrain is
recovered by replaying one win trace and stamping every visible cell into a world
grid as the level scrolls past (:func:`build_ground`). The result is a
:class:`GroundMap` you can query by world pixel:

    from env.ground import get_ground
    g = get_ground("level1")          # built once, cached in-process
    g.surface_y(1200)                 # top of the walkable floor at world x=1200
    g.is_floor(1200, 108)             # standing-surface check
    list(g.pits())                    # (x0, x1) world-x spans with no floor

Downstream (progress-task) code should depend on this module, not on the raw
collision addresses.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

from env.entity import ADDR_XSCROLL

# ── Live collision probe (port of get_bg_collision, bank7.asm) ────────────────
ADDR_BG_COLLISION_DATA = 0x0680
ADDR_VSCROLL = 0xFC      # fine vertical scroll (mirror of PPU scroll)
ADDR_HSCROLL = 0xFD      # fine horizontal scroll (0-255 within the loaded screens)
ADDR_PPUCTRL = 0xFF      # bit0 selects the base nametable of the left screen

COLL_EMPTY, COLL_FLOOR, COLL_WATER, COLL_SOLID = 0x00, 0x01, 0x02, 0x80
COLL_NAMES = {COLL_EMPTY: "empty", COLL_FLOOR: "floor",
              COLL_WATER: "water", COLL_SOLID: "solid"}
_COLL_CODE_LUT = (0x00, 0x01, 0x02, 0x80)   # collision_code_lookup_tbl
_SCREEN_OFF_TBL = (0x00, 0x40)              # level_screen_mem_offset_tbl_01

CELL = 16                # native collision-cell size in pixels
PPU_W, PPU_H = 256, 240


def get_bg_collision(ram: np.ndarray, x: int, y: int) -> int:
    """Background collision code at *screen* pixel (x, y) — a port of the game's
    ``get_bg_collision`` (bank7.asm). Returns one of ``COLL_EMPTY`` / ``COLL_FLOOR``
    / ``COLL_WATER`` / ``COLL_SOLID``. ``x``/``y`` are on-screen coords (0-255); the
    routine folds in horizontal/vertical scroll and the nametable select itself."""
    vscroll = int(ram[ADDR_VSCROLL])
    hscroll = int(ram[ADDR_HSCROLL])
    ppuctrl = int(ram[ADDR_PPUCTRL])

    # world y = y + vscroll, with the game's >=0xf0 / overflow +0x10 fixup
    a = y + vscroll
    if a > 0xFF or (a & 0xFF) >= 0xF0:
        wy = ((a & 0xFF) + 0x10) & 0xFF
    else:
        wy = a & 0xFF

    ax = x + hscroll
    carry_x = 1 if ax > 0xFF else 0
    x12 = ax & 0xFF
    nt = (ppuctrl & 0x01) ^ carry_x                 # which of the two loaded screens

    off = ((wy >> 2) & 0x3C) | ((x12 >> 6) & 0x03) | _SCREEN_OFF_TBL[nt]
    col = (x12 >> 4) & 0x03                          # which of the 4 points in the byte

    if y >= 0xE0:                                    # past last collision row → empty
        return COLL_EMPTY
    byte = int(ram[ADDR_BG_COLLISION_DATA + off])
    code2 = (byte >> ((3 - col) * 2)) & 0x03
    return _COLL_CODE_LUT[code2]


def left_edge(ram: np.ndarray) -> int:
    """Absolute world x of screen column 0 (level scroll: screen<<8 | offset)."""
    return (int(ram[ADDR_XSCROLL]) << 8) | int(ram[ADDR_XSCROLL + 1])


# ── Whole-level terrain: GroundMap ────────────────────────────────────────────

_STANDABLE = (COLL_FLOOR, COLL_SOLID)   # surfaces the player can stand on


@dataclass(frozen=True)
class GroundMap:
    """A level's terrain as a ``codes`` grid at 16-px cell resolution.

    ``codes[row, col]`` is the collision code of the cell whose top-left world pixel
    is ``(x0 + col*cell, y0 + row*cell)``. All query methods take/return **world
    pixel** coordinates. y grows downward (screen convention).
    """

    level: str
    cell: int
    x0: int                 # world x (pixels) of column 0
    y0: int                 # world y (pixels) of row 0
    codes: np.ndarray       # (n_rows, n_cols) uint8 collision codes; 0 = empty

    # -- extent -------------------------------------------------------------
    @property
    def x_max(self) -> int:
        return self.x0 + self.codes.shape[1] * self.cell

    @property
    def y_max(self) -> int:
        return self.y0 + self.codes.shape[0] * self.cell

    def _col(self, wx: int) -> int:
        return (wx - self.x0) // self.cell

    def _row(self, wy: int) -> int:
        return (wy - self.y0) // self.cell

    # -- point queries ------------------------------------------------------
    def code_at(self, wx: int, wy: int) -> int:
        """Collision code at world pixel (wx, wy); ``COLL_EMPTY`` if out of range."""
        c, r = self._col(wx), self._row(wy)
        if 0 <= r < self.codes.shape[0] and 0 <= c < self.codes.shape[1]:
            return int(self.codes[r, c])
        return COLL_EMPTY

    def is_floor(self, wx: int, wy: int) -> bool:
        return self.code_at(wx, wy) == COLL_FLOOR

    def is_solid(self, wx: int, wy: int) -> bool:
        return self.code_at(wx, wy) == COLL_SOLID

    def is_water(self, wx: int, wy: int) -> bool:
        return self.code_at(wx, wy) == COLL_WATER

    def is_standable(self, wx: int, wy: int) -> bool:
        """True where the player can rest (floor or the top of a solid block)."""
        return self.code_at(wx, wy) in _STANDABLE

    # -- column queries -----------------------------------------------------
    def surface_y(self, wx: int) -> int | None:
        """World y (pixel, top of the cell) of the topmost standable surface at
        column ``wx`` — the height the player would stand at. ``None`` if the column
        has no floor/solid cell (a pit or off-map)."""
        c = self._col(wx)
        if not (0 <= c < self.codes.shape[1]):
            return None
        col = self.codes[:, c]
        rows = np.nonzero((col == COLL_FLOOR) | (col == COLL_SOLID))[0]
        return None if len(rows) == 0 else self.y0 + int(rows[0]) * self.cell

    def has_floor(self, wx: int) -> bool:
        """Whether column ``wx`` has any standable surface at all."""
        return self.surface_y(wx) is not None

    def pits(self):
        """Yield ``(x0, x1)`` world-x spans (inclusive-exclusive) where no column
        has a standable surface — pits / gaps the player must jump or fall through."""
        run_start = None
        for c in range(self.codes.shape[1]):
            wx = self.x0 + c * self.cell
            empty = self.surface_y(wx) is None
            if empty and run_start is None:
                run_start = wx
            elif not empty and run_start is not None:
                yield (run_start, wx)
                run_start = None
        if run_start is not None:
            yield (run_start, self.x_max)

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, level=self.level, cell=self.cell,
                            x0=self.x0, y0=self.y0, codes=self.codes)

    @classmethod
    def load(cls, path: str) -> "GroundMap":
        d = np.load(path, allow_pickle=False)
        return cls(level=str(d["level"]), cell=int(d["cell"]),
                   x0=int(d["x0"]), y0=int(d["y0"]), codes=d["codes"])


# ── Walkable line segments (for traversal anchors) ────────────────────────────

@dataclass(frozen=True)
class GroundSegment:
    """A maximal horizontal run of *standable surface* — one walkable line segment
    (a platform / ground stretch) at a single height. Bounding box in world pixels
    is ``[x0, x1) × [y, y+cell)``; ``y`` is the surface the player stands on."""

    sid: int
    row: int                # cell row of the surface
    c0: int                 # first cell column (inclusive)
    c1: int                 # last cell column (inclusive)
    cell: int
    x0: int                 # world x of the left edge (pixels)
    y: int                  # surface world y (pixels, top of the surface cell)

    @property
    def x1(self) -> int:
        return self.x0 + (self.c1 - self.c0 + 1) * self.cell

    @property
    def length(self) -> int:
        return self.x1 - self.x0

    @property
    def cx(self) -> int:
        return (self.x0 + self.x1) // 2


class SurfaceIndex:
    """The level's walkable surfaces, segmented into :class:`GroundSegment` runs,
    plus a fast "which segment is the player standing on?" lookup.

    A *surface* cell is a standable cell (floor or solid) with open space directly
    above it — i.e. a place the player can actually rest. Adjacent surface cells on
    the same row form one segment.
    """

    def __init__(self, g: GroundMap, *, max_run: int | None = 8):
        """``max_run`` caps a segment's width in cells so long flat stretches get
        several evenly-spaced anchors instead of one (a single 1400-px ground would
        otherwise yield just one landmark). ``None`` disables the split."""
        self.g = g
        self.max_run = max_run
        codes = g.codes
        standable = (codes == COLL_FLOOR) | (codes == COLL_SOLID)
        open_above = np.ones_like(standable)
        open_above[1:, :] = ~standable[:-1, :]        # cell above is not standable
        surface = standable & open_above

        self.labels = np.full(codes.shape, -1, dtype=np.int32)   # (row, col) -> sid
        self.segments: list[GroundSegment] = []
        nrows, ncols = codes.shape
        step = max_run if max_run else ncols
        for r in range(nrows):
            c = 0
            while c < ncols:
                if not surface[r, c]:
                    c += 1
                    continue
                c0 = c
                while c < ncols and surface[r, c]:
                    c += 1
                # split the contiguous run [c0, c) into <=max_run-wide segments
                for sc in range(c0, c, step):
                    ec = min(sc + step, c)            # [sc, ec)
                    self.labels[r, sc:ec] = len(self.segments)
                    self.segments.append(GroundSegment(
                        sid=len(self.segments), row=r, c0=sc, c1=ec - 1, cell=g.cell,
                        x0=g.x0 + sc * g.cell, y=g.y0 + r * g.cell))

    def segment_at(self, wx: int, feet_y: int, tol_rows: int = 1) -> int | None:
        """Segment id the player standing at world (wx, feet_y) is on, or ``None``.

        ``feet_y`` is the player's foot pixel (sprite y + a small offset). Matches
        the standable surface in that column nearest the feet within ``tol_rows``.
        """
        g = self.g
        c = (wx - g.x0) // g.cell
        if not (0 <= c < g.codes.shape[1]):
            return None
        target = (feet_y - g.y0) // g.cell
        best, best_d = None, None
        for dr in range(-tol_rows, tol_rows + 1):
            r = target + dr
            if 0 <= r < g.codes.shape[0] and self.labels[r, c] >= 0:
                if best_d is None or abs(dr) < best_d:
                    best, best_d = int(self.labels[r, c]), abs(dr)
        return best


_SI_CACHE: dict[tuple[int, int | None], SurfaceIndex] = {}


def surface_index(g: GroundMap, *, max_run: int | None = 8) -> SurfaceIndex:
    """A :class:`SurfaceIndex` for ``g`` (memoised per GroundMap object + max_run)."""
    k = (id(g), max_run)
    if k not in _SI_CACHE:
        _SI_CACHE[k] = SurfaceIndex(g, max_run=max_run)
    return _SI_CACHE[k]


# ── Building a GroundMap from traces ──────────────────────────────────────────

def find_win_trace(level: str, root: str = "game_trace/mc_trace") -> list[str]:
    """All win traces for ``level`` (e.g. ``win_level1_*.npz``), sorted."""
    return sorted(glob.glob(os.path.join(root, "**", f"win_{level}_*.npz"),
                            recursive=True))


def _sample_frame(ram: np.ndarray, terrain: dict) -> None:
    """Stamp every non-empty collision cell of the two visible screens into
    ``terrain`` (keyed by world cell), keeping the first code seen per cell."""
    left = left_edge(ram)
    first_cell, last_cell = left // CELL, (left + PPU_W - 1) // CELL
    for wc in range(first_cell, last_cell + 1):
        sx = wc * CELL - left + CELL // 2            # cell centre in screen x
        if not (0 <= sx < PPU_W):
            continue
        for row in range(0, PPU_H, CELL):            # y has no scroll on L1
            code = get_bg_collision(ram, sx, row + CELL // 2)
            if code != COLL_EMPTY:
                terrain.setdefault((wc, row // CELL), code)


def build_ground(level: str = "level1", *, trace: str | None = None,
                 max_traces: int = 1) -> GroundMap:
    """Replay win trace(s) for ``level`` and reconstruct its :class:`GroundMap`.

    One trace usually suffices — the collision buffer holds the current + next
    screen, so a full playthrough exposes every cell. ``trace`` pins a specific
    file; otherwise the first ``max_traces`` win traces are merged.
    """
    from util.replay import make_env, rewind_state

    files = [trace] if trace else find_win_trace(level)[:max_traces]
    if not files:
        raise FileNotFoundError(f"no win traces for {level!r} under game_trace/mc_trace")

    terrain: dict[tuple[int, int], int] = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        actions, skip = d["actions"], int(d["skip"])
        env = make_env()
        rewind_state(env, bytes(d["initial_state"]))
        for act in actions:
            a = np.asarray(act, dtype=np.uint8)
            for _ in range(skip):
                env.step(a)
            _sample_frame(env.unwrapped.get_ram(), terrain)
        env.close()

    xs = [c[0] for c in terrain]
    ys = [c[1] for c in terrain]
    x0c, x1c = min(xs), max(xs)
    y0c, y1c = min(ys), max(ys)
    codes = np.zeros((y1c - y0c + 1, x1c - x0c + 1), dtype=np.uint8)
    for (wc, rc), code in terrain.items():
        codes[rc - y0c, wc - x0c] = code
    return GroundMap(level=level, cell=CELL,
                     x0=x0c * CELL, y0=y0c * CELL, codes=codes)


# in-process cache so repeated get_ground(level) calls don't re-replay
_CACHE: dict[str, GroundMap] = {}


def get_ground(level: str = "level1", *, cache: str | None = None,
               rebuild: bool = False) -> GroundMap:
    """Return the :class:`GroundMap` for ``level``, building it on first use.

    Memoised in-process. Pass ``cache=PATH`` to also persist to / load from an
    ``.npz`` on disk (skips the replay entirely on later runs); ``rebuild=True``
    forces a fresh build even if a cache exists.
    """
    if not rebuild and level in _CACHE:
        return _CACHE[level]
    if cache and not rebuild and os.path.exists(cache):
        g = GroundMap.load(cache)
    else:
        g = build_ground(level)
        if cache:
            g.save(cache)
    _CACHE[level] = g
    return g
