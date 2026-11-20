"""Maze panel: parses gameworld (Pac-Man style) frames into compact text.

The panel is shown to the inner LLM in every LOG and PLAN prompt (via the
{map_analysis} slot).  It provides, derived from the CURRENT frame only:

  - an ASCII tile map of the maze (#=wall .=corridor o=pellet O=power
    @=you, B/P/I/C=ghosts, F=frightened)
  - your corner position, the open directions, and a note on whether the
    previous action made progress
  - pellet / power / ghost counts, food remaining vs the full board, and
    which ghosts are still in the house
  - an estimated score from food eaten this run, plus the best run so far
  - the nearest pellet with the first step to take (BFS over the maze)
  - the nearest reachable ghost
  - a READY flag (you just died, score preserved) and a stall warning

All geometry (frame border, cell size, wall grid) is detected from the
episode's own frames at the first step and re-verified against a wall-grid
signature every frame (with a tolerance so moving sprites do not trigger
rebuilds), so a new level is handled automatically.  Palette thresholds
target the standard arcade look (red walls, gray pellets, white power
pellets, yellow player, classic ghost colours); they are properties of the
rendering, not of any particular layout or episode.

Detection (all from the current frame's own pixels):

  - geometry: the red frame border gives the outer box; the border thickness
    is scanned from the top/bottom edges; the cell size is the grid that
    maximises per-cell wall-fraction bimodality (walls are solid cell
    blocks, so the true grid has cells that are almost all-wall or almost
    all-corridor).
  - walls: a tile is wall if >half its pixels are red/lavender.
  - pellets: small gray components (un-eroded; erosion destroys the 6x5
    dots).  Power pellets: white components that are not wall (the lavender
    no-pellet zones are white-ish and wall, so they are excluded).
  - player: the largest yellow component (eroded bright non-wall).
  - ghosts: all ghost-coloured components (eroded bright non-wall), plus the
    red Blinky, which is excluded from the bright pass because red is in the
    wall mask.  Blinky is found as a medium solid red component.
  - READY text: red letter components in the central band (the player's red
    head deco is excluded by proximity).

Pellets and power pellets are static objects that only disappear, so the
current food sets are maintained remove-only against the board sets
captured at level start / after each death; this makes the counts immune
to anti-aliasing flicker and to the power-pellet blink.

Failure is graceful: any parse problem returns "" and the agent behaves
like the parent.  The state dict is plain JSON-serialisable data (lists,
strings, ints, bools, None, str-keyed dicts) so it survives the host
checkpoint round trip.  Numpy is used only for ephemeral per-frame
computation.
"""
from collections import deque

import numpy as np
from scipy import ndimage

# ---- palette / geometry tuning (standard arcade maze, not a layout) ----
BORDER_FRAC = 0.4                # border row/col must exceed this red fraction
BIMOD_MIN = 0.25                 # min wall-fraction bimodality for a valid grid
PELLET_S = (10, 40)              # pixel area of an un-eroded normal pellet
PELLET_BOX = 10
PELLET_MIN_FILL = 0.35           # real pellet comps fill ~0.40; slivers are lower
POWER_S = (110, 250)             # pixel area of an un-eroded power pellet
POWER_BOX = (10, 20)
PLAYER_MIN_S = 150
PLAYER_BOX = 60
GHOST_MIN_S = 200
GHOST_BOX = (20, 60)
GHOST_MIN_FILL = 0.45            # solid ghosts; thin ring arcs are lower
GHOST_MAX_COUNT = 4
GHOST_DEDUPE_PX = 16             # two ghosts closer than this are one
BLINKY_BOX = (28, 38)
BLINKY_S = (400, 900)
BLINKY_MAX_FILL = 0.9
LETTER_S = (40, 300)
LETTER_BOX = 20
LETTER_MAX_FILL = 0.92           # solid wall strips fill ~1.0, letters less
READY_MIN_LETTERS = 4
READY_Y_TOL = 10
READY_X_SPREAD = 40
READY_CENTER_X = 0.4             # max |dx| from maze centre, in maze widths
READY_CENTER_Y = 0.3
PLAYER_LETTER_DIST = 22          # red comps this close to you are not letters
REMOVE_STREAK = 3                # unoccluded absent frames before removal
POWER_REMOVE_STREAK = 8          # power pellets blink; longer memory
BOARD_SEEN_NEEDED = 2            # frames a tile must be seen to join the board
STALL_WARN = 8                   # steps without an eaten pellet -> warning
TRANSITION_TIMEOUT = 16          # forced exit: max pause frames (true pauses are 8-13)
RESUME_MOVE_MIN = 3              # corner units moved in one frame that count as resuming play
POCKET_MAX_TILES = 60            # enclosed no-pellet corridor = house-like
WALL_FRAC = 0.5                  # tile is wall if >half its pixels are wall
WALL_REBUILD_FRAC = 0.02         # wall-grid diff fraction that forces rebuild
WALL_REBUILD_MIN = 8
PELLET_PTS = 10                  # classic Pac-Man pellet score
POWER_PTS = 50                   # classic Pac-Man power pellet score
RESTART_JUMP = 15                # food-count jump signalling a full board refill


def new_state():
    return {
        "initialized": False,
        "border": None,
        "cell": None,
        "ix0": None, "iy0": None, "ix1": None, "iy1": None,
        "n_cols": None, "n_rows": None,
        "wall": None,
        "house": None,
        "deaths": [],
        "last_ready_step": None,
        "ready": False,
        "start_pc": None,
        "start_pw": None,
        "est_score": None,
        "best_est": 0,
        "ghosts": [],
        "stall": 0,
        "last_pc": None,
        "p": None,
        "p_stale": 0,
        "p_tile": None,
        "board_pellets": [],
        "pellets": [],
        "p_absent": {},
        "board_power": [],
        "power": [],
        "power_absent": {},
        "board_counts": {},
        "max_ghosts": 0,
        "pellet_anchor": "center",
        "last_move_note": "",
        "last_step": -1,
        "ghost0_run": 0,
        "notready_run": 0,
        "in_transition": False,
        "trans_frames": 0,
        "ready_seen_trans": False,
        "nav_target": None,
        "nav_hint": None,
        "last_dir": None,
        "lost_ghosts": {},
        "restart_run": 0,
    }


# ---------------------------------------------------------------- masks ----
def _masks_full(a):
    r = a[..., 0].astype(np.int16)
    g = a[..., 1].astype(np.int16)
    b = a[..., 2].astype(np.int16)
    red = (r > 150) & (g < 120) & (b < 120)
    lav = (r > 230) & (g > 180) & (b > 230) & ((r - g) > 12) & ((b - g) > 12)
    wall = red | lav
    bright = (a.astype(np.int32).sum(axis=2) > 60)
    return red, wall, bright


def _masks_sub(sub):
    r = sub[..., 0].astype(np.int16)
    g = sub[..., 1].astype(np.int16)
    b = sub[..., 2].astype(np.int16)
    red = (r > 150) & (g < 120) & (b < 120)
    lav = (r > 230) & (g > 180) & (b > 230) & ((r - g) > 12) & ((b - g) > 12)
    wall = red | lav
    bright = (sub.astype(np.int32).sum(axis=2) > 60)
    gray = (np.abs(r - g) < 15) & (np.abs(g - b) < 15) & (r > 45) & (r < 185)
    white = (r > 200) & (g > 200) & (b > 200)
    return red, wall, bright, gray, white


def _components(mask, a):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if not n:
        return
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    sl = ndimage.find_objects(lab)
    means = [ndimage.mean(a[..., ch], lab, range(1, n + 1)) for ch in range(3)]
    for i in range(n):
        y0, x0 = sl[i][0].start, sl[i][1].start
        w = sl[i][1].stop - x0
        h = sl[i][0].stop - y0
        yield (sizes[i], x0, y0, w, h,
               (float(means[0][i]), float(means[1][i]), float(means[2][i])))


# ------------------------------------------------------------ structure ----
def detect_border(a):
    r = a[..., 0].astype(np.int16)
    g = a[..., 1].astype(np.int16)
    b = a[..., 2].astype(np.int16)
    red = (r > 150) & (g < 120) & (b < 120)
    rows = np.where(red.sum(axis=1) > BORDER_FRAC * red.sum(axis=1).max())[0]
    cols = np.where(red.sum(axis=0) > BORDER_FRAC * red.sum(axis=0).max())[0]
    if len(rows) < 2 or len(cols) < 2:
        return None
    return [int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())]


def _frame_t(wall, x0, x1, y0, y1):
    """Border thickness: distance from the outer edge to the last wall pixel,
    scanned inward (16px window) from the top and bottom at the centre
    column.  Returns the mean of the two."""
    runs = []
    x = x0 + (x1 - x0) // 2
    for (y_start, y_end, step) in [(y0, y0 + 16, 1), (y1, y1 - 16, -1)]:
        idx = -1
        for y in range(y_start, y_end, step):
            if 0 <= y < wall.shape[0] and wall[y, x]:
                idx = y
        runs.append(abs(idx - y_start) if idx >= 0 else 0)
    return int(round(sum(runs) / len(runs)))


def _detect_grid(wall, x0, x1, y0, y1, t0):
    """Pick (cell, thickness, n_cols, n_rows) by maximising per-cell
    wall-fraction bimodality.  Walls are solid cell blocks, so the true grid
    has cells that are almost all-wall or almost all-corridor (|frac-0.5|
    high); a wrong grid straddles boundaries (frac~0.5, low score)."""
    best = None
    for c in range(10, 31):
        for dt in range(-2, 3):
            tt = t0 + dt
            if tt < 2 or tt > 24:
                continue
            wi = x1 - x0 - 2 * tt + 1
            hi = y1 - y0 - 2 * tt + 1
            if wi <= 0 or hi <= 0 or wi % c or hi % c:
                continue
            nc = wi // c
            nr = hi // c
            if nc < 12 or nr < 12:
                continue
            sub = wall[y0 + tt:y0 + tt + hi, x0 + tt:x0 + tt + wi]
            fr = sub.reshape(nr, c, nc, c).mean(axis=(1, 3))
            score = float(np.abs(fr - 0.5).mean())
            if score < BIMOD_MIN:
                continue
            if best is None or score > best[0]:
                best = (score, c, tt, nc, nr)
    return best


def _wall_grid(a, wall, ix0, iy0, nc, nr, c):
    sub = wall[iy0:iy0 + nr * c, ix0:ix0 + nc * c]
    return sub.reshape(nr, c, nc, c).mean(axis=(1, 3)) > WALL_FRAC


def _detect_anchor(comps, c):
    """Sprite anchor style (tile corner vs tile centre), decided per
    episode from the detection centres' offsets modulo the cell size."""
    if not comps:
        return "center"
    mods = sorted(cx % c for cx, _cy in comps)
    med = mods[len(mods) // 2]
    return "corner" if (med < c / 4.0 or med > 3.0 * c / 4.0) else "center"


def _tile_assign(cx, cy, c, anchor):
    if anchor == "corner":
        return int(round(cx / c)), int(round(cy / c))
    return int(cx // c), int(cy // c)


def _pellet_tiles(gray, sub, c, nc, nr):
    comps = []
    for s, x0, y0, w, h, _rgb in _components(gray, sub):
        if not (PELLET_S[0] <= s <= PELLET_S[1]):
            continue
        if not (4 <= w <= PELLET_BOX and 4 <= h <= PELLET_BOX):
            continue
        if s / float(w * h) < PELLET_MIN_FILL:
            continue
        comps.append((x0 + w / 2.0, y0 + h / 2.0))
    anchor = _detect_anchor(comps, c)
    pt = np.zeros((nr, nc), bool)
    for cx, cy in comps:
        k, j = _tile_assign(cx, cy, c, anchor)
        if 0 <= k < nc and 0 <= j < nr:
            pt[j, k] = True
    return pt, anchor


def _mark_pockets(W, pt):
    """Enclosed corridor regions without pellets (e.g. the ghost house)
    are not traversable; mark them as walls.  Returns (W, pockets)."""
    nr, nc = W.shape
    seen = np.zeros_like(W, bool)
    pockets = []
    for j in range(nr):
        for k in range(nc):
            if W[j, k] or seen[j, k]:
                continue
            comp = []
            q = deque([(j, k)])
            seen[j, k] = True
            while q:
                jj, kk = q.popleft()
                comp.append((jj, kk))
                for dj, dk in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nj, nk = jj + dj, kk + dk
                    if 0 <= nj < nr and 0 <= nk < nc and not W[nj, nk] and not seen[nj, nk]:
                        seen[nj, nk] = True
                        q.append((nj, nk))
            if len(comp) <= POCKET_MAX_TILES and not pt[comp].any():
                for jj, kk in comp:
                    W[jj, kk] = True
                pockets.append(comp)
    return W, pockets


def _house_bbox(pockets):
    if not pockets:
        return None
    p = max(pockets, key=len)
    js = [j for j, _k in p]
    ks = [k for _j, k in p]
    return [min(js), min(ks), max(js), max(ks)]


def _reset_board(st, seen_p, seen_w):
    """(Re)capture the full-board food sets from a fresh frame.  Called only
    when the maze structure is (re)built, i.e. a new level layout."""
    st["board_pellets"] = sorted([list(t) for t in seen_p])
    st["pellets"] = sorted([list(t) for t in seen_p])
    st["p_absent"] = {}
    st["board_power"] = sorted([list(t) for t in seen_w])
    st["power"] = sorted([list(t) for t in seen_w])
    st["power_absent"] = {}
    st["board_counts"] = {}
    st["max_ghosts"] = 0


def _build_structure(st, a):
    _red, wall, _bright = _masks_full(a)
    bd = detect_border(a)
    if bd is None:
        return False
    x0, x1, y0, y1 = bd
    t0 = _frame_t(wall, x0, x1, y0, y1)
    grid = _detect_grid(wall, x0, x1, y0, y1, t0)
    if grid is None:
        return False
    _score, c, t, nc, nr = grid
    ix0, iy0 = x0 + t, y0 + t
    ix1, iy1 = x1 - t, y1 - t
    if nc < 5 or nr < 5:
        return False
    W = _wall_grid(a, wall, ix0, iy0, nc, nr, c)
    sub = a[iy0:iy0 + nr * c, ix0:ix0 + nc * c]
    _r, _w, _b, gray, _white = _masks_sub(sub)
    pt, anchor = _pellet_tiles(gray, sub, c, nc, nr)
    W, pockets = _mark_pockets(W, pt)
    st.update({
        "initialized": True,
        "border": bd,
        "cell": int(c),
        "ix0": int(ix0), "iy0": int(iy0), "ix1": int(ix1), "iy1": int(iy1),
        "n_cols": int(nc), "n_rows": int(nr),
        "wall": [[int(v) for v in row] for row in W],
        "house": _house_bbox(pockets),
        "pellet_anchor": anchor,
    })
    # capture the board food sets from this frame (un-eroded gray/white)
    seen_p, seen_w = _food_tiles(sub, c, anchor)
    _reset_board(st, seen_p, seen_w)
    st.update({
        "p": None, "p_stale": 0, "p_tile": None,
        "stall": 0, "last_pc": None,
        "ready": False, "last_ready_step": None,
        "last_move_note": "",
        "ghost0_run": 0, "notready_run": 0, "in_transition": False,
        "trans_frames": 0, "ready_seen_trans": False,
        "nav_target": None, "nav_hint": None, "last_dir": None, "lost_ghosts": {},
        "restart_run": 0,
    })
    return True


# ------------------------------------------------------- classification ----
def _is_player(s, w, h, R, G, B):
    return (s >= PLAYER_MIN_S and w <= PLAYER_BOX and h <= PLAYER_BOX
            and R > 180 and G > 130 and B < 100 and (R - G) < 70)


def _ghost_kind(R, G, B):
    if R > 180 and G < 110 and B < 110:
        return "B"            # blinky (red)
    if R > 180 and 110 <= G <= 220 and B > 150 and (R - G) >= 20:
        return "P"            # pinky
    if R < 120 and G > 150 and B > 150:
        return "I"            # inky (cyan)
    if R > 180 and 100 <= G <= 190 and B < 100 and (R - G) >= 40:
        return "C"            # clyde (orange)
    if R < 100 and G < 100 and B > 150:
        return "F"            # frightened (blue)
    if R > 200 and G > 200 and B > 200:
        return "F"            # frightened white flash
    return None


def _food_tiles(sub, c, anchor):
    """Current-frame food tile sets from the un-eroded gray (pellets) and
    white non-wall (power) masks."""
    _r, wall_s, _b, gray, white = _masks_sub(sub)
    seen_p = set()
    seen_w = set()
    for s, x0, y0, w, h, _rgb in _components(gray & ~wall_s, sub):
        if PELLET_S[0] <= s <= PELLET_S[1] and 4 <= w <= PELLET_BOX and 4 <= h <= PELLET_BOX:
            if s / float(w * h) < PELLET_MIN_FILL:
                continue
            cx, cy = x0 + w / 2.0, y0 + h / 2.0
            seen_p.add(_tile_assign(cx, cy, c, anchor))
    for s, x0, y0, w, h, _rgb in _components(white & ~wall_s, sub):
        if (POWER_S[0] <= s <= POWER_S[1] and POWER_BOX[0] <= w <= POWER_BOX[1]
                and POWER_BOX[0] <= h <= POWER_BOX[1]):
            cx, cy = x0 + w / 2.0, y0 + h / 2.0
            seen_w.add((int(cx // c), int(cy // c)))
    return seen_p, seen_w


# ------------------------------------------------------------- dynamics ----
def _corner(st, x, y):
    c = st["cell"]
    i = int(round(x / c))
    j = int(round(y / c))
    i = max(0, min(st["n_cols"] - 1, i))
    j = max(0, min(st["n_rows"] - 1, j))
    return i, j


def _hyst(st, key, absent_key, seen, board_key, remove_streak, occ):
    """Maintain the current food set.

    Board growth: a tile joins the board set once seen in
    BOARD_SEEN_NEEDED frames (anti-aliasing noise is one-frame only), so
    the board is the union of everything ever really seen since the last
    structure build and can never shrink.

    Removal: a tile is dropped only after `remove_streak` consecutive
    frames in which it is BOTH unseen and NOT covered by the player or a
    ghost sprite (a sprite standing on a pellet hides it; the counter is
    paused while the tile is occluded).  A tile that is seen again is
    re-added immediately if it belongs to the board set."""
    absent = st[absent_key]
    cur = set(tuple(t) for t in st[key])
    bset = set(tuple(t) for t in st[board_key])
    counts = st["board_counts"]
    for t in seen:
        k = "%d,%d" % t
        counts[k] = counts.get(k, 0) + 1
        if counts[k] == BOARD_SEEN_NEEDED:
            bset.add(t)
    st[board_key] = [list(t) for t in sorted(bset)]
    for t in list(cur):
        k = "%d,%d" % t
        if t in seen:
            absent.pop(k, None)
        elif t in occ:
            pass
        else:
            absent[k] = absent.get(k, 0) + 1
            if absent[k] >= remove_streak:
                cur.discard(t)
                del absent[k]
    for t in seen:
        if t in bset:
            cur.add(t)
    st[key] = [list(t) for t in sorted(cur)]


# ---------------------------------------------------------------- graph ----
def _open_edges(W):
    nr, nc = W.shape

    def tw(k, j):
        return bool(W[j, k]) if 0 <= k < nc and 0 <= j < nr else True

    # An edge is open only when BOTH straddling tiles are corridor; out of
    # bounds counts as wall, so the board border closes correctly.
    # oh[j][i] = right edge of cell (i,j) (straddles (i,j)/(i+1,j));
    # ov[j][i] = down edge of cell (i,j) (straddles (i,j)/(i,j+1)).
    # The extra last row/column of each list is padding, never indexed.
    oh = [[not (tw(i, j) or tw(i + 1, j)) for i in range(nc)] for j in range(nr + 1)]
    ov = [[not (tw(i, j) or tw(i, j + 1)) for i in range(nc + 1)] for j in range(nr)]
    return oh, ov


def _bfs(oh, ov, start):
    nc = len(oh[0])
    nr = len(ov)
    dist = {start: 0}
    parent = {}
    q = deque([start])
    while q:
        i, j = q.popleft()
        d = dist[(i, j)] + 1
        if i < nc and oh[j][i] and (i + 1, j) not in dist:
            dist[(i + 1, j)] = d
            parent[(i + 1, j)] = (i, j)
            q.append((i + 1, j))
        if i > 0 and oh[j][i - 1] and (i - 1, j) not in dist:
            dist[(i - 1, j)] = d
            parent[(i - 1, j)] = (i, j)
            q.append((i - 1, j))
        if j < nr and ov[j][i] and (i, j + 1) not in dist:
            dist[(i, j + 1)] = d
            parent[(i, j + 1)] = (i, j)
            q.append((i, j + 1))
        if j > 0 and ov[j - 1][i] and (i, j - 1) not in dist:
            dist[(i, j - 1)] = d
            parent[(i, j - 1)] = (i, j)
            q.append((i, j - 1))
    return dist, parent


_GNAME = {"B": "blinky", "P": "pinky", "I": "inky", "C": "clyde", "F": "frightened"}
_DIRNAME = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}
_DIRS = (("up", (0, -1)), ("down", (0, 1)), ("left", (-1, 0)), ("right", (1, 0)))
_OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _escape_dir(oh, ov, p, ghosts):
    """Pick the open direction from corner `p` that maximises the distance to
    the nearest ghost (BFS from every visible ghost).  Returns a direction
    name, or None when no open direction is strictly safer than staying."""
    i, j = p
    nc = len(oh[0])
    nr = len(ov)
    gdists = []
    for g in ghosts:
        if g[3]:
            continue
        gdists.append(_bfs(oh, ov, (g[0], g[1]))[0])
    if not gdists:
        return None
    best = None
    for name, (di, dj) in _DIRS:
        nk, nj = i + di, j + dj
        if not (0 <= nk < nc and 0 <= nj < nr):
            continue
        if di == 0 and dj == -1 and not (j > 0 and ov[j - 1][i]):
            continue
        if di == 0 and dj == 1 and not (j < nr and ov[j][i]):
            continue
        if di == -1 and dj == 0 and not (i > 0 and oh[j][i - 1]):
            continue
        if di == 1 and dj == 0 and not (i < nc and oh[j][i]):
            continue
        dmin = min(gd.get((nk, nj), 10 ** 6) for gd in gdists)
        dcur = min(gd.get((i, j), 10 ** 6) for gd in gdists)
        if dmin <= dcur:
            continue
        if best is None or dmin > best[0]:
            best = (dmin, name)
    return best[1] if best else None


# --------------------------------------------------------------- render ----
def _first_step(parent, target, start):
    if target == start or target not in parent:
        return None
    cur = target
    for _ in range(len(parent) + 2):
        if parent.get(cur) == start:
            return _DIRNAME.get((cur[0] - start[0], cur[1] - start[1]))
        cur = parent.get(cur)
        if cur is None:
            return None
    return None


def _render(st, W):
    nc, nr = st["n_cols"], st["n_rows"]
    p = st["p"]
    ghosts = st["ghosts"]
    lines = []
    if p:
        stale = " (stale %d frames, may have moved)" % st["p_stale"] if st["p_stale"] >= 2 else ""
        lines.append("MAZE PANEL: you@(%d,%d)%s" % (p[0], p[1], stale))
    else:
        st["nav_hint"] = None
        lines.append("MAZE PANEL: you@? (not detected)")

    hidden = max(0, st["max_ghosts"] - len(ghosts))
    total_board = len(st["board_pellets"]) + len(st["board_power"])
    gl = "ghosts=%d" % len(ghosts)
    if hidden > 0:
        gl += " visible (%d hidden in house)" % hidden
    lines.append("pellets=%d power=%d %s food-left=%d/%d" % (
        len(st["pellets"]), len(st["power"]), gl,
        len(st["pellets"]) + len(st["power"]), total_board))
    if st["est_score"] is not None:
        lines.append("est score: run~%d best~%d (food points only; ghost kills add more)" % (
            st["est_score"], st["best_est"]))

    if st.get("in_transition"):
        if st.get("restart_run", 0) > 0:
            if p:
                lines.append("TRANSITION: GAME RESTARTED (READY); the board refilled and your score reset to 0 - your best so far still counts. You start at (%d,%d). Moves are ignored now - use wait." % (p[0], p[1]))
            else:
                lines.append("TRANSITION: GAME RESTARTED (READY); the board refilled and your score reset to 0 - your best so far still counts. Moves are ignored now - use wait.")
        elif st["ready"] and p:
            if st["deaths"]:
                lines.append("TRANSITION: post-death pause (READY); you respawn at (%d,%d). Moves are ignored now - use wait." % (p[0], p[1]))
            else:
                lines.append("TRANSITION: pre-game pause (READY); you start at (%d,%d). Moves are ignored now - use wait." % (p[0], p[1]))
        else:
            # last_dir is set once the player has ever moved, so mid-episode
            # pauses (death, incl. the corpse phase before READY is seen)
            # are worded as post-death, not start-up.
            if st["deaths"] or st["last_dir"] is not None:
                lines.append("TRANSITION: post-death pause; moves are ignored - use wait.")
            else:
                lines.append("TRANSITION: start-up pause; moves are ignored - use wait.")
        if st["deaths"]:
            lines.append("deaths=%d" % len(st["deaths"]))
        return "\n".join(lines)
    if st.get("restart_run", 0) > 0:
        lines.append("NOTE: the game just RESTARTED - the board refilled and your score reset to 0; your best so far still counts.")

    oh, ov = _open_edges(W)
    if p:
        i, j = p
        od = []
        if j > 0 and ov[j - 1][i]:
            od.append("up")
        if j < nr and ov[j][i]:
            od.append("down")
        if i > 0 and oh[j][i - 1]:
            od.append("left")
        if i < nc and oh[j][i]:
            od.append("right")
        lines.append("open: %s" % (",".join(od) if od else "none (surrounded)"))
    if st["last_move_note"]:
        lines.append(st["last_move_note"])

    if p:
        i, j = p
        dist, parent = _bfs(oh, ov, tuple(p))
        cands = []
        for k, jj in st["pellets"]:
            if (k, jj) == (i, j):
                continue  # under you (occluded); clears on its own
            d = dist.get((k, jj))
            if d is not None:
                cands.append((d, (k, jj), "pellet"))
        if not cands:
            for k, jj in st["power"]:
                if (k, jj) == (i, j):
                    continue
                d = dist.get((k, jj))
                if d is not None:
                    cands.append((d, (k, jj), "power"))
        if cands:
            best_d = min(c[0] for c in cands)
            nt = tuple(st.get("nav_target") or (-1, -1))
            last_dir = st.get("last_dir")
            stalled = st["stall"] >= STALL_WARN and not st["ready"]
            chosen = None
            for d, t, kind in cands:
                if d > best_d + 2:
                    continue
                fs = _first_step(parent, t, tuple(p))
                # While stalled a reversal is the right answer, so drop the
                # anti-reverse preference; otherwise keep it (rule 1: commit).
                opp = 1 if (not stalled and last_dir and fs == _OPP.get(last_dir)) else 0
                sticky = 0 if t == nt else 1
                rank = (opp, sticky, d, t)
                if chosen is None or rank < chosen[0]:
                    chosen = (rank, t, kind, fs)
            if chosen is not None:
                _rank, (tk, tj), kind, fs = chosen
                st["nav_target"] = [tk, tj]
                st["nav_hint"] = [kind, tk, tj, fs, _rank[2]]
                lag = ""
                if (fs and last_dir and fs != last_dir and fs != _OPP.get(last_dir)):
                    lag = (" [turn from %s may lag one step - re-issue if it does not take effect]"
                           % last_dir)
                lines.append("nearest %s at (%d,%d) dist=%d first=%s%s" % (
                    kind, tk, tj, _rank[2], fs if fs else "?", lag))
        else:
            st["nav_target"] = None
            st["nav_hint"] = None
            if len(st["pellets"]) + len(st["power"]) > 0:
                lines.append("food remains (%d left) but no route found this frame (map noise) - move along any open direction" % (len(st["pellets"]) + len(st["power"])))
            else:
                lines.append("no food left on the board")
        gd = None
        gk = None
        gg = None
        for g in ghosts:
            d = dist.get((g[0], g[1]))
            if d is not None and (gd is None or d < gd):
                gd = d
                gk = g[2]
                gg = g
        if gk is not None:
            lines.append("nearest reachable ghost: %s dist=%d" % (_GNAME.get(gk, gk), gd))
            if gd <= 3 and gg is not None and not gg[3]:
                if gk == "F":
                    lines.append("GHOST DANGER: a frightened ghost is right next to you - move toward it to eat it")
                else:
                    esc = _escape_dir(oh, ov, tuple(p), ghosts)
                    if esc:
                        lines.append("GHOST DANGER: %s is %d step(s) away - escape via %s" % (_GNAME.get(gk, gk), gd, esc))
                    else:
                        lines.append("GHOST DANGER: %s is %d step(s) away - no clear escape; wait or reverse" % (_GNAME.get(gk, gk), gd))

    if st["lost_ghosts"]:
        parts = []
        for gkind in sorted(st["lost_ghosts"]):
            li, lj, age = st["lost_ghosts"][gkind]
            parts.append("%s last seen at (%d,%d) %d frame(s) ago (may be overlapping you)" % (
                _GNAME.get(gkind, gkind), li, lj, age))
        lines.append("lost: " + "; ".join(parts))
    if st["ready"]:
        lines.append("READY: pause before play; your score is preserved. Re-orient using this panel and pick a safe direction.")
    if st["stall"] >= STALL_WARN and not st["ready"]:
        hint = st.get("nav_hint")
        if hint and hint[3]:
            avoid = " (do not keep sliding %s)" % st["last_dir"] if st["last_dir"] else ""
            lines.append("STALLED: no pellet eaten for %d steps - your next move must be %s, toward the %s at (%d,%d)%s." % (
                st["stall"], hint[3], hint[0], hint[1], hint[2], avoid))
        else:
            lines.append("STALLED: no pellet eaten for %d steps - change direction at the next junction, do not keep looping." % st["stall"])
    if st["deaths"]:
        lines.append("deaths=%d" % len(st["deaths"]))

    lines.append("MAP (#=wall .=corridor o=pellet O=power @=you B/P/I/C=ghost F=frightened):")
    pset = set(tuple(t) for t in st["pellets"])
    wset = set(tuple(t) for t in st["power"])
    rows = []
    for j in range(nr):
        row = []
        for k in range(nc):
            if W[j, k]:
                row.append("#")
            elif (k, j) in wset:
                row.append("O")
            elif (k, j) in pset:
                row.append("o")
            else:
                row.append(".")
        rows.append(row)
    for g in ghosts:
        gi, gj = g[0], g[1]
        if 0 <= gi < nc and 0 <= gj < nr:
            rows[gj][gi] = _GNAME.get(g[2], g[2])[0].upper()
    pt = st["p_tile"]
    if pt and 0 <= pt[0] < nc and 0 <= pt[1] < nr:
        rows[pt[1]][pt[0]] = "@"
    lines.extend("".join(row) for row in rows)
    return "\n".join(lines)


# ---------------------------------------------------------------- update ----
def update(st, a, step, prev_action=None):
    """Parse frame `a` (HxWx3 uint8) into the panel text for this step.

    Mutates the JSON-safe state dict `st` in place.  Returns "" when the
    frame cannot be parsed (agent then behaves like the parent)."""
    if a is None:
        return ""
    try:
        return _update(st, a, step, prev_action)
    except Exception:
        return ""


def _update(st, a, step, prev_action):
    if not st["initialized"]:
        if not _build_structure(st, a):
            return ""
    c = st["cell"]
    ix0, iy0, ix1, iy1 = st["ix0"], st["iy0"], st["ix1"], st["iy1"]
    nc, nr = st["n_cols"], st["n_rows"]

    sub = a[iy0:iy0 + nr * c, ix0:ix0 + nc * c]
    red_s, wall_s, bright_s, gray_s, white_s = _masks_sub(sub)
    W = _wall_grid(a, wall_s, 0, 0, nc, nr, c)
    stored = np.array(st["wall"], dtype=bool)
    if (W != stored).sum() > max(WALL_REBUILD_MIN, WALL_REBUILD_FRAC * nc * nr):
        if not _build_structure(st, a):
            return ""
        c = st["cell"]
        ix0, iy0, ix1, iy1 = st["ix0"], st["iy0"], st["ix1"], st["iy1"]
        nc, nr = st["n_cols"], st["n_rows"]
        sub = a[iy0:iy0 + nr * c, ix0:ix0 + nc * c]
        red_s, wall_s, bright_s, gray_s, white_s = _masks_sub(sub)
        stored = np.array(st["wall"], dtype=bool)
    W = stored

    prev_p = st["p"]

    # --- component lists (computed once per mask) ---
    fg = ndimage.binary_erosion(bright_s & ~wall_s, iterations=1)
    red_comps = list(_components(red_s, sub))
    fg_comps = list(_components(fg, sub))
    gray_comps = list(_components(gray_s & ~wall_s, sub))
    white_comps = list(_components(white_s & ~wall_s, sub))

    # --- food (un-eroded gray/white) ---
    anchor = st.get("pellet_anchor", "center")
    seen_p = set()
    seen_w = set()
    for s, x0, y0, w, h, _rgb in gray_comps:
        if PELLET_S[0] <= s <= PELLET_S[1] and 4 <= w <= PELLET_BOX and 4 <= h <= PELLET_BOX:
            if s / float(w * h) < PELLET_MIN_FILL:
                continue
            cx, cy = x0 + w / 2.0, y0 + h / 2.0
            seen_p.add(_tile_assign(cx, cy, c, anchor))
    for s, x0, y0, w, h, _rgb in white_comps:
        if (POWER_S[0] <= s <= POWER_S[1] and POWER_BOX[0] <= w <= POWER_BOX[1]
                and POWER_BOX[0] <= h <= POWER_BOX[1]):
            cx, cy = x0 + w / 2.0, y0 + h / 2.0
            seen_w.add((int(cx // c), int(cy // c)))

    # --- player + non-red ghosts (eroded bright non-wall) ---
    p_cand = None
    ghost_cands = []
    for s, x0, y0, w, h, (R, G, B) in fg_comps:
        if _is_player(s, w, h, R, G, B):
            if p_cand is None or s > p_cand[0]:
                p_cand = (s, x0, y0, w, h, x0 + w / 2.0, y0 + h / 2.0)
            continue
        kind = _ghost_kind(R, G, B)
        if kind and GHOST_MIN_S <= s and GHOST_BOX[0] <= w <= GHOST_BOX[1] and GHOST_BOX[0] <= h <= GHOST_BOX[1]:
            if s / float(w * h) >= GHOST_MIN_FILL:
                ghost_cands.append((s, kind, x0, y0, w, h, x0 + w / 2.0, y0 + h / 2.0))

    # --- Blinky (red, un-eroded; red is in the wall mask so the bright
    #     pass never sees him) ---
    blinky = None
    for s, x0, y0, w, h, _rgb in red_comps:
        if BLINKY_BOX[0] <= w <= BLINKY_BOX[1] and BLINKY_BOX[0] <= h <= BLINKY_BOX[1]:
            if BLINKY_S[0] <= s <= BLINKY_S[1] and s / float(w * h) < BLINKY_MAX_FILL:
                if blinky is None or s > blinky[0]:
                    blinky = (s, x0, y0, w, h, x0 + w / 2.0, y0 + h / 2.0)

    # --- occlusion: tiles covered by the player or a ghost sprite ---
    occ = set()

    def _add_box(x0, y0, w, h):
        k0, k1 = int(x0 // c), int((x0 + w - 1) // c)
        j0, j1 = int(y0 // c), int((y0 + h - 1) // c)
        for kk in range(k0, k1 + 1):
            for jj in range(j0, j1 + 1):
                if 0 <= kk < nc and 0 <= jj < nr:
                    occ.add((kk, jj))

    if p_cand is not None:
        _s, px0, py0, pw, ph, pcx, pcy = p_cand
        _add_box(px0, py0, pw, ph)
        # the red head deco is a separate component riding on the player
        for s, x0, y0, w, h, _rgb in red_comps:
            if 30 <= s <= 320 and w <= 20 and h <= 20:
                if abs(x0 + w / 2.0 - pcx) <= 25 and abs(y0 + h / 2.0 - pcy) <= 25:
                    _add_box(x0, y0, w, h)
    for _s, _k, gx0, gy0, gw, gh, _cx, _cy in ghost_cands:
        _add_box(gx0, gy0, gw, gh)
    if blinky is not None:
        _s, bx0, by0, bw, bh, _cx, _cy = blinky
        _add_box(bx0, by0, bw, bh)

    # --- pellets / power (board = union of all seen; removal is
    #     occlusion-aware so sprites hiding food cannot corrupt counts) ---
    _hyst(st, "pellets", "p_absent", seen_p, "board_pellets", REMOVE_STREAK, occ)
    _hyst(st, "power", "power_absent", seen_w, "board_power", POWER_REMOVE_STREAK, occ)

    # --- player (keep last known position briefly if occluded) ---
    p = None
    p_tile = None
    if p_cand is not None:
        p = list(_corner(st, p_cand[5], p_cand[6]))
        p_tile = [int(p_cand[5] // c), int(p_cand[6] // c)]
        st["p_stale"] = 0
    else:
        st["p_stale"] += 1
        if st["p_stale"] > 6:
            st["p"] = None
            st["p_tile"] = None
        else:
            p = st["p"]
            p_tile = st["p_tile"]

    # --- last travel direction (corner units; ~3 corners per action) ---
    if p is not None and prev_p is not None and st["p_stale"] == 0:
        di = p[0] - prev_p[0]
        dj = p[1] - prev_p[1]
        mag = abs(di) + abs(dj)
        if mag > 6:
            st["last_dir"] = None  # teleport (respawn); previous direction is stale
        elif mag >= 1:
            if abs(di) >= abs(dj):
                st["last_dir"] = "right" if di > 0 else "left"
            else:
                st["last_dir"] = "down" if dj > 0 else "up"

    # --- ghosts (all single-pass + Blinky, dedupe by position) ---
    all_ghosts = []
    for s, kind, _x0, _y0, _w, _h, cx, cy in sorted(ghost_cands, reverse=True):
        if len(all_ghosts) >= GHOST_MAX_COUNT:
            break
        if all((cx - ax) ** 2 + (cy - ay) ** 2 > GHOST_DEDUPE_PX ** 2
               for _s, _k, ax, ay, _x0, _y0, _w, _h in all_ghosts):
            all_ghosts.append((s, kind, cx, cy, _x0, _y0, _w, _h))
    if blinky is not None and len(all_ghosts) < GHOST_MAX_COUNT:
        if all((blinky[5] - ax) ** 2 + (blinky[6] - ay) ** 2 > GHOST_DEDUPE_PX ** 2
               for _s, _k, ax, ay, _x0, _y0, _w, _h in all_ghosts):
            all_ghosts.append((blinky[0], "B", blinky[5], blinky[6],
                               blinky[1], blinky[2], blinky[3], blinky[4]))

    glist = []
    seen_corners = {}
    for s, kind, cx, cy, _x0, _y0, _w, _h in all_ghosts:
        gi, gj = _corner(st, cx, cy)
        k = int(cx // c)
        j = int(cy // c)
        inhouse = 1 if (0 <= k < nc and 0 <= j < nr and W[j, k]) else 0
        if (gi, gj) in seen_corners:
            continue
        seen_corners[(gi, gj)] = True
        glist.append([gi, gj, kind, inhouse])

    if len(glist) > st["max_ghosts"]:
        st["max_ghosts"] = len(glist)

    # --- lost-ghost markers: a visible (out-of-house) ghost that vanished
    #     was probably merged with the player sprite (pre-death) or eaten;
    #     flag its last corner for a few frames so the model treats that
    #     area as dangerous.  Expiry keeps the line from lingering. ---
    lost = st["lost_ghosts"]
    cur_kinds = set(g[2] for g in glist)
    for gkind in list(lost):
        if gkind in cur_kinds:
            del lost[gkind]  # visible again (or reclassified); drop the marker
        else:
            lost[gkind][2] += 1  # age every frame the kind stays absent
    for g in st["ghosts"]:
        gkind = g[2]
        if gkind in cur_kinds:
            continue
        if g[3] or gkind in lost:
            continue
        lost[gkind] = [g[0], g[1], 1]
    for gkind in list(lost):
        if lost[gkind][2] > 4:
            del lost[gkind]
    st["lost_ghosts"] = lost

    # --- READY letters (red, un-eroded; exclude the player's head deco) ---
    letters = []
    cx_maze = nc * c / 2.0
    cy_maze = nr * c / 2.0
    px, py = (p_cand[5], p_cand[6]) if p_cand is not None else (None, None)
    for s, x0, y0, w, h, _rgb in red_comps:
        cx, cy = x0 + w / 2.0, y0 + h / 2.0
        if not (LETTER_S[0] <= s <= LETTER_S[1] and w <= LETTER_BOX and h <= LETTER_BOX):
            continue
        if s / float(w * h) >= LETTER_MAX_FILL:
            continue
        if abs(cx - cx_maze) > READY_CENTER_X * nc * c:
            continue
        if abs(cy - cy_maze) > READY_CENTER_Y * nr * c:
            continue
        if px is not None and abs(cx - px) <= PLAYER_LETTER_DIST and abs(cy - py) <= PLAYER_LETTER_DIST:
            continue
        letters.append((cx, cy))

    ready = False
    if len(letters) >= READY_MIN_LETTERS:
        # cluster by row (y within READY_Y_TOL); a READY row has >=4 letters
        letters.sort(key=lambda t: t[1])
        cluster = [letters[0]]
        for lx, ly in letters[1:]:
            if ly - cluster[-1][1] <= READY_Y_TOL:
                cluster.append((lx, ly))
            else:
                cluster = [(lx, ly)]
            if len(cluster) >= READY_MIN_LETTERS:
                if max(x for x, _y in cluster) - min(x for x, _y in cluster) >= READY_X_SPREAD:
                    ready = True
                break

    # --- death (READY) tracking and board refill ---
    # --- death (READY) tracking.  The board does NOT refill on death in
    # this game: eaten pellets stay gone, so the current sets persist.
    # If a variant did refill, _hyst re-adds seen board tiles at once. ---
    if ready:
        if st["last_step"] >= 0 and (
                st["last_ready_step"] is None or step - st["last_ready_step"] > 1):
            st["deaths"].append(step)
        st["last_ready_step"] = step
    else:
        # fallback: death without visible READY text (food count jumps up)
        pc_now = len(seen_p) + len(seen_w)
        if (st["last_pc"] is not None and pc_now >= st["last_pc"] + 15
                and st["last_ready_step"] is not None
                and step - st["last_ready_step"] > 3):
            st["deaths"].append(step)

    st["ready"] = ready

    prev_pc = st["last_pc"]  # confirmed count before this frame

    # --- stall tracking (confirmed pellet+power count) ---
    pc = len(st["pellets"]) + len(st["power"])
    if st["last_pc"] is None:
        st["last_pc"] = pc
        st["stall"] = 0
    elif pc == st["last_pc"]:
        st["stall"] += 1
    else:
        st["last_pc"] = pc
        st["stall"] = 0

    if st["start_pc"] is None:
        st["start_pc"] = pc
        st["start_pw"] = len(st["power"])
    elif pc >= st["start_pc"] + RESTART_JUMP:
        st["start_pc"] = pc
        st["start_pw"] = len(st["power"])
        st["restart_run"] = 4  # terminal restart: full board refill detected
    elif (prev_pc is not None
          and pc >= prev_pc + RESTART_JUMP
          and pc >= st["start_pc"]):
        # Food count jumped back up to (at least) the full-board level:
        # the board refilled, i.e. a terminal-failure game restart.
        # start_pc already equals the full-board level, so no re-anchor.
        st["restart_run"] = 4
    # --- score estimate: food points only (a lower bound on the true
    #     score).  The board sets hold every tile really seen since the
    #     last structure build, so a tile seen before and now gone was
    #     eaten.  On a terminal restart the board refills, the current
    #     sets re-add the tiles and the estimate drops back to ~0; the
    #     best-so-far is kept across restarts. ---
    est = (max(0, len(st["board_pellets"]) - len(st["pellets"])) * PELLET_PTS
           + max(0, len(st["board_power"]) - len(st["power"])) * POWER_PTS)
    st["est_score"] = est
    if est > st["best_est"]:
        st["best_est"] = est

    # --- transition state machine (death/READY pause).  Death sequence:
    #     3 frames player hidden (ghosts visible) -> 5-6 frames all sprites
    #     hidden -> 3 READY frames -> resume.  Moves issued during the pause
    #     are ignored, so the panel tells the model to wait.  Entry: no
    #     ghosts visible for 2+ frames AND the player at least partly hidden
    #     (2+ stale frames), or READY text, or player undetected for 3+
    #     frames (hidden pre-death).  Exit: no READY for 2 frames AND either
    #     the ghosts are visible again or the player is tracked again (play
    #     has actually resumed); requiring ghost0_run==0 alone would keep the
    #     machine open for the whole window in which all ghosts are
    #     undetected (in-house, undrawn) after a p_stale>=3 entry.  NOTE:
    #     ghost0 alone must NOT trigger a transition: in this engine all
    #     four ghosts can be undetected during normal play, which produced
    #     multi-frame false "use wait" windows that suppressed the safety
    #     lines (lost:, GHOST DANGER) right before a death. ---
    if len(glist) == 0:
        st["ghost0_run"] += 1
    else:
        st["ghost0_run"] = 0
    if ready:
        st["notready_run"] = 0
    else:
        st["notready_run"] += 1
    # Play genuinely resumed: a real slide (>= RESUME_MOVE_MIN corner units
    # in one frame), a teleport respawn (last_dir reset to None above), or
    # the READY phase has ended.  The post-death "corpse" sprite is a
    # detected (p_stale=0) frozen yellow sprite at the death location that
    # may drift 0-1 tiles while every move is still ignored, so the machine
    # must NOT exit on mere detection: the old exit fired mid-pause and
    # exposed a live panel with "no progress" notes during ignored moves.
    # A timeout bounds any false-positive window (see comment above).
    resume_move = False
    if p is not None and prev_p is not None and st["p_stale"] == 0:
        resume_move = abs(p[0] - prev_p[0]) + abs(p[1] - prev_p[1]) >= RESUME_MOVE_MIN
    if st["in_transition"]:
        st["trans_frames"] += 1
        if ready:
            st["ready_seen_trans"] = True
        if (st["notready_run"] >= 2
                and (resume_move or st["last_dir"] is None or st["ready_seen_trans"])
                or st["trans_frames"] >= TRANSITION_TIMEOUT):
            st["in_transition"] = False
            st["ready_seen_trans"] = False
            st["stall"] = 0
    else:
        st["trans_frames"] = 0
        st["ready_seen_trans"] = False
        if (st["ghost0_run"] >= 2 and st["p_stale"] >= 2) or ready or st["p_stale"] >= 3:
            st["in_transition"] = True
            st["ready_seen_trans"] = bool(ready)
    if st["restart_run"] > 0:
        st["restart_run"] -= 1

    # --- previous action progress note ---
    note = ""
    dirs = {"move_up": (0, -1), "move_down": (0, 1), "move_left": (-1, 0), "move_right": (1, 0)}
    if prev_action in dirs and not st["in_transition"] and st["p_stale"] == 0 and p is not None and prev_p is not None:
        di, dj = dirs[prev_action]
        mv = (p[0] - prev_p[0]) * di + (p[1] - prev_p[1]) * dj
        if mv == 0:
            note = "last %s made no progress (blocked or turned)" % prev_action
        elif mv < 0:
            note = "last %s did not take effect (you moved away)" % prev_action
    st["last_move_note"] = note

    st["p"] = p
    st["p_tile"] = p_tile
    st["ghosts"] = glist
    st["last_step"] = step
    return _render(st, W)
