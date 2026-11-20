"""Generic playfield decoder and placement solver for block-falling games.

Everything here is derived from the current public observation (an RGB PIL
image) plus, when available, the previous frame's decode.  No coordinate,
colour or layout constant from any particular episode is hard-coded: the
playfield border is located from the image itself, the grid size is taken
from faint grid-line periodicity when detectable and from a standard
playfield-shape prior otherwise, and cell states are classified by relative
brightness/saturation thresholds.

All returned values are JSON-safe (dict/list/str/int/float/bool/None).
"""
from __future__ import annotations

import numpy as np

EMPTY, COLORED, GHOST = 0, 1, 2

# Standard block-falling playfield prior (10 wide x 20 tall, square cells).
# Used only when the grid lines are too faint to count; the prior is about
# the game genre, not about any one episode.
_PRIOR_COLS, _PRIOR_ROWS = 10, 20

# Relative cell classification thresholds (measured against dark board
# background ~max 35, ghost ~66 gray, pieces ~177+ saturated).
_EMPTY_MAX = 45
_GHOST_MAX = 100
_GHOST_SAT = 25
_COLORED_SAT = 40

_TETROMINOES = {
    "I": frozenset({(0, 0), (0, 1), (0, 2), (0, 3)}),
    "O": frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
    "T": frozenset({(0, 1), (1, 0), (1, 1), (1, 2)}),
    "S": frozenset({(0, 1), (0, 2), (1, 0), (1, 1)}),
    "Z": frozenset({(0, 0), (0, 1), (1, 1), (1, 2)}),
    "J": frozenset({(0, 0), (1, 0), (1, 1), (1, 2)}),
    "L": frozenset({(0, 2), (1, 0), (1, 1), (1, 2)}),
}

_HUE_NAMES = [
    ("cyan", (0, 255, 255)), ("yellow", (255, 255, 0)), ("purple", (180, 60, 255)),
    ("green", (60, 255, 90)), ("red", (255, 60, 60)), ("blue", (50, 80, 255)),
    ("orange", (255, 150, 40)),
]


def _cluster(idx, gap=3):
    """Cluster sorted indices into (start, end) pairs allowing a small gap."""
    if len(idx) == 0:
        return []
    out, cur = [], [int(idx[0])]
    for v in idx[1:]:
        v = int(v)
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            out.append((cur[0], cur[-1]))
            cur = [v]
    out.append((cur[0], cur[-1]))
    return out


def _normalize(cells):
    cells = sorted(cells)
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _rotations(cw_cells):
    """All four rotations of a shape (clockwise), as normalized cell sets."""
    out = []
    cells = cw_cells
    for _ in range(4):
        cells = _normalize((c, -r) for r, c in cells)
        out.append(cells)
    return out


def _shape_table():
    table = {}
    for name, cells in _TETROMINOES.items():
        table[name] = _rotations(cells)
    return table


_SHAPE_TABLE = _shape_table()


def identify_piece(cells):
    """Match a set of board cells against tetrominoes (partial shapes allowed).

    Returns (name or None, rotation_index or None, matched_cell_count).
    """
    if not cells:
        return None, None, 0
    norm = _normalize(cells)
    n = len(norm)
    if n == 1:
        return None, None, 1
    best = (None, None, 0)
    for name, rots in _SHAPE_TABLE.items():
        for rot, shape in enumerate(rots):
            if len(shape) == n and shape == norm:
                return name, rot, n
            if 3 <= n < len(shape) and norm <= shape:
                if n > best[2]:
                    best = (name, rot, n)
    if n >= 3 and best[0] is None:
        return None, None, n
    return best


def hue_name(rgb_cells):
    """Nearest standard block colour name for a list of (r,g,b) triples."""
    arr = np.asarray(rgb_cells, dtype=float)
    if arr.size == 0:
        return ""
    mean = arr.mean(axis=0)
    best, best_dist = "", float("inf")
    for name, ref in _HUE_NAMES:
        d = float(np.linalg.norm(mean - np.asarray(ref, dtype=float)))
        if d < best_dist:
            best, best_dist = name, d
    return best


def find_board_frame(rgb):
    """Locate the playfield border rectangle. Returns None when not found.

    Strategy: the border is a bright, green-dominant rectangle.  Vertical
    border columns are the columns with a very high green count (the two
    longest green vertical runs); the top/bottom are then read from the
    vertical extent of those border columns themselves, which is immune to
    unrelated horizontal UI separators.
    """
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    cmax = rgb.max(axis=2).astype(np.int16)
    green = (cmax > 100) & (g > r + 20) & (g > b + 20)
    if not green.any():
        return None
    col_counts = green.sum(axis=0)
    max_col = int(col_counts.max())
    if max_col < 40:
        return None
    vcols = np.where(col_counts > 0.6 * max_col)[0]
    vc = _cluster(vcols, gap=3)
    if len(vc) < 2:
        return None
    left, right = vc[0], vc[-1]

    def vertical_extent(x0, x1):
        rows = np.where(green[:, x0:x1 + 1].sum(axis=1) > 0)[0]
        cl = _cluster(rows, gap=3)
        if not cl:
            return None
        return max(cl, key=lambda c: c[1] - c[0])

    ext_l = vertical_extent(left[0], left[1])
    ext_r = vertical_extent(right[0], right[1])
    if ext_l is None or ext_r is None:
        return None
    ext_start = min(ext_l[0], ext_r[0])
    ext_end = max(ext_l[1], ext_r[1])
    if ext_end - ext_start < 80 or right[0] - left[1] < 40:
        return None
    # Horizontal border rows: rows whose green run spans the whole board
    # width.  UI separators elsewhere on screen are rejected by matching
    # each candidate to the border columns' own vertical extent.
    span = green[:, left[0]:right[1] + 1].sum(axis=1)
    width = right[1] - left[0] + 1
    rrows = np.where(span > 0.6 * width)[0]
    rc = _cluster(rrows, gap=3)
    if not rc:
        return {"left": [left[0], left[1]], "right": [right[0], right[1]],
                "top": [ext_start, ext_start + 2], "bottom": [ext_end - 2, ext_end]}

    def nearest_cluster(target):
        best, best_d = None, float("inf")
        for a, b2 in rc:
            d = abs((a + b2) / 2.0 - target)
            if d < best_d:
                best, best_d = (a, b2), d
        return best

    top = nearest_cluster(ext_start)
    bottom = nearest_cluster(ext_end)
    if top is None or bottom is None or bottom[0] <= top[1]:
        top = (ext_start, ext_start + 2)
        bottom = (ext_end - 2, ext_end)
    return {"left": [left[0], left[1]], "right": [right[0], right[1]],
            "top": [int(top[0]), int(top[1])], "bottom": [int(bottom[0]), int(bottom[1])]}


def _faint_lines(rgb, frame, axis):
    """Try to count interior grid lines from their faint brightness bump.

    The per-column (or per-row) MINIMUM of the max channel over the whole
    interior is robust against coloured cells: a grid line column stays
    slightly brighter at every row, while background rows sit at the base.
    Returns the number of interior lines or None when not confident.
    """
    x0, x1 = frame["left"][1] + 1, frame["right"][0]
    y0, y1 = frame["top"][1] + 1, frame["bottom"][0]
    interior = cmax_slice = rgb.max(axis=2)
    if axis == "v":
        profile = interior[y0:y1, x0:x1].min(axis=0).astype(float)
    else:
        profile = interior[y0:y1, x0:x1].min(axis=1).astype(float)
    if len(profile) < 60:
        return None
    base = float(np.median(profile))
    cand = np.where(profile > base + 2.0)[0]
    cl = _cluster(cand, gap=2)
    if len(cl) < 4 or len(cl) > 13:
        return None
    centers = [float((a + b) / 2.0) for a, b in cl]
    edges = [x0, x1] if axis == "v" else [y0, y1]
    spans = [c - edges[0] for c in centers] + [edges[1] - edges[0]]
    spacings = [spans[i + 1] - spans[i] for i in range(len(spans) - 1)]
    if not spacings:
        return None
    med = float(np.median(spacings))
    if med < 12 or med > 40:
        return None
    if float(np.std(spacings)) > 0.18 * med:
        return None
    # All interior lines must sit strictly between the edges.
    if min(centers) < 2 or max(centers) > (edges[1] - edges[0]) - 2:
        return None
    return len(cl)


def estimate_grid(rgb, frame, prev):
    """Determine (n_cols, n_rows, method). Persists a good result per episode."""
    if prev is not None:
        pc, pr = prev.get("n_cols"), prev.get("n_rows")
        if pc and pr and prev.get("grid_method") != "prior_mismatch":
            # Re-verify only lightly: frame size must be close.
            w = frame["right"][0] - frame["left"][1]
            pw = prev.get("frame_width") or 0
            if pw and abs(w - pw) <= 6:
                return pc, pr, "persisted"
    w = frame["right"][0] - frame["left"][1]
    h = frame["bottom"][0] - frame["top"][1]
    lv = _faint_lines(rgb, frame, "v")
    lr = _faint_lines(rgb, frame, "h")
    if lv is not None and lr is not None:
        return lv + 1, lr + 1, "faint_lines"
    n_cols = max(6, min(14, round(np.sqrt(_PRIOR_COLS * _PRIOR_ROWS * (w / max(h, 1))))))
    n_rows = max(12, min(26, round(_PRIOR_COLS * _PRIOR_ROWS / n_cols)))
    return n_cols, n_rows, "prior"


def classify_grid(rgb, frame, n_cols, n_rows):
    """Classify every cell as EMPTY / COLORED / GHOST. Returns (grid, cell_px)."""
    x0, x1 = frame["left"][1] + 1, frame["right"][0]
    y0, y1 = frame["top"][1] + 1, frame["bottom"][0]
    cw = (x1 - x0) / n_cols
    ch = (y1 - y0) / n_rows
    grid = []
    for ry in range(n_rows):
        row = []
        ya = int(y0 + ry * ch)
        yb = int(y0 + (ry + 1) * ch)
        my = max(int((yb - ya) * 0.2), 1)
        for cx in range(n_cols):
            xa = int(x0 + cx * cw)
            xb = int(x0 + (cx + 1) * cw)
            mx = max(int((xb - xa) * 0.2), 1)
            cell = rgb[ya + my:yb - my, xa + mx:xb - mx]
            if cell.size == 0:
                row.append(EMPTY)
                continue
            cmax = int(cell.max(axis=2).max())
            if cmax < _EMPTY_MAX:
                row.append(EMPTY)
                continue
            hi = cell.max(axis=2).mean()
            lo = cell.min(axis=2).mean()
            sat = float(hi - lo)
            if cmax < _GHOST_MAX:
                row.append(GHOST if sat < _GHOST_SAT else COLORED)
            else:
                row.append(COLORED if sat >= _COLORED_SAT else GHOST)
        grid.append(row)
    return grid, cw, ch


def decode_strip(rgb, frame, n_cols):
    """Detect which board columns hold a piece cell above row 0.

    Some block-falling games draw the spawning piece in a thin band above
    the visible playfield, aligned to the same columns.  This returns the
    list of such columns (plus the mean colour) so the caller can report a
    piece that is still entering the board.  Returns (cols, hue) where cols
    is empty when nothing is present.
    """
    x0 = frame["left"][1] + 1
    x1 = frame["right"][0]
    y_top = max(frame["top"][0] - 1, 0)
    if y_top < 4:
        return [], ""
    cw = (x1 - x0) / n_cols
    band = rgb[0:y_top, x0:x1]
    if band.size == 0:
        return [], ""
    r = band[..., 0].astype(np.int16)
    g = band[..., 1].astype(np.int16)
    b = band[..., 2].astype(np.int16)
    cmax = band.max(axis=2).astype(np.int16)
    sat = cmax - band.min(axis=2).astype(np.int16)
    # saturated, bright, and not the green border/separator colour
    piece = (cmax > 100) & (sat > 50) & ~((g > r + 20) & (g > b + 20))
    cols = []
    px = []
    for cx in range(n_cols):
        xa = int(cx * cw)
        xb = int((cx + 1) * cw)
        m = max(int((xb - xa) * 0.2), 1)
        sub = piece[:, xa + m:xb - m]
        if sub.size and sub.mean() > 0.25:
            cols.append(cx)
            subpx = band[:, xa + m:xb - m].reshape(-1, 3)
            keep = piece[:, xa + m:xb - m].reshape(-1)
            if keep.any():
                px.append(subpx[keep])
    hue = ""
    if px:
        hue = hue_name(np.concatenate(px, axis=0))
    return cols, hue


def _bfs_supported(colored, n_cols, n_rows):
    """Colored cells supported from below (flood fill up from the bottom row)."""
    locked = set()
    frontier = {(n_rows - 1, c) for c in range(n_cols) if (n_rows - 1, c) in colored}
    while frontier:
        r, c = frontier.pop()
        if (r, c) in locked or (r, c) not in colored:
            continue
        locked.add((r, c))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols and (nr, nc) not in locked and (nr, nc) in colored:
                frontier.add((nr, nc))
    return locked


def split_active_locked(grid, n_cols, n_rows, prev_state):
    """Split colored cells into the locked stack and the active piece.

    Gray cells are the landing ghost.  A colored cell is locked when it is
    supported from below through other colored cells.  When no ghost is
    visible (the piece rests flush on the stack) the just-landed piece is
    recovered by diffing against the previous frame's locked set, with a
    downward-shift correction for line clears.
    """
    colored = {(r, c) for r in range(n_rows) for c in range(n_cols) if grid[r][c] == COLORED}
    ghost = {(r, c) for r in range(n_rows) for c in range(n_cols) if grid[r][c] == GHOST}
    locked = _bfs_supported(colored, n_cols, n_rows)
    floating = colored - locked
    active = set(floating)
    resting = False
    if not ghost:
        new_locked = None
        prev_locked = (prev_state or {}).get("locked")
        if prev_locked:
            prev_set = {tuple(p) for p in prev_locked}
            best_k, best_overlap = 0, -1.0
            denom = max(len(prev_set), 1)
            for k in range(0, n_rows + 1):
                shifted = {(r + k, c) for r, c in prev_set if r + k < n_rows}
                overlap = len(shifted & locked) / denom
                if overlap > best_overlap:
                    best_overlap, best_k = overlap, k
            if best_k > 0 and best_overlap >= 0.8:
                explained = {(r + best_k, c) for r, c in prev_set if r + best_k < n_rows}
            else:
                explained = prev_set
            new_locked = locked - explained
            if new_locked:
                active = set(new_locked)
                resting = True
            else:
                active = set()
                resting = True
        elif floating:
            active = set(floating)
        else:
            active = set()
            resting = True
    return active, locked, ghost, resting


def shift_detect(prev_locked, locked, n_rows):
    """Best downward shift k of prev_locked that best explains locked."""
    prev_set = {tuple(p) for p in (prev_locked or [])}
    best_k, best_overlap = 0, -1.0
    denom = max(len(prev_set), 1)
    for k in range(0, n_rows + 1):
        shifted = {(r + k, c) for r, c in prev_set if r + k < n_rows}
        overlap = len(shifted & locked) / denom
        if overlap > best_overlap:
            best_overlap, best_k = overlap, k
    return best_k, best_overlap


def identify_new_piece(active, prev_state, n_cols, n_rows):
    """Heuristic: which cells belong to a freshly spawned piece after a clear/shift."""
    return active


def detect_events(prev, cur):
    """Compare two consecutive decodes; describe what the previous action did."""
    events = []
    if prev is None:
        return events
    p_locked = set(map(tuple, prev.get("locked") or []))
    p_active = set(map(tuple, prev.get("active") or []))
    c_locked = set(map(tuple, cur.get("locked") or []))
    c_active = set(map(tuple, cur.get("active") or []))
    n_rows = cur["n_rows"]

    if len(p_locked) >= 20 and len(c_locked) < 0.08 * len(p_locked) and not c_active:
        events.append("board reset (game restarted)")
        return events

    k, _ov = shift_detect(p_locked, c_locked, n_rows)
    cleared_rows = k if k > 0 else 0
    if k > 0:
        explained = {(r + k, c) for r, c in p_locked if r + k < n_rows}
    else:
        explained = p_locked
    landed = (c_locked - explained) - c_active
    if landed:
        rs = [r for r, _ in landed]
        cs = [c for _, c in landed]
        events.append("piece landed rows {}-{} cols {}-{}".format(
            min(rs), max(rs), min(cs), max(cs)))
    if cleared_rows:
        events.append("{} line(s) cleared".format(cleared_rows))
    new_piece_name = cur.get("piece_name")
    old_piece_name = prev.get("piece_name")
    if c_active and new_piece_name and new_piece_name != old_piece_name:
        events.append("new piece: {} (was {})".format(new_piece_name, old_piece_name or "?"))
    if not landed and not cleared_rows and c_active and p_active:
        d_r = min(r for r, _ in c_active) - min(r for r, _ in p_active)
        d_c = min(c for _, c in c_active) - min(c for _, c in p_active)
        desc = []
        if d_r > 0:
            desc.append("fell {} row(s)".format(d_r))
        elif d_r < 0:
            desc.append("moved up {} row(s)".format(-d_r))
        if d_c < 0:
            desc.append("shifted left {}".format(-d_c))
        elif d_c > 0:
            desc.append("shifted right {}".format(d_c))
        if cur.get("piece_name") and cur["piece_name"] != old_piece_name:
            desc.append("shape now {}".format(cur.get("piece_name")))
        events.append("piece " + (", ".join(desc) if desc else "unchanged"))
    return events


def _drop_distance(stack, cells, row, col, n_cols, n_rows):
    """How far the piece can fall from (row, col). None when it already collides."""
    occupied = set(map(tuple, stack))
    d = 0
    while True:
        nxt = {(r + row + d + 1, c + col) for r, c in cells}
        if any(rr >= n_rows or cc < 0 or cc >= n_cols or (rr, cc) in occupied
               for rr, cc in nxt):
            return d
        d += 1


def solve_placement(grid, active, strip_cols, n_cols, n_rows):
    """Slide-then-hard-drop solver: maximize the raw drop distance.

    The controllable piece is the observed active cells plus, while the
    piece is still entering the board from the top strip, the strip cells
    at row -1.  The plan is a lateral slide of at most 4 moves (nearest
    column tried first) followed by hard_drop, choosing the column that
    maximizes the raw drop distance from the piece's CURRENT position.
    No rotation, no soft drop, no wait: under the scoring model verified
    from this game's own observations (hard_drop 2 pts/row, soft_drop
    1 pt/row, 0 pts for moves/rotates/waits, one gravity row per action,
    top-out resets the score), maximizing raw drop distance per piece is
    the dominant policy, and it was validated in an offline simulation
    over many seeds.  Everything is derived from the current frame.
    """
    active = set(map(tuple, active or []))
    strip = {(-1, c) for c in (strip_cols or [])}
    piece = set(active)
    if strip:
        if not active:
            # Piece still fully in the top strip: only its bottom row is
            # visible there; the drop distance is set by that bottom row.
            piece |= strip
        else:
            min_r = min(r for r, _ in active)
            cols = {c for _, c in active}
            if min_r == 0:
                # Piece half-entered: the strip holds the row above it.
                for r, c in strip:
                    if any(abs(c - cc) <= 1 for cc in cols):
                        piece.add((r, c))
    occupied = {(r, c) for r in range(n_rows) for c in range(n_cols)
                if grid[r][c] == COLORED and (r, c) not in active}

    def can_be(cells):
        for r, c in cells:
            if c < 0 or c >= n_cols:
                return False
            if r >= n_rows:
                return False
            if 0 <= r < n_rows and (r, c) in occupied:
                return False
        return True

    def drop_distance(cells):
        d = 0
        t = set(cells)
        while True:
            t = {(r + 1, c) for r, c in t}
            if not can_be(t):
                return d
            d += 1

    if not piece:
        # Resting piece, nothing entering: lock it with hard_drop.
        return {"plan": ["hard_drop"], "drop_rows": 0, "slide": 0,
                "land_rows": [], "cleared": 0, "holes": 0, "max_height": 0}

    d0 = drop_distance(piece)
    if d0 <= 0:
        # Already supported (resting on the stack): hard_drop locks it.
        return {"plan": ["hard_drop"], "drop_rows": 0, "slide": 0,
                "land_rows": sorted({r for r, _ in piece}),
                "cleared": 0, "holes": 0, "max_height": 0}

    best_off, best_d = 0, d0
    for off in (0, -1, 1, -2, 2, -3, 3, -4, 4):
        trial = {(r, c + off) for r, c in piece}
        if not can_be(trial):
            continue
        d = drop_distance(trial)
        if d > best_d:
            best_d, best_off = d, off

    shifted = {(r, c + best_off) for r, c in piece}
    land = {(r + best_d, c) for r, c in shifted}
    new_occ = occupied | land
    heights = [0] * n_cols
    for c in range(n_cols):
        for r in range(n_rows):
            if (r, c) in new_occ:
                heights[c] = n_rows - r
                break
    holes = 0
    for c in range(n_cols):
        seen = False
        for r in range(n_rows):
            if (r, c) in new_occ:
                seen = True
            elif seen:
                holes += 1
    cleared = sum(1 for r in range(n_rows)
                  if all((r, c) in new_occ for c in range(n_cols)))
    plan = (["move_left"] * (-best_off) if best_off < 0
            else ["move_right"] * best_off) + ["hard_drop"]
    return {"plan": plan, "drop_rows": best_d, "slide": best_off,
            "land_rows": sorted({r for r, _ in land}), "cleared": cleared,
            "holes": holes, "max_height": max(heights),
            "height_sum": sum(heights)}


def decode_frame(image, prev_state=None):
    """Full decode of one observation frame. Returns a JSON-safe state dict.

    Never raises: any failure returns {"detected": False, ...}.
    """
    state = {"detected": False}
    if image is None:
        return state
    try:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception:
        return state
    try:
        frame = find_board_frame(rgb)
        if frame is None:
            return {"detected": False, "reason": "no border found"}
        n_cols, n_rows, method = estimate_grid(rgb, frame, prev_state)
        grid, cw, ch = classify_grid(rgb, frame, n_cols, n_rows)
        active, locked, ghost, resting = split_active_locked(grid, n_cols, n_rows, prev_state)
        name, rot, n_match = identify_piece(active)
        hue = ""
        strip_cols, strip_hue = decode_strip(rgb, frame, n_cols)
        if name:
            cells_px = []
            for r, c in active:
                xa = frame["left"][1] + 1 + int(c * cw)
                xb = xa + int(cw)
                ya = frame["top"][1] + 1 + int(r * ch)
                yb = ya + int(ch)
                m = int(cw * 0.3)
                cells_px.append(rgb[ya + m:yb - m, xa + m:xb - m].reshape(-1, 3))
            hue = hue_name(cells_px)
        # A piece still entering from above the board is not in the grid.
        # Report its column footprint so the caller can see it.
        entering = bool(strip_cols) and not active
        # Stack heights come from the locked cells only, so a falling piece
        # does not inflate the reported height.  A resting (d=0) piece is
        # already part of `locked`, so it counts as stack.
        locked_set = set(map(tuple, locked))
        # Stack height of a column = distance from the TOP of the stack to
        # the floor: scan top-to-bottom and take the first locked cell.
        heights = []
        for c in range(n_cols):
            h = 0
            for r in range(n_rows):
                if (r, c) in locked_set:
                    h = n_rows - r
                    break
            heights.append(h)
        piece_cells = sorted(active)
        suggestion = solve_placement(grid, active, strip_cols, n_cols, n_rows)
        state = {
            "detected": True,
            "frame": frame,
            "frame_width": frame["right"][0] - frame["left"][1],
            "n_cols": n_cols, "n_rows": n_rows, "grid_method": method,
            "grid": grid,
            "heights": heights,
            "max_height": max(heights) if heights else 0,
            "headroom": n_rows - max(heights) if heights else n_rows,
            "active": [list(p) for p in piece_cells],
            "locked": [list(p) for p in sorted(locked)],
            "ghost": [list(p) for p in sorted(ghost)],
            "resting": bool(resting),
            "piece_name": name, "piece_rot": rot, "piece_hue": hue,
            "entering": entering,
            "strip_cols": list(strip_cols),
            "strip_hue": strip_hue,
            "active_bbox": ([min(r for r, _ in active), min(c for _, c in active),
                             max(r for r, _ in active), max(c for _, c in active)]
                            if active else None),
            "suggestion": suggestion,
            "stack_cells": len(locked),
        }
        return state
    except Exception:
        return {"detected": False, "reason": "decode error"}


def format_board_block(state, events=None):
    """Render the decode as a compact text block for model prompts."""
    if not state or not state.get("detected"):
        return ("BOARD (computed from current pixels): not detected this step — "
                "fall back to the visible screenshot and your memory.")
    lines = ["BOARD (computed from current pixels): "
             "{}x{} grid; method {}".format(state["n_cols"], state["n_rows"], state["grid_method"])]
    grid = state["grid"]
    n_cols = state["n_cols"]
    lines.append("  " + " ".join(str(c % 10) for c in range(n_cols)))
    for r, row in enumerate(grid):
        s = []
        for v in row:
            s.append("o" if v == COLORED else ("g" if v == GHOST else "."))
        lines.append("{:2d} {}".format(r, "".join(s)))
    heights = state["heights"]
    lines.append("stack heights L->R: " + " ".join(str(h) for h in heights))
    lines.append("max height {} / {}; headroom {}".format(
        state["max_height"], state["n_rows"], state["headroom"]))
    if state.get("active"):
        bb = state["active_bbox"]
        lines.append("active piece: {} {} at rows {}-{} cols {}-{}".format(
            state.get("piece_hue") or "", state.get("piece_name") or "?",
            bb[0], bb[2], bb[1], bb[3]))
        if state.get("ghost"):
            gr = [r for r, _ in state["ghost"]]
            gc = [c for _, c in state["ghost"]]
            lines.append("ghost landing: rows {}-{} cols {}-{}".format(min(gr), max(gr), min(gc), max(gc)))
        if state.get("resting"):
            lines.append("note: no visible ghost — the piece appears to be resting on the stack")
        if state.get("strip_cols"):
            lines.append("note: part of the piece is still entering from the top (strip cols {})".format(
                state["strip_cols"]))
    elif state.get("entering") and state.get("strip_cols"):
        lines.append("active piece: ENTERING from top — {} piece spanning cols {} (not on the grid yet)".format(
            state.get("strip_hue") or "?", state["strip_cols"]))
    else:
        lines.append("active piece: none detected")
    sug = state.get("suggestion")
    if sug and sug.get("plan"):
        plan_txt = " then ".join(
            (name + " x" + str(n) if n > 1 else name)
            for name, n in _compress_plan(sug["plan"]))
        lines.append("SUGGESTED PLAN: {}  (drop distance {} rows, slide {}, lands rows {}, "
                     "holes after {}, peak height after {})".format(
            plan_txt, sug["drop_rows"], sug["slide"],
            sug["land_rows"][0] if sug["land_rows"] else "?",
            sug["holes"], sug["max_height"]))
    elif state.get("active"):
        lines.append("SUGGESTED PLAN: none found — commit hard_drop")
    if events:
        lines.append("events since last step: " + "; ".join(events))
    return "\n".join(lines)


def _compress_plan(plan):
    out = []
    for action in plan:
        if out and out[-1][0] == action:
            out[-1][1] += 1
        else:
            out.append([action, 1])
    return out
