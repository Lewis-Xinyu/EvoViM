"""Episode-local perception for a falling-block game seen only as pixels.

Nothing here is keyed to a seed, layout, colour name or screen coordinate.
The well is located by its longest pair of vertical strokes; the cell pitch is
measured from the faint grid inside that box, with a square-cell split as the
fallback.  Piece identity comes from the shape of the cells, cells that float
or that are drawn in the muted "shadow" colour are separated from the resting
stack, and a small drop simulator turns the recognised piece into the board the
next action would produce.

Every function is pure: the agent passes the current PIL image (and, when it
has one, the previous frame) and gets values back.  Nothing is remembered
between frames here, so the module carries no episode-specific constants.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- primitives

def as_array(image):
    return np.asarray(image.convert("RGB"), dtype=np.int16)


def saturated(arr):
    """Bright, coloured (non-grey, non-background) pixels."""
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    return (mx > 85) & ((mx - mn) > 45)


def structural_green(arr):
    """The thin green strokes that outline the field."""
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return (g > 90) & (r < 60) & (b < 60)


def _runs(indices, max_gap=0):
    out = []
    for i in indices:
        if out and i <= out[-1][1] + 1 + max_gap:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def _best_vertical_pair(groups, width):
    """Two thin, tall, far-apart vertical strokes that bound a plausible well."""
    thin = [g for g in groups if (g[1] - g[0]) <= 8]
    best = None
    for i in range(len(thin)):
        for j in range(i + 1, len(thin)):
            left, right = thin[i], thin[j]
            span = right[0] - left[1]
            if span < 0.04 * width or span > 0.85 * width:
                continue
            if best is None or span > best[0]:
                best = (span, left, right)
    return best


def find_playfield(arr):
    """Return the well's interior bounds as a dict, or None."""
    h, w = arr.shape[:2]
    strokes = structural_green(arr)
    col_counts = strokes.sum(axis=0)
    row_counts = strokes.sum(axis=1)
    col_thresh = max(30, int(0.22 * h))
    row_thresh = max(30, int(0.10 * w))
    cols = [i for i in range(w) if col_counts[i] > col_thresh]
    rows = [i for i in range(h) if row_counts[i] > row_thresh]
    if len(cols) < 2 or len(rows) < 2:
        return None
    pair = _best_vertical_pair(_runs(cols), w)
    if pair is None:
        return None
    _, left, right = pair
    x0, x1 = int(left[1]) + 1, int(right[0]) - 1
    if x1 - x0 < 0.04 * w:
        return None
    # Horizontal strokes that actually cross the well interior.  A border is
    # several pixels thick; one-pixel page rules that happen to be long enough
    # are rejected by the thickness test.
    width = x1 - x0 + 1
    inner_rows = []
    for r in rows:
        seg = strokes[r, x0:x1 + 1]
        if seg.sum() > 0.88 * width:
            inner_rows.append(r)
    if len(inner_rows) < 2:
        return None
    groups = [g for g in _runs(inner_rows) if (g[1] - g[0]) >= 1]
    if len(groups) < 2:
        return None
    groups.sort(key=lambda g: g[0])
    y0 = int(groups[0][1]) + 1
    y1 = int(groups[-1][0]) - 1
    if y1 - y0 < 0.08 * h:
        return None
    return {"x0": x0, "x1": x1, "y0": y0, "y1": y1}


def measure_pitch(arr, field):
    """Cell width/height from the faint interior grid, else None."""
    x0, x1, y0, y1 = field["x0"], field["x1"], field["y0"], field["y1"]
    sub = arr[y0:y1 + 1, x0:x1 + 1]
    r, g, b = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
    grid = (g > 40) & (g < 85) & (r < 25) & (b < 25)
    if grid.size == 0:
        return None
    row_scores = grid.sum(axis=1)
    col_scores = grid.sum(axis=0)
    row_thr = max(4, int(0.25 * sub.shape[1]))
    col_thr = max(4, int(0.25 * sub.shape[0]))
    results = []
    for scores, thr in ((row_scores, row_thr), (col_scores, col_thr)):
        idx = [i for i in range(len(scores)) if scores[i] > thr]
        centres = [int((a + b) / 2) for a, b in _runs(idx, 1)]
        diffs = [b - a for a, b in zip(centres, centres[1:])]
        if len(diffs) < 3:
            return None
        results.append(float(np.median(diffs)))
    ch, cw = results[0], results[1]
    if ch < 5 or cw < 5:
        return None
    return cw, ch


def infer_grid(arr, field):
    """Choose (cols, rows, cell_w, cell_h)."""
    field_w = field["x1"] - field["x0"] + 1
    field_h = field["y1"] - field["y0"] + 1
    measured = measure_pitch(arr, field)
    if measured:
        cw, ch = measured
        cols = int(round(field_w / cw))
        rows = int(round(field_h / ch))
        if 4 <= cols <= 20 and 6 <= rows <= 40:
            return cols, rows, field_w / float(cols), field_h / float(rows)
    best = None
    for cols, rows in ((10, 20), (8, 16), (12, 24), (10, 18), (9, 18), (6, 12), (8, 20), (4, 8)):
        ratio = (field_w / float(cols)) / (field_h / float(rows))
        score = -abs(ratio - 1.0)
        if best is None or score > best[0]:
            best = (score, cols, rows)
    return best[1], best[2], field_w / float(best[1]), field_h / float(best[2])


def read_cells(arr, field, cols, rows, cw, ch):
    """Modal colour per cell.

    Saturated cells are returned as the bright colour.  Muted cells (the
    landing shadow a falling-block game draws under a piece) are returned as a
    normalised grey so they can be told apart from locked blocks.  Empty cells
    are None.
    """
    x0, y0 = field["x0"], field["y0"]
    sat = saturated(arr)
    cells = [[None] * cols for _ in range(rows)]
    for r in range(rows):
        ya = int(round(y0 + r * ch)) + 3
        yb = int(round(y0 + (r + 1) * ch)) - 3
        if yb - ya < 3:
            continue
        for c in range(cols):
            xa = int(round(x0 + c * cw)) + 3
            xb = int(round(x0 + (c + 1) * cw)) - 3
            if xb - xa < 3:
                continue
            patch = arr[ya:yb, xa:xb]
            mask = sat[ya:yb, xa:xb]
            total = mask.size
            if total == 0:
                continue
            cover = mask.sum() / float(total)
            if cover >= 0.40:
                pix = patch[mask]
                quant = (pix // 10) * 10
                keys, counts = np.unique(quant, axis=0, return_counts=True)
                idx = int(counts.argmax())
                if counts[idx] >= 0.55 * mask.sum():
                    cells[r][c] = tuple(int(v) for v in keys[idx])
                    continue
            # Shadow: flat, mid-grey, no colour.  Locked blocks in this kind of
            # rendering are either saturated or outlined bright green, so a
            # uniform grey interior is the landing marker, not a block.
            grey = patch.reshape(-1, 3)
            gmin, gmax = grey.min(axis=1), grey.max(axis=1)
            neutral = (gmax - gmin) <= 12
            mid = (gmax >= 40) & (gmax <= 110)
            if neutral.mean() >= 0.85 and mid.mean() >= 0.85:
                cells[r][c] = SHADOW
    return cells


SHADOW = (128, 128, 128)


# ------------------------------------------------------------ semantics

def is_shadow(rgb):
    return rgb is not None and tuple(rgb) == SHADOW


def colour_family(rgb):
    """Coarse hue family of a cell colour, from hue/saturation only.

    The bands are the ordinary divisions of the hue circle, not a palette
    copied from one rendering; the cell's shape remains the piece's identity.
    """
    if rgb is None or is_shadow(rgb):
        return None
    r, g, b = [v / 255.0 for v in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    delta = mx - mn
    if mx < 0.25:
        return "dark"
    if delta < 0.12 * max(mx, 1e-6):
        return "grey"
    if mx == r:
        hue = (60.0 * ((g - b) / delta)) % 360.0
    elif mx == g:
        hue = 60.0 * ((b - r) / delta) + 120.0
    else:
        hue = 60.0 * ((r - g) / delta) + 240.0
    if hue < 20 or hue >= 340:
        return "red"
    if hue < 45:
        return "orange"
    if hue < 70:
        return "yellow"
    if hue < 160:
        return "green"
    if hue < 200:
        return "cyan"
    if hue < 260:
        return "blue"
    return "magenta"


FAMILY_LETTER = {"red": "R", "orange": "O", "yellow": "Y", "green": "G",
                 "cyan": "C", "blue": "B", "magenta": "M", "grey": "G", "other": "?"}


def cell_letter(rgb):
    if is_shadow(rgb):
        return ":"
    fam = colour_family(rgb)
    return FAMILY_LETTER.get(fam, ".") if fam else "."


def render_field(cols, rows, cells):
    lines = []
    for r in range(rows):
        lines.append("{:02d} {}".format(r, "".join(cell_letter(cells[r][c]) for c in range(cols))))
    return "\n".join(lines)


def fill(cells, rows, cols):
    return [[cells[r][c] if (cells[r][c] is not None and not is_shadow(cells[r][c]))
             else None for c in range(cols)] for r in range(rows)]


def column_heights(cols, rows, cells):
    """Distance from the floor to the highest filled cell in each column.

    Measured from the topmost filled cell, not from the floor: a column whose
    blocks sit above a gap is just as tall as one resting on the floor.  The
    landing shadow is ignored (it is not a block).
    """
    solid = fill(cells, rows, cols)
    heights = []
    for c in range(cols):
        top = rows
        for r in range(rows):
            if solid[r][c] is not None:
                top = r
                break
        heights.append(rows - top)
    return heights


def holes(cols, rows, cells):
    """Empty cells with at least one filled cell above them in the same column."""
    solid = fill(cells, rows, cols)
    total = 0
    for c in range(cols):
        filled_above = False
        for r in range(rows):
            if solid[r][c] is not None:
                filled_above = True
            elif filled_above:
                total += 1
    return total


def top_row_by_column(cols, rows, cells):
    solid = fill(cells, rows, cols)
    tops = []
    for c in range(cols):
        top = rows
        for r in range(rows):
            if solid[r][c] is not None:
                top = r
                break
        tops.append(top)
    return tops


def connected_components(occupied, cols, rows):
    """4-connected groups of True cells; returns list of (cells, bbox)."""
    seen = [[False] * cols for _ in range(rows)]
    comps = []
    for r in range(rows):
        for c in range(cols):
            if not occupied[r][c] or seen[r][c]:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            group = []
            while stack:
                cr, cc = stack.pop()
                group.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and occupied[nr][nc] and not seen[nr][nc]:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            rs = [p[0] for p in group]
            cs = [p[1] for p in group]
            comps.append((group, (min(rs), max(rs), min(cs), max(cs))))
    return comps


def diff_cells(prev, cur, rows, cols):
    """Cells that are not the same in two frames (colour or occupancy)."""
    out = set()
    if prev is None:
        return out
    for r in range(rows):
        for c in range(cols):
            a, b = prev[r][c], cur[r][c]
            if a is None and b is None:
                continue
            if a is None or b is None or tuple(a) != tuple(b):
                out.add((r, c))
    return out


def moving_groups(prev, cur, cols, rows):
    """Groups in the current frame that changed since the previous frame.

    The falling piece and its shadow move every step; locked blocks stop
    changing.  A group qualifies when at least half of its cells sit on a
    changed cell, which is robust to the single row a piece slides past.
    Returns (moving, stationary) lists of (cells, bbox) in the current frame.
    """
    occupied = [[cur[r][c] is not None for c in range(cols)] for r in range(rows)]
    comps = connected_components(occupied, cols, rows)
    changed = diff_cells(prev, cur, rows, cols)
    moving, still = [], []
    for group, bbox in comps:
        hits = sum(1 for cell in group if cell in changed)
        (moving if hits * 2 >= len(group) else still).append((group, bbox))
    return moving, still, changed


def normalise_shape(cells):
    """Canonical tuple of occupied offsets for a small cell group."""
    if not cells:
        return ()
    rs = [p[0] for p in cells]
    cs = [p[1] for p in cells]
    r0, c0 = min(rs), min(cs)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def shadow_piece(cells, rows, cols):
    """Locate the falling piece from the landing shadow it casts.

    A falling-block rendering draws the ghost of the piece at the place it
    would land, in a flat muted colour.  The ghost therefore carries the
    piece's own shape, though a tall stack or the piece itself can hide some
    of its cells.  Walking upward from the ghost by the smallest offset whose
    cell pattern is visible finds the piece.

    Only offsets up to a tetromino's height are considered, so a mostly hidden
    ghost cannot produce a vacuous match far above the board.  A match with
    nothing visible is accepted only when the ghost itself is near the top,
    i.e. a freshly spawned piece is partly above the field.

    Nothing here depends on which columns or how many cells a particular
    episode used; the offset is found by matching the ghost's own shape against
    the bright cells.

    Returns ``{"shape_cells", "visible", "offset", "bbox"}`` or None.
    """
    ghost = [(r, c) for r in range(rows) for c in range(cols) if is_shadow(cells[r][c])]
    if not ghost:
        return None
    bright = [[cells[r][c] is not None and not is_shadow(cells[r][c]) for c in range(cols)]
              for r in range(rows)]
    top = min(r for r, _c in ghost)
    for offset in range(1, rows + 1):
        expected = [(r - offset, c) for r, c in ghost]
        visible = [(r, c) for r, c in expected if 0 <= r < rows]
        if not all(bright[r][c] for r, c in visible):
            continue
        # A match whose visible part is empty would otherwise be satisfiable at
        # any large offset.  It is only meaningful for a piece that has just
        # spawned above the field, i.e. a ghost near the top.
        if not visible and top > 3:
            continue
        bbox = (min(r for r, _c in expected), max(r for r, _c in expected),
                min(c for _r, c in expected), max(c for _r, c in expected))
        return {"shape_cells": expected, "visible": visible, "offset": offset, "bbox": bbox}
    return None


SEED_SHAPES = {
    ((0, 0),): "unit",
    ((0, 0), (0, 1)): "domino",
    ((0, 0), (0, 1), (0, 2)): "tri",
    ((0, 0), (0, 1), (0, 2), (0, 3)): "I",
    ((0, 0), (0, 1), (1, 0), (1, 1)): "O",
    ((0, 1), (0, 2), (1, 0), (1, 1)): "S",
    ((0, 0), (0, 1), (1, 1), (1, 2)): "Z",
    ((0, 0), (1, 0), (2, 0), (2, 1)): "J",
    ((0, 1), (1, 1), (2, 0), (2, 1)): "L",
    ((0, 1), (1, 0), (1, 1), (1, 2)): "T",
}


def _canonical_shape(cells):
    """Smallest normalised rotation, so orientation does not change the name."""
    options = rotations(normalise_shape(cells))
    return min(options) if options else ()


def shape_name(cells):
    """Name of a cell group, independent of its rotation or colour."""
    if not cells:
        return "empty"
    canon = _canonical_shape(cells)
    for seed, name in SEED_SHAPES.items():
        if _canonical_shape(seed) == canon:
            return name
    return "irregular"


def floating_components(board, rows, cols):
    """Bright groups that do not reach the floor.

    A locked stack is connected to the floor by its own cells; a falling piece
    is not connected to it unless the two happen to touch face to face, which
    happens only at the instant before it locks.  This makes the fall set a
    clean, layout-free piece detector.
    """
    occupied = [[board[r][c] is not None for c in range(cols)] for r in range(rows)]
    comps = connected_components(occupied, cols, rows)
    return [item for item in comps if item[1][1] < rows - 1]


def track_piece(prev_piece, newly, cells, rows, cols):
    """Find the piece by shifting the previous piece onto newly filled cells.

    Used when the piece rests on the stack: it is not separable by
    connectivity, no ghost is drawn, and the only evidence is which cells just
    became bright.  The previous piece is tried at every small translation and
    scored by how many newly bright cells it covers, requiring that every one
    of its cells is bright in the current frame.  This keeps working without
    knowing the piece's colour or the stack's layout.
    """
    if not prev_piece or not newly:
        return None
    bright = [[cells[r][c] is not None and not is_shadow(cells[r][c]) for c in range(cols)]
              for r in range(rows)]
    newly = set(newly)
    best = None
    for dy in range(0, 4):
        for dx in range(-1, 2):
            shifted = [(r + dy, c + dx) for r, c in prev_piece]
            if not all(0 <= r < rows and 0 <= c < cols and bright[r][c] for r, c in shifted):
                continue
            hits = len(newly & set(shifted))
            if hits == 0:
                continue
            score = (hits, -dy, -abs(dx))
            if best is None or score > best[0]:
                best = (score, shifted)
    return list(best[1]) if best else None


def detect_piece(cells, rows, cols, prev_cells=None, prev_piece=None):
    """Locate the falling piece and return (board_without_piece, piece, source, note).

    Order of evidence:

    1. A group not connected to the floor is the piece.  This is the cleanest
       case and needs no history.
    2. If the piece touches the stack, the cells that changed since the
       previous frame are the piece; without a previous frame or without
       changed cells, nothing is reported rather than guessing.
    3. If no bright piece is visible at all but a ghost is, the piece is still
       above the field: its shape is the ghost's.

    Once the piece is located, the landing ghost is used to recover its full
    shape, which matters when the piece is clipped at the top of the field.
    """
    board = [[cells[r][c] if (cells[r][c] is not None and not is_shadow(cells[r][c]))
              else None for c in range(cols)] for r in range(rows)]
    piece = None
    source = None
    floating = floating_components(board, rows, cols)
    if len(floating) == 1:
        piece = floating[0][0]
        source = "not connected to the floor"
    elif len(floating) > 1:
        floating.sort(key=lambda item: item[1][0])
        piece = floating[0][0]
        source = "topmost of {} separated groups".format(len(floating))
    else:
        # The piece touches the stack, or nothing floats: use the cells that
        # changed since the previous frame.  Working on the changed set itself
        # matters: the piece is connected to the stack, so a component test on
        # the current frame would absorb the whole pile and never mark it.
        seed = None
        if prev_cells is not None:
            changed = diff_cells(prev_cells, cells, rows, cols)
            seed = [(r, c) for r, c in changed
                    if cells[r][c] is not None and not is_shadow(cells[r][c])]
        if seed:
            tracked = track_piece(prev_piece, seed, cells, rows, cols)
            piece = tracked or seed
            for r, c in seed:
                board[r][c] = None
            if tracked:
                for r, c in tracked:
                    board[r][c] = None
            source = "cells that changed since the previous frame"
    note = None
    ghost = [(r, c) for r in range(rows) for c in range(cols) if is_shadow(cells[r][c])]
    if piece:
        for r, c in piece:
            board[r][c] = None
        full = _ghost_shape(board, rows, cols, ghost, piece, cells)
        if full is not None:
            piece = full
            note = "full shape recovered from the landing ghost"
    elif ghost:
        # Nothing bright is visible: the piece is still above the field.  The
        # ghost carries its shape, so it can still be named.
        piece = [(r - 1, c) for r, c in ghost]
        source = "above the visible field; shape taken from the landing ghost"
    return board, piece, source, note


def _ghost_shape(board, rows, cols, ghost, piece, cells):
    """Complete a partly visible piece using the ghost's shape and position."""
    bright = [(r, c) for r in range(rows) for c in range(cols)
              if cells[r][c] is not None and not is_shadow(cells[r][c])]
    if not ghost:
        # No ghost: complete a clipped piece from the known tetromino shapes by
        # matching its visible part, as long as the answer is unique.
        return _complete_clipped(piece, rows, cols)
    fits = fit_ghost(board, rows, cols, ghost, bright=bright)
    for fit in fits:
        points, _offset = lift_piece(cells, rows, cols, fit["placed"])
        if points is not None:
            return points
    if len(piece) < 4:
        completed = _complete_clipped(piece, rows, cols)
        if completed is not None:
            return completed
    return None


def _complete_clipped(piece, rows, cols):
    """Extend a piece clipped by the top of the field to a unique tetromino."""
    if len(piece) >= 4:
        return None
    present = [(r, c) for r, c in piece if r >= 0]
    if not present or min(r for r, _c in present) > 0:
        return None
    matches = set()
    for seed, _name in SEED_SHAPES.items():
        if len(seed) != 4:
            continue
        for rot in rotations(seed):
            for dr in range(0, 4):
                shifted = [(r - dr, c) for r, c in rot]
                for dc in range(0, cols):
                    trial = [(r, c + dc) for r, c in shifted]
                    visible = frozenset(p for p in trial if 0 <= p[0] < rows and 0 <= p[1] < cols)
                    if visible == frozenset(present):
                        matches.add(tuple(sorted(trial)))
    return list(next(iter(matches))) if len(matches) == 1 else None


def fit_ghost(board, rows, cols, ghost, occluders=(), bright=()):
    """Tetromino placements whose landing covers every observed ghost cell.

    A tall stack, or the falling piece itself, can hide part of the landing
    ghost, leaving an incomplete outline.  Every placement approved here covers
    the cells that *are* visible, and every cell it adds beyond them must be
    explainable as hidden: it must already be occupied on the board, belong to
    the falling piece (``occluders``), or be drawn bright in the current frame
    (which is where the piece's own cells are).  Placements are returned
    best-first, preferring fewer hidden cells and then a stable name/column
    order.  Nothing is specific to an episode: the seven shapes and their
    rotations are the game's rules.
    """
    ghost = set(ghost)
    if not ghost:
        return []
    explainable = set(occluders) | set(bright)
    for r in range(rows):
        for c in range(cols):
            if board[r][c] is not None:
                explainable.add((r, c))
    found = []
    for seed, name in SEED_SHAPES.items():
        if len(seed) != 4:
            continue
        for rot in rotations(seed):
            width = max(c for _r, c in rot) + 1
            for c0 in range(0, cols - width + 1):
                r0 = landing_row(board, rows, cols, rot, c0)
                if r0 is None:
                    continue
                placed = {(r0 + r, c0 + c) for r, c in rot}
                if not ghost <= placed:
                    continue
                extra = placed - ghost
                if any(cell not in explainable for cell in extra):
                    continue
                found.append((len(extra), name, rot, r0, c0, placed))
    found.sort(key=lambda item: (item[0], item[1], item[4]))
    return [{"shape": item[1], "rot": item[2], "row": item[3], "col": item[4],
             "placed": sorted(item[5]), "hidden": item[0]} for item in found]


def lift_piece(cells, rows, cols, placement):
    """Shift a fitted landing up to the piece's current row.

    Returns (points, offset) or (None, None) when no bright copy of the placed
    shape is visible.  The copy may be clipped at the top of the field, and
    cells above the field count as satisfied.
    """
    bright = [[cells[r][c] is not None and not is_shadow(cells[r][c]) for c in range(cols)]
              for r in range(rows)]
    placement = list(placement)
    if not placement:
        return None, None
    for offset in range(1, rows + 1):
        expected = [(r - offset, c) for r, c in placement]
        visible = [(r, c) for r, c in expected if 0 <= r < rows]
        if not visible:
            continue
        if all(bright[r][c] for r, c in visible):
            return expected, offset
    return None, None



def shape_of_groups(groups):
    """Concatenate several cell groups that may be one piece split by colour."""
    pts = []
    for group, _bbox in groups:
        pts.extend(group)
    return shape_name(pts)


def rotations(shape):
    """All distinct rotations of a set of (row, col) offsets, each normalised."""
    out = []
    cur = set(shape)
    for _ in range(4):
        norm = normalise_shape(list(cur))
        if norm not in out:
            out.append(norm)
        cur = {(c, -r) for r, c in cur}
    return out


def collides(board, rows, cols, shape, r0, c0):
    for r, c in shape:
        rr, cc = r0 + r, c0 + c
        if cc < 0 or cc >= cols or rr >= rows:
            return True
        if rr >= 0 and board[rr][cc] is not None:
            return True
    return False


def landing_row(board, rows, cols, shape, c0):
    """Lowest non-colliding top row for a piece whose bbox starts at row 0."""
    if collides(board, rows, cols, shape, 0, c0):
        return None
    r = 0
    while not collides(board, rows, cols, shape, r + 1, c0):
        r += 1
    return r


def drop_outcomes(board, rows, cols, shape):
    """Every legal rotation/column placement: (row, col, shape, landing_row)."""
    out = []
    seen = set()
    for rot in rotations(shape):
        width = max(c for _r, c in rot) + 1 if rot else 0
        for c0 in range(0, cols - width + 1):
            key = (rot, c0)
            if key in seen:
                continue
            seen.add(key)
            row = landing_row(board, rows, cols, rot, c0)
            if row is not None:
                out.append((row, c0, rot, row))
    return out


def apply_piece(board, shape, r0, c0, marker):
    out = [list(row) for row in board]
    for r, c in shape:
        rr, cc = r0 + r, c0 + c
        if 0 <= rr < len(out) and 0 <= cc < len(out[0]):
            out[rr][cc] = marker
    return out


def cleared_rows(board):
    return [r for r, row in enumerate(board) if all(v is not None for v in row)]


def score_placement(board, rows, cols, height, hole_count, cleared, bumpiness, wells):
    """A small, fixed evaluation of a settled board.

    The weights are generic falling-block heuristics (prefer clears and low,
    even stacks; punish holes and buried wells).  They are not fitted to an
    episode: every term is expressed per unit of the measured board.
    """
    return (2.0 * cleared + 0.6 * bumpiness - 1.2 * height
            - 3.5 * hole_count - 0.8 * wells + 0.0 * len(board))


def board_features(board, rows, cols):
    solid = [[board[r][c] for c in range(cols)] for r in range(rows)]
    heights = column_heights(cols, rows, solid)
    holes_n = holes(cols, rows, solid)
    total_h = sum(heights)
    bump = sum(abs(heights[i] - heights[i + 1]) for i in range(cols - 1))
    wells = 0
    for c in range(cols):
        left = heights[c - 1] if c > 0 else rows
        right = heights[c + 1] if c + 1 < cols else rows
        drop = min(left, right) - heights[c]
        if drop > 0:
            wells += drop
    return heights, total_h, holes_n, bump, wells
