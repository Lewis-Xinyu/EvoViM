"""Derived placement table and lock ledger.

Every number here is computed from the pixels the host supplied for one frame
(the legal drops of the falling piece) or from the two frames around one hard
drop (the observed fall of a locked piece).  Nothing is learned from an episode
and no weight is fitted.  The only game rule used is the one verified in the
evidence: a hard drop scores 2 points per row the piece falls.

The table ranks drops in this order:
  1. danger: the stack after the drop reaches the spawn region;
  2. covered holes created (a hole cannot clear until the cells above it do);
  3. points per key press (points / (rotations + sideways moves + hard drop)),
     which is the rate that matters when the budget is a number of key presses;
  4. lower maximum column height, then fewer keys, then leftmost column.
"""
from __future__ import annotations

from agents import tetris_vision as vision

POINTS_PER_ROW = 2   # verified: each hard drop scores 2 points per row fallen
DANGER_MARGIN = 4    # a 4-row piece needs four free rows at the top to spawn


def _cw(cells):
    return {(c, -r) for r, c in cells}


def cw_turns(current, target):
    """Clockwise quarter turns that turn `current` into `target`; None if none."""
    target = vision.normalise_shape([tuple(p) for p in target])
    cur = set(tuple(p) for p in current)
    for k in range(4):
        if vision.normalise_shape(list(cur)) == target:
            return k
        cur = _cw(cur)
    return None


def rotation_keys(turns):
    """Key presses needed: one rotate_ccw is the short way for three clockwise turns."""
    if turns is None:
        return None
    return min(turns, 4 - turns) % 4


def _rot_text(turns):
    return {0: "as spawned", 1: "cw 1", 2: "180", 3: "ccw 1"}.get(turns, "?")


def _rank_key(entry):
    return (entry["danger"], entry["new_holes"] > 0, -entry["points_per_key"],
            entry["max_height"], entry["keys"], entry["col"])


def placement_table(stack, rows, cols, points):
    """All legal hard drops of the falling piece, each with derived consequences."""
    if not points:
        return []
    shape = vision.normalise_shape([tuple(p) for p in points])
    top = min(r for r, _c in points)
    left = min(c for _r, c in points)
    holes_before = vision.holes(cols, rows, stack)
    entries = []
    for land, col, rot, _same in vision.drop_outcomes(stack, rows, cols, shape):
        turns = cw_turns(shape, rot)
        rot_keys = rotation_keys(turns)
        if rot_keys is None:
            continue
        move_keys = abs(col - left)
        keys = rot_keys + move_keys + 1
        fall = land - top
        board = vision.apply_piece(stack, rot, land, col, "X")
        full = vision.cleared_rows(board)
        kept = [row for r, row in enumerate(board) if r not in full]
        while len(kept) < rows:
            kept.insert(0, [None] * cols)
        heights, total_h, hole_n, _bump, _wells = vision.board_features(kept, rows, cols)
        max_h = max(heights) if heights else 0
        entries.append({
            "col": col, "turns": turns, "rot_keys": rot_keys, "move_keys": move_keys,
            "keys": keys, "fall": fall, "points": POINTS_PER_ROW * fall,
            "clears": len(full), "new_holes": hole_n - holes_before,
            "max_height": max_h, "total_height": total_h,
            "danger": max_h >= rows - DANGER_MARGIN,
            "points_per_key": round(POINTS_PER_ROW * fall / keys, 3),
            "rot": [list(p) for p in rot],
        })
    entries.sort(key=_rank_key)
    return entries


def format_row(rank, e):
    return ("#{rank} col {col} {rot} keys {keys} (rotate {rk}, sideways {mk}, drop 1) "
            "fall {fall} rows = {pts} points; new covered holes {nh:+d}; clears {cl}; "
            "max height {mh}{danger}; points per key {ppk:.2f}").format(
                rank=rank, col=e["col"], rot=_rot_text(e["turns"]), keys=e["keys"],
                rk=e["rot_keys"], mk=e["move_keys"], fall=e["fall"], pts=e["points"],
                nh=e["new_holes"], cl=e["clears"], mh=e["max_height"],
                danger=" DANGER" if e["danger"] else "", ppk=e["points_per_key"])


def summary(entries):
    if not entries:
        return "no legal drop was found for the falling piece"
    free = sum(1 for e in entries if not e["danger"] and e["new_holes"] <= 0)
    danger = sum(1 for e in entries if e["danger"])
    in_place = [e for e in entries if e["keys"] == 1]
    text = "legal drops {}; hole-free and below danger {}; danger {}".format(
        len(entries), free, danger)
    if in_place:
        e = in_place[0]
        text += "; drop in place (1 key): fall {} rows, new covered holes {:+d}".format(
            e["fall"], e["new_holes"])
    return text


def format_table(entries, limit=5):
    lines = [format_row(i, e) for i, e in enumerate(entries[:limit], 1)]
    return summary(entries) + ("\n" + "\n".join(lines) if lines else "")


def _clearing_landing(points, before, after_full, next_piece, rows, cols):
    """Fallback for a drop that cleared rows: the landing whose simulated board, after
    the full rows are removed, has the same occupancy as the frame.  Cells of the
    newly spawned piece are ignored, since they are not part of the stack."""
    skip = {tuple(p) for p in next_piece or []}
    best = None
    for d in range(0, rows + 1):
        cells = [(r + d, c) for r, c in points]
        if any(c < 0 or c >= cols or r >= rows for r, c in cells):
            break
        if any(r >= 0 and before[r][c] is not None for r, c in cells):
            break
        board = [list(row) for row in before]
        for r, c in cells:
            if r >= 0:
                board[r][c] = "P"
        full = [r for r, row in enumerate(board) if all(v is not None for v in row)]
        if not full:
            continue
        kept = [row for r, row in enumerate(board) if r not in full]
        while len(kept) < rows:
            kept.insert(0, [None] * cols)
        landed = set(cells)
        same = True
        for r in range(rows):
            for c in range(cols):
                if (r, c) in skip and (r, c) not in landed:
                    continue
                if (kept[r][c] is None) != (after_full[r][c] is None):
                    same = False
                    break
            if not same:
                break
        if same:
            best = d
    return best


def observed_fall(points, before, after, rows, cols, next_piece=None):
    """Distance the piece fell, read from the two frames, and why it may not count.

    Walks the piece's cells down while every cell is still empty in `before`
    (the frame before the drop).  The last such distance whose cells are all
    filled in `after` (the frame after the drop) is the landing.  A landing
    counts as a hard drop only if the piece is then at rest: one more row down
    is blocked.  A piece that moved but is still free to fall (a gravity tick,
    or a key the game ignored) is not a lock, so it earns no points.
    Returns (fall, reason); fall is None when no lock is confirmed.
    """
    best = None
    for d in range(0, rows + 1):
        cells = [(r + d, c) for r, c in points]
        if any(c < 0 or c >= cols or r >= rows for r, c in cells):
            break
        if any(r >= 0 and before[r][c] is not None for r, c in cells):
            break
        inside = [(r, c) for r, c in cells if r >= 0]
        if inside and all(after[r][c] is not None for r, c in inside):
            best = d
    if best is None:
        best = _clearing_landing(points, before, after, next_piece, rows, cols)
    if best is None:
        return None, "no landing confirmed by the frame after the drop"
    rests = any(r + best + 1 >= rows or (r + best + 1 >= 0 and before[r + best + 1][c] is not None)
                for r, c in points)
    if not rests:
        return None, "the piece moved but did not come to rest (a gravity step or an ignored key, not a hard drop)"
    return best, None


def _filled(board):
    return sum(1 for row in board for v in row if v is not None)


def lock_outcome(step, action, points, shape, before, after_full, rows, cols, predicted_fall,
                 next_piece=None, prev_action=None):
    """One ledger row for a hard drop, built only from the frames around it.

    `after_full` is the measured stack after the drop with nothing removed.
    `next_piece` is the falling piece detected in that frame; its cells are
    taken out of the stack only where they do not coincide with the landing,
    so a detector that mistakes the landed piece for the new one cannot erase it.
    """
    fall, why = observed_fall(points, before, after_full, rows, cols, next_piece)
    landed = set() if fall is None else {(r + fall, c) for r, c in points}
    after = [list(row) for row in after_full]
    for r, c in next_piece or []:
        if (r, c) not in landed and 0 <= r < rows and 0 <= c < cols:
            after[r][c] = None
    diff = _filled(before) + len(points) - _filled(after)
    clears = diff // cols if diff >= 0 and diff % cols == 0 else None
    heights = vision.column_heights(cols, rows, after)
    if fall is not None:
        reason = None
    elif before == after_full:
        reason = "the board did not change: no falling piece was visible to drop"
    else:
        reason = why
    return {
        "step": step, "action": action, "shape": shape,
        "prev_action": prev_action,
        "fall": fall, "points": None if fall is None else POINTS_PER_ROW * fall,
        "predicted_fall": predicted_fall,
        "match": fall is not None and predicted_fall is not None and fall == predicted_fall,
        "clears": clears,
        "new_holes": vision.holes(cols, rows, after) - vision.holes(cols, rows, before),
        "max_height": max(heights) if heights else 0,
        "reason": reason,
    }


def format_lock(row):
    if row["fall"] is None:
        return ("Your previous action (step {step}) was a hard drop, but the lock could not be "
                "verified: {reason}.").format(step=row["step"], reason=row["reason"])
    pred = row["predicted_fall"]
    check = ("the drop-in-place prediction was {} rows: {}".format(
        pred, "exact" if row["match"] else "different")) if pred is not None else \
        "no drop-in-place prediction was available"
    return ("Your previous action (step {step}) was a hard drop of a {shape} piece. Measured fall "
            "{fall} rows = {pts} points by the verified rule (2 per row); {check}. Covered holes "
            "{nh:+d}; rows cleared {cl}; max column height {mh}.").format(
                step=row["step"], shape=row["shape"], fall=row["fall"], pts=row["points"],
                check=check, nh=row["new_holes"],
                cl="unknown" if row["clears"] is None else row["clears"], mh=row["max_height"])


def repeat_groups(rows_list):
    """Group hard drops by what the previous drop did, from this episode's ledger only.

    A drop is grouped by the previous action.  When that action was a hard drop, the
    group depends on whether that earlier drop was measured and locked:
      - "after_unlocked": the earlier drop was measured and did not lock (same piece still falling);
      - "after_locked": the earlier drop was measured and locked;
      - unmeasured earlier drops are left out, since their outcome is unknown.
    Drops with no known previous action (the first action of the episode) are left out.
    """
    by_step = {r["step"]: r for r in rows_list}
    groups = {"after_unlocked": [], "after_locked": [], "after_other": []}
    for r in rows_list:
        prev = r.get("prev_action")
        if prev is None:
            continue
        if prev != "hard_drop":
            groups["after_other"].append(r)
            continue
        earlier = by_step.get(r["step"] - 1)
        if earlier is None or not earlier.get("measured", True):
            continue
        key = "after_unlocked" if earlier["fall"] is None else "after_locked"
        groups[key].append(r)
    return groups


def repeat_text(rows_list):
    """One clause per non-empty group, e.g. 'hard drops pressed right after a hard drop that did not lock locked 0 of 5'."""
    groups = repeat_groups(rows_list)
    labels = [
        ("after_unlocked", "hard drops pressed right after a hard drop that did not lock"),
        ("after_locked", "hard drops pressed right after a hard drop that locked"),
        ("after_other", "hard drops made after any other action"),
    ]
    parts = []
    for key, label in labels:
        group = groups[key]
        if group:
            parts.append("{} locked {} of {}".format(
                label, sum(1 for r in group if r["fall"] is not None), len(group)))
    return "; ".join(parts)


def calibration(rows_list):
    verified = [r for r in rows_list if r["fall"] is not None]
    exact = [r for r in verified if r["match"]]
    total = sum(r["points"] or 0 for r in rows_list)
    text = ("locks recorded {}; landing verified for {}; drop-in-place prediction exact for {}; "
            "points by the rule so far {}").format(len(rows_list), len(verified), len(exact), total)
    split = repeat_text(rows_list)
    if split:
        text += "; " + split
    return text
