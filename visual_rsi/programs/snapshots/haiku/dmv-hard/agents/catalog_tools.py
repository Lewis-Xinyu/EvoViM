"""LLM-invoked tools over the code-maintained page catalog.

The catalog indexes what the agent already observed; these tools let the LLM
decide when to search it, when to run pixel matching, which candidates to look
at, and what appearance notes to keep.  Storage itself stays in page_catalog.py
and in the agent's memory files.

Parameter keys are matched case-insensitively and required parameters are checked
here, not by the host schema check.  Models sometimes write JSON keys in capitals
(e.g. "PRODUCT_ID"); the host would otherwise reject the whole call as an unknown
parameter before the handler runs.  The tool still receives exactly the same
information, and missing or invalid values get a plain-text message the model can
act on.
"""
from __future__ import annotations

import inspect

import numpy as np

from registry import Tool
from agents.page_catalog import content_box, content_crop

MAX_VIEW = 4
TILE_MAX_SIDE = 384
# ALIGN_PAGES: both sides of each aligned pair are fitted to this many pixels on the
# longer side.  The host only shrinks images, so smaller ones keep their native size.
MAX_ALIGN = 4
ALIGN_SIDE = 160


def _catalog(agent):
    catalog = getattr(agent, "catalog", None)
    if catalog is None:
        raise ValueError("page catalog is not configured")
    return catalog


def _tolerant(tool_name, fn):
    """Wrap a handler so keys are case-insensitive and required keys are checked here."""
    params = inspect.signature(fn).parameters
    required = [name for name, p in params.items()
                if name != "agent" and p.default is inspect.Parameter.empty]

    def wrapper(**raw):
        kwargs = {}
        for key, value in raw.items():
            name = str(key).lower()
            if name in params and name != "agent":
                kwargs[name] = value
        missing = [name for name in required if name not in kwargs]
        if missing:
            return ("{}: missing parameter(s) {}; call again with lowercase JSON keys, e.g. {{\"{}\": ...}}."
                    ).format(tool_name, ", ".join(missing), missing[0])
        return fn(**kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


def install_catalog(agent, registry):
    def query(scope="recent", session=None, category=None, product_id=None, limit=40):
        catalog = _catalog(agent)
        limit = max(1, min(int(limit), 100))
        if scope == "session":
            if session is None:
                return ("CATALOG_QUERY: scope 'session' needs the session number; "
                        "call again with session=<1-5>, or use scope 'recent'.")
            return catalog.render(session=int(session), limit=limit)
        if scope == "category":
            if not category:
                return ("CATALOG_QUERY: scope 'category' needs the category name; "
                        "read a page line first or use scope 'recent'.")
            return catalog.render(category=str(category), limit=limit)
        if scope == "product":
            if not product_id:
                return ("CATALOG_QUERY: scope 'product' needs an 8-hex product id; "
                        "use scope 'recent' to see ids.")
            entry = catalog.get(str(product_id))
            if entry is None:
                return "PAGE CATALOG: {} is not indexed yet.".format(product_id)
            answered = catalog.answered_index().get(str(product_id).lower())
            return "\n".join(["PAGE CATALOG: one entry",
                              "  " + catalog._line(entry) + catalog._answer_tag(answered),
                              "  note step: {} | confidence: {}".format(
                                  entry.get("note_step"), entry.get("confidence"))])
        if scope == "pool":
            question = catalog.question_for(agent.step_count)
            after = question["after_session"] if question else None
            rows = catalog.pool(after_session=after, before_step=agent.step_count)
            head = "ELIGIBLE POOL: {} page(s) from sessions earlier than {}".format(
                len(rows), after if after else "the current one")
            return "\n".join([head] + ["  " + catalog._line(r) for r in rows[-limit:]])
        if scope == "all":
            return catalog.render(limit=100)
        return catalog.render(limit=limit)

    def match_current(after_session=None, sessions=None, rotation=False, top_k=10, min_score=None):
        catalog = _catalog(agent)
        step = agent.step_count
        question = catalog.question_for(step)
        session_filter = None
        if after_session is None and question:
            after_session = question["after_session"]
        if sessions:
            session_filter = [int(s) for s in sessions]
        try:
            results, per_session = catalog.match(
                getattr(agent, "current_image", None),
                after_session=None if after_session is None else int(after_session),
                before_step=step,
                sessions=session_filter,
                rotation=bool(rotation),
                top_k=int(top_k) if top_k else 10,
                min_score=min_score,
            )
        except ValueError as exc:
            # An empty pool (e.g. before any page is indexed) is a normal state,
            # not a tool failure; tell the model what to do instead.
            return "IMAGE MATCH: {}. Nothing to rank yet; keep observing pages.".format(exc)
        pool_size = len(catalog.pool(after_session=after_session, before_step=step))
        if session_filter:
            pool_size = sum(1 for r in catalog.pool(after_session=after_session, before_step=step)
                            if r.get("session") in set(session_filter))
        agent.catalog_last_matches = [item["entry"]["product_id"] for item in results[:MAX_VIEW]]
        return catalog.render_matches(results, per_session, pool_size, question=question,
                                       limit=max(1, min(int(top_k) if top_k else 10, 25)))

    def view_pages(product_ids=None):
        catalog = _catalog(agent)
        if isinstance(product_ids, str):
            product_ids = [product_ids]
        if not product_ids:
            product_ids = list(getattr(agent, "catalog_last_matches", []) or [])[:MAX_VIEW]
            if not product_ids:
                recent = catalog.entries()[-MAX_VIEW:]
                product_ids = [entry["product_id"] for entry in recent]
        if not product_ids:
            return ("VIEW_PAGES: no catalog pages exist yet, so there is nothing to render. "
                    "Keep observing product pages first.")
        ops = []
        sources = []
        labels = []
        for index, product_id in enumerate(product_ids[:MAX_VIEW], 1):
            entry = catalog.get(str(product_id))
            if entry is None:
                return "VIEW_PAGES: {} is not indexed yet.".format(product_id)
            frame_id = entry.get("frame_id")
            if not frame_id:
                return ("VIEW_PAGES: no host frame was recorded for {} (it was indexed without pixels); "
                        "pick another candidate.".format(product_id))
            src = "f{}".format(index)
            ops.append({"op": "frame", "id": frame_id, "as": src})
            ops.append({"op": "resize", "src": src, "max_side": TILE_MAX_SIDE, "as": src + "r"})
            sources.append(src + "r")
            labels.append("{} {}".format(index, product_id))
        cols = 2 if len(sources) > 1 else 1
        ops.append({"op": "tile", "srcs": sources, "cols": cols, "labels": labels, "as": "out"})
        if len(ops) > 32:
            raise ValueError("visual program too long")
        agent.visual_request = {"ops": ops, "output": "out"}
        return ("Attached next: a labelled tile of {} catalog page(s) rendered by the host from its own "
                "frames: {}. Compare it with the current observation image and the match scores."
                ).format(len(sources), ", ".join(str(p) for p in product_ids[:MAX_VIEW]))

    def align_pages(product_ids=None):
        catalog = _catalog(agent)
        probe = getattr(agent, "current_image", None)
        frame = getattr(agent, "current_frame_id", None)
        if probe is None or not frame:
            return ("ALIGN_PAGES: the current image is not available as a host frame, so it cannot be cut "
                    "out beside the pages; use VIEW_PAGES instead.")
        if isinstance(product_ids, str):
            product_ids = [product_ids]
        if not product_ids:
            product_ids = list(getattr(agent, "catalog_last_matches", []) or [])[:MAX_ALIGN]
        if not product_ids:
            return "ALIGN_PAGES: no candidates yet; run MATCH_CURRENT_IMAGE first or name up to four product_ids."
        gray = np.asarray(probe.convert("L"), dtype=np.float64)
        content = content_crop(gray)
        if content.shape[0] < 12 or content.shape[1] < 12:
            return "ALIGN_PAGES: the current image has no usable content to align."
        cx0, cy0, cx1, cy1 = content_box(gray)
        ops = [{"op": "frame", "id": frame, "as": "probe_frame"}]
        sources, labels, skipped, scored = [], [], [], []
        for index, product_id in enumerate(product_ids[:MAX_ALIGN], 1):
            found = catalog.window(str(product_id), content)
            if found.get("box") is None:
                skipped.append("{} ({})".format(product_id, found.get("reason", "no window")))
                continue
            x0, y0, x1, y1 = found["box"]
            win_w, win_h = x1 - x0, y1 - y0
            # Both sides get the same display size: the smaller native long side, capped.
            # resize only shrinks, so neither side is ever enlarged.
            side = max(1, min(ALIGN_SIDE, max(cx1 - cx0, cy1 - cy0), max(win_w, win_h)))
            probe_alias, window_alias = "pc{}".format(index), "wc{}".format(index)
            page_alias = "page{}".format(index)
            ops.append({"op": "crop", "src": "probe_frame", "box": [cx0, cy0, cx1, cy1], "as": probe_alias})
            ops.append({"op": "resize", "src": probe_alias, "max_side": side, "as": probe_alias + "r"})
            ops.append({"op": "frame", "id": found["frame_id"], "as": page_alias})
            ops.append({"op": "crop", "src": page_alias, "box": [x0, y0, x1, y1], "as": window_alias})
            ops.append({"op": "resize", "src": window_alias, "max_side": side, "as": window_alias + "r"})
            sources += [probe_alias + "r", window_alias + "r"]
            labels += ["probe", "{} {:.3f}".format(product_id, found["score"])]
            scored.append("{} {:.3f} ({}x{} px)".format(product_id, found["score"], win_w, win_h))
        if not sources:
            return "ALIGN_PAGES: no candidate could be cut out ({}).".format("; ".join(skipped) or "none given")
        ops.append({"op": "tile", "srcs": sources, "cols": 2, "labels": labels, "as": "out"})
        if len(ops) > 32:
            raise ValueError("visual program too long")
        agent.visual_request = {"ops": ops, "output": "out"}
        text = ("Attached next: aligned pairs. Each row has the current image's content region on the left and, "
                "on the right, the window of one candidate page that MATCH_CURRENT_IMAGE matched, cut out at the "
                "matched location. Both sides of a row are shown at the same display size (at most {} px). A small "
                "window is weak evidence: its label gives its size in page pixels. Window scores: {}."
                ).format(ALIGN_SIDE, ", ".join(scored))
        if skipped:
            text += " Not aligned: " + "; ".join(skipped) + "."
        return text

    def note(product_id, note=None, confidence=None):
        catalog = _catalog(agent)
        entry = catalog.get(str(product_id))
        if entry is None:
            return ("CATALOG_NOTE: {} is not indexed yet; notes are only accepted for pages you have "
                    "actually seen.".format(product_id))
        if not note:
            return ("CATALOG_NOTE: {} currently has note: {!r} (step {}, confidence {!r})."
                    ).format(product_id, entry.get("note") or "", entry.get("note_step"), entry.get("confidence"))
        if confidence is not None and str(confidence).lower() not in {"low", "medium", "high"}:
            return ("CATALOG_NOTE: confidence must be low, medium or high (got {!r}); the note was "
                    "not recorded. Call again with a valid confidence or omit it.").format(confidence)
        confidence = None if confidence is None else str(confidence).lower()
        entry = catalog.note(str(product_id), note, agent.step_count, confidence)
        return "CATALOG_NOTE: recorded for {} at step {}: {}".format(
            product_id, entry["note_step"], entry["note"])

    query = _tolerant("CATALOG_QUERY", query)
    match_current = _tolerant("MATCH_CURRENT_IMAGE", match_current)
    view_pages = _tolerant("VIEW_PAGES", view_pages)
    align_pages = _tolerant("ALIGN_PAGES", align_pages)
    note = _tolerant("CATALOG_NOTE", note)

    # Schemas list the lowercase keys; the wrapper above accepts other casings.
    # additionalProperties is open and "required" is empty so the host check does not
    # reject a call before the handler can read it.  Required keys are enforced in the
    # wrapper and reported as text.
    obj = {"type": "object", "additionalProperties": True}
    registry.register(Tool(
        "CATALOG_QUERY",
        "read the product-page catalog built from pages you have seen: recent pages, one session, one "
        "category, one product, the pool eligible for the current recall question, or everything",
        {**obj, "properties": {
            "scope": {"type": "string", "enum": ["recent", "session", "category", "product", "pool", "all"]},
            "session": {"type": "integer", "minimum": 1},
            "category": {"type": "string"},
            "product_id": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100}},
         "required": []},
        query, ("LOG", "PLAN"), "memory_read", "text"))
    registry.register(Tool(
        "MATCH_CURRENT_IMAGE",
        "rank earlier-session product pages by pixel similarity to the current image; use this on a recall "
        "question, then confirm the top candidates with VIEW_PAGES. rotation=true also searches rotated "
        "copies of each page: slower, and it can raise false matches, so use it only as a second look",
        {**obj, "properties": {
            "after_session": {"type": "integer", "minimum": 1},
            "sessions": {"type": "array", "items": {"type": "integer", "minimum": 1}, "maxItems": 4},
            "rotation": {"type": "boolean"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 25},
            "min_score": {"type": "number", "minimum": 0.0}},
         "required": []},
        match_current, ("LOG", "PLAN"), "memory_read", "text"))
    registry.register(Tool(
        "VIEW_PAGES",
        "attach host-rendered images of up to four catalog pages, tiled and labelled with their product ids; "
        "call it with no ids to view the current match shortlist",
        {**obj, "properties": {
            "product_ids": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_VIEW}},
         "required": []},
        view_pages, ("LOG", "PLAN"), "memory_read", "image"))
    registry.register(Tool(
        "ALIGN_PAGES",
        "attach aligned pairs for up to four candidate pages: the current image's content region beside the "
        "window of that page which MATCH_CURRENT_IMAGE matched, both at the same display size; labels give the "
        "window's size in page pixels. Use it when a "
        "probe may be a magnified or cropped part of a page, where a whole-page view hides the scale. Call with "
        "no ids to align the current shortlist",
        {**obj, "properties": {
            "product_ids": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_ALIGN}},
         "required": []},
        align_pages, ("LOG", "PLAN"), "memory_read", "image"))
    registry.register(Tool(
        "CATALOG_NOTE",
        "record or correct what a catalogued product actually looks like, with an optional confidence; "
        "call with no note to read the stored note. Required: product_id",
        {**obj, "properties": {
            "product_id": {"type": "string"},
            "note": {"type": "string"},
            "confidence": {"type": "string", "description": "low, medium or high"}},
         "required": []},
        note, ("LOG", "PLAN"), "memory_write", "text"))
