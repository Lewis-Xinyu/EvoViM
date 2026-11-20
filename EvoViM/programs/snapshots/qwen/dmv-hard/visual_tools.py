"""Candidate-side visual memory tools; pixels return only through host RPC.

Line M gen 3: MATCH_PROBE is a decision engine.  On a question step it runs
three matchings of the question image against the saved candidate pages --
full page, small pattern inset, and central product photo -- each a
scale/rotation/position-invariant NCC ranking computed in code.  From the
three top-1 ids and margins it prints a DECISION (COMMIT with the exact id,
CONTESTED, or FLAT) and attaches a tile of candidates ordered by the relevant
matching (best match = cell #1).  RECALL_TILE gains a 'product' view (the
central product photo of each page) and, on a question step right after
MATCH_PROBE, follows the matching's ranking so paging walks the ranked list.

The model still commits every answer; the tools only compute, rank, validate
and show.  All thresholds are generic NCC margins, fitted to no unit.
"""
from __future__ import annotations

from pathlib import Path

from registry import Tool


def install_visual(agent, registry, scratch=None, event=None):
    from agents.visual_memory import VisualMemoryStore
    agent.visual = VisualMemoryStore(scratch or agent._memory_dir, event)
    agent.visual_request = None

    def save(label=None, note=None):
        if not agent.current_frame_id:
            raise ValueError("No host frame is available to recall later")
        info = agent.visual.save(agent.current_image, agent.step_count,
                                 label if label is not None else f"frame step {agent.step_count}",
                                 note if note is not None else "",
                                 frame_id=agent.current_frame_id)
        # Derived bookkeeping (product index, inset box, frame size) is
        # code-side computation on public observations; the decision to save
        # is the model's.  Missing on non-dmv agents: plain save.
        recorder = getattr(agent, "_record_saved_frame", None)
        if callable(recorder):
            recorder(info["id"])
        return info

    def recall(image_id):
        metadata, _ = agent.visual.recall(image_id)
        frame_id = agent.visual.data["images"][image_id].get("frame_id")
        if not frame_id:
            raise ValueError("visual memory entry has no host frame id")
        agent.visual_request = {"ops": [{"op": "frame", "id": frame_id, "as": "out"}], "output": "out"}
        return metadata

    def recall_tile(view=None, sessions=None, ids=None, offset=0):
        builder = getattr(agent, "_build_tile_request", None)
        if not callable(builder):
            raise ValueError("RECALL_TILE is only available in product-browsing episodes")
        return builder(view=view, sessions=sessions, ids=ids, offset=offset)

    def match_probe(**kwargs):
        # g3: permissive schema + **kwargs so stray keys (e.g. {"keyword":""})
        # no longer produce protocol/tool errors; the host decoder still
        # requires a JSON object, so the playbook says to pass {}.
        info = getattr(agent, "_current_dmv_info", None) or {}
        if info.get("kind") != "probe":
            return ("MATCH_PROBE only works on a question step (the current observation is "
                    "a product page, not a question). On a page step: SAVE_FRAME if the page "
                    "is not saved, then commit continue.")
        probe = getattr(agent, "current_image", None)
        if probe is None:
            return ("MATCH_PROBE: no question image is available right now - recall a tile "
                    "or commit your best saved id.")
        from agents.inset import validate_inset_box
        from agents.probe_match import (gray_from_path, rank_ext, rank_inset,
                                        rank_product)
        pages = []
        boxes = {}
        for product_id, entry in agent._dmv_candidate_entries(None, None):
            image_id = entry.get("image_id")
            if not image_id:
                continue
            frame_path = Path(agent.visual.root) / "{}.png".format(image_id)
            if not frame_path.exists():
                continue
            gray = gray_from_path(frame_path)
            pages.append((product_id, gray))
            box = entry.get("inset_box")
            if box and validate_inset_box(box, gray.shape[1], gray.shape[0]):
                boxes[product_id] = [int(value) for value in box]
        if not pages:
            return ("MATCH_PROBE: no saved candidate pages to match against yet. "
                    "Save pages with SAVE_FRAME first, then match again.")
        ext = rank_ext(probe, pages)
        inset = rank_inset(probe, pages, boxes)
        prod = rank_product(probe, pages)

        def margin(ranked):
            if not ranked:
                return 0.0
            if len(ranked) > 1:
                return ranked[0][1] - ranked[1][1]
            return ranked[0][1]

        ext1, extm = (ext[0][0], margin(ext)) if ext else (None, 0.0)
        ins1, insm = (inset[0][0], margin(inset)) if inset else (None, 0.0)
        ins_s1 = inset[0][1] if inset else 0.0
        pr1, prm = (prod[0][0], margin(prod)) if prod else (None, 0.0)
        pr_s1 = prod[0][1] if prod else 0.0

        # Gates (checked in order).  Each commit gate requires a decisive
        # margin (>= 0.12) in its own matching AND corroboration: the inset
        # must beat the product score outright, the product must agree with
        # the full-page top-1, and the full-page top-1 must agree with one of
        # the other two matchings.  A large margin without corroboration is
        # CONTESTED, everything else FLAT.
        if (ins1 is not None and insm >= 0.12 and ins_s1 > pr_s1
                and (ins1 == ext1 or extm < 0.05)):
            decision, proposal, dec_margin = "commit_inset", ins1, insm
        elif prm >= 0.12 and pr1 == ext1:
            decision, proposal, dec_margin = "commit_prod", pr1, prm
        elif extm >= 0.12 and (ext1 == ins1 or ext1 == pr1):
            decision, proposal, dec_margin = "commit_ext", ext1, extm
        elif extm >= 0.12:
            decision, proposal, dec_margin = "contested", None, extm
        else:
            decision, proposal, dec_margin = "flat", None, extm

        if decision == "contested":
            view = "full"
        elif decision == "flat":
            if ins1 is not None and ext1 == ins1:
                view = "inset"
            elif ext1 == pr1:
                view = "product"
            else:
                view = "full"
        else:
            view = {"commit_inset": "inset", "commit_prod": "product",
                    "commit_ext": "full"}[decision]

        agent._dmv_match = {
            "step": getattr(agent, "step_count", None),
            "view": view,
            "decision": decision,
            "proposal": proposal,
            "ext": [pid for pid, _ in ext],
            "inset": [pid for pid, _ in inset],
            "product": [pid for pid, _ in prod],
        }

        lines = [
            "MATCH_PROBE over {} saved candidate pages (sessions 1-{}). Three matchings "
            "of the question image (scale/rotation/position-invariant NCC; higher = better):".format(
                len(pages), info.get("after_session", "?")),
            "1) FULL PAGE (whole saved frame):",
        ]
        for position, (pid, score) in enumerate(ext[:8], start=1):
            lines.append("   {}. {}  {:.3f}".format(position, pid, score))
        lines.append("   TOP-1 MARGIN: {:.3f}".format(extm))
        lines.append("2) PATTERN INSET (small corner square; {} pages have a valid one):".format(
            len(inset)))
        if inset:
            for position, (pid, score) in enumerate(inset[:8], start=1):
                lines.append("   {}. {}  {:.3f}".format(position, pid, score))
            lines.append("   TOP-1 MARGIN: {:.3f}".format(insm))
        else:
            lines.append("   (no candidate page has a valid pattern inset)")
        lines.append("3) PRODUCT (central product photo region):")
        for position, (pid, score) in enumerate(prod[:8], start=1):
            lines.append("   {}. {}  {:.3f}".format(position, pid, score))
        lines.append("   TOP-1 MARGIN: {:.3f}".format(prm))
        if proposal is not None:
            lines.append(
                "DECISION: COMMIT - {} (decisive, top-1 margin {:.3f}). The attached tile "
                "shows it as cell #1; commit exactly: <|ACTION|>answer {}<|END|>".format(
                    proposal, dec_margin, proposal))
        elif decision == "contested":
            lines.append(
                "DECISION: CONTESTED - the matchings disagree (full-page #1 = {}, inset #1 = "
                "{}, product #1 = {}), so the scores do not decide it. Do not commit from the "
                "scores: find the cell of the attached tile whose page the question image "
                "re-photographs (cell #1 = {}) and commit THAT id.".format(
                    ext1, ins1, pr1, ext1))
        else:
            lines.append(
                "DECISION: FLAT - no matching is decisive (all top-1 margins < 0.12). Pick "
                "the cell of the attached tile that the question image matches and commit its "
                "id; if your answer is not visible there, page with RECALL_TILE (same view "
                "'{}', offset=8) a few tiles at most.".format(view))
        lines.append(agent._build_tile_request(view=view, limit=8))
        return "\n".join(lines)

    obj = {"type": "object", "additionalProperties": False}
    # A cue visible only at reset must be savable before the first action:
    # step 0 has no LOG phase because LOG requires a preceding action.
    registry.register(Tool("SAVE_FRAME", "save the current pixels, including an initial cue at step 0; label and note are optional and should only describe what you can verify; in store episodes use label 's<session> p<page> <product id>' so the page is findable later", {**obj, "properties": {"label": {"type": "string"}, "note": {"type": "string"}}, "required": []}, save, ("LOG", "PLAN"), "memory_write", "metadata"))
    registry.register(Tool("SEARCH_IMAGE", "search visual metadata", {**obj, "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]}, agent.visual.search, ("LOG", "PLAN"), "memory_read", "metadata"))
    registry.register(Tool("RECALL_IMAGE", "recall one saved page by image_id; use a JSON object, then inspect the historical image in the next model request rather than trusting its saved note; use it to verify a close match before answering", {**obj, "properties": {"image_id": {"type": "string"}}, "required": ["image_id"]}, recall, ("LOG", "PLAN"), "memory_read", "image"))
    registry.register(Tool("ANNOTATE_IMAGE", "correct or add visual metadata after inspecting pixels", {**obj, "properties": {"image_id": {"type": "string"}, "label": {"type": "string"}, "note": {"type": "string"}}, "required": ["image_id"]}, agent.visual.annotate, ("LOG", "PLAN"), "memory_write", "metadata"))
    registry.register(Tool("DELETE_IMAGE", "delete a saved image", {**obj, "properties": {"image_id": {"type": "string"}}, "required": ["image_id"]}, agent.visual.delete, ("LOG", "PLAN"), "memory_write", "metadata"))
    registry.register(Tool("RECALL_TILE", "compare many saved product pages at once: renders a labelled grid (tile) of saved frames, one cell per product, cell label = product id; view 'inset' shows the small square pattern from each page, view 'product' shows the central product photo of each page, view 'full' shows the whole page; on a question step the cells follow the MATCH_PROBE ranking for that view (best match first) - page through the tiles (offset) and answer with the id of the matching cell", {**obj, "properties": {"view": {"type": "string", "enum": ["inset", "product", "full"]}, "sessions": {"type": "string", "description": "restrict to session numbers, e.g. '2' or '1-3'; default: the question's candidate sessions (or all saved)"}, "ids": {"type": "array", "items": {"type": "string"}, "maxItems": 40, "description": "explicit product ids to show"}, "offset": {"type": "integer", "minimum": 0, "maximum": 1000, "description": "start after this many candidates"}}, "required": []}, recall_tile, ("LOG", "PLAN"), "memory_read", "image"))

    # g3: permissive argument schema (no additionalProperties:False) so stray
    # keys from the model do not fail validation; the handler ignores them.
    # The host decoder still requires a JSON object: the playbook tells the
    # model to pass {} explicitly.
    registry.register(Tool("MATCH_PROBE", "rank every saved candidate page against the current question image with three scale/rotation/position-invariant matchings (full page, pattern inset, product photo), print each top-8 list with a TOP-1 MARGIN and a DECISION (COMMIT names the exact id to commit; CONTESTED / FLAT mean verify on the attached tile), and attach the tile ordered by best match; call it FIRST on every question step with {} as its argument", {"type": "object", "properties": {}, "required": []}, match_probe, ("LOG", "PLAN"), "memory_read", "image"))
