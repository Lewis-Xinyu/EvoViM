"""Candidate-side visual memory tools; pixels return only through host RPC."""
from __future__ import annotations

from registry import Tool


def install_visual(agent, registry, scratch=None, event=None):
    from agents.visual_memory import VisualMemoryStore
    agent.visual = VisualMemoryStore(scratch or agent._memory_dir, event)
    agent.visual_request = None

    def save(label=None, note=None):
        if not agent.current_frame_id:
            raise ValueError("No host frame is available to recall later")
        return agent.visual.save(agent.current_image, agent.step_count,
                                 label if label is not None else f"frame step {agent.step_count}",
                                 note if note is not None else "",
                                 frame_id=agent.current_frame_id)

    def recall(image_id):
        metadata, _ = agent.visual.recall(image_id)
        frame_id = agent.visual.data["images"][image_id].get("frame_id")
        if not frame_id:
            raise ValueError("visual memory entry has no host frame id")
        agent.visual_request = {"ops": [{"op": "frame", "id": frame_id, "as": "out"}], "output": "out"}
        return metadata

    obj = {"type": "object", "additionalProperties": False}
    # A cue visible only at reset must be savable before the first action:
    # step 0 has no LOG phase because LOG requires a preceding action.
    registry.register(Tool("SAVE_FRAME", "save the current pixels, including an initial cue at step 0; label and note are optional and should only describe what you can verify", {**obj, "properties": {"label": {"type": "string"}, "note": {"type": "string"}}, "required": []}, save, ("LOG", "PLAN"), "memory_write", "metadata"))
    registry.register(Tool("SEARCH_IMAGE", "search visual metadata", {**obj, "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]}, agent.visual.search, ("LOG", "PLAN"), "memory_read", "metadata"))
    registry.register(Tool("RECALL_IMAGE", "recall saved pixels by image_id; use a JSON object, then inspect the historical image in the next model request rather than trusting its saved note", {**obj, "properties": {"image_id": {"type": "string"}}, "required": ["image_id"]}, recall, ("LOG", "PLAN"), "memory_read", "image"))
    registry.register(Tool("ANNOTATE_IMAGE", "correct or add visual metadata after inspecting pixels", {**obj, "properties": {"image_id": {"type": "string"}, "label": {"type": "string"}, "note": {"type": "string"}}, "required": ["image_id"]}, agent.visual.annotate, ("LOG", "PLAN"), "memory_write", "metadata"))
    registry.register(Tool("DELETE_IMAGE", "delete a saved image", {**obj, "properties": {"image_id": {"type": "string"}}, "required": ["image_id"]}, agent.visual.delete, ("LOG", "PLAN"), "memory_write", "metadata"))
