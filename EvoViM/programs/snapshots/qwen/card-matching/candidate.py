from agents.memory_agent import MemoryAgent
from visual_tools import install_visual


Agent = MemoryAgent

_OBJ = {"type": "object", "additionalProperties": False}
_COORD = {"type": "string"}
_STATE = {"type": "string", "enum": ["unknown", "known", "face_up", "removed"]}


def _board(agent):
    return getattr(agent, "board", None)


def install(agent, registry):
    """Install the host-mediated M-line visual tools plus the board-tracker
    correction tools.  The tracker is derived computation over the episode's
    own observations, and its full model (per-cell states, ready pairs and
    the suggested flip) is shown to the inner LLM on every request and
    mirrored in board_state.txt, match_index.txt and round_log.txt.  The
    JSON query tools BOARD_STATUS / FIND_MATCH / COMPARE_CARDS were removed:
    their answers are already always visible, their development-episode
    invocations all failed as pipe-style protocol errors, and the same
    information stays reachable through SEARCH on match_index.txt and TAIL
    on round_log.txt.  The LLM keeps explicit, LLM-decided memory tools:
    the seven file operations, the five visual tools and the two tracker
    corrections below."""
    install_visual(agent, registry)

    from registry import Tool

    def set_cell_state(coordinate, state, note=None):
        board = _board(agent)
        if board is None:
            return "board tracker not initialised yet"
        return board.set_cell_state(str(coordinate).strip(), state, note)

    def clear_fingerprint(coordinate):
        board = _board(agent)
        if board is None:
            return "board tracker not initialised yet"
        return board.clear_fingerprint(str(coordinate).strip())

    registry.register(Tool(
        "SET_CELL_STATE",
        "correct the tracker's state for one cell (unknown, known, face_up or "
        "removed); use only when the tracker contradicts the observation",
        dict(_OBJ,
             properties={"coordinate": _COORD, "state": _STATE,
                         "note": {"type": "string", "maxLength": 200}},
             required=["coordinate", "state"]),
        set_cell_state, ("LOG", "PLAN"), "memory_write", "text"))
    registry.register(Tool(
        "CLEAR_FINGERPRINT",
        "forget the recorded picture of one cell so it is matched again next "
        "time it is revealed; use only when the recorded picture is wrong",
        dict(_OBJ, properties={"coordinate": _COORD}, required=["coordinate"]),
        clear_fingerprint, ("LOG", "PLAN"), "memory_write", "text"))
