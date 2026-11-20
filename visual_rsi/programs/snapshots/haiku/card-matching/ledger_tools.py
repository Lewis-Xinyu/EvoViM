"""Candidate-side registration of the read-only card-ledger query tool."""
from __future__ import annotations

from registry import Tool

LEDGER_QUERIES = ["summary", "pending", "cell", "picture", "ready", "unseen", "verdicts", "flags"]


def install_ledger(agent, registry):
    def query(query, cell="", picture=""):
        return agent.ledger.query(query, cell=cell, picture=picture)

    obj = {"type": "object", "additionalProperties": False}
    parameters = {**obj,
                  "properties": {
                      "query": {"type": "string", "enum": LEDGER_QUERIES},
                      "cell": {"type": "string", "maxLength": 8},
                      "picture": {"type": "string", "maxLength": 8},
                  },
                  "required": ["query"]}
    registry.register(Tool(
        "CARD_LEDGER",
        "read the pixel-derived card ledger. query=summary|pending|ready|unseen|verdicts|flags "
        "(pending = the first card of this round and its known face-down partner), "
        "query=cell with cell=aA (one cell's state and picture), query=picture with picture=p07 "
        "(where a picture was seen). Read-only; the ledger records what the pixels show.",
        parameters, query, ("LOG", "PLAN"), "memory_read", "text"))

