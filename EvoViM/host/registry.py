"""Host-side access to the child runtime registry (single implementation)."""
from .child.registry import (  # noqa: F401
    LEGACY_KEYS, COMMAND_RE, ProtocolError, Registry, Tool, decode, register_legacy, sample_arguments, validate,
)
