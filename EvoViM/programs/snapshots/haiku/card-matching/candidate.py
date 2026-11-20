from agents.ledger_memory_agent import LedgerMemoryAgent
from ledger_tools import install_ledger
from visual_tools import install_visual


Agent = LedgerMemoryAgent


def install(agent, registry):
    """Install the host-mediated M-line tools after the legacy tools.

    The visual tools are kept unchanged; the card-ledger query tool is added
    after them. Both are read-only with respect to the environment.
    """
    install_visual(agent, registry)
    return install_ledger(agent, registry)
