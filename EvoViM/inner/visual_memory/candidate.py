from agents.memory_agent import MemoryAgent
from visual_tools import install_visual


Agent = MemoryAgent


def install(agent, registry):
    """Install the host-mediated M-line tools after the legacy tools."""
    return install_visual(agent, registry)
