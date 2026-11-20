from agents.memory_agent import MemoryAgent
from visual_tools import install_visual
from agents.painting_memory import install_paintings


Agent = MemoryAgent


def install(agent, registry):
    """Install the host-mediated M-line tools after the legacy tools."""
    install_visual(agent, registry)
    install_paintings(agent, registry)
