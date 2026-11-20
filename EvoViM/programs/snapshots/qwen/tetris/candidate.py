from agents.memory_agent import MemoryAgent
import board_tools
from visual_tools import install_visual


Agent = MemoryAgent


def install(agent, registry):
    """Install the host-mediated M-line tools after the legacy tools."""
    install_visual(agent, registry)
    board_tools.install_board_tools(agent, registry)
