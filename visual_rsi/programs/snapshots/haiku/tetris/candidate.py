"""Candidate entry point.

The agent keeps the parent's two-phase memory loop and adds a derived board
record plus explicit board tools.  `install` registers those tools after the
host's legacy memory tools and the visual memory tools.
"""
from agents.tetris_agent import TetrisMemoryAgent
from agents.tetris_tools import install_tetris
from visual_tools import install_visual


Agent = TetrisMemoryAgent


def install(agent, registry):
    """Install the M-line visual tools, then the board tools."""
    install_visual(agent, registry)
    return install_tetris(agent, registry)
