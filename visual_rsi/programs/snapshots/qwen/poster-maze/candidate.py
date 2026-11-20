from agents.postermaze_agent import PosterMazeAgent
from poster_tools import install_poster
from visual_tools import install_visual

Agent = PosterMazeAgent


def install(agent, registry):
    """Install the host-mediated M-line tools (originals kept) plus the
    postermaze poster tools."""
    install_visual(agent, registry)
    install_poster(agent, registry)
