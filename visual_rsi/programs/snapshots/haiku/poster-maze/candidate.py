from agents.poster_memory_agent import PosterMemoryAgent
from visual_tools import install_visual
from poster_tools import install_poster_tools


Agent = PosterMemoryAgent


def install(agent, registry):
    """Keep the parent's legacy and visual tools; add the poster tools after them."""
    install_visual(agent, registry)
    return install_poster_tools(agent, registry)
