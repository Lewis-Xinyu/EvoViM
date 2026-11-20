"""Line M candidate: whole-view memory with a match-gated answer step."""
from view_agent import ViewMemoryAgent
from view_tools import install as install_view_tools

Agent = ViewMemoryAgent


def install(agent, registry):
    """Register view tools. Parent visual tools (whole-frame SAVE_FRAME, RECALL_IMAGE) are not installed."""
    return install_view_tools(agent, registry)

