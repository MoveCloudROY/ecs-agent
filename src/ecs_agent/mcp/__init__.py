from __future__ import annotations

try:
    import mcp  # type: ignore[import-not-found]
except ImportError as exc:
    raise ImportError(
        "MCP support requires the 'mcp' package. "
        'Install with: pip install "ecs-agent[mcp]"'
    ) from exc

from ecs_agent.mcp.client import MCPClient
from ecs_agent.mcp.components import MCPClientComponent, MCPConfigComponent

_ = mcp

__all__ = ["MCPClient", "MCPClientComponent", "MCPConfigComponent"]
