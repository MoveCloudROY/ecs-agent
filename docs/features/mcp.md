# Model Context Protocol (MCP) Integration

The framework supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), allowing your agents to connect to external tool servers.

## Overview

MCP integration enables:
-   **Interoperability**: Connect to any MCP-compliant server (e.g., SQLite, GitHub, Google Drive).
-   **Automatic Conversion**: MCP tools are automatically converted into ECS-compatible tools.
-   **Namespacing**: Tools are namespaced as `server_name/tool_name` to prevent collisions.
-   **Skill Integration**: MCP servers are wrapped in an `MCPSkillAdapter`, making them compatible with the [Skills system](skills.md).

## Setup

MCP support is an optional dependency. Install it using `uv`:

```bash
uv pip install -e ".[mcp]"
```

## Configuration

Use the `MCPConfigComponent` to define how to connect to the server.

### stdio Transport
Used for local executable servers.

```python
from ecs_agent.mcp.components import MCPConfigComponent

config = MCPConfigComponent(
    server_name="sqlite",
    transport_type="stdio",
    config={
        "command": "uvx",
        "args": ["mcp-server-sqlite", "--db-path", "my-database.db"]
    }
)
```

### SSE / HTTP Transport
Used for remote servers.

```python
config = MCPConfigComponent(
    server_name="remote-server",
    transport_type="sse", # or "http"
    config={"url": "https://mcp.example.com/sse"}
)
```

## Using the MCP Adapter

The `MCPSkillAdapter` wraps an `MCPClient` and implements the `Skill` protocol.

```python
from ecs_agent.mcp.client import MCPClient
from ecs_agent.mcp.adapter import MCPSkillAdapter

# 1. Initialize client
client = MCPClient(config)

# 2. Wrap in adapter
skill = MCPSkillAdapter(client, server_name="sqlite")

# 3. Install via SkillManager
from ecs_agent import SkillManager
manager = SkillManager()
manager.install(world, agent_entity, skill)
```

## Custom Tool Conversion

By default, the adapter converts all MCP tools using `mcp_tool_to_ecs_tool`. You can provide a custom converter to filter or transform tools.

```python
def my_converter(mcp_tool):
    # Filter: only expose read-only tools
    if not mcp_tool["name"].startswith("read_"):
        return None
    
    # Use default conversion for others
    from ecs_agent.mcp.adapter import mcp_tool_to_ecs_tool
    return mcp_tool_to_ecs_tool(client, "my-server", mcp_tool)

skill = MCPSkillAdapter(client, server_name="my-server", converter=my_converter)
```

## API Reference

### MCPClient
-   `connect()`: Establishes connection to the server.
-   `disconnect()`: Closes the connection.
-   `list_tools()`: Fetches raw tool definitions from the server.
-   `call_tool(name, args)`: Executes a tool on the server.
-   `is_connected`: Property indicating connection status.

### MCPSkillAdapter
-   `name`: `mcp-{server_name}`.
-   `tools()`: Returns namespaced tools (`server_name/tool_name`).
-   `system_prompt()`: Provides a Tier 1 summary of available MCP tools.

## Examples

See [`examples/mcp_agent.py`](../../examples/mcp_agent.py) for a complete demo using a mocked MCP server.
