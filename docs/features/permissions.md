# Permission System and Sandboxing

The ECS Agent framework provides a robust permission system to control tool access and a sandboxing mechanism for secure tool execution.

## Permission System

The `PermissionComponent` and `PermissionSystem` allow you to define whitelists and blacklists for tool calls. This is essential for restricting agents to a safe subset of available tools.

### PermissionComponent

Attach this component to an agent entity to define its permissions.

```python
from ecs_agent.components import PermissionComponent

# Only allow specific tools
permissions = PermissionComponent(allowed_tools=["read_file", "search"])

# Deny specific tools
permissions = PermissionComponent(denied_tools=["bash", "delete_file"])
```

- `allowed_tools`: If not empty, only tools in this list can be executed.
- `denied_tools`: Tools in this list are always blocked, even if they are in `allowed_tools`.

### PermissionSystem

The `PermissionSystem` should be registered with a high priority (default -10) so it runs before the `ToolExecutionSystem`. It filters `PendingToolCallsComponent` based on the defined policies.

```python
from ecs_agent.systems.permission import PermissionSystem

world.register_system(PermissionSystem(priority=-10), priority=-10)
```

When a tool call is denied:
1. It's removed from `PendingToolCallsComponent`.
2. An error message is appended to the `ConversationComponent`.
3. A `ToolDeniedEvent` is published to the `EventBus`.

## Sandboxed Execution

For tools that execute external commands or potentially dangerous logic, the framework supports sandboxing via `bwrap` (bubblewrap).

### Bwrap Sandbox

The `bwrap` sandbox provides strong isolation on Linux systems. It creates a new namespace with:
- Read-only root filesystem (`--ro-bind / /`)
- Private `/tmp` directory (`--tmpfs /tmp`)
- No network access (via `--unshare-all` by default)
- Isolated device and process namespaces

#### Enabling the Sandbox

To enable `bwrap` isolation, configure the `SandboxConfigComponent` and ensure the tool is marked as `sandbox_compatible`.

```python
from ecs_agent.components import SandboxConfigComponent

world.add_component(agent, SandboxConfigComponent(
    sandbox_mode="bwrap",
    timeout=30.0,
    max_output_size=10000
))
```

#### Transparency

The sandboxing is transparent to tool authors. The framework wraps tool handlers at registration time if they are marked as compatible. If `bwrap` is not installed on the system, the framework gracefully falls back to a standard `asyncio` subprocess execution (while still enforcing timeouts).

### Code Examples

#### Permission-Restricted Agent

```python
from ecs_agent.core import World
from ecs_agent.components import PermissionComponent, PendingToolCallsComponent
from ecs_agent.systems.permission import PermissionSystem
from ecs_agent.types import ToolCall

world = World()
agent = world.create_entity()

# Only allow 'read' tool
world.add_component(agent, PermissionComponent(allowed_tools=["read"]))
world.register_system(PermissionSystem())

# This call will be allowed
world.add_component(agent, PendingToolCallsComponent(
    tool_calls=[ToolCall(id="1", name="read", arguments={})]
))

# This call will be blocked in the next tick
world.add_component(agent, PendingToolCallsComponent(
    tool_calls=[ToolCall(id="2", name="bash", arguments={"command": "rm -rf /"})]
))
```

#### Sandboxed Tool Execution

```python
from ecs_agent.components import SandboxConfigComponent
from ecs_agent.types import ToolSchema

# Tool marked as sandbox compatible
schema = ToolSchema(
    name="bash",
    description="Execute shell command",
    parameters={...},
    sandbox_compatible=True
)

# Enable bwrap in config
world.add_component(agent, SandboxConfigComponent(sandbox_mode="bwrap"))
```
