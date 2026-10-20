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

- `allowed_tools=None` (default): no allowlist restriction.
- `allowed_tools=[]`: deny every tool.
- A non-empty `allowed_tools` list permits only its members.
- `denied_tools`: Tools in this list are always blocked, even if they are in `allowed_tools`.

### PermissionSystem

ToolExecutionSystem enforces the current permission policy before every handler invocation, even without PermissionSystem. PermissionSystem is optional preflight at priority 1, after ReasoningSystem (0). It records decisions without consuming pending calls or appending messages.

```python
from ecs_agent.systems.permission import PermissionSystem

world.register_system(PermissionSystem(), priority=1)
```

When execution denies a call:

1. The handler is not invoked and no execution-start event is emitted.
2. Execution writes a `role="tool"` result with the original call ID, in batch order.
3. One `ToolDeniedEvent` is published after the result is recorded.

Preflight systems never remove calls. ToolExecutionSystem owns consumption, result ordering and session persistence.

## Migrating Permission Policies

Replace explicit `allowed_tools=[]` with `allowed_tools=None` if the intended policy is unrestricted. An empty list now means deny all. Deny entries always take precedence.

Checkpoint schema version 2 stores this distinction. Loading a version 1 checkpoint (including a payload without a version) migrates missing or empty allowlists to None to preserve its old runtime behavior. Old DSL denials already lost from a checkpoint cannot be reconstructed; compile the original DSL again.

A phase without a tool declaration sets None; a phase declaring an empty tuple denies all. A child agent preserves inherited restrictions and intersects them with its own `SubagentConfig.tools` policy. Session tool installation uses the same permission evaluator as execution.

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
world.register_system(PermissionSystem(), priority=1)

# Preflight records a decision; execution enforces it and writes the result.
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
