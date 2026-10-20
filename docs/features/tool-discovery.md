# Tool Discovery, Sandbox, and Approval

The ECS Agent framework provides a complete pipeline for tool management: automatic discovery, sandboxed execution, and human-in-the-loop approval.

## Enhanced Events

The framework features an extensive event system for monitoring tool execution, skill lifecycles, and MCP connectivity. These events are published to the `EventBus` and can be used for logging, UI updates, or debugging.

### New Event Types

| Event Name | Fired When | Key Fields |
|------------|------------|------------|
| `ToolExecutionStartedEvent` | Before a tool handler is called | `tool_call`, `entity_id` |
| `ToolExecutionCompletedEvent` | After a tool handler returns | `tool_call_id`, `result`, `success` |
| `SkillInstalledEvent` | After a skill is installed | `skill_name`, `tool_names`, `entity_id` |
| `SkillUninstalledEvent` | After a skill is uninstalled | `skill_name`, `entity_id` |
| `SkillDiscoveryEvent` | After scanning a discovery source | `source`, `skills_found`, `errors` |
| `MCPConnectedEvent` | After an MCP server connects | `server_name` |
| `MCPDisconnectedEvent` | After an MCP server disconnects | `server_name` |
| `MCPToolCallEvent` | After an MCP tool is called | `server_name`, `tool_name`, `success` |

### Code Example: Subscribing to Events

You can subscribe to these events to track the agent's progress in real-time.

```python
async def on_tool_start(event: ToolExecutionStartedEvent):
    print(f"Tool {event.tool_call.name} started execution.")

async def on_discovery(event: SkillDiscoveryEvent):
    print(f"Discovered {len(event.skills_found)} skills from {event.source}")

world.event_bus.subscribe(ToolExecutionStartedEvent, on_tool_start)
world.event_bus.subscribe(SkillDiscoveryEvent, on_discovery)
```

## Tool Auto-Discovery

The `scan_module` function automatically discovers tool-decorated functions in a Python module.

### The `@tool` Decorator

The parameters schema is derived from the function signature and type hints
(`Annotated[str, "description"]` adds per-parameter descriptions).

Inference supports JSON scalar types, nested `list[T]`, `dict[str, T]`,
`Literal` values, unions, nullable values, and `Annotated` descriptions. Union
branches retain their full constraints. Required parameters are determined by
Python defaults, independently of whether the type permits null:
`value: str | None` is required, while `value: str | None = None` is optional.
Omitted parameters use their Python defaults. Generated parameter objects reject
unknown keys.

Unsupported annotations, missing input annotations, unresolved forward
references, positional-only parameters, `*args`, and `**kwargs` fail during
automatic definition instead of silently becoming strings or disappearing.
Errors identify the tool and parameter. Use an explicit `ToolDefinition` and an
adapter handler for custom objects; schemas do not convert JSON into Python
Enum, dataclass, or Pydantic instances. `@tool()` preserves the original callable
and its type signature.

### Registering functions

```python
from typing import Literal
from ecs_agent.tools import register_tools, tool

@tool()
def total(values: list[int], mode: Literal["sum", "count"] = "sum") -> str:
    return str(sum(values) if mode == "sum" else len(values))

register_tools(world, agent, [total])
```

The registration function validates the whole batch before merging into the
existing entity registry. Duplicate names, invalid schemas, and detectable
signature mismatches fail without partially registering the batch. Existing
skill tools are retained. Registration does not grant permission or approval.
Same-name replacement is not supported by this API.

`get_tool_definition(fn)` exposes a decorated function's schema and handler as
a `ToolDefinition`; callers do not need `_tool_schema` or `_tool_handler`.
Manual tools can pass `ToolDefinition(schema, async_handler)` directly. The
legacy ToolRegistryComponent constructor remains available; direct dictionary
mutation does not receive registration validation.

```python
from typing import Annotated

from ecs_agent.tools import tool

@tool(name="calculate", description="Perform arithmetic calculations")
async def calculate(
    expression: Annotated[str, "Math expression to evaluate"],
) -> str:
    return str(eval(expression))
```

Tools without conflicting side effects (pure lookups, read-only access) can opt
into concurrent execution. `ToolExecutionSystem` runs consecutive
`concurrency_safe` calls of one batch in parallel and lands results in the
original order; unmarked tools stay serial barriers.

```python
@tool(description="Look up a city's weather.", concurrency_safe=True)
async def lookup_weather(city: Annotated[str, "City name."]) -> str:
    ...
```

Use `@tool(..., inline_result=True)` when the next model call must receive the
result itself even if scratchbook sinking is enabled. The default is `False`.
File readers and delegation/control tools use inline results so the model can
read archived output, review feedback, session IDs, and lifecycle status.
Failed tool results remain visible inline even when their full record is archived.

### Scanning a Module

```python
from ecs_agent.tools import register_tools, scan_module
import my_tools_module

discovered = scan_module(my_tools_module)
# dict[str, tuple[ToolSchema, async_handler]]
register_tools(world, agent, discovered)
```

## Sandboxed Execution

The `sandboxed_execute` function runs tool handlers with timeout and output size limits.

```python
from ecs_agent.tools import sandboxed_execute

result = await sandboxed_execute(
    func=my_handler,
    args={"expression": "2 + 2"},
    timeout=30.0,         # Maximum execution time in seconds
    max_output_size=10000  # Maximum output size in bytes
)
```

### Configuration via SandboxConfigComponent

```python
from ecs_agent.components import SandboxConfigComponent

world.add_component(agent, SandboxConfigComponent(timeout=10.0, max_output_size=5000))
```

## Tool Approval System

The `ToolApprovalSystem` provides human-in-the-loop approval for tool calls before execution.

### Approval Policies

```python
from ecs_agent.types import ApprovalPolicy

# Three policies available:
ApprovalPolicy.ALWAYS_APPROVE   # All tool calls pass through
ApprovalPolicy.ALWAYS_DENY      # All tool calls are blocked
ApprovalPolicy.REQUIRE_APPROVAL # Human must approve each call
```

### Setup

```python
from ecs_agent.components import ToolApprovalComponent
from ecs_agent.systems.tool_approval import ToolApprovalSystem

world.add_component(agent, ToolApprovalComponent(
    policy=ApprovalPolicy.REQUIRE_APPROVAL,
    timeout=60.0,  # Seconds to wait for approval; None for infinite wait
))
world.register_system(ToolApprovalSystem(), priority=2)
```

### Approval Flow

1. LLM generates tool calls (via `ReasoningSystem`).
2. Optional `ToolApprovalSystem` preflight shares authorization decisions with `ToolExecutionSystem`. Execution performs the same checks when preflight is absent.
3. For `REQUIRE_APPROVAL` policy:
   - A `ToolApprovalRequestedEvent` is published with a `Future[bool]`.
   - External code resolves the future: `event.approval_future.set_result(True)` to approve, `False` to deny.
   - Missing handlers, subscriber failures and timeout deny the call. The timeout covers subscriber execution and waiting for the answer. External cancellation propagates without execution.
4. Execution consumes the original batch and writes allowed and denied tool results in order. Denial produces a tool message with the original call ID, never a system-message substitute.
5. Approval is bound to the entity, batch, call ID, tool name, arguments and policy. Changes invalidate the decision; restoring a checkpoint requires fresh approval. Historical `approved_calls` entries do not grant authority.

### Events

- `ToolApprovalRequestedEvent(entity_id, tool_call, approval_future)` — Requesting approval
- `ToolApprovedEvent(entity_id, tool_call_id)` — Call was approved
- `ToolDeniedEvent(entity_id, tool_call_id)` — Call was denied

### Priority Order

The recommended order is:

- `ReasoningSystem` at priority `0`
- Optional `PermissionSystem` at priority `1`
- Optional `ToolApprovalSystem` at priority `2`
- `ToolExecutionSystem` at priority `5`

Execution enforces authorization even when preflight systems are missing or scheduled differently. Concurrent preflights share one approval request. Execution freezes arguments and the selected handler when authorization commits at start; later policy changes affect calls not yet started.

## Complete Example

See [`examples/tool_approval_agent.py`](../../examples/tool_approval_agent.py) for a full working demo.
