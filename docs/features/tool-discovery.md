# Tool Discovery, Sandbox, and Approval

The ECS Agent framework provides a complete pipeline for tool management: automatic discovery, sandboxed execution, and human-in-the-loop approval.

## Tool Auto-Discovery

The `scan_module` function automatically discovers tool-decorated functions in a Python module.

### The `@tool` Decorator

```python
from ecs_agent.tools import tool

@tool(
    name="calculate",
    description="Perform arithmetic calculations",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression to evaluate"}
        },
        "required": ["expression"]
    }
)
async def calculate(expression: str) -> str:
    return str(eval(expression))
```

### Scanning a Module

```python
from ecs_agent.tools import scan_module
import my_tools_module

tools, handlers = scan_module(my_tools_module)
# tools: dict[str, ToolSchema] — tool name to schema mapping
# handlers: dict[str, Callable] — tool name to async handler mapping

world.add_component(agent, ToolRegistryComponent(tools=tools, handlers=handlers))
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
world.register_system(ToolApprovalSystem(priority=-5), priority=-5)
```

### Approval Flow

1. LLM generates tool calls (via `ReasoningSystem`).
2. `ToolApprovalSystem` intercepts pending calls before `ToolExecutionSystem`.
3. For `REQUIRE_APPROVAL` policy:
   - A `ToolApprovalRequestedEvent` is published with a `Future[bool]`.
   - External code resolves the future: `event.future.set_result(True)` to approve, `False` to deny.
   - On timeout (or `ToolTimeoutError`), the call is denied.
4. Approved calls proceed to `ToolExecutionSystem`; denied calls are removed.

### Events

- `ToolApprovalRequestedEvent(entity_id, tool_call, future)` — Requesting approval
- `ToolApprovedEvent(entity_id, tool_call_id)` — Call was approved
- `ToolDeniedEvent(entity_id, tool_call_id)` — Call was denied

### Priority Order

The approval system must run **before** tool execution:
- `ToolApprovalSystem` at priority `-5`
- `ToolExecutionSystem` at priority `5`

## Complete Example

See [`examples/tool_approval_agent.py`](../../examples/tool_approval_agent.py) for a full working demo.
