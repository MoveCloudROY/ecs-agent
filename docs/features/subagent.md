# Subagent Delegation

The `SubagentSystem` enables parent agents to spawn child agents for subtask execution via the `subagent` and `delegate` tools. The system supports synchronous and background execution modes, with isolated environments and policy-based capability inheritance.


## Overview

Subagent delegation provides:
- **Named Subagent Registry**: Pre-configure subagent profiles with specific capabilities
- **Isolated Execution**: Each subagent runs in its own `World` with independent state
- **Automatic Result Aggregation**: Results flow back to parent via tool result messages.
- **Event Tracking**: Monitor delegation lifecycle with `DelegationStartedEvent` and `DelegationCompletedEvent`.
- **Skill Inheritance**: Subagents can inherit specific skills, system prompts (via `SystemPromptComponent`), and tools from their parent agent via `InheritancePolicy`.
- **Sync and Background Modes**: Execute tasks immediately or as background sessions with ID tracking.
- **Lifecycle Management**: Track background sessions through `Idle`, `Working`, `Dead`, `Timeout`, and `Cancelled` states.
- **Control Tools**: Tools to query status, retrieve results, and cancel background sessions.
- **Timeout Policy**: Per-call timeout overrides with global fallback and automated handling.
- **Retry Reliability**: Transparent `RetryProvider` wrapping for transient LLM failures.
- **Explicit Control**: Install the delegation tool manually with custom names using the `install_delegate_tool` API.


## Core Components

### SubagentConfig

Define subagent profiles:

```python
from ecs_agent.types import SubagentConfig
from ecs_agent.providers import FakeProvider

researcher_config = SubagentConfig(
    name="researcher",
    provider=FakeProvider(responses=[...]),
    model="gpt-4o",
    system_prompt="You are a research specialist. Provide detailed, factual information.",
    max_ticks=10,
    skills=[],  # Skill names to load and install on this subagent
    inheritance_policy=InheritancePolicy(inherit_system_prompt=True),  # Optional configuration
)
```

### SubagentRegistryComponent

Register named subagents:

```python
from ecs_agent.components import SubagentRegistryComponent

world.add_component(
    entity,
    SubagentRegistryComponent(
        subagents={
            "researcher": researcher_config,
            "writer": writer_config,
            "analyst": analyst_config,
        }
    ),
)
```
### SubagentSessionRecord

Track background session metadata:

```python
from ecs_agent.types import SubagentSessionRecord

record = SubagentSessionRecord(
    session_id="session_123",
    category="researcher",
    prompt="...",
    status="Working",
    timeout_seconds=30.0,
    # ... other fields
)
```

### SubagentLifecycleStatus

Background sessions transition through a strict state machine:

| State | Description |
| :--- | :--- |
| `Idle` | Completed successfully. |
| `Working` | Currently executing. |
| `Dead` | Failed with an error. |
| `Timeout` | Terminated after exceeding timeout limit. |
| `Cancelled` | Terminated by explicit cancel request. |

The system handles delegation and auto-registers the `delegate` tool. It supports both manual registration and backward-compatible auto-discovery for entities with the required components.

### Explicit Installer API

For fine-grained control, use the `install_delegate_tool` method. This allows you to customize the tool name or override existing registrations.

```python
subagent_system = SubagentSystem()
world.register_system(subagent_system)

# Install tool with custom name and override policy
subagent_system.install_delegate_tool(
    world, 1, tool_name="ask_subagent", override=True
)
```
### Subagent Control Installer

Entities using background subagents must have the `SubagentSessionTableComponent` and call `install_subagent_control_tools` to enable control tools.

```python
subagent_system = SubagentSystem()
world.register_system(subagent_system)
subagent_system.install_subagent_control_tools(world, entity_id)
```

### Auto-Registration Semantics (Backward Compatibility)

The `SubagentSystem` automatically registers the `delegate` tool for any entity that has both:
1. `SubagentRegistryComponent`
2. `ToolRegistryComponent`

If the `delegate` tool is already registered in the `ToolRegistryComponent`, the system skips registration for that entity to avoid overwriting custom implementations unless the installer API is used with `override=True`.

Note: `ToolRegistryComponent` is required because the delegate tool handler is registered into its `handlers` dict.

### Registration

Register the `SubagentSystem` during world setup:

```python
from ecs_agent.systems.subagent import SubagentSystem

# Register system (priority -1 recommended to run before ReasoningSystem)
world.register_system(SubagentSystem(priority=-1), priority=-1)
```


## Usage

### Basic Delegation

1. **Register subagents** with `SubagentRegistryComponent`
2. **Register SubagentSystem** (priority -1, before ReasoningSystem)
3. **SubagentSystem creates a new child entity** with the subagent's provider, model, and `SystemPromptComponent` (template/content from effective system prompt), plus a `ChildStubComponent` to mark the parent-world stub entity
4. **LLM calls delegate tool** to invoke subagent
5. **SubagentSystem executes** child and returns result
```python
from ecs_agent.core import World
from ecs_agent.components import (
    LLMComponent,
    ConversationComponent,
    ToolRegistryComponent,
    SubagentRegistryComponent,
)
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import Message, SubagentConfig

# Create parent world
world = World()
parent = world.create_entity()

# Configure subagent
researcher = SubagentConfig(
    name="researcher",
    provider=your_provider,
    model="gpt-4o",
    system_prompt="You are a research assistant.",
    max_ticks=5,
    skills=[],
)

# Register subagent
world.add_component(
    parent,
    SubagentRegistryComponent(subagents={"researcher": researcher}),
)

# Add empty ToolRegistryComponent (SubagentSystem will auto-register delegate tool)
world.add_component(
    parent,
    ToolRegistryComponent(tools={}, handlers={}),
)

# Add LLM and conversation
world.add_component(
    parent,
    LLMComponent(provider=your_provider, model="gpt-4o"),
)
world.add_component(
    parent,
    ConversationComponent(
        messages=[
            Message(role="user", content="Research quantum computing and summarize.")
        ]
    ),
)

# Register systems (SubagentSystem BEFORE ReasoningSystem)
world.register_system(SubagentSystem(priority=-1), priority=-1)
world.register_system(ReasoningSystem(priority=0), priority=0)
world.register_system(ToolExecutionSystem(priority=5), priority=5)
# ... register other systems (MemorySystem, ErrorHandlingSystem, etc.)

# Run
runner = Runner()
await runner.run(world, max_ticks=20)
```

### Unified `subagent` Tool Usage

The `subagent` tool replaces the legacy `delegate` tool for most use cases, adding support for background execution and skill overrides:

```json
{
  "name": "subagent",
  "arguments": {
    "category": "researcher",
    "prompt": "Explain quantum entanglement",
    "load_skills": ["web_search"],
    "background": true,
    "timeout": 30.0
  }
}
```

**Parameters:**
- `category`: Name of the subagent configuration to use.
- `prompt`: The task description.
- `load_skills`: Optional list of additional skills to install on the child.
- `background`: If `true`, returns a session ID immediately; if `false`, waits for completion.
- `timeout`: Optional per-call timeout in seconds (overrides global default).

### Background Control Usage

When running in `background: true` mode, use control tools to manage the session:

1. **Check Status**: `subagent_status(session_id="session_123")` returns current lifecycle state and a summary table if `session_id` is omitted.
2. **Retrieve Result**: `subagent_result(session_id="session_123", timeout=10.0)` waits for completion and returns the result.
3. **Cancel**: `subagent_cancel(session_id="session_123")` terminates the session.

### Timeout Precedence

Timeouts are resolved in the following order:
1. **Per-call override**: The `timeout` argument in the `subagent` tool call.
2. **Global default**: The `default_timeout` passed to `SubagentSystem` constructor.
3. **None**: No timeout limit applied.

### Multi-Subagent Workflow

Use multiple specialized subagents:

```python
subagents = {
    "researcher": SubagentConfig(
        name="researcher",
        provider=provider,
        model="gpt-4o",
        system_prompt="Research specialist. Provide detailed facts.",
        max_ticks=10,
        skills=[],
    ),
    "writer": SubagentConfig(
        name="writer",
        provider=provider,
        model="gpt-4o",
        system_prompt="Content writer. Create engaging prose.",
        max_ticks=10,
        skills=[],
    ),
    "critic": SubagentConfig(
        name="critic",
        provider=provider,
        model="gpt-4o",
        system_prompt="Critical reviewer. Identify weaknesses.",
        max_ticks=5,
        skills=[],
    ),
}

world.add_component(parent, SubagentRegistryComponent(subagents=subagents))
```

The parent LLM can orchestrate:

```
User: "Write a blog post about AI safety."

Parent LLM:
1. Call delegate(subagent_name="researcher", task="Research AI safety concerns")
2. Call delegate(subagent_name="writer", task="Write blog post: [research results]")
3. Call delegate(subagent_name="critic", task="Review this draft: [blog post]")
4. Revise based on feedback
```

### Retry and Reliability

By default, all subagent LLM providers are wrapped in a `RetryProvider` using a standard `RetryConfig`. This handles transient network errors and rate limits automatically. `FakeProvider` used in tests is exempt from this wrapping to maintain deterministic behavior.
## Events

### DelegationStartedEvent

Fired when subagent begins execution:

```python
from ecs_agent.types import DelegationStartedEvent

async def on_delegation_started(event: DelegationStartedEvent) -> None:
    print(f"Delegating to {event.subagent_name}: {event.task}")

world.event_bus.subscribe(DelegationStartedEvent, on_delegation_started)
```

### DelegationCompletedEvent

Fired when subagent completes:

```python
from ecs_agent.types import DelegationCompletedEvent

async def on_delegation_completed(event: DelegationCompletedEvent) -> None:
    print(f"Subagent {event.subagent_name} completed: {event.result}")

world.event_bus.subscribe(DelegationCompletedEvent, on_delegation_completed)
```

## Error Handling

If a subagent fails or produces an error, the result contains the error:

```python
# Subagent error is returned as tool result
result = "Error: Subagent 'researcher' failed: <error details>"
```

The parent can handle this via normal tool result processing.

> **Note on validation errors**: Parameter validation (empty `category`, empty `prompt`, non-list `load_skills`) happens *before* execution and raises `ValueError` immediately rather than returning an error string. Ensure tool call arguments are well-formed before invocation.

## Inheritance Policy (Detailed)

The `InheritancePolicy` controls which capabilities are inherited from parent to child agents during delegation. This enables parent-to-child capability sharing while maintaining isolation.

### Configuration

```python
from ecs_agent.types import InheritancePolicy, SubagentConfig

policy = InheritancePolicy(
    enabled=True,                      # Master toggle for inheritance
    inherit_system_prompt=True,        # Append parent system prompt to child
    inherit_tools=["search", "read"],  # Whitelist of tool names to inherit
    inherit_permissions=False,         # Inherit parent permission restrictions
    allow_delegate_tool=True,          # Enable delegate tool on child
    tool_conflict_policy="skip",       # How to handle tool name conflicts: skip|error|override
    missing_skill_policy="warn",       # How to handle missing inherited skills: warn|error
)

config = SubagentConfig(
    name="researcher",
    provider=provider,
    model="gpt-4o",
    system_prompt="You are a research assistant.",
    inheritance_policy=policy,  # Attach policy to config
)
```

### Policy Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Master toggle. If `False`, all inheritance is disabled. |
| `inherit_system_prompt` | `bool` | `True` | Append parent's system prompt to child's. Merged with `\n\n` separator. |
| `inherit_tools` | `list[str]` | `[]` | Whitelist of tool names to inherit from parent. Empty list = no tools inherited. |
| `inherit_permissions` | `bool` | `False` | Copy parent's `PermissionComponent` to child (tool whitelist/blacklist). |
| `allow_delegate_tool` | `bool` | `True` | Enable `delegate` tool on child (allows recursive delegation). |
| `tool_conflict_policy` | `str` | `"skip"` | How to resolve tool name conflicts: `"skip"` (ignore duplicate), `"error"` (raise), `"override"` (replace). |
| `missing_skill_policy` | `str` | `"warn"` | How to handle missing parent skills: `"warn"` (log warning), `"error"` (raise). |

### Inheritance Behavior

#### System Prompt Inheritance

When `inherit_system_prompt=True`, the parent's system prompt is **appended** to the child's:

```python
# Parent system prompt
parent_prompt = "You are a collaborative agent. Always verify sources."

# Child config
child_prompt = "You are a research specialist."

# Effective child prompt (merged)
effective_prompt = "You are a research specialist.\n\nYou are a collaborative agent. Always verify sources."
```

#### Tool Inheritance

Only tools explicitly listed in `inherit_tools` are copied from parent to child:

```python
# Parent has tools: ["search", "read", "write", "calculate"]

policy = InheritancePolicy(
    enabled=True,
    inherit_tools=["search", "read"],  # Only these two are inherited
)

# Child will receive: ["search", "read"]
# Child will NOT receive: ["write", "calculate"]
```

**Tool Conflict Resolution:**

- `skip` (default): If child already has a tool with the same name, parent's tool is ignored
- `error`: Raise `ValueError` if conflict detected
- `override`: Parent's tool replaces child's tool

#### Permission Inheritance

When `inherit_permissions=True`, the parent's `PermissionComponent` is copied to the child:

```python
# Parent has PermissionComponent with whitelist=["search", "read"]

policy = InheritancePolicy(
    enabled=True,
    inherit_permissions=True,
)

# Child receives identical PermissionComponent
# Child can only use tools in ["search", "read"]
```

#### Skill-Based Inheritance

If inherited tools come from skills, the SubagentSystem attempts to install those skills on the child:

```python
# Parent has SkillComponent with "web-search" skill (provides "search" tool)

policy = InheritancePolicy(
    enabled=True,
    inherit_tools=["search"],  # Tool from "web-search" skill
)

# SubagentSystem will:
# 1. Detect "search" tool comes from "web-search" skill
# 2. Attempt to install "web-search" skill on child
# 3. If skill is missing from parent, handle per missing_skill_policy
```

**Missing Skill Handling:**

- `warn` (default): Log warning and continue (tool will not be available on child)
- `error`: Raise `ValueError` and fail delegation

### Usage Examples

#### Example 1: Inherit Search Tool Only

```python
policy = InheritancePolicy(
    enabled=True,
    inherit_tools=["search"],  # Only search tool
    inherit_system_prompt=False,  # No prompt inheritance
)

config = SubagentConfig(
    name="researcher",
    provider=provider,
    model="gpt-4o",
    system_prompt="You are a research assistant.",
    inheritance_policy=policy,
)
```

#### Example 2: Full Capability Sharing

```python
policy = InheritancePolicy(
    enabled=True,
    inherit_system_prompt=True,
    inherit_tools=["search", "read", "write", "calculate"],
    inherit_permissions=True,
    tool_conflict_policy="override",
)

config = SubagentConfig(
    name="worker",
    provider=provider,
    model="gpt-4o",
    system_prompt="You are a worker agent.",
    inheritance_policy=policy,
)
```

#### Example 3: Isolated Child (No Inheritance)

```python
policy = InheritancePolicy(
    enabled=False,  # Disable all inheritance
)

config = SubagentConfig(
    name="isolated-agent",
    provider=provider,
    model="gpt-4o",
    system_prompt="You are isolated.",
    inheritance_policy=policy,
)
```


## Best Practices

### 1. Limit Subagent max_ticks

Prevent runaway subagents:

```python
SubagentConfig(
    name="worker",
    provider=provider,
    model="gpt-4o",
    system_prompt="...",
    max_ticks=5,  # Strict limit
    skills=[],
)
```

### 2. Use Specific System Prompts

Specialize each subagent:

```python
# Good
system_prompt="You are a fact-checker. Verify claims and cite sources."

# Bad
system_prompt="You are a helpful assistant."
```

### 3. Monitor Delegation Events

Track subagent usage:

```python
delegation_count = 0

async def track_delegations(event: DelegationStartedEvent) -> None:
    global delegation_count
    delegation_count += 1
    if delegation_count > 10:
        print("Warning: Excessive delegations detected")

world.event_bus.subscribe(DelegationStartedEvent, track_delegations)
```

### 4. Provide Clear Tasks

Delegate specific, well-defined tasks:

```python
# Good
task="Extract all dates mentioned in this text: [text]"

# Bad
task="Help me with this"
```
### 5. Inheritance Policy Best Practices

1. **Whitelist Tools Explicitly**: Only inherit tools the child actually needs. Avoid inheriting all parent tools.
2. **Use `skip` for Conflict Policy**: Prevents accidental tool overwrites. Use `override` only when intentional.
3. **Test Missing Skills**: Ensure parent has required skills installed before delegation if using `inherit_tools`.
4. **Disable Recursive Delegation**: Set `allow_delegate_tool=False` to prevent children from spawning sub-children.

## Limitations

- Subagents can delegate to other subagents if `allow_delegate_tool=True` in `InheritancePolicy` (recursive delegation supported)
- Subagent state is not persisted after execution completes
- Tool calls from subagents are isolated (cannot access parent tools)
- `TerminalComponent` from child world is NOT copied to parent (prevents premature runner termination). Additionally, the parent-world stub entity for each delegation carries a `ChildStubComponent`, which causes `ReasoningSystem` to skip it — preventing unintended LLM inference on completed delegation stubs
- After child world completes, the stub entity's `LLMComponent.system_prompt` is updated to reflect the effective (post-inheritance) system prompt used by the child
## See Also

- [Multi-Agent Collaboration](./multi-agent.md) — Entity-to-entity messaging
- [Tool Execution System](../systems.md#toolexecutionsystem) — Tool call processing
- [Event Bus](../core-concepts.md#eventbus) — Pub/sub events
