# Subagent Delegation

The `SubagentSystem` enables parent agents to spawn child agents for subtask execution via the `subagent` tool. The system supports synchronous and background execution modes, with isolated environments and policy-based capability inheritance.


## Overview

Subagent delegation provides:
- **Named Subagent Registry**: Pre-configure subagent profiles with specific capabilities
- **Isolated Execution**: Each subagent runs in its own `World` with independent state
- **Automatic Result Aggregation**: Results flow back to parent via tool result messages.
- **Event Tracking**: Monitor delegation lifecycle with `DelegationStartedEvent` and `DelegationCompletedEvent`.
- **Skill Inheritance**: Subagents can inherit specific skills, system prompts (via `SystemPromptConfigSpec` and `SystemPromptRenderSystem`), and tools from their parent agent via `InheritancePolicy`.
- **Sync and Background Modes**: Execute tasks immediately or as background sessions with ID tracking.
- **Lifecycle Management**: Track background sessions through `queued`, `running`, `succeeded`, `failed`, `timed_out`, and `cancelled` states.
- **Scheduler & Concurrency**: Process-global FIFO queue with configurable concurrency limits and automatic re-enqueuing on world restore.
- **Control Tools**: Tools to query status (including queue position), retrieve results (explicit wait-based), and cancel background sessions (atomic for queued tasks). Includes `subagent_wait` for non-polling workflows.
- **Timeout Policy**: Per-call timeout overrides with global fallback and automated handling.
- **Retry Reliability**: Transparent `RetryModel` wrapping for transient LLM failures.


## Core Components

### SubagentConfig

Define subagent profiles:

```python
from ecs_agent.types import SubagentConfig
from ecs_agent.providers import FakeModel

researcher_config = SubagentConfig(
    name="researcher",
    model=FakeModel(responses=[...]),
    system_prompt="You are a research specialist. Provide detailed, factual information.",
    max_ticks=10,
    skills=[],  # Skill names to load and install on this subagent
    inheritance_policy=InheritancePolicy(inherit_system_prompt=True),  # Optional configuration
)
```
### Automatic Placeholder Injection

When `_assemble_child_world` builds the child world, it calls `_build_child_prompt_template` on the effective system prompt before storing it in `SystemPromptConfigSpec`. This helper appends standard sections for `${_installed_tools}` and `${_installed_skills}` unless those placeholders are already present in the prompt string:

```python
# If SubagentConfig.system_prompt does NOT contain ${_installed_tools} or ${_installed_skills},
# the following sections are automatically appended:
#
#   \n\n## Available Tools
#   ${_installed_tools}
#   \n\n## Available Skills
#   ${_installed_skills}
```

`SystemPromptRenderSystem` (priority -20) then resolves these placeholders at runtime from the child entity's `ToolRegistryComponent` and `SkillComponent`, so the child agent's rendered system prompt always reflects its actual installed tools and skills.

To suppress the auto-append for a specific placeholder, simply include it yourself in `SubagentConfig.system_prompt`:

```python
SubagentConfig(
    name="researcher",
    model=model,
    system_prompt=(
        "You are a research specialist.\n\n"
        "## My Tools\n${_installed_tools}"  # prevents auto-append of tools section
    ),
)
```

### SubagentRegistryComponent

Register named subagents and, optionally, enable free-form delegation:

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

By default, `category` must match a key in `subagents`. To let the parent agent call arbitrary named workers, opt in with `FreeSubagentConfig`:

```python
from ecs_agent.components import SubagentRegistryComponent
from ecs_agent.types import FreeSubagentConfig

world.add_component(
    entity,
    SubagentRegistryComponent(
        free_subagent_config=FreeSubagentConfig(enabled=True),
    ),
)
```

When free-form mode is enabled, `subagent(category="security-reviewer", prompt="...")` creates a dynamic `SubagentConfig` using the parent entity's `LLMComponent.model`, the free-mode system prompt template, and the configured default skills/inheritance policy. Registered subagents still take precedence, so teams can mix curated names with ad hoc specialists.
### SubagentSessionRecord

Track background session metadata:

```python
from ecs_agent.types import SubagentSessionRecord

record = SubagentSessionRecord(
    session_id="session_123",
    category="researcher",
    prompt="...",
    status="running",
    timeout_seconds=30.0,
    artifact_id="subagent_1234567890abcdef12345678",
    artifact_record_path="scratchbook/records/subagent/subagent_1234567890abcdef12345678",
    artifact_inline_content=None,
    # ... other fields
)
```

Subagent result persistence is registry-backed:

- Durable outputs are persisted to `scratchbook/records/subagent/subagent_<uuid24>`.
- `session_id` is runtime/session-local and ephemeral.
- `artifact_id` is the durable artifact identifier for long-term references.
- `artifact_record_path` and `artifact_inline_content` expose canonical record location and inline-threshold behavior on `SubagentSessionRecord`.

### SubagentLifecycleStatus

Background sessions transition through a strict state machine:

| State | Description |
| :--- | :--- |
| `queued` | Waiting in the FIFO queue for an available concurrency slot. |
| `running` | Currently executing in a child world. |
| `succeeded` | Completed successfully; result is available. |
| `failed` | Terminated with an error. |
| `timed_out` | Terminated after exceeding timeout limit. |
| `cancelled` | Terminated by explicit cancel request (atomic for queued tasks). |

### Scheduler & Concurrency

The `SubagentSystem` manages a process-global FIFO queue for background sessions.

- **Concurrency Limit**: Configured via `SubagentSystem(max_background_concurrency=N)`. Default is 5.
- **FIFO Queue**: Sessions are processed in the order they were launched.
- **Cap Conflict**: If multiple `SubagentSystem` instances are registered with different concurrency caps, a `ValueError` is raised to prevent ambiguous scheduling behavior.

### Restore Semantics

When a `World` is restored from a serialized state (e.g., after a process restart):

- **Queued Sessions**: Automatically re-enqueued in the scheduler to resume their wait.
- **Running Sessions**: Since the live task handle is lost, these are marked as `failed` with the error `restored_without_live_task_handle`. They do not automatically resume execution and do not emit parent notifications, because this is restore-time lifecycle reconciliation rather than a subagent execution result.

### Subagent Control Installer

Entities using background subagents must have the `SubagentSessionTableComponent` and call `install_subagent_control_tools` to enable control tools.

```python
subagent_system = SubagentSystem()
world.register_system(subagent_system)
subagent_system.install_subagent_control_tools(world, entity_id)
```

### Registration

Register the `SubagentSystem` and `SubagentWaitSystem` during world setup:

```python
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.systems.subagent_wait import SubagentWaitSystem

# Register SubagentSystem (priority -1 recommended to run before ReasoningSystem).
world.register_system(SubagentSystem(priority=-1), priority=-1)

# To enable free-form subagent names for all entities with ToolRegistryComponent
# and expose that capability in ${_installed_subagents}, run SubagentSystem before
# SystemPromptRenderSystem (recommended priority -30) instead of the previous line:
# world.register_system(
#     SubagentSystem(priority=-30, allow_unregistered_subagents=True),
#     priority=-30,
# )

# Register SubagentWaitSystem (priority -5 REQUIRED to run before ReasoningSystem)
# This system handles the subagent_wait tool and notification delivery.
world.register_system(SubagentWaitSystem(priority=-5), priority=-5)
```


## Usage

### Basic Delegation

1. **Register subagents** with `SubagentRegistryComponent`
2. **Register SubagentSystem** (priority -1, before ReasoningSystem)
3. **SubagentSystem creates a new child entity** with the subagent's provider, model, and a `SystemPromptConfigSpec` whose inline template is built by `_build_child_prompt_template` (auto-appending `${_installed_tools}` and `${_installed_skills}` sections if not already present), plus a `ChildStubComponent` to mark the parent-world stub entity
4. **LLM calls subagent tool** to invoke subagent
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
from ecs_agent.systems.subagent_wait import SubagentWaitSystem
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

# Add empty ToolRegistryComponent (SubagentSystem will auto-register subagent tool)
world.add_component(
    parent,
    ToolRegistryComponent(tools={}, handlers={}),
)

# Add LLM and conversation
world.add_component(
    parent,
    LLMComponent(model=your_model),
)
world.add_component(
    parent,
    ConversationComponent(
         messages=[
            Message(role="user", content="Research quantum computing and summarize.")
         ]
     ),
)

# Register systems (SubagentWaitSystem and SubagentSystem BEFORE ReasoningSystem)
world.register_system(SubagentWaitSystem(priority=-5), priority=-5)
world.register_system(SubagentSystem(priority=-1), priority=-1)
world.register_system(ReasoningSystem(priority=0), priority=0)
world.register_system(ToolExecutionSystem(priority=5), priority=5)
# ... register other systems (ErrorHandlingSystem, etc.)

# Run
runner = Runner()
await runner.run(world, max_ticks=20)
```

### Free-Form Delegation

Free-form delegation is opt-in. It is useful when you want the model to invent focused roles on demand instead of pre-registering every possible worker name.

```python
from ecs_agent.components import LLMComponent, ToolRegistryComponent
from ecs_agent.systems.subagent import SubagentSystem

world.add_component(parent, LLMComponent(model=your_model))
world.add_component(parent, ToolRegistryComponent(tools={}, handlers={}))

world.register_system(
    SubagentSystem(priority=-30, allow_unregistered_subagents=True),
    priority=-30,
)
```

With that option enabled, `SubagentSystem` creates a `SubagentRegistryComponent` if the entity does not already have one, installs the `subagent` tool, and updates the tool description plus `${_installed_subagents}` prompt inventory to say that arbitrary unregistered category names are allowed. Use a priority earlier than `SystemPromptRenderSystem` so this inventory hint is available during the first rendered prompt. If a requested category is registered, its explicit `SubagentConfig` is used; otherwise a dynamic config is built from the parent model.

### `subagent` Tool Usage

The `subagent` tool enables parent agents to delegate subtasks with support for background execution and skill overrides:

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

1. **Check Status**: `subagent_status(session_id="session_123")` returns current lifecycle state (including `queue_position` if `queued`) and a summary table if `session_id` is omitted.
2. **Retrieve Result**: `subagent_result(session_id="session_123", read_method="full", timeout=10.0)` polls durable metadata and returns the result once the session is terminal. It does not require a live task handle. Supports `read_method="summary"` for cached summary retrieval.
3. **Cancel**: `subagent_cancel(session_id="session_123")` terminates the session. For `queued` sessions, this is an atomic removal from the scheduler.

## Non-Polling Background Workflow

The recommended way to handle background subagents is using the explicit wait-notification model. This avoids polling and allows the parent agent to sleep until results are ready.

### `subagent_wait` Tool

The `subagent_wait` tool puts the parent agent into a future-based wait state until one or more background sessions complete.

```json
{
  "name": "subagent_wait",
  "arguments": {
    "session_ids": ["session-a", "session-b"],
    "timeout": 60.0
  }
}
```

- **`session_ids`**: Optional list of session IDs to wait for. If `null` or omitted, waits for ANY background session to complete.
- **`timeout`**: Optional maximum seconds to wait.
- **Behavior**: The parent agent is woken up automatically when a matching session reaches a terminal state.

### Durable Notification Semantics

When a background session completes, the system enqueues a durable unread notification for the parent:
- **Wake-worthy states**: `succeeded`, `failed`, and `timed_out` sessions generate notifications.
- **Non-wake-worthy states**: `cancelled` sessions do NOT generate notifications.
- **Persistence**: Notifications survive world save/load (restore-safe).

### Wake Notification Delivery

When the parent agent is woken by `SubagentWaitSystem`, it receives ONE compact `system` message per wake cycle, even if multiple sessions completed (batched delivery).

**Example Notification:**
> Background subagent updates:
> - session-abc succeeded. Call subagent_result(session_id="session-abc") for the full result or subagent_result(session_id="session-abc", read_method="summary") for the cached summary.

Delivered notifications are marked as read and are not re-delivered after a world restore.

### `subagent_result` with Summary Reads

The `subagent_result` tool supports a `read_method` parameter to optimize context usage:

| Field | Type | Description |
|-------|------|-------------|
| `read_method` | `"full"` \| `"summary"` | Default is `"full"`. `"summary"` returns the cached summary if available. |

- **`read_method="full"`**: Returns the complete result (backward-compatible).
- **`read_method="summary"`**: Returns the cached summary inline. If no summary is available (e.g., the subagent didn't emit a summary envelope), it returns an error payload.

### Summary Envelope Format

Background subagents can emit a specific XML-like envelope to provide a cacheable summary:

```xml
<subagent_background_result>
<summary>Brief summary of the work performed.</summary>
<full_result>The complete, detailed output.</full_result>
</subagent_background_result>
```

If this envelope is present in the subagent's final message, the `summary` content is cached and made available via `read_method="summary"`.

### Recommended Parent Flow

```python
# 1. Launch background subagents
subagent(category="researcher", prompt="...", background=true) # returns "session-a"

# 2. Enter explicit wait
subagent_wait()

# 3. Receive system notification when session-a completes

# 4. Read result (using summary for efficiency)
subagent_result(session_id="session-a", read_method="summary")
```

#### Result Payload Fields

The `subagent_result` tool returns a JSON payload with the following fields. **Internal timestamp fields and excerpts are excluded** to avoid polluting LLM context:

| Field | Type | Description |
|-------|------|-------------|
| `status` | `"success"` \| `"error"` | Whether the subagent execution succeeded or failed. |
| `session_id` | `str` | The unique session identifier for this delegation. |
| `category` | `str` | The subagent category/name used for this delegation. |
| `lifecycle_status` | `str` | Current state machine status: `"queued"`, `"running"`, `"succeeded"`, `"failed"`, `"timed_out"`, or `"cancelled"`. |
| `read_method` | `"full"` \| `"summary"` | The method used to read the result. Default is `"full"`. |
| `artifact_id` | `str` or `null` | Durable artifact identifier (e.g., `subagent_<uuid24>`) for persisted results. |
| `record_path` | `str` or `null` | Canonical record location (e.g., `scratchbook/records/subagent/subagent_<uuid24>`). |
| `inline_content` | `str` or `null` | Full result text (if `read_method="full"` and size ≤ 8192 bytes) or cached summary (if `read_method="summary"`). |
| `error` | `str` or `null` | Error message if execution failed (status is `"error"`), otherwise `null`. |
| `queue_position` | `int` (optional) | Queue position (0-indexed) only present when `lifecycle_status` is `"queued"`. |

**Fields NOT in payload** (internal use only, present on `SubagentSessionRecord` dataclass but excluded from JSON):
- `created_at` — Session creation timestamp (internal scheduling)
- `updated_at` — Last update timestamp (internal tracking)
- `started_at` — Execution start time (internal telemetry)
- `finished_at` — Execution end time (internal telemetry)
- `result_excerpt` — First 200 characters of result (internal UI/rendering for reminder tables)

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
1. Call subagent(category="researcher", prompt="Research AI safety concerns")
2. Call subagent(category="writer", prompt="Write blog post: [research results]")
3. Call subagent(category="critic", prompt="Review this draft: [blog post]")
4. Revise based on feedback
```

### Retry and Reliability

By default, all subagent LLM models are wrapped in a `RetryModel` using a standard `RetryConfig`. This handles transient network errors and rate limits automatically. `FakeModel` used in tests is exempt from this wrapping to maintain deterministic behavior.
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

 ## Limitations

 - Subagent state is not persisted after execution completes
 - Tool calls from subagents are isolated (cannot access parent tools)

- `TerminalComponent` from child world is NOT copied to parent (prevents premature runner termination). Additionally, the parent-world stub entity for each delegation carries a `ChildStubComponent`, which causes `ReasoningSystem` to skip it — preventing unintended LLM inference on completed delegation stubs
- After child world completes, the stub entity's `LLMComponent.system_prompt` reflects the effective rendered prompt (including expanded `${_installed_tools}` and `${_installed_skills}` sections) produced by `SystemPromptRenderSystem` in the child world during execution.
## See Also

- [Multi-Agent Collaboration](./multi-agent.md) — Entity-to-entity messaging
- [Tool Execution System](../systems.md#toolexecutionsystem) — Tool call processing
- [Event Bus](../core-concepts.md#eventbus) — Pub/sub events
