# Subagent Delegation

The `SubagentSystem` enables parent agents to spawn child agents for subtask execution via the `delegate` tool. The system auto-registers this tool, manages isolated child execution, and returns results to the parent.

## Overview

Subagent delegation provides:
- **Named Subagent Registry**: Pre-configure subagent profiles with specific capabilities
- **Isolated Execution**: Each subagent runs in its own `World` with independent state
- **Automatic Result Aggregation**: Results flow back to parent via tool result messages
- **Event Tracking**: Monitor delegation lifecycle with `DelegationStartedEvent` and `DelegationCompletedEvent`

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
    skills=[],  # Optional: list of skill names to load
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

### SubagentSystem

The system handles delegation and auto-registers the `delegate` tool:

```python
from ecs_agent.systems.subagent import SubagentSystem

# SubagentSystem should be registered BEFORE ReasoningSystem
world.register_system(SubagentSystem(priority=-1), priority=-1)
```

**IMPORTANT**: The SubagentSystem automatically registers the `delegate` tool for entities that have both `SubagentRegistryComponent` and `ToolRegistryComponent`. You do not need to manually register the delegate tool.

## Usage

### Basic Delegation

1. **Register subagents** with `SubagentRegistryComponent`
2. **Add ToolRegistryComponent** to enable delegate tool auto-registration
3. **Register SubagentSystem** (priority -1, before ReasoningSystem)
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

### Delegate Tool Usage

The LLM can call the `delegate` tool to invoke subagents:

```json
{
  "name": "delegate",
  "arguments": {
    "subagent_name": "researcher",
    "task": "Explain quantum entanglement in simple terms"
  }
}
```

The SubagentSystem will:
1. Look up "researcher" in the `SubagentRegistryComponent`
2. Create a new isolated `World` for the subagent
3. Run the subagent to completion with the task as a user message
4. Extract the final assistant response from the child conversation
5. Return result to parent as a tool result message
6. Parent ReasoningSystem receives the tool result and generates final response

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
    print(f"Result: {event.result}")

world.event_bus.subscribe(DelegationCompletedEvent, on_delegation_completed)
```

## Error Handling

If a subagent fails or produces an error, the result contains the error:

```python
# Subagent error is returned as tool result
result = "Error: Subagent 'researcher' failed: <error details>"
```

The parent can handle this via normal tool result processing.

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

## Limitations

- Subagents cannot delegate to other subagents (no recursive delegation)
- Subagent state is not persisted after execution completes
- Tool calls from subagents are isolated (cannot access parent tools)
- `TerminalComponent` from child world is NOT copied to parent (prevents premature runner termination)

## See Also

- [Multi-Agent Collaboration](./multi-agent.md) — Entity-to-entity messaging
- [Tool Execution System](../systems.md#toolexecutionsystem) — Tool call processing
- [Event Bus](../core-concepts.md#eventbus) — Pub/sub events
