# Prompt Normalization

The prompt normalization pipeline replaces raw string concatenation with a declarative, two-phase system for assembling LLM prompts. It separates system prompt rendering (placeholder resolution) from user prompt normalization (trigger injection and context pooling), producing immutable rendered components that LLM callers consume directly.

## Architecture Overview

Prompt normalization runs as two ECS systems in every tick, before reasoning:

```
PromptConfigSpec          PromptConfigComponent
       |                         |
       v                         v
SystemPromptRenderSystem   UserPromptNormalizationSystem
  (priority -20)              (priority -10)
       |                         |
       v                         v
RenderedSystemPromptComponent  RenderedUserPromptComponent
       |                         |
       +--------+--------+------+
                |
                v
     ReasoningSystem / PlanningSystem / ReplanningSystem
```

1. **SystemPromptRenderSystem** (priority -20) reads `PromptConfigSpec`, resolves all `${name}` placeholders, and writes a `RenderedSystemPromptComponent`.
2. **UserPromptNormalizationSystem** (priority -10) reads `PromptConfigComponent`, injects keyword/event triggers and context pool entries into the last user message, and writes a `RenderedUserPromptComponent`.
3. **LLM callers** (`ReasoningSystem`, `PlanningSystem`, `ReplanningSystem`) read the rendered components instead of assembling prompts themselves.

Stored conversation history is never mutated. All injections are transient and scoped to the current tick.

## Quick Start

```python
import asyncio
from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    RenderedSystemPromptComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.prompts.contracts import PromptConfigSpec, PromptTemplateSource
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.types import CompletionResult, Message, ToolSchema


async def main() -> None:
    world = World()
    entity = world.create_entity()

    provider = FakeProvider(responses=[
        CompletionResult(message=Message(role="assistant", content="Hello!"))
    ])
    world.add_component(entity, LLMComponent(provider=provider, model="fake"))
    world.add_component(entity, ConversationComponent(
        messages=[Message(role="user", content="Hi")]
    ))

    # Declare a system prompt template with a built-in placeholder
    world.add_component(entity, PromptConfigSpec(
        template_source=PromptTemplateSource(
            inline="You are a helpful assistant.\n\nAvailable tools:\n${_installed_tools}"
        ),
    ))
    world.add_component(entity, ToolRegistryComponent(
        tools={
            "search": ToolSchema(
                name="search",
                description="Search the web",
                parameters={"type": "object", "properties": {}},
            )
        },
        handlers={},
    ))

    world.register_system(SystemPromptRenderSystem(priority=-20), priority=-20)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    runner = Runner()
    await runner.run(world, max_ticks=1)

    rendered = world.get_component(entity, RenderedSystemPromptComponent)
    print(rendered.text if rendered else "<missing>")
    # Output:
    # You are a helpful assistant.
    #
    # Available tools:
    # - search


asyncio.run(main())
```

## Contracts Reference

### PromptTemplateSource

Frozen dataclass specifying the template origin. Exactly one of `inline` or `file_path` must be set.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `inline` | `str \| None` | `None` | Inline template string |
| `file_path` | `str \| None` | `None` | Path to a UTF-8 template file |

Raises `ValueError` if both or neither field is provided.

```python
from ecs_agent.prompts.contracts import PromptTemplateSource

# Inline template
source = PromptTemplateSource(inline="You are ${role}.")

# File-based template
source = PromptTemplateSource(file_path="prompts/system.md")
```

### PlaceholderSpec

Frozen dataclass defining a single placeholder resolver.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `str` | (required) | Placeholder identifier (must match `[A-Za-z_][A-Za-z0-9_]*`) |
| `value` | `str \| Callable[[], str]` | (required) | Static string or zero-arg callable returning a string |

Names starting with `_` are reserved for built-in placeholders. Invalid names raise `ValueError`.

```python
from ecs_agent.prompts.contracts import PlaceholderSpec

# Static value
static = PlaceholderSpec(name="project_name", value="ecs-agent")

# Dynamic callable (invoked once per render)
dynamic = PlaceholderSpec(name="timestamp", value=lambda: "2025-01-01T00:00:00Z")
```

### PromptConfigSpec

Frozen dataclass used as an ECS component. Declares the system prompt template and user-defined placeholders. Processed by `SystemPromptRenderSystem`.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `template_source` | `PromptTemplateSource` | (required) | Template origin (inline or file) |
| `placeholders` | `list[PlaceholderSpec]` | `[]` | User-defined placeholder resolvers |

```python
from ecs_agent.prompts.contracts import PromptConfigSpec, PromptTemplateSource, PlaceholderSpec

world.add_component(entity, PromptConfigSpec(
    template_source=PromptTemplateSource(
        inline="You are ${role}. Tools:\n${_installed_tools}"
    ),
    placeholders=[PlaceholderSpec(name="role", value="a coding assistant")],
))
```

### TriggerSpec

Frozen dataclass defining a pattern-based trigger rule applied by `UserPromptNormalizationSystem`.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `pattern` | `str` | (required) | Text pattern to match against the user message |
| `match_mode` | `"keyword" \| "prefix" \| "contains"` | (required) | How to match the pattern |
| `action` | `"replace" \| "skill" \| "script"` | (required) | What to do on match |
| `content` | `str` | (required) | Content to inject or replace with |
| `priority` | `int` | `0` | Higher priority triggers are evaluated first |

```python
from ecs_agent.prompts.contracts import TriggerSpec

trigger = TriggerSpec(
    pattern="@refactor",
    match_mode="keyword",
    action="replace",
    content="Focus on refactoring: improve code quality without changing behavior.",
    priority=10,
)
```

## Components Reference

### PromptConfigComponent

Opts an entity into the user prompt normalization pipeline. Configures trigger templates and context pool behavior.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `trigger_templates` | `dict[str, str]` | `{}` | Mapping of trigger patterns to injection content |
| `enable_context_pool` | `bool` | `False` | Whether to inject context pool entries into user messages |
| `context_pool_max_chars` | `int` | `8192` | Maximum characters for rendered context pool |

**Used by:** `UserPromptNormalizationSystem`

```python
from ecs_agent.components import PromptConfigComponent

world.add_component(entity, PromptConfigComponent(
    trigger_templates={
        "@code": "Prioritize deterministic code-first reasoning.",
        "event:tool_success": "Prefer successful tool outputs as evidence.",
    },
    enable_context_pool=True,
))
```

### OneShotContextPoolComponent

Session-scoped injection pool holding ranked context entries from tools, subagents, or other sources.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `items` | `list[tuple[int, int, str, str]]` | `[]` | Tuples of `(priority, order, source_label, content)` |
| `state` | `str` | `"idle"` | Reservation state: `"idle"`, `"reserved"`, or `"committed"` |
| `reserved_turn_id` | `str` | `""` | Turn ID that made the current reservation |
| `reserved_counter_snapshot` | `int` | `-1` | Counter value at reservation time |
| `reserved_items` | `list[tuple[int, int, str, str]]` | `[]` | Snapshot of items at reservation time |
| `_counter` | `int` | `0` | Monotonically increasing registration counter |

**Used by:** `UserPromptNormalizationSystem`

```python
from ecs_agent.components import OneShotContextPoolComponent

world.add_component(entity, OneShotContextPoolComponent(
    items=[
        (30, 0, "tool:search", "source: tool:search\nstatus: success\nresult: citation-A"),
        (20, 1, "subagent:researcher", "source: subagent:researcher\nresult: synthesis-B"),
    ],
    _counter=2,
))
```

### TurnStateComponent

Tracks per-turn identifiers for idempotent injection and deduplication.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `current_turn_id` | `str` | `""` | ID of the current turn |
| `last_injected_turn_id` | `str` | `""` | ID of the last turn where injection occurred |

**Used by:** `UserPromptNormalizationSystem`

```python
from ecs_agent.components import TurnStateComponent

world.add_component(entity, TurnStateComponent(current_turn_id="turn-001"))
```

### RenderedSystemPromptComponent

Output component written by `SystemPromptRenderSystem`. Contains the fully resolved system prompt.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `text` | `str` | (required) | Rendered system prompt with all placeholders resolved |
| `placeholder_snapshot` | `dict[str, str]` | `{}` | Snapshot of all resolved placeholder values |

**Produced by:** `SystemPromptRenderSystem`
**Consumed by:** `ReasoningSystem`, `PlanningSystem`, `ReplanningSystem`

### RenderedUserPromptComponent

Output component written by `UserPromptNormalizationSystem`. Contains the normalized user message with all injections applied.

| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `text` | `str` | (required) | Normalized user message |
| `turn_id` | `str` | (required) | Turn ID for this render |

**Produced by:** `UserPromptNormalizationSystem`
**Consumed by:** `ReasoningSystem`, `PlanningSystem`, `ReplanningSystem`

## Systems Reference

### SystemPromptRenderSystem

Resolves `${name}` placeholders in system prompt templates and produces a `RenderedSystemPromptComponent`.

**Constructor:** `SystemPromptRenderSystem(priority: int = 0)`
**Recommended priority:** `-20`

Processing steps:
1. Query all entities with a `PromptConfigSpec` component.
2. Read the template from the configured source (inline string or file path).
3. Resolve user-defined placeholders via `PlaceholderSpec` values.
4. Resolve built-in placeholders (`_installed_tools`, `_installed_skills`, etc.) from entity components.
5. Substitute all `${name}` references using Python `string.Template`.
6. Write a `RenderedSystemPromptComponent` to the entity.
7. Update `LLMComponent.system_prompt` for backward compatibility.

Raises `ValueError` on missing placeholders. No silent fallbacks.

```python
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem

world.register_system(SystemPromptRenderSystem(priority=-20), priority=-20)
```

### UserPromptNormalizationSystem

Injects trigger templates and context pool entries into the last user message without mutating stored conversation history.

**Constructor:** `UserPromptNormalizationSystem(priority: int = 0, trigger_specs: list[TriggerSpec] | None = None)`
**Recommended priority:** `-10`

Processing steps:
1. Query entities with `ConversationComponent` or `ConversationTreeComponent`.
2. Find the last user message.
3. Apply keyword triggers: if user text contains a trigger key (e.g., `@code`), prepend the trigger block with a `[PROMPT_INJECT:@keyword]` marker.
4. Apply event triggers: match `event:<name>` patterns against active events derived from context pool sources.
5. Apply context pool injection: render `OneShotContextPoolComponent` items sorted by priority (descending), wrapped in a `[PROMPT_CONTEXT_POOL]` marker.
6. Apply `TriggerSpec` rules passed via constructor (`replace` action replaces the entire message; other actions prepend).
7. Write a `RenderedUserPromptComponent` to the entity.

Deduplication: if a `[PROMPT_INJECT:...]` marker is already present in the user text, it is not doubled.

```python
from ecs_agent.systems.user_prompt_normalization_system import UserPromptNormalizationSystem

world.register_system(UserPromptNormalizationSystem(priority=-10), priority=-10)
```

## Built-in Placeholders

These placeholders are automatically resolved by `SystemPromptRenderSystem` from entity components. They do not need to be declared in `PromptConfigSpec.placeholders`.

| Placeholder | Source Component | Description |
| :--- | :--- | :--- |
| `${_installed_tools}` | `ToolRegistryComponent` | Bullet list of registered tool names |
| `${_installed_skills}` | `SkillComponent` | Bullet list of installed skill names |
| `${_installed_mcps}` | `MCPClientComponent` | Bullet list of MCP tool names |
| `${_installed_subagents}` | `SubagentRegistryComponent` | Bullet list of subagent names |

If a source component is absent from the entity, the placeholder resolves to `"- none"`.

## Trigger Templates

Trigger templates inject contextual prompt blocks into user messages based on pattern matching. They are configured via `PromptConfigComponent.trigger_templates`.

### Keyword Triggers

Keys that do not start with `event:` are treated as keyword triggers. When the pattern appears anywhere in the user message, the corresponding content is prepended.

```python
trigger_templates = {
    "@code": "Prioritize deterministic code-first reasoning.",
    "@help": "Provide step-by-step guidance with examples.",
}
```

A user message `"Please @code summarize findings"` produces:

```
[PROMPT_INJECT:@code]
Prioritize deterministic code-first reasoning.

Please @code summarize findings
```

### Event Triggers

Keys starting with `event:` match against active events derived from context pool source labels. If a matching event is found, the content is injected.

```python
trigger_templates = {
    "event:tool_success": "Prefer successful tool outputs as evidence.",
}
```

Events are extracted from `OneShotContextPoolComponent` item source labels (e.g., `tool:search` produces events `tool:search`, `tool`, and `tool_success` if the content contains `status: success`).

### Idempotency

If a `[PROMPT_INJECT:...]` marker for a given keyword is already present in the user text, the system does not inject it again.

## Context Pool Injection

When `PromptConfigComponent.enable_context_pool` is `True` and an `OneShotContextPoolComponent` is present, context entries are injected into the user message.

Items are sorted by priority (descending), then by registration order (ascending). The rendered block is wrapped in a `[PROMPT_CONTEXT_POOL]` marker and placed before the original user text.

```
[PROMPT_INJECT:@code]
Prioritize deterministic code-first reasoning.

[PROMPT_CONTEXT_POOL]
source: tool:search
status: success
result: citation-A

---

source: subagent:researcher
status: success
result: synthesis-B

Please @code summarize latest findings
```

The original user message is always preserved as the tail of the normalized text.

## Callable Placeholders

Placeholder values can be zero-argument callables that return a string. The callable is invoked exactly once per render cycle.

```python
import time
from ecs_agent.prompts.contracts import PlaceholderSpec

PlaceholderSpec(name="render_time", value=lambda: str(int(time.time())))
```

Requirements:
- Must return `str` (raises `ValueError` otherwise).
- Must be side-effect-free.
- Exceptions propagate immediately (no silent fallback).

## File-Based Templates

Templates can be loaded from the filesystem instead of inline strings.

```python
from ecs_agent.prompts.contracts import PromptTemplateSource

source = PromptTemplateSource(file_path="prompts/system_prompt.md")
```

The file is read as UTF-8. The same `${name}` placeholder syntax applies. Raises `ValueError` if the file does not exist or is unreadable.

## Error Handling

The prompt normalization pipeline fails loudly. There are no silent fallbacks.

| Condition | Exception | Message |
| :--- | :--- | :--- |
| Both/neither source in `PromptTemplateSource` | `ValueError` | `"requires exactly one of inline or file_path"` |
| Unknown placeholder in template | `ValueError` | `"unknown placeholders in template: {name}"` |
| Invalid placeholder name | `ValueError` | `"Invalid placeholder name: must match [A-Za-z_][A-Za-z0-9_]*"` |
| Reserved placeholder name (starts with `_`) | `ValueError` | `"names starting with '_' are reserved"` |
| Callable returns non-str | `ValueError` | `"must return str"` |
| Missing template file | `ValueError` | `"missing template file: {path}"` |

All errors are logged via structlog before re-raising.

## Example

Run the included demo to see the full pipeline in action:

```bash
# Fake mode (no API key needed)
uv run python examples/prompt_normalization_demo.py

# Real LLM mode
LLM_API_KEY=your-key uv run python examples/prompt_normalization_demo.py
```

The demo exercises:
- System prompt rendering with `${_installed_tools}` built-in placeholder
- Keyword trigger injection (`@code`)
- Event trigger configuration (`event:tool_success`)
- Context pool injection from tool and subagent sources
- Rendered component inspection (`RenderedSystemPromptComponent`, `RenderedUserPromptComponent`)

See [`examples/prompt_normalization_demo.py`](../../examples/prompt_normalization_demo.py) for the full source.

## See Also

- [Components](../components.md) — Full component reference including prompt components
- [Systems](../systems.md) — All built-in systems with priority recommendations
- [Skills](./skills.md) — Skill system and trigger template registration
