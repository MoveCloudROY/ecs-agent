# Agent Scratchbook

`ArtifactRegistry` is the canonical persistence API for scratchbook artifacts. It centralizes durable IDs, canonical filesystem paths, atomic writes, and inline-content behavior so every system persists artifacts the same way.

## Why ArtifactRegistry is Central

`ArtifactRegistry` is used by planning, replanning, tool execution, trigger script normalization, and subagent result persistence. This eliminates path drift and prevents category-by-category storage conventions from diverging.

Key guarantees:

- Canonical IDs for immutable records (`tool_<uuid24>`, `subagent_<uuid24>`)
- Canonical plan/Boulder paths derived from normalized plan names
- Atomic writes (`temp file + os.replace`) for crash-safe updates
- Unified inline-content threshold behavior (UTF-8 byte based)

## Canonical Storage Paths

The registry owns all canonical scratchbook paths:

- Tool result record: `scratchbook/records/tool/tool_<uuid24>`
- Subagent result record: `scratchbook/records/subagent/subagent_<uuid24>`
- Human-readable plan document: `scratchbook/<plan_slug>/plan.md`
- Machine-readable execution state: `scratchbook/<plan_slug>/executes/boulder.json`

Legacy category examples (`tool_results/`, `planning/`, `replanning/`) are migration-only history and are not canonical runtime paths.

## Inline Content Threshold (8 KB)

When persisting immutable records (`persist()` or `capture_stream()`), `inline_content` follows a strict UTF-8 byte threshold:

- Payload `<= 8192` bytes: `inline_content` is populated with full content
- Payload `> 8192` bytes: `inline_content` is `None` (content remains file-backed)

This keeps small artifacts cheap to inline while keeping large payloads out of hot context paths.

## Boulder Lifecycle (Trigger → Execution)

`boulder.json` models mutable execution state for a plan:

1. **Creation**: `UserPromptNormalizationSystem` creates Boulder on plan-type script trigger entry (`plan`, `replan`, `planning` trigger signatures).
2. **Planning/Replanning updates**: planning systems update step status/progress.
3. **Tool transition updates**: tool execution updates status and last tool metadata.

Required Boulder identity/session fields are preserved through updates:

- `schema_version`
- `plan_name`
- `active_plan`
- `started_at`

## Prompt Provider

The `ScratchbookPromptPlaceholderProvider` injects scratchbook context into system prompts via the `ScratchbookPromptConfig` component. This allows the agent to be aware of its persistent scratchbook, available artifact types, and specific artifact paths.

### Configuration

To enable scratchbook placeholders, attach a `ScratchbookPromptConfig` component to the agent entity:

```python
from ecs_agent.scratchbook.prompt_definition import (
    ScratchbookArtifactPromptDef,
    ScratchbookPromptConfig,
)

world.add_component(entity_id, ScratchbookPromptConfig(
    scratchbook_root_path="scratchbook",
    overview_default_template=(
        "You have access to a scratchbook at ${_scratchbook_path}.\n"
        "Artifact types available:\n${_scratchbook_artifact_types}\n"
        "Use builtin tools to read/write artifacts."
    ),
    artifacts=[
        ScratchbookArtifactPromptDef(
            artifact_type_id="tool_output",
            path="scratchbook/records/tool",
            purpose="Immutable records of tool call outputs.",
            readonly=True,
            read_when="When you need to reference a past tool result.",
        ),
        ScratchbookArtifactPromptDef(
            artifact_type_id="plan",
            path="scratchbook/plan.md",
            purpose="Active plan and execution state.",
            readonly=False,
            read_when="Before each reasoning step.",
        ),
    ],
))
```

### Placeholder Surface

The scratchbook provider exposes several built-in placeholders:

| Placeholder | Description |
| :--- | :--- |
| `${_scratchbook_path}` | The root-relative path to the scratchbook directory. |
| `${_scratchbook_artifact_types}` | A sorted bullet list of registered artifact type IDs. |
| `${_scratchbook_artifacts}` | A sorted join of all per-type artifact blocks. |
| `${_scratchbook_overview}` | The full overview block rendered from `overview_default_template`. |
| `${_scratchbook_artifact_<type>}` | The rendered block for a specific artifact type (e.g., `_scratchbook_artifact_plan`). |
| `${_scratchbook_artifact_path_<type>}` | The path for a specific artifact type (e.g., `_scratchbook_artifact_path_plan`). |

### Empty-State Behavior

If a placeholder is not applicable (e.g., no artifacts are registered), its value defaults to `"- none"`. This ensures that the system prompt remains valid even when the scratchbook is empty.

### Artifact Type Normalization

Artifact type IDs are normalized to ensure they are safe for use as placeholder names:
- Converted to lowercase.
- Non-alphanumeric characters are replaced with underscores (`_`).
- Leading and trailing underscores are trimmed.
- Multiple consecutive underscores are collapsed into one.

For example, `"Plan Notes"` becomes `plan_notes`, and the corresponding placeholder is `${_scratchbook_artifact_plan_notes}`.

### Template Overrides

You can override the default template for a specific artifact type using `user_override_template`:

```python
ScratchbookArtifactPromptDef(
    artifact_type_id="plan",
    path="scratchbook/plan.md",
    purpose="Active plan.",
    readonly=False,
    read_when="Before each step.",
    user_override_template=(
        "## Active Plan\n"
        "File: ${_scratchbook_artifact_path_plan}\n"
        "This file is WRITABLE. Update it as you complete steps.\n"
    ),
)
```

### System Prompt Example

A system prompt template can consume these placeholders to provide the agent with scratchbook context:

```
You are an expert AI assistant with access to a persistent scratchbook.

${_scratchbook_overview}

## Plan
${_scratchbook_artifact_plan}

## Tool Outputs
${_scratchbook_artifact_tool_output}
```

## API Reference (ArtifactRegistry)

```python
from pathlib import Path
from ecs_agent.scratchbook import ArtifactKind, ArtifactRegistry

registry = ArtifactRegistry(root=Path("."))

# Immutable records
result = registry.persist(kind=ArtifactKind.TOOL, content='{"ok": true}')
print(result.record_path)      # scratchbook/records/tool/tool_<uuid24>
print(result.inline_content)   # str | None (8 KB threshold)

# Human-facing mutable plan document
plan_path = registry.write_plan(
    plan_name="Release Plan",
    content="# Plan: Release Plan\n",
)

# Machine-facing mutable plan state
boulder_path = registry.create_boulder(
    plan_name="Release Plan",
    initial_data={"trigger_pattern": "@plan"},
)
registry.update_boulder(
    plan_name="Release Plan",
    updates={"status": "running", "current_step": 1},
)

# Async producer capture (stream -> immutable record)
async def stream_source() -> AsyncIterator[str]:
    yield "chunk-1"
    yield "chunk-2"

descriptor = await registry.capture_stream(
    kind=ArtifactKind.SUBAGENT,
    source=stream_source(),
)
print(descriptor.record_path)  # scratchbook/records/subagent/subagent_<uuid24>
```

## Constructor and Key Methods

- `ArtifactRegistry(root: Path)`
- `persist(kind: ArtifactKind, content: str) -> ArtifactPersistResult`
- `write_plan(plan_name: str, content: str) -> str`
- `create_boulder(plan_name: str, initial_data: dict[str, Any]) -> str`
- `update_boulder(plan_name: str, updates: dict[str, Any]) -> str`
- `capture_stream(kind: ArtifactKind, source: AsyncIterator[str]) -> ArtifactDescriptor`

## See Also

- [Task Orchestration System](task-system.md) — How tasks resolve canonical artifact references.
- [Subagent Delegation](subagent.md) — Subagent output persistence and artifact metadata.
- [API Reference](../api-reference.md) — Full signatures and module-level APIs.
