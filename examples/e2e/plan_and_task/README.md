# Plan and Task E2E Example

This example demonstrates an interactive plan→review→execute workflow using the ECS-based LLM Agent framework. It features a robust state machine, review-gated planning, artifact persistence, and recovery semantics.

## Overview

The workflow follows a structured lifecycle:
1. **Planning**: The agent interviews the user to build a draft plan.
2. **Review**: The plan must be approved by both an Advisor and a QA subagent.
3. **Execution**: Once finalized, the plan is decomposed into a task queue and executed.

## Architecture

- **Built-in Tools** — The main agent has `read_file`, `write_file`, `edit_file`, `bash`, and `glob` tools pre-installed via `BuiltinToolsSkill`, workspace-bound to the example directory.
- **ECS Core**: Uses `SystemPromptRenderSystem`, `UserPromptNormalizationSystem`, `ReasoningSystem`, `ToolExecutionSystem`, and `MemorySystem`.
- **Prompt Configuration**: The planner entity declares `SystemPromptConfigSpec` with `PLAN_INTERVIEW_SYSTEM_PROMPT`, and `SystemPromptRenderSystem` bridges the rendered value into `LLMComponent.system_prompt` before reasoning.
- **State Machine**: Explicit phase transitions managed by `WorkflowStateMachine`.
- **Artifacts**: Durable persistence of plans, state, and execution evidence via `PlanTaskScratchbookAdapter`.
- **Controller**: `PlanController` manages the high-level workflow logic and review gates.
- **Subagent Reviews**: Advisor and QA review steps are wired as ECS subagents via `SubagentRegistryComponent`. The planner invokes them with `subagent(category="advisor", ...)` and verdicts are automatically extracted from subagent results via `DelegationCompletedEvent` subscription.
- **Task Execution**: `TaskExec` handles plan loading, dependency resolution, and subagent dispatch.
- **Slash Commands**: Dispatched via ECS `TriggerSpec` script handlers on `UserPromptConfigComponent`. Commands appear as transformed messages in conversation history.

## Supported Commands

The interactive runtime supports exactly eight slash commands:

- `/plan:start <description>`: Initialize a new workflow with a draft description.
- `/plan:status`: Show the current workflow phase, status, and review verdicts.
- `/plan:finalize`: Finalize the plan and transition to task execution (requires approved reviews).
- `/task:start <task_id>`: Start execution of a specific task.
- `/task:status`: Show the status of the current task and subagent sessions.
- `/task:resume`: Resume a blocked or replanned task.
- `/task:replan <reason>`: Request a replan for the current task.
- `/task:abort`: Abort the current task and transition to a terminal state.

## Artifact Layout

All workflow data is persisted in `scratchbook/<workflow_id>/`:

- `plan/`: Contains `draft.md` (working draft, included as `draft_plan` artifact) and `workflow_plan.md` (the single living plan file, edited in-place).
- `state/`: Contains `runtime_state.json`, `events.jsonl`, and `task_queue.json`.
- `memory/`: Contains `knowledge.jsonl` for cross-task context.
- `evidence/`: Directory for task execution artifacts.
- `review/`: Contains JSON verdicts from Advisor and QA reviews.

## Usage

### Interactive Mode

Run the entry point to start an interactive session.

```bash
LLM_API_KEY=your-api-key uv run python examples/e2e/plan_and_task/main.py
```

#### Multi-line input

The prompt supports multi-line messages. Press **Enter** to start a new line; submit with a **blank line** (press Enter on an empty line):

```
You> /plan:start 我想开发一个辅助写作软件，
... 支持长篇小说和剧本创作，
... 需要多 Agent 协作完成各章节生成。
...
         ↑ blank line submits
```

Single-line commands work as before — just type and press Enter, then Enter again on the empty continuation line:

```
You> /plan:status
...
```

`exit` or `quit` typed as the **first line** (followed by Enter + blank line) terminates the session. `Ctrl+D` (EOF) also exits cleanly.

### Automation Mode (piped input)

Automate interactions by piping commands. In pipe mode each `\n\n` (double newline) acts as a submit boundary:

```bash
printf '/plan:start Build demo\n\n/plan:status\n\nexit\n\n' | uv run python examples/e2e/plan_and_task/main.py
```

## Recovery / Restart

The workflow can be restarted at any time. On startup, no workflow ID is resolved and no scratchbook folder is created. Instead:
1. Call `/plan:start <original description>` — `slug_from_description()` re-derives the same ID from the same description.
2. State is restored from `scratchbook/<workflow_id>/state/runtime_state.json`.
3. Any in-flight subagents are marked `stale` and the machine transitions to `TASK_BLOCKED` for safe resumption.

> **Note**: Use the same description text (or the same slug) as the original `/plan:start` call so the derived workflow ID matches the existing scratchbook directory.

## Testing

### Integration tests

Run the integration suite to verify command parsing, state machine logic, artifact persistence, and credential-gated CLI coverage:

```bash
uv run pytest tests/integration/test_plan_and_task_flow.py -v
```

- `uv run pytest tests/integration/test_plan_and_task_flow.py -k "subagent"` — verifies subagent component wiring

### Specific test filters

```bash
uv run pytest tests/integration/test_plan_and_task_flow.py -k "commands"
uv run pytest tests/integration/test_plan_and_task_flow.py -k "artifacts"
```

### Real-LLM acceptance tests

Requires `LLM_API_KEY`. Verifies the controller and task execution with a real provider:

```bash
LLM_API_KEY="$LLM_API_KEY" \
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
LLM_MODEL=qwen3.5-flash \
uv run pytest tests/live/test_plan_and_task_flow_live.py -v
```

## Environment Variables

- `LLM_API_KEY`: OpenAI-compatible API key.
- `LLM_BASE_URL`: API base URL (defaults to DashScope).
- `LLM_MODEL`: Model ID (defaults to `qwen3.5-flash`).
- `PLAN_TASK_INTERACTIVE`: Set to `0` to disable interactive stdin.
- `DEBUG`: Set to `1` to enable debug logging.

## Implementation Details

- **Testable World Factory**: `build_plan_task_world(provider, model, base_dir)` is a public function that returns `(world, agent_id, adapter_ref, runtime_state)`, enabling direct world setup in tests without running the CLI. `adapter_ref` is a `list[ArtifactAdapter | None]` — starts as `[None]` and is populated in-place by the `/plan:start` handler after the workflow ID is derived.
- **workflow_id Auto-Derivation**: `/plan:start <description>` calls `slug_from_description()` to convert the natural language description into a URL-safe workflow ID. For example, `"Build a task manager"` becomes `"build-a-task-manager"`. CJK text is normalized and joined with hyphens. The derived ID controls the scratchbook directory for all subsequent operations in that session.
- **Progressive Draft Editing**: The planning interview fills `draft.md` one section at a time using `read_file` + `edit_file`. The LLM asks one question per turn and edits the corresponding section's placeholder text. Full-file rewrites via `write_file` are explicitly prohibited by the system prompt.
- **Atomic Writes**: All artifact updates use atomic file operations to prevent corruption.
- **Circuit Breaker**: `TaskExec` implements a retry budget to prevent infinite loops on failing tasks.
- **Review Gating**: Finalization is strictly blocked until both `PLAN_ADVISOR_REVIEW` and `PLAN_QA_REVIEW` have `approved` verdicts.
- **Dependency Resolution**: Tasks are executed in topological order based on their `dependencies` list.
