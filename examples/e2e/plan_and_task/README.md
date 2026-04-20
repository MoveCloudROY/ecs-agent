# Plan and Task E2E Example

This example demonstrates an interactive plan→review→execute workflow using the ECS-based LLM Agent framework. It features a robust state machine, review-gated planning, artifact persistence, and recovery semantics.

## Overview

The workflow follows a structured lifecycle:
1. **Draft Interview**: The agent interviews the user to build a draft plan (`DRAFT_INTERVIEW`).
2. **Draft Reviews**: The draft must be approved by both an Advisor (`DRAFT_ADVISOR_REVIEW`) and a QA subagent (`DRAFT_QA_REVIEW`).
3. **Write Plan**: Once QA approves the draft, the system **automatically** transitions to `WRITE_PLAN` and triggers a dedicated `plan_writer` subagent (equipped with the `writing-plans` skill) to convert the approved draft into a structured `workflow_plan.md`. No manual command is needed.
4. **Plan QA Review**: When the plan writer finishes, the system **automatically** transitions to `PLAN_QA_REVIEW` where QA reviews the final plan document.
5. **Execution**: Once finalized, the plan is decomposed into a task queue and executed.

## Architecture

- **Built-in Tools** — The main agent has `read_file`, `write_file`, `edit_file`, `bash`, and `glob` tools pre-installed via `BuiltinToolsSkill`, workspace-bound to the example directory. `edit_file` uses hash-anchored `edits_json` only: supply a `pos` of `"N#HASH"` obtained from a prior `read_file` call.
- **ECS Core**: Uses `SystemPromptRenderSystem`, `UserPromptNormalizationSystem`, `ReasoningSystem`, `ToolExecutionSystem`, and `MemorySystem`.
- **Prompt Configuration**: The planner entity declares `SystemPromptConfigSpec` with `DRAFT_INTERVIEW_SYSTEM_PROMPT`, and `SystemPromptRenderSystem` bridges the rendered value into `LLMComponent.system_prompt` before reasoning.
- **State Machine**: Explicit phase transitions managed by `WorkflowStateMachine`.
- **Artifacts**: Durable persistence of plans, state, and execution evidence via `PlanTaskScratchbookAdapter`.
- **Controller**: `PlanController` manages the high-level workflow logic and review gates.
- **Subagent Reviews**: Advisor and QA review steps are wired as ECS subagents via `SubagentRegistryComponent`. The planner invokes them with `subagent(category="advisor", ...)` and verdicts are automatically extracted from subagent results via `DelegationCompletedEvent` subscription.
- **Plan Writer Subagent**: The `WRITE_PLAN` phase is executed by a dedicated `plan_writer` subagent registered in `SubagentRegistryComponent`. It is pre-loaded with the `writing-plans` skill (discovered from `.claude/skills/writing_plans/SKILL.md`) and inherits `read_file`, `write_file`, `edit_file`, and `glob` tools. When it completes, `handle_write_plan_completed()` transitions the state to `PLAN_QA_REVIEW`.
- **Task Execution**: `TaskExec` handles plan loading, dependency resolution, and subagent dispatch.
- **Slash Commands**: Dispatched via ECS `TriggerSpec` script handlers on `UserPromptConfigComponent`. Commands appear as transformed messages in conversation history.

## Supported Commands

The interactive runtime supports eleven slash commands:

- `/plan:start <description>`: Initialize a new workflow with a draft description.
- `/plan:resume <workflow_id>`: Restore a previously-started workflow from disk by its workflow ID (e.g. `creative-writing-assistant-with-llm-workflow`). Marks any in-flight subagents as stale and resumes from the persisted phase.
- `/plan:status`: Show the current workflow phase, status, and review verdicts.
- `/plan:finalize`: Finalize the plan and transition to task execution (requires all three approved reviews).
- `/plan:write`: Transition from `DRAFT_QA_REVIEW` to `WRITE_PLAN` phase to produce `workflow_plan.md`. **Optional** — this transition now happens automatically when QA approves the draft, but can still be invoked manually.
- `/plan:qa_review <approved|revise|blocked> [notes]`: Record a QA verdict on the final plan document.
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
1. Call `/plan:start <original description>` — the LLM re-derives the same slug from the same description (or uses `slug_from_description()` as fallback).
2. State is restored from `scratchbook/<workflow_id>/state/runtime_state.json`.
3. Any in-flight subagents are marked `stale` and the machine transitions to `TASK_BLOCKED` for safe resumption.

> **Note**: Use the same description text (or the same slug) as the original `/plan:start` call so the derived workflow ID matches the existing scratchbook directory.

### Mid-flight State Reconciliation

When `/plan:resume <workflow_id>` is called, the system reads the persisted state and automatically reconciles any in-progress phases so the workflow can continue without manual intervention:

| Resumed phase | Condition | Automatic action |
|---|---|---|
| `DRAFT_QA_REVIEW` | `review_verdicts` contains an `approved` verdict for this phase | Transitions to `WRITE_PLAN`, injects a write-plan trigger message to start the `plan_writer` subagent |
| `WRITE_PLAN` | (any — plan_writer was mid-flight) | Injects a write-plan trigger message to restart the `plan_writer` subagent |
| `PLAN_QA_REVIEW` | `review_verdicts` contains an `approved` verdict for this phase | Transitions to `PLAN_FINALIZED` |
| All other phases | — | No automatic action; resumes normally |

This means after a process restart you can call `/plan:resume <workflow_id>` and, if QA had already approved the draft before the restart, the `plan_writer` will be triggered automatically — no need to manually issue `/plan:write`.

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
- `DEBUG`: Set to `1` to enable debug logging. All `plan_task_*` structured log events will appear on stderr via structlog.

### Log Events (observable with `DEBUG=1`)

| Event | Level | File | Description |
|-------|-------|------|-------------|
| `plan_task_workflow_id_derived` | info | `runtime.py` | Workflow ID derived from LLM or fallback; `method=llm\|fallback`, `slug=` |
| `plan_task_workflow_id_llm_failed` | warning | `runtime.py` | LLM slug derivation failed; `exception=` |
| `plan_task_draft_written` | info | `scratchbook_adapter.py` | Draft written to disk; `path=` |
| `plan_task_state_loaded` | debug | `scratchbook_adapter.py` | Runtime state read from disk; `phase=` |
| `plan_task_event_appended` | debug | `scratchbook_adapter.py` | Event appended to events.jsonl; `event_type=` |
| `plan_task_memory_appended` | debug | `scratchbook_adapter.py` | Memory entry appended; `task_id=` |
| `plan_task_subagents_marked_stale` | info | `scratchbook_adapter.py` | In-flight subagents staled on restart; `stale_count=`, `task_ids=` |
| `plan_task_task_queue_initialized` | info | `task_exec.py` | Task queue built and state updated; `task_count=`, `current_task_id=`, `phase=` |
| `plan_task_subagent_dispatched` | info | `task_exec.py` | Subagent session recorded for a task; `task_id=`, `session_id=` |
| `plan_task_task_completed` | info | `task_exec.py` | Task completed; `next_task_id=`, `workflow_done=` |
| `plan_task_circuit_breaker_triggered` | warning | `task_exec.py` | Task retry budget exhausted; `retry_count=`, `max_retries=` |
| `plan_task_dependency_cycle_detected` | warning | `task_exec.py` | Cyclic dependency found before raise; `cycle_ids=` |
| `plan_task_reviews_not_approved` | warning | `task_exec.py` | Task start blocked by missing reviews; `missing_phases=` |
| `plan_task_finalize_blocked` | warning | `controller.py` | Plan finalization blocked by missing verdicts; `missing_phases=` |
| `plan_task_plan_artifact_missing` | warning | `controller.py` | Plan artifact file not found before raise; `path=` |
| `plan_task_plan_not_finalized` | warning | `plan_schema.py` | Plan status is not finalized before raise; `status=` |
| `plan_task_command_plan_start` | info | `main.py` | `/plan:start` succeeded; `workflow_id=`, `description_len=` |
| `plan_task_command_plan_resume` | info | `main.py` | `/plan:resume` succeeded; `workflow_id=`, `phase=` |
| `plan_task_command_plan_finalize` | info | `main.py` | `/plan:finalize` succeeded; `workflow_id=` |
| `plan_task_command_task_start` | info | `main.py` | `/task:start` succeeded; `task_count=`, `current_task_id=` |
| `plan_task_command_task_resume` | info | `main.py` | `/task:resume` succeeded; `workflow_id=` |
| `plan_task_command_task_replan` | info | `main.py` | `/task:replan` succeeded; `task_id=` |
| `plan_task_command_task_abort` | info | `main.py` | `/task:abort` succeeded; `task_id=` |
| `plan_task_command_plan_status` | debug | `main.py` | `/plan:status` invoked |
| `plan_task_command_task_status` | debug | `main.py` | `/task:status` invoked; `phase=` |
| `plan_task_command_error` | warning | `main.py` | A slash command raised ValueError; `command=`, `exception=` |
| `plan_task_llm_usage` | info | `billing.py` | Per-invocation token counts; `prompt_tokens=`, `completion_tokens=`, `total_tokens=`, `cached_input_tokens=`, `cache_creation_tokens=`, `cache_read_tokens=` |
| `plan_task_llm_cache_stats` | info | `billing.py` | Per-invocation cache hit-rate; `cache_read_tokens=`, `total_prompt_tokens=`, `cache_hit_rate=`. **Not emitted with DashScope** — DashScope Chat Completions API does not return cached token counts, so this event is only triggered when using providers that expose `cached_tokens` (e.g. OpenAI) or `cache_read_input_tokens` (e.g. Anthropic). |
| `plan_task_session_billing_summary` | info | `billing.py` | Cumulative token totals at end of session; `invocation_count=`, `total_prompt_tokens=`, `total_completion_tokens=`, `total_tokens=`, `total_cached_input_tokens=` |
| `accounting_invocation_recorded` | info | `ecs_agent.accounting` | Per-invocation cost + cache hit-rate from `AccountingSubscriber`; `total_cost=`, `cache_hit_rate=` |
| `plan_task_auto_transition_write_plan` | info | `controller.py` | QA review approved — auto-transitioned from `DRAFT_QA_REVIEW` to `WRITE_PLAN`; `workflow_id=` |
| `plan_task_auto_transition_plan_finalized` | info | `controller.py` | Plan QA review approved — auto-transitioned from `PLAN_QA_REVIEW` to `PLAN_FINALIZED`; `workflow_id=` |
| `plan_task_auto_trigger_plan_writer` | info | `main.py` | QA approved — injected write-plan trigger message to start `plan_writer` subagent automatically; `workflow_id=`, `source=` (omitted when triggered by live QA event; `source=reconcile_after_resume` when triggered on `/plan:resume`) |

## Implementation Details

- **Testable World Factory**: `build_plan_task_world(provider, model, base_dir)` is a public function that returns `(world, agent_id, adapter_ref, runtime_state)`, enabling direct world setup in tests without running the CLI. `adapter_ref` is a `list[ArtifactAdapter | None]` — starts as `[None]` and is populated in-place by the `/plan:start` handler after the workflow ID is derived.
- **workflow_id Auto-Derivation**: `/plan:start <description>` calls `derive_workflow_id_from_llm()` to ask the LLM to generate a short, meaningful English slug from the description (e.g., `"writing-assistant-multi-agent"`). Falls back to `slug_from_description()` on provider error or invalid output. The derived ID controls the scratchbook directory for all subsequent operations in that session.
- **Progressive Draft Editing**: The planning interview fills `draft.md` one section at a time using `read_file` (to get LINE#HASH annotated content) + `edit_file(edits_json=...)` (hash-anchored). The LLM reads the file first to capture `N#HASH` references, then replaces exactly the placeholder line. Full-file rewrites via `write_file` are explicitly prohibited by the system prompt.
- **Atomic Writes**: All artifact updates use atomic file operations to prevent corruption.
- **Circuit Breaker**: `TaskExec` implements a retry budget to prevent infinite loops on failing tasks.
- **Review Gating**: Finalization is strictly blocked until both `PLAN_ADVISOR_REVIEW` and `PLAN_QA_REVIEW` have `approved` verdicts.
- **Advisor Retry Loop**: When the advisor returns `revise` or `blocked`, the system prompt instructs the planner LLM to apply the feedback to `draft.md` via `edit_file` and re-call the advisor. Only an `approved` advisor verdict unlocks the QA step. All advisor verdicts are appended to `review_verdicts` and the last verdict per phase is used for gating.
- **Dependency Resolution**: Tasks are executed in topological order based on their `dependencies` list.
- **Token Prefix Caching**: DashScope Chat Completions API enables server-side prefix caching automatically — no client-side configuration needed. For the Responses API (`ApiFormat.OPENAI_RESPONSES`), set `enable_store=True` in `ProviderConfig` to opt into server-side storage/caching. **However, DashScope Chat Completions API does not return cached token counts in its response** (`prompt_tokens_details` contains only `text_tokens`, no `cached_tokens` field). As a result, `normalize_openai_usage()` always returns `cached_input_tokens=None` for DashScope, `compute_cache_stats()` returns `None`, and the `plan_task_llm_cache_stats` log event is **never emitted** when using DashScope. Cache hit-rate is only observable via `plan_task_llm_cache_stats` when using providers that expose cached token counts in their response — e.g. OpenAI (returns `prompt_tokens_details.cached_tokens`) or Anthropic (returns `cache_read_input_tokens`).
