# Implementation Notes

## Modules

- `commands.py`: Defines a closed slash-command grammar and parser for the workflow.
- `runtime.py`: Manages the interactive stdin loop and event-driven user input.
- `artifacts.py`: Implements `ArtifactAdapter` for atomic, durable persistence of all workflow data.
- `state_models.py`: Dataclass definitions for `RuntimeState`, `TaskRecord`, `ReviewVerdict`, and `SubagentRecord`.
- `plan_schema.py`: Handles parsing and validation of the Markdown-based workflow plans with YAML frontmatter.
- `controller.py`: `PlanController` manages the high-level workflow logic, including plan initialization and review gating.
- `task_exec.py`: `TaskExec` handles plan loading, dependency-aware task queueing, and subagent execution context assembly.
- `state_machine.py`: `WorkflowStateMachine` defines valid phase transitions and handles process restart recovery.
- `prompts.py`: Contains system prompt templates. `PLAN_INTERVIEW_SYSTEM_PROMPT` is an f-string that embeds `build_advisor_prompt()` and `build_qa_prompt()` example outputs so the planner LLM knows the expected subagent prompt format before calling `record_advisor_verdict` / `record_qa_verdict`.
- `main.py`: Entrypoint that bootstraps the ECS world. Exposes `build_plan_task_world(provider, model, workflow_id, base_dir)` as a public factory that installs `SubagentRegistryComponent` (with "advisor" and "qa" subagents), `ToolRegistryComponent` (with `record_advisor_verdict` and `record_qa_verdict` handlers), `SubagentSessionTableComponent`, and `SubagentSystem(priority=-1)`. `main()` calls this factory internally.

## Architecture Decisions

- **No TaskSystem/TaskComponent usage**: This example intentionally uses a custom `TaskExec` and `RuntimeState` to demonstrate manual orchestration and artifact-based persistence instead of the built-in ECS task components.
- **Prompt systems run before reasoning**: `SystemPromptRenderSystem` renders `PLAN_INTERVIEW_SYSTEM_PROMPT` from `SystemPromptConfigSpec` at priority `-20`, and `UserPromptNormalizationSystem` normalizes outbound user prompts at priority `-10` before the LLM turn begins.
- **Persisted State + Atomic Writes**: All state changes are persisted to disk using atomic file operations (temp file + rename) to ensure consistency even on crashes.
- **Strong State Machine**: Explicit phase transitions prevent invalid operations (e.g., starting a task before the plan is finalized).
- **Review-Gated Planning**: The workflow requires approved verdicts from both an Advisor and a QA subagent before a plan can be finalized.
- **Subagent-Driven Reviews**: Advisor and QA reviews are implemented as ECS subagents registered in `SubagentRegistryComponent` — not as manual prompt injection. The planner LLM calls `subagent(category="advisor", ...)` with the draft content, then calls `record_advisor_verdict(verdict, notes)` to persist the result via `PlanController.handle_advisor_review()`.
- **Circuit-Breaker for Delegation**: `TaskExec` tracks retry counts for each task and blocks execution if a task fails repeatedly.

## Artifact Layout

The system uses a canonical directory structure under `.artifacts/workflows/<workflow_id>/`:
- `plan/`: Drafts and finalized plans.
- `state/`: Runtime state, event logs, and task queues.
- `memory/`: Shared knowledge across tasks.
- `review/`: Structured review verdicts.

## Testing

 - **Integration Tests**: `tests/integration/test_plan_and_task_flow.py` covers the command surface, state machine, artifact persistence, and credential-gated CLI checks without depending on `FakeProvider`.
   - New tests: `test_main_world_setup_installs_subagent_infrastructure`, `test_verdict_tool_handlers_update_runtime_state`, `test_prompt_builders_return_non_empty_strings`
 - **Live Tests**: `tests/live/test_plan_and_task_flow_live.py` provides credential-gated acceptance tests using a real LLM provider for the controller logic.
