# Implementation Notes

## Modules

- `commands.py`: Defines a closed slash-command grammar and parser for the workflow.
- `runtime.py`: Manages the interactive stdin loop and event-driven user input.
- `scratchbook_adapter.py`: Implements `PlanTaskScratchbookAdapter` for durable persistence of all workflow data via `ScratchbookService` (structured I/O) and `ArtifactRegistry` (canonical artifact records). Also exposes `build_scratchbook_prompt_config(workflow_id)` which returns a `ScratchbookPromptConfig` component wired on the agent entity so `SystemPromptRenderSystem` injects scratchbook context into system prompts.
- `state_models.py`: Dataclass definitions for `RuntimeState`, `TaskRecord`, `ReviewVerdict`, and `SubagentRecord`.
- `plan_schema.py`: Handles parsing and validation of the Markdown-based workflow plans with YAML frontmatter.
- `controller.py`: `PlanController` manages the high-level workflow logic, including plan initialization and review gating.
- `task_exec.py`: `TaskExec` handles plan loading, dependency-aware task queueing, and subagent execution context assembly.
- `state_machine.py`: `WorkflowStateMachine` defines valid phase transitions and handles process restart recovery.
- `prompts.py`: Contains system prompt templates. `PLAN_INTERVIEW_SYSTEM_PROMPT` is an f-string that embeds `build_advisor_prompt()` and `build_qa_prompt()` example outputs so the planner LLM knows the expected subagent prompt format.
- `main.py`: Entrypoint that bootstraps the ECS world. Exposes `build_plan_task_world(provider, model, workflow_id, base_dir)` as a public factory that installs `SubagentRegistryComponent` (with "advisor" and "qa" subagents), `SubagentSessionTableComponent`, and `SubagentSystem(priority=-1)`. `main()` calls this factory internally.

## Architecture Decisions

- **No TaskSystem/TaskComponent usage**: This example intentionally uses a custom `TaskExec` and `RuntimeState` to demonstrate manual orchestration and artifact-based persistence instead of the built-in ECS task components.
- **Prompt systems run before reasoning**: `SystemPromptRenderSystem` renders `PLAN_INTERVIEW_SYSTEM_PROMPT` from `SystemPromptConfigSpec` at priority `-20`, and `UserPromptNormalizationSystem` normalizes outbound user prompts at priority `-10` before the LLM turn begins.
- **ScratchbookService for I/O**: `write_state`/`read_state`/`append_event`/`append_memory`/`write_review_verdict` all go through `ScratchbookService`, which provides atomic index writes (`write_index`) and append-only logs (`append_log`). Plan file writes (`write_plan`, `write_draft`) use a local `_write_text_atomic` for Markdown content not suited to JSON serialization.
- **ScratchbookPromptConfig wired as ECS component**: `build_scratchbook_prompt_config(workflow_id)` is registered as a component on the agent entity. `SystemPromptRenderSystem` detects it and automatically creates a `ScratchbookPromptPlaceholderProvider` to inject scratchbook artifact context into system prompts — no manual provider registration required.
- **Strong State Machine**: Explicit phase transitions prevent invalid operations (e.g., starting a task before the plan is finalized).
- **Review-Gated Planning**: The workflow requires approved verdicts from both an Advisor and a QA subagent before a plan can be finalized.
- **Subagent-Driven Reviews**: Advisor and QA reviews are implemented as ECS subagents registered in `SubagentRegistryComponent`. The planner LLM calls `subagent(category="advisor", ...)` with the draft content.
- **Verdict Recording via DelegationCompletedEvent**: An event bus subscription automatically extracts verdicts (`approved`, `revise`, or `blocked`) from subagent result text using the regex `\b(approved|revise|blocked)\b` (case-insensitive). It defaults to `revise` if no match is found.
- **Trigger Dispatch**: Eight `TriggerSpec(action='script')` entries handle all slash commands inside the ECS pipeline, transforming them into workflow actions.
- **Circuit-Breaker for Delegation**: `TaskExec` tracks retry counts for each task and blocks execution if a task fails repeatedly.

## Artifact Layout

The system uses a canonical directory structure under `scratchbook/<workflow_id>/`:
- `plan/`: Drafts and finalized plans.
- `state/`: Runtime state, event logs, and task queues.
- `memory/`: Shared knowledge across tasks.
- `review/`: Structured review verdicts.

## Testing

  - **Integration Tests**: `tests/integration/test_plan_and_task_flow.py` covers the command surface, state machine, artifact persistence, and credential-gated CLI checks without depending on `FakeProvider`.
   - New tests: `test_main_world_setup_installs_subagent_infrastructure`, `test_delegation_event_subscription_updates_runtime_state`, `test_prompt_builders_return_non_empty_strings`, `test_build_scratchbook_prompt_config_returns_valid_config`, `test_main_world_adds_scratchbook_prompt_config_component`

 - **Live Tests**: `tests/live/test_plan_and_task_flow_live.py` provides credential-gated acceptance tests using a real LLM provider for the controller logic.
