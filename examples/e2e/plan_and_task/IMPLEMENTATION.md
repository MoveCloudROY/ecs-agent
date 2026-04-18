# Implementation Notes

## Modules

- `commands.py`: Defines a closed slash-command grammar and parser for the workflow.
- `runtime.py`: Manages the interactive stdin loop and event-driven user input. Provides `slug_from_description(text)` (URL-safe fallback slug from text) and `derive_workflow_id_from_llm(description, provider)` (async — calls the LLM to generate a short, meaningful English slug; falls back to `slug_from_description` on error or invalid output).
- `scratchbook_adapter.py`: Implements `PlanTaskScratchbookAdapter` for durable persistence of all workflow data via `ScratchbookService` (structured I/O) and `ArtifactRegistry` (canonical artifact records). Also exposes `build_scratchbook_prompt_config(workflow_id)` which returns a `ScratchbookPromptConfig` component wired on the agent entity so `SystemPromptRenderSystem` injects scratchbook context into system prompts.
- `state_models.py`: Dataclass definitions for `RuntimeState`, `TaskRecord`, `ReviewVerdict`, and `SubagentRecord`.
- `plan_schema.py`: Handles parsing and validation of the Markdown-based workflow plans with YAML frontmatter.
- `controller.py`: `PlanController` manages the high-level workflow logic, including plan initialization and review gating.
- `task_exec.py`: `TaskExec` handles plan loading, dependency-aware task queueing, and subagent execution context assembly.
- `state_machine.py`: `WorkflowStateMachine` defines valid phase transitions and handles process restart recovery.
- `prompts.py`: Contains system prompt templates. `PLAN_INTERVIEW_SYSTEM_PROMPT` enforces a **progressive interview protocol**: the LLM asks one question at a time, calls `read_file` to get the LINE#HASH annotated content, then uses `edit_file(edits_json=...)` (hash-anchored) to fill in the matching section of `draft.md` incrementally. Full rewrites via `write_file` are explicitly discouraged.
- `main.py`: Entrypoint that bootstraps the ECS world. Exposes `build_plan_task_world(provider, model, base_dir)` as a public factory returning `(world, agent_id, adapter_ref: list[ArtifactAdapter | None], runtime_state)`. `adapter_ref` starts as `[None]` — no workflow ID is resolved and no scratchbook folder is created at startup. The `/plan:start` handler calls `derive_workflow_id_from_llm(description, provider)` to get a meaningful slug, creates the `ArtifactAdapter`, adds `ScratchbookPromptConfig` to the ECS world, and sets `adapter_ref[0]` in-place. Installs `SubagentRegistryComponent` (with "advisor" and "qa" subagents), `SubagentSessionTableComponent`, and `SubagentSystem(priority=-1)`.

## Architecture Decisions

- **No TaskSystem/TaskComponent usage**: This example intentionally uses a custom `TaskExec` and `RuntimeState` to demonstrate manual orchestration and artifact-based persistence instead of the built-in ECS task components.
- **Prompt systems run before reasoning**: `SystemPromptRenderSystem` renders `PLAN_INTERVIEW_SYSTEM_PROMPT` from `SystemPromptConfigSpec` at priority `-20`, and `UserPromptNormalizationSystem` normalizes outbound user prompts at priority `-10` before the LLM turn begins.
- **ScratchbookService for I/O**: `write_state`/`read_state`/`append_event`/`append_memory`/`write_review_verdict` all go through `ScratchbookService`, which provides atomic index writes (`write_index`) and append-only logs (`append_log`). Plan file writes (`write_plan`, `write_draft`) use a local `_write_text_atomic` for Markdown content not suited to JSON serialization.
- **ScratchbookPromptConfig wired as ECS component**: `build_scratchbook_prompt_config(workflow_id)` is registered as a component on the agent entity. `SystemPromptRenderSystem` detects it and automatically creates a `ScratchbookPromptPlaceholderProvider` to inject scratchbook artifact context into system prompts — no manual provider registration required.
- **Progressive Draft Editing**: `PLAN_INTERVIEW_SYSTEM_PROMPT` establishes a strict `read_file` → `edit_file(edits_json=...)` workflow. The initial `draft.md` template contains seven structured sections each with unique placeholder text. The LLM reads the file first (getting LINE#HASH annotated output), then fills each section one at a time per conversation turn using hash-anchored edits. This prevents full-file rewrites and makes each edit traceable.

- **workflow_id Auto-Derivation**: When `/plan:start <description>` is called, `derive_workflow_id_from_llm(description, provider)` asks the LLM to generate a short English slug (e.g., `"writing-assistant-multi-agent"`). Falls back to `slug_from_description(description)` on provider error or invalid output. The `adapter_ref[0]` is swapped in-place to the derived id so all downstream handlers target the correct scratchbook directory.

- **edit_file hash-anchored interface**: `edit_file` accepts only `file_path`, `edits_json`, and `workspace_root`. The `edits_json` parameter is an array of `{"op": "replace", "pos": "N#HASH", "lines": [...]}` operations where `N#HASH` is the line reference obtained from a prior `read_file` call. The system prompt instructs the LLM to always read first to capture LINE#HASH values, then apply targeted replacements.


- **Subagent-Driven Reviews**: Advisor and QA reviews are implemented as ECS subagents registered in `SubagentRegistryComponent`. The planner LLM calls `subagent(category="advisor", ...)` with the draft content.
- **Verdict Recording via DelegationCompletedEvent**: An event bus subscription automatically extracts verdicts (`approved`, `revise`, or `blocked`) from subagent result text using the regex `\b(approved|revise|blocked)\b` (case-insensitive). It defaults to `revise` if no match is found.
- **Advisor Retry Loop**: When the advisor returns `revise` or `blocked`, `PLAN_INTERVIEW_SYSTEM_PROMPT` instructs the planner LLM to read the feedback, apply edits to `draft.md` via `edit_file`, and re-call the advisor subagent. The QA subagent is only called once the advisor returns `approved`. All advisor verdicts are appended to `review_verdicts`; `_missing_approved_reviews` uses the last verdict per phase to determine gating.
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
   - New tests: `test_main_world_setup_installs_subagent_infrastructure`, `test_delegation_event_subscription_updates_runtime_state`, `test_prompt_builders_return_non_empty_strings`, `test_build_scratchbook_prompt_config_returns_valid_config`, `test_main_world_does_not_add_scratchbook_prompt_config_at_init`, `test_plan_interview_system_prompt_instructs_edit_file_usage`, `test_plan_interview_system_prompt_defines_interview_flow`, `test_draft_template_has_structured_sections`, `test_draft_template_has_placeholder_content`, `test_edit_file_schema_exposes_edits_json`, `test_edit_file_old_str_replaces_content`, `test_edit_file_raises_when_old_str_not_found`, `test_edit_file_raises_on_invalid_edits_json`, `test_derive_workflow_id_uses_llm_slug`, `test_derive_workflow_id_normalizes_llm_output`, `test_derive_workflow_id_falls_back_on_empty_response`, `test_derive_workflow_id_falls_back_on_provider_error`

 - **Live Tests**: `tests/live/test_plan_and_task_flow_live.py` provides credential-gated acceptance tests using a real LLM provider. Includes `test_live_derive_workflow_id_from_llm_returns_valid_slug` which verifies the LLM returns a valid `^[a-z][a-z0-9-]*$` slug (3–50 chars) for a real Chinese task description.
