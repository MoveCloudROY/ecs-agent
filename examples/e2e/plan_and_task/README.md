# Plan and Task E2E Example

This example demonstrates an interactive plan→review→execute workflow using the ECS-based LLM Agent framework. It features a recoverable phase graph (`ecs_agent.phases`), review-gated planning, artifact persistence, recovery semantics, and framework-native auto compaction for both the main agent and spawned subagents.

## Overview

The workflow follows a structured lifecycle:
1. **Draft Interview**: The agent proactively drafts the plan — proposing a recommended choice per section and asking the user to confirm or redirect — rather than interrogating them (`DRAFT_INTERVIEW`).
2. **Draft Reviews**: The draft must be approved by both an Advisor (`DRAFT_ADVISOR_REVIEW`) and a QA subagent (`DRAFT_QA_REVIEW`).
3. **Write Plan**: Once QA approves the draft, the `DRAFT_QA_REVIEW` phase's `ApprovalGate` **automatically** routes the workflow to `WRITE_PLAN` and triggers a dedicated `plan_writer` subagent (equipped with the `writing-plans` skill) to convert the approved draft into a structured `workflow_plan.md`. No manual command is needed.
4. **Plan QA Review**: When the plan writer finishes, the system **automatically** advances to `PLAN_QA_REVIEW` where QA reviews the final plan document.
5. **Execution**: Once finalized, the plan is decomposed into a task queue and executed.

The lifecycle is declared once as a phase graph (`PLAN_TASK_PHASE_GRAPH` in `phase_graph.py`, built on `ecs_agent.phases`). The current phase lives in a `PhaseComponent` on the agent entity — the runtime source of truth — and every transition goes through `advance()` (validated against the graph's edges) or `force()` (used only by `/plan:start`, which may begin a new workflow from any phase, including terminal ones). Review verdicts are routed declaratively by each review phase's `ApprovalGate`, and the persisted `RuntimeState` (`phase`/`graph_hash`/`status`) is a snapshot of `PhaseComponent` stamped at persist time by `phase_sync.save_state()` — the single state-write path.

## Architecture

- **Built-in Tools** — The main agent has `read_file`, `write_file`, `edit_file`, `bash`, `glob`, and `webfetch` tools pre-installed via `BuiltinToolsSkill`, workspace-bound to the example directory. `read_file` returns clean content, while `edit_file` keeps its original `op`/`pos`/`end`/`content` signature with `pos`/`end` as 1-based file line numbers validated against framework-managed internal snapshots. The main agent has unrestricted access to all these tools.
- **Web Search (optional)** — When `BRAVE_API_KEY` is set, `build_plan_task_world` installs `WebSearchSkill` on the main agent, adding a `web_search` tool (Brave Search API) that is advertised to the interviewer through `${_installed_tools}` like any other built-in tool. It lets the planner ground its draft proposals — scope, requirements, constraints — in current facts, and stays available across every phase the main agent drives. Without the key the tool is simply omitted and the workflow runs unchanged (the always-on `webfetch` tool remains available for fetching a known URL).
- **Subagent Tool Permissions** — `advisor`, `qa`, and `plan_qa` review subagents inherit only `read_file` and `glob` (read-only) via `InheritancePolicy(inherit_tools=["read_file", "glob"], inherit_permissions=True)`. They cannot write files or run shell commands. `plan_writer` inherits `read_file`, `write_file`, `edit_file`, and `glob` since it must produce the final plan document.
- **Two Distinct QA Subagents** — Draft QA (`qa`) and Plan QA (`plan_qa`) are registered as separate subagents with separate system prompts and separate transition paths in the phase graph:
  - `qa` uses `QA_SYSTEM_PROMPT` (draft review lens) and routes `DelegationCompletedEvent` → `controller.handle_qa_review()` → `DRAFT_QA_REVIEW` verdict.
  - `plan_qa` uses `PLAN_QA_REVIEW_SYSTEM_PROMPT` (final plan review lens) and routes `DelegationCompletedEvent` → `controller.handle_plan_qa_review()` → `PLAN_QA_REVIEW` verdict.
  - The planner system prompt calls `subagent(category="qa", ...)` for draft review and `subagent(category="plan_qa", ...)` for plan review.
- **Review Prompts via File Path** — When invoking `advisor`, `qa`, `plan_qa`, or `plan_writer` subagents for review, the prompt passes the artifact file path (e.g. `scratchbook/<workflow_id>/plan/draft.md`) rather than embedding the file content inline. The subagent reads the file itself using `read_file`, avoiding prompt token bloat.
- **Reviewer Prompt Layering** — Each reviewer subagent's system prompt owns its full review checklist and the shared `VERDICT: approved|revise|blocked` output contract (`_VERDICT_CONTRACT` in `prompts.py`); the main agent only sends short dispatch prompts (`build_advisor_prompt` / `build_qa_prompt` / `build_plan_qa_prompt`) pointing at the artifact file. Checklists are therefore applied even if the main agent paraphrases, and each checklist has exactly one authority. `IDLE` uses a dedicated minimal prompt (`IDLE_MAIN_AGENT_SYSTEM_PROMPT`) that routes the user to `/plan:start` instead of the interview protocol, since no draft exists before a workflow starts.
- **Auto Compaction** — `build_plan_task_world(...)` installs `CompactionConfigComponent(threshold_tokens=300_000, compaction_method="predrop_then_compact")` by default, `ConversationArchiveComponent`, and `CompactionSystem` at priority `-30` so compaction runs before workflow prompt rendering and before reasoning. `SystemPromptRenderSystem` then injects the current summary into the effective system prompt as `<chat_history_summary>...</chat_history_summary>` XML.
- **Subagent Compaction Inheritance** — Child worlds created by `SubagentSystem` inherit the parent `CompactionConfigComponent`, receive their own `ConversationArchiveComponent`, and register `CompactionSystem` at the same priority. Long-running review and task subagents therefore compact independently without requiring plan-and-task-specific special cases.
- **Workflow Reset Safety** — `/plan:start`, `/plan:resume`, and `/task:start <workflow_id>` clear stale `CurrentCompactionSummaryComponent`, reset archived summaries, and invalidate `RenderedSystemPromptComponent` before restoring or switching workflow state. This prevents an old summary from leaking into a newly loaded workflow phase.
- **Log Truncation** — Structured log fields `last_user_prompt` and user-normalization `prompt_text` are truncated to 200 characters to keep logs readable without losing signal. System-prompt render logs still report `prompt_length`, but the rendered prompt text itself is not truncated in this example.
- **ECS Core**: Uses `SystemPromptRenderSystem`, `UserPromptNormalizationSystem`, `ReasoningSystem`, and `ToolExecutionSystem`.
- **Prompt Configuration**: The agent entity declares `SystemPromptConfigSpec` with the inline template `${_phase_prompt}`. `SystemPromptRenderSystem` resolves that placeholder from the phase graph's per-phase prompt bindings for the current `PhaseComponent.phase` (each `PhaseSpec` maps the `main` agent key to `IDLE_MAIN_AGENT_SYSTEM_PROMPT`, `PLAN_MAIN_AGENT_SYSTEM_PROMPT`, or `TASK_MAIN_AGENT_SYSTEM_PROMPT`) and bridges the rendered value into `LLMComponent.system_prompt` before reasoning.
- **Phase Graph**: `build_plan_task_world(...)` binds `PLAN_TASK_PHASE_GRAPH` via `await bind_phase_graph(world, agent_id, ..., agent_key="main")`. Per-phase system prompts are selected automatically through the `${_phase_prompt}` placeholder — no polling system is registered; prompt changes take effect on the transition itself.
- **Phase Transitions**: Explicit transitions via `ecs_agent.phases` — `advance()` for graph-validated moves, `force()` for `/plan:start`, `record_approval()` for review-verdict routing through each review phase's `ApprovalGate`. Every transition is recorded in `PhaseComponent.history` with a reason.
- **Artifacts**: Durable persistence of plans, state, and execution evidence via `PlanTaskScratchbookAdapter`. Main-agent tool results are currently kept inline in ECS conversation/tool-result state rather than being written through `ToolResultsSink`.
- **Controller**: `PlanController` manages the high-level workflow logic and review gates.
- **Subagent Reviews**: Advisor, QA, and Plan QA review steps are wired as ECS subagents via `SubagentRegistryComponent`. The planner invokes them with `subagent(category="advisor", ...)`, `subagent(category="qa", ...)`, and `subagent(category="plan_qa", ...)` respectively. Verdicts are automatically extracted from subagent results via `DelegationCompletedEvent` subscription, routed to the correct controller method based on the subagent name.
- **Plan Writer Subagent**: The `WRITE_PLAN` phase is executed by a dedicated `plan_writer` subagent registered in `SubagentRegistryComponent`. It is pre-loaded with the `writing-plans` skill (discovered from `.claude/skills/writing_plans/SKILL.md`) and inherits `read_file`, `write_file`, `edit_file`, and `glob` tools. When it completes, `handle_write_plan_completed()` transitions the state to `PLAN_QA_REVIEW`.
- **Task Execution**: `TaskExec` handles plan loading, dependency resolution, and subagent dispatch.
- **Slash Commands**: Dispatched via ECS `TriggerSpec` script handlers on `UserPromptConfigComponent`. Commands appear as transformed messages in conversation history.
- **System Execution Order**: `UserInputSystem` runs at priority **-15** (before `UserPromptNormalizationSystem` at -10). This ensures the user's message is already in `ConversationComponent` when script handlers fire, so slash commands like `/task:start` are matched in the same tick they are entered.

## Interactive `ask_question` Tool

`ask_tool.py` installs an `ask_question` tool on the main agent (via `install_ask_question_tool(world, agent_id)` in `build_plan_task_world`) that lets the agent pause mid-turn and put structured questions to the user. It is the interview flow's primary way to ask: the planner system prompt directs the agent to put every proposal's confirm/tweak/redirect choice — and any genuinely open question — through `ask_question` rather than plain prose.

- **Shape** — the agent passes 1-4 questions, each with a short `header`, the full `question` text, an optional `multi_select` flag, and 2-4 `options` (`{label, description}`); omitting `options` makes it a free-text question. Payloads are validated in `parse_questions`; a bad call returns an `Error: ...` string the model can correct.
- **Mechanism** — the tool is UI-agnostic. Its handler publishes a `UserQuestionRequestedEvent` carrying an `asyncio.Future` and blocks on it; a front end presents the questions and resolves the future with the collected `QuestionAnswer` list (or `None` if dismissed). The tool is **not** `concurrency_safe`, so the concurrent tool executor treats it as a barrier — it never overlaps sibling tool calls. If no front end is attached (e.g. a non-interactive run), the handler fails fast (`EventBus.has_subscribers`) instead of hanging.
- **TUI front end** — `PlanTaskTuiBridge` captures the future and the view model folds the event into a `question` `UiChange`; the app opens `QuestionScreen` (a `ModalScreen`) with radio groups, selection lists, and free-text fields. Submitting resolves the future via `bridge.submit_answers`; `Ctrl+Q`/quit resolves any pending question as dismissed so the world tick can settle.
- **stdin front end** — `runtime.setup_interactive_input` subscribes the same event and prompts for each answer on the terminal (numbered choices for option questions, free text otherwise), so `ask_question` works identically in the REPL.
- **Result** — answers return to the model as JSON `{answers: [{header, question, answer}]}`, where `answer` flattens the chosen label(s) and any custom text; a dismissed prompt returns `{cancelled: true, ...}`.

## Supported Commands

The interactive runtime supports eleven slash commands:

- `/plan:start <description>`: Initialize a new workflow with a draft description.
- `/plan:resume <workflow_id>`: Restore a previously-started workflow from disk by its workflow ID (e.g. `creative-writing-assistant-with-llm-workflow`). Marks any in-flight subagents as stale and resumes from the persisted phase; if that phase is `TASK_RUNNING`, the graph's `on_resume` policy demotes it to `TASK_BLOCKED` at load.
- `/plan:status`: Show the current workflow phase, status, and review verdicts.
- `/plan:finalize`: Finalize the plan and transition to task execution (requires all three approved reviews).
- `/plan:write`: Transition from `DRAFT_QA_REVIEW` to `WRITE_PLAN` phase to produce `workflow_plan.md`. **Optional** — the `DRAFT_QA_REVIEW` `ApprovalGate` performs this transition automatically when QA approves the draft, but it can still be invoked manually.
- `/plan:qa_review <approved|revise|blocked> [notes]`: Record a QA verdict on the final plan document.
- `/task:start <workflow_id>`: Start execution of a specific task. If no workflow is active in the current session, providing a `workflow_id` auto-loads the persisted state from scratchbook (equivalent to `/plan:resume <workflow_id>` followed by starting task execution). Accepts phases `PLAN_FINALIZED`, `TASK_READY`, `TASK_RUNNING`, and `TASK_BLOCKED`. If the auto-loaded state is still in `PLAN_QA_REVIEW` but already has an `approved` `PLAN_QA_REVIEW` verdict, the auto-load replays that phase's `ApprovalGate` (advancing to `PLAN_FINALIZED`) before initializing the task queue; with a non-approved `PLAN_QA_REVIEW` verdict the command fails with the task-queue initialization error and the phase stays `PLAN_QA_REVIEW`. Re-initialization carries completed tasks forward: tasks already in `completed_task_ids` rebuild as `"completed"` and are not re-run — only pending work is dispatched; if nothing is pending the workflow advances to `TASK_COMPLETED`.
- `/task:status`: Show the status of the current task and subagent sessions.
- `/task:resume`: Resume a blocked or replanned task.
- `/task:replan <reason>`: Request a replan for the current task.
- `/task:abort`: Abort the current task and transition to a terminal state.

## Artifact Layout

All workflow data is persisted in `scratchbook/<workflow_id>/`:

- `plan/`: Contains `draft.md` (working draft, included as `draft_plan` artifact) and `workflow_plan.md` (the single living plan file, edited in-place).
- `state/`: Contains `runtime_state.json` (whose `tasks` field is the live task queue) and `events.jsonl` — one `phase_transition` line per committed phase transition (from/to/reason/forced/tick), written by a `PhaseChangedEvent` subscriber.
- `memory/`: Contains `knowledge.jsonl` for cross-task context.
- `evidence/`: Directory for task execution artifacts.
- `review/`: Contains JSON verdicts from Advisor and QA reviews.

Main-agent tool call results are not currently persisted as separate canonical records by this example. They remain inline in ECS tool result state and conversation tool messages while durable workflow artifacts continue to live under `scratchbook/<workflow_id>/`.

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

### TUI Mode

A [Textual](https://textual.textualize.io/)-based terminal UI over the same world, commands, and environment variables (install the `tui` extra first — `uv sync --group dev` already includes it):

```bash
LLM_API_KEY=your-api-key uv run python -m examples.e2e.plan_and_task.tui
```

What it renders, all driven by the ECS event bus (design notes: `docs/plans/2026-07-11-plan-and-task-tui.md`):

- **Transcript pane** — token-streamed assistant output (reasoning dimmed, content as Markdown), slash-command results as titled panels, tool call/result lines, and subagent lifecycle lines.
- **Sidebar** — the 13-phase graph with the current `PhaseComponent` phase highlighted, recorded review verdicts, the live task queue table (current task marked), and recent subagent activity (each running subagent shows a live streamed-character count).
- **Subagent inspector** — `Ctrl+O` (or clicking the sidebar subagent panel) opens a modal that lists every delegated subagent run and, for the selected one, shows its full streamed **reasoning**, streamed output, and final result. It reads the live view model, so while a review/writer subagent is running its thinking appears token by token; after it finishes the full result stays available (the transcript only keeps a one-line summary). `↑`/`↓` pick a run, `Esc` closes. This is powered by `SubagentSystem(stream_subagents=True)`, which bridges each synchronous subagent's `StreamReasoningDeltaEvent`/`StreamContentDeltaEvent` to the parent event bus as `SubagentStreamDeltaEvent` (keyed by the delegation `correlation_id`).
- **Status bar** — while the agent is busy, an animated spinner line above the usage bar shows the current activity (`waiting for model` / `thinking` / `generating` / `tool <name>` / `subagent <name>`); on its right, the count of tokens generated in the current turn ticks live. During streaming the count is estimated from streamed characters (CJK ≈ 1 token/char, other ≈ 4 chars/token, marked with `~`) and reconciles to the real `completion_tokens` when each invocation's usage record arrives. The bar hides while the agent waits for input.
- **Usage bar** — cumulative tokens, cache hit-rate, and cost from `LLMInvocationEvent` accounting.
- **Input** — single-line prompt with slash-command completion. Typing `/` opens a completion list directly above the input showing every matching command with its hint (argument shape + purpose, from each `TriggerSpec.description`); further typing narrows the list by prefix. `↑`/`↓` move the highlight, `Tab` (or `Enter` on a partial command) fills the highlighted command into the input with a trailing space, clicking an entry does the same, and `Esc` closes the list. `Enter` on a fully-typed command submits it; the inline grey suggestion (accept with `→`) stays available. `exit`/`quit` or `Ctrl+Q` ends the session; `Ctrl+L` clears the transcript; `Ctrl+O` opens the subagent inspector.
- **Question modal** — when the agent calls the `ask_question` tool (see [Interactive `ask_question` Tool](#interactive-ask_question-tool)), a centered modal presents its 1-4 questions: a radio group (single choice), a selection list (multi-select), or a free-text field, plus a free-text field on option questions for a custom "other" answer. `Enter` submits, `Ctrl+S` submits, `Esc` cancels; the collected answers resolve the tool call and the agent resumes. The stdin REPL handles the same tool by prompting for each answer on the terminal.
- **Copy-on-select** — mouse-selecting text (transcript, sidebar panels, live tail) automatically copies it to the system clipboard and shows a brief `clipboard` toast. Selection in the transcript is character-precise (double-width CJK included) and highlights with the theme selection background; dragging into the blank area below the text selects to the end of the content. The clipboard write uses the OSC 52 escape sequence via Textual's `App.copy_to_clipboard`, which requires a terminal that accepts OSC 52 (kitty, Alacritty, WezTerm, iTerm2, recent VTE/GNOME Terminal; over SSH it reaches the local clipboard). The `Input` field keeps its own native selection and shortcuts and is not auto-copied.

Implementation layout (`examples/e2e/plan_and_task/tui/`):

| Module | Role |
|---|---|
| `view_model.py` | Pure reducer: ECS events → renderable state (no textual imports; unit-testable) |
| `bridge.py` | Event-bus subscriptions → view model; owns the pending `UserInputRequestedEvent` and `UserQuestionRequestedEvent` futures; mirrors the REPL re-arm loop of `runtime.py` |
| `app.py` | Textual `App`: widgets, CSS layout, key bindings; consumes `UiChange` messages. Includes `SelectableRichLog` (text selection over `RichLog`), the `on_text_selected` copy-on-select handler, `QuestionScreen` (the `ask_question` modal), and `SubagentInspectorScreen` (the `Ctrl+O` subagent reasoning/result modal) |
| `session.py` | Wires view model + bridge + app over a built world (shared by `__main__` and tests) |
| `__main__.py` | Entrypoint: builds the world via `build_plan_task_world(...)`, runs `Runner.run` and `App.run_async` concurrently on one asyncio loop |

The TUI enables `StreamingComponent(enabled=True, non_blocking_delta_publish=False)` on the agent entity; ordered delta publishing is required so `StreamEndEvent` arrives after all `StreamContentDeltaEvent`s (non-blocking publish would flush an empty assistant message). The stdin REPL (`main.py`) keeps streaming off and prints whole messages, unchanged.

Subagent streaming is enabled by `SubagentSystem(stream_subagents=True)` in `build_plan_task_world`. Normally the stream bridge only runs for *background* subagent sessions; this opt-in flag (default off, so existing behavior and locked tests are untouched) makes each **synchronous** reviewer/writer subagent also bridge its child-world reasoning/content stream to the parent event bus (keyed by `correlation_id` since a sync subagent has no session id). The view model accumulates those deltas per run and the inspector renders them. Both the stdin REPL and the TUI benefit from the flag, but only the TUI surfaces the stream.

## Recovery / Restart

The workflow can be restarted at any time. On startup, no workflow ID is resolved and no scratchbook folder is created. Instead:
1. Call `/plan:start <original description>` — the LLM re-derives the same slug from the same description (or uses `slug_from_description()` as fallback).
2. State is restored from `scratchbook/<workflow_id>/state/runtime_state.json`.
3. Any in-flight subagents are marked `stale`. The load flow overwrites the runtime `PhaseComponent` with the persisted phase and re-binds the phase graph; the graph's `on_resume` policy then demotes a persisted `TASK_RUNNING` to `TASK_BLOCKED` for safe resumption (the demotion is recorded in the phase audit history).

Framework checkpoint restore may also mark a previously running background subagent as `failed` with `restored_without_live_task_handle` when its live task handle no longer exists. That lifecycle marker remains available through explicit subagent status/result inspection, but is not surfaced as a fresh review notification.

> **Note**: Use the same description text (or the same slug) as the original `/plan:start` call so the derived workflow ID matches the existing scratchbook directory.

### Mid-flight State Reconciliation

When `/plan:resume <workflow_id>` is called, the system reads the persisted state and automatically reconciles any in-progress phases so the workflow can continue without manual intervention. Reconciliation replays the resumed phase's declarative `ApprovalGate` mapping (the same one used for live verdict routing) against the latest persisted verdict via `ecs_agent.phases.resume_phase_graph(approvals=...)` — the routing rule lives only in `phase_graph.py`, and no duplicate entry is written to the approval ledger. The load flow also rehydrates the in-world `PhaseApprovalsComponent` audit ledger as a one-record-per-phase snapshot of the persisted verdicts, so `latest_verdicts()` stays truthful across restarts; `/plan:start` and a scope-changed `/task:replan` clear it:

| Resumed phase | Condition | Automatic action |
|---|---|---|
| `DRAFT_QA_REVIEW` | `review_verdicts` contains an `approved` verdict for this phase | Transitions to `WRITE_PLAN`, injects a write-plan trigger message to start the `plan_writer` subagent |
| `WRITE_PLAN` | (any — plan_writer was mid-flight) | Injects a write-plan trigger message to restart the `plan_writer` subagent |
| `PLAN_QA_REVIEW` | `review_verdicts` contains an `approved` verdict for this phase | Transitions to `PLAN_FINALIZED` |
| All other phases | — | No automatic action; resumes normally |

This means after a process restart you can call `/plan:resume <workflow_id>` and, if QA had already approved the draft before the restart, the `plan_writer` will be triggered automatically — no need to manually issue `/plan:write`.

## Testing

### Integration tests

Run the integration suite to verify command parsing, phase-transition logic, artifact persistence, and credential-gated CLI coverage:

```bash
uv run pytest tests/integration/test_plan_and_task_flow.py -v
```

TUI coverage lives in `tests/integration/test_plan_and_task_tui.py`: pure view-model reducer tests, bridge input-future tests against a real `World`, headless Textual smoke tests (`App.run_test()`), and an end-to-end session that ticks a real `Runner` with a `FakeModel` through `/plan:status` and streaming output:

```bash
uv run pytest tests/integration/test_plan_and_task_tui.py -v
```

`ask_question` coverage lives in `tests/integration/test_plan_and_task_ask_tool.py`: `parse_questions`/`format_answers` validation, the handler's publish-and-await path (including the no-front-end and dismissed cases), view-model folding into a `question` `UiChange`, bridge `submit_answers`, `QuestionScreen` modal interaction (free-text, radio, cancel), and a full handler → bridge → modal round trip:

```bash
uv run pytest tests/integration/test_plan_and_task_ask_tool.py -v
```

- `uv run pytest tests/integration/test_plan_and_task_flow.py -k "subagent"` — verifies subagent component wiring
- `uv run pytest tests/integration/test_plan_and_task_flow.py -k "compaction or stale_compaction_state" -v` — verifies main-agent auto compaction, subagent inheritance, and stale-summary reset on workflow switch/resume

### Specific test filters

```bash
uv run pytest tests/integration/test_plan_and_task_flow.py -k "commands"
uv run pytest tests/integration/test_plan_and_task_flow.py -k "artifacts"
```

### Real-LLM acceptance tests

Requires `LLM_API_KEY`. Verifies the controller and task execution with a real model:

```bash
LLM_API_KEY="$LLM_API_KEY" \
LLM_BASE_URL=https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1 \
LLM_MODEL=qwen3.5-flash \
uv run pytest tests/live/test_plan_and_task_flow_live.py -v
```

To exercise the new compaction path against an Anthropic-compatible endpoint:

```bash
LLM_API_FORMAT=anthropic_messages \
LLM_BASE_URL=https://cc2.caaa.tech \
LLM_API_KEY="$LLM_API_KEY" \
LLM_MODEL=glm-5.1 \
uv run pytest tests/live/test_plan_and_task_flow_live.py::test_anthropic_plan_task_auto_compaction_summarizes_context -v
```

## Environment Variables

- `LLM_API_KEY`: API key for the chosen provider.
- `BRAVE_API_KEY`: Optional [Brave Search](https://brave.com/search/api/) API key. When set, the main agent gains a `web_search` tool for grounding draft proposals in current facts; when unset, the tool is omitted and the workflow runs unchanged.
- `LLM_BASE_URL`: API base URL (defaults to DashScope Responses API: `https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1`).
- `LLM_MODEL`: Model ID (defaults to `qwen3.5-flash`).
- `LLM_API_FORMAT`: Provider/format selector. Accepted values:
  - `openai_responses` (default) — OpenAI Responses API via `OpenAIModel`
  - `openai_chat_completions` — OpenAI Chat Completions API via `OpenAIModel`
  - `anthropic_messages` — Anthropic Messages API via `ClaudeModel` (also works with Kimi-compatible Anthropic endpoints)
- `LLM_ENABLE_STORE`: Set to `1`/`true` to enable Responses API stored-response chaining (`previous_response_id`). **Off by default** for this example: it is human-in-the-loop, so `ask_question` and the input prompt can pause a turn for minutes, during which a server-side stored response can expire. The adapter always sends the full history in `input`, so chaining saves no tokens here; leave it off unless a gateway is known to benefit.
- `LLM_STREAM_READ_TIMEOUT`: Seconds of streaming silence before the model call fails, used as a stall detector on the TUI (streaming) path. **Defaults to `120`.** Because a turn can pause for minutes on `ask_question`, the resumed streaming call sometimes lands on a dead gateway connection; without this bound the turn hangs forever and the session looks frozen right after you answer. The timeout resets on every streamed byte, so a live-but-slow model is never cut off — only true silence for the whole window fails the call, which then returns control to you. Set to `0` to restore the unbounded behavior (only advisable for a byte-silent slow reasoning model behind a trusted gateway). Applies to both the OpenAI and Anthropic streaming paths; a malformed value falls back to `120`.
- `PLAN_TASK_LANGFUSE`: Set to `1`, `true`, `yes`, or `on` to install Langfuse observability on the plan-and-task `World` before `Runner.run()` starts.
- `PLAN_TASK_LANGFUSE_ENVIRONMENT`: Optional Langfuse environment label. Defaults to `plan-and-task`.
- `PLAN_TASK_LANGFUSE_RELEASE`: Optional release label sent with plan-and-task traces.
- `PLAN_TASK_LANGFUSE_SESSION_ID`: Optional session ID for grouping plan-and-task traces. The Langfuse SDK v4 adapter sends this as a trace-level session attribute via `propagate_attributes(...)`; metadata-only session IDs do not power the Langfuse Sessions UI.
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` or `LANGFUSE_BASE_URL`: Langfuse connection settings used when `PLAN_TASK_LANGFUSE` is enabled.
- `PLAN_TASK_INTERACTIVE`: Set to `0` to disable interactive stdin.
- `DEBUG`: Set to `1` to make this example call `configure_logging()` with debug logging. All `plan_task_*` structured log events will then appear on stderr via structlog.

### Langfuse Observability

Install the optional extra before enabling Langfuse for this example:

```bash
uv pip install -e ".[langfuse]"
```

When `PLAN_TASK_LANGFUSE` is enabled, `main.py` calls `install_plan_task_langfuse_observability()` after `build_plan_task_world(...)` creates the `World` and before `Runner.run(...)` starts. In interactive mode, every `UserInputReceivedEvent` starts a `user.turn` trace that covers the complete chain from that user input until the next user input or process exit: prompt normalization, retrieval/compaction, LLM generations, tool calls, subagent spans, retries, errors, context pressure, and completion scores all stay inside that turn trace. One-shot runs without interactive input keep the runner trace for backward compatibility. Completed LLM, tool, and subagent observations export their ECS-recorded end timestamps through the Langfuse SDK v4 public lifecycle; preserving historical start timestamps requires explicitly validating your SDK version and enabling `LangfuseConfig(enable_private_v4_historical_otel=True)` because that path uses private SDK hooks. Raw prompts, tool arguments, and outputs are captured by default for backward compatibility; use `LangfuseConfig(capture_input=False, capture_output=False)` if raw content should not leave the process.

Subagents are exported as `subagent.<name>` spans inside the active `user.turn` trace. Their child-world LLM calls are exported as `generation` observations under that subagent span, and child-world tool/retrieval/API work is exported as child spans/events under the same turn trace rather than creating another top-level Langfuse trace. When a child-world generation requests a tool, that tool observation stays attached to the requesting generation so the Langfuse hierarchy shows the exact delegation chain.

Use environment variables or a secret manager for `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and the Langfuse host value. Do not put concrete keys in scripts, docs, or command history. On exit, the CLI calls `flush()` and `shutdown()` on the observability handle so buffered trace events are sent before the process terminates.

Anthropic-compatible Langfuse smoke run:

```bash
PLAN_TASK_LANGFUSE=1 \
PLAN_TASK_LANGFUSE_ENVIRONMENT=dev \
PLAN_TASK_LANGFUSE_RELEASE=local-test \
PLAN_TASK_LANGFUSE_SESSION_ID="plan-task-dev-1" \
LLM_API_FORMAT=anthropic_messages \
LLM_MODEL=deepseek-v4-flash \
uv run python examples/e2e/plan_and_task/main.py
```

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`, `LLM_BASE_URL`, and `LLM_API_KEY` in your shell or secret manager before running the command.

### Anthropic / Kimi Example

```bash
DEBUG=1 \
  LLM_API_FORMAT=anthropic_messages \
  LLM_BASE_URL=https://api.anthropic.com \
  LLM_API_KEY=sk-... \
  LLM_MODEL=kimi-for-coding \
  uv run python examples/e2e/plan_and_task/main.py
```

The `ClaudeModel` appends `/v1/messages` to `LLM_BASE_URL`, so the actual endpoint called is `https://api.anthropic.com/v1/messages`. Anthropic cache stats (`cache_read_input_tokens`, `cache_creation_input_tokens`) are normalized and surfaced as `plan_task_llm_cache_stats` events with `cache_hit_rate`.

### Log Events (observable with `DEBUG=1`)

This example explicitly enables logging in `main.py` when `DEBUG=1`; the base `ecs-agent` library remains silent until `configure_logging()` is called.

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
| `plan_task_circuit_breaker_triggered` | warning | `task_exec.py` | Task retry count exhausted; `retry_count=`, `max_retries=` |
| `plan_task_dependency_cycle_detected` | warning | `task_exec.py` | Cyclic dependency found before raise; `cycle_ids=` |
| `plan_task_reviews_not_approved` | warning | `task_exec.py` | Task start blocked by missing reviews; `missing_phases=` |
| `plan_task_finalize_blocked` | warning | `controller.py` | Plan finalization blocked by missing verdicts; `missing_phases=` |
| `plan_task_plan_artifact_missing` | warning | `controller.py` | Plan artifact file not found before raise; `path=` |
| `plan_task_plan_not_finalized` | warning | `plan_schema.py` | Plan status is not finalized before raise; `status=` |
| `plan_task_command_plan_start` | info | `main.py` | `/plan:start` succeeded; `workflow_id=`, `description_len=` |
| `plan_task_command_plan_resume` | info | `main.py` | `/plan:resume` succeeded; `workflow_id=`, `phase=` |
| `plan_task_command_plan_finalize` | info | `main.py` | `/plan:finalize` succeeded; `workflow_id=` |
| `plan_task_command_task_start` | info | `main.py` | `/task:start` succeeded; `task_count=`, `current_task_id=` |
| `plan_task_task_start_auto_loaded_state` | info | `main.py` | `/task:start <workflow_id>` auto-loaded persisted state; `workflow_id=`, `phase=` |
| `plan_task_command_task_resume` | info | `main.py` | `/task:resume` succeeded; `workflow_id=` |
| `plan_task_command_task_replan` | info | `main.py` | `/task:replan` succeeded; `task_id=` |
| `plan_task_command_task_abort` | info | `main.py` | `/task:abort` succeeded; `task_id=` |
| `plan_task_command_plan_status` | debug | `main.py` | `/plan:status` invoked |
| `plan_task_command_task_status` | debug | `main.py` | `/task:status` invoked; `phase=` |
| `plan_task_command_error` | warning | `main.py` | A slash command raised ValueError; `command=`, `exception=` |
| `plan_task_llm_usage` | info | `billing.py` | Per-invocation token counts; `prompt_tokens=`, `completion_tokens=`, `total_tokens=`, `cached_input_tokens=`, `cache_creation_tokens=`, `cache_read_tokens=` |
| `plan_task_llm_cache_stats` | info | `billing.py` | Per-invocation cache hit-rate; `cache_read_tokens=`, `total_prompt_tokens=`, `cache_hit_rate=`. Emitted when using `ApiFormat.OPENAI_RESPONSES` with DashScope (returns `input_tokens_details.cached_tokens`), OpenAI (returns `prompt_tokens_details.cached_tokens`), or Anthropic (returns `cache_read_input_tokens`). Not emitted with DashScope Chat Completions API, which does not expose cached token counts. |
| `plan_task_session_billing_summary` | info | `billing.py` | Cumulative token totals at end of session; `invocation_count=`, `total_prompt_tokens=`, `total_completion_tokens=`, `total_tokens=`, `total_cached_input_tokens=` |
| `accounting_invocation_recorded` | info | `ecs_agent.accounting` | Per-invocation cost + cache hit-rate from `AccountingSubscriber`; `total_cost=`, `cache_hit_rate=` |
| `plan_task_advisor_review_recorded` | info | `controller.py` | Advisor verdict recorded (the advisor gate never advances the phase); `workflow_id=`, `verdict=` |
| `plan_task_qa_review_recorded` | info | `controller.py` | Draft QA verdict recorded; `approved` routes `DRAFT_QA_REVIEW` → `WRITE_PLAN` through the phase's `ApprovalGate`; `workflow_id=`, `verdict=` |
| `plan_task_plan_qa_review_recorded` | info | `controller.py` | Plan QA verdict recorded; `approved` routes `PLAN_QA_REVIEW` → `PLAN_FINALIZED` through the phase's `ApprovalGate`; `workflow_id=`, `verdict=` |
| `plan_task_review_verdict_ignored` | info | `controller.py` | Later same-phase verdict discarded because the phase already holds a sticky `approved` verdict; `workflow_id=`, `phase=`, `verdict=` |
| `plan_task_reconcile_advanced` | info | `main.py` | Gate replay on `/plan:resume` or task-command auto-load advanced the phase from the persisted verdict; `workflow_id=`, `to_phase=`, `source=reconcile_after_resume` |
| `plan_task_verdict_recording_failed` | error | `main.py` | Subagent verdict rejected by the controller (e.g. review phase neither current nor adjacent); `subagent_name=`, `exception=` |
| `plan_task_auto_trigger_plan_writer` | info | `main.py` | QA approved — injected write-plan trigger message to start `plan_writer` subagent automatically; `workflow_id=`, `source=` (omitted when triggered by live QA event; `source=reconcile_after_resume` when triggered on `/plan:resume`) |

## Implementation Details

- **Testable World Factory**: `build_plan_task_world(model, base_dir=None, *, compaction_threshold_tokens=..., compaction_method=...)` is a public function that returns `(world, agent_id, adapter_ref, runtime_state)`, enabling direct world setup in tests without running the CLI. `adapter_ref` is a `list[ArtifactAdapter | None]` — starts as `[None]` and is populated in-place by the `/plan:start` handler after the workflow ID is derived.
- **Framework-Native Auto Compaction**: `build_plan_task_world(...)` accepts `compaction_threshold_tokens` and `compaction_method`, installs `CompactionConfigComponent`, initializes `ConversationArchiveComponent`, and registers `CompactionSystem()` at priority `-30`. The example reuses the shared framework compaction pipeline rather than maintaining a bespoke plan-and-task summarizer.
- **workflow_id Auto-Derivation**: `/plan:start <description>` calls `derive_workflow_id_from_llm()` to ask the LLM to generate a short, meaningful English slug from the description (e.g., `"writing-assistant-multi-agent"`). Falls back to `slug_from_description()` on provider error or invalid output. The derived ID controls the scratchbook directory for all subsequent operations in that session.
- **Proactive Draft Editing**: The planning agent drives `draft.md` one section per turn: it explores 2-3 options for the active section, writes its recommendation in place tagged `(proposed)`, and puts the choice to the user via the `ask_question` tool (recommendation plus alternatives as selectable options), only asking a genuine open question when a decision is high-stakes and under-determined. Edits use clean `read_file` output plus snapshot-protected `edit_file(op="replace", pos="<line-number>", content=...)` calls — the LLM reads first so the framework has a fresh internal snapshot, then replaces exactly the placeholder line or range. Full-file rewrites via `write_file` are explicitly prohibited by the system prompt.
- **Language Matching**: Every agent replies in the language the user writes in. `prompts.py` defines four `## Language` directives — `_LANGUAGE_MAIN_DIRECTIVE` (shared by the `PLAN` and `TASK` main agents, which see the user directly), `_LANGUAGE_IDLE_DIRECTIVE` (the `IDLE` entry agent), `_LANGUAGE_REVIEWER_DIRECTIVE` (the `advisor`/`qa`/`plan_qa` subagents), and `_LANGUAGE_PLAN_WRITER_DIRECTIVE` (the `plan_writer`). Main and idle agents match the user's latest message and switch when the user switches; the review and writer subagents never see the user, so they match the language of the artifact they `read_file` (the draft or plan). All four keep machine-parsed and schema-validated tokens verbatim English regardless of language — the `VERDICT: approved|revise|blocked` line, phase names, subagent categories, slash commands, tool/file names, the `draft.md` scaffold headings and `(to be filled …)` placeholder, and every `workflow_plan.md` YAML key / `status: finalized` / `## Tasks` / `### Task:` token — so verdict extraction, phase routing, and `plan_schema.py` validation keep working when the prose is e.g. Chinese. The directive strings contain no `{`, `}`, or `$`, so they stay inert under both the f-string prompts and the `${…}` `string.Template` render pass.
- **Atomic Writes**: All artifact updates use atomic file operations to prevent corruption.
- **Circuit Breaker**: `TaskExec` blocks tasks whose `TaskRecord.retry_count` reaches the retry limit, preventing infinite loops on failing tasks.
- **Review Gating**: Finalization is strictly blocked until `DRAFT_ADVISOR_REVIEW`, `DRAFT_QA_REVIEW`, and `PLAN_QA_REVIEW` all have `approved` verdicts.
- **Advisor Retry Loop**: When the advisor returns `revise` or `blocked`, the system prompt instructs the planner LLM to apply the feedback to `draft.md` via `edit_file` and re-call the advisor. Only an `approved` advisor verdict unlocks the QA step. Non-approved verdicts for a phase replace any prior non-approved verdict (upsert semantics). Once a phase reaches `approved`, that verdict is **sticky** — any subsequent upsert attempt for the same phase is silently ignored. `approved` verdicts always have `notes=None`; non-approved verdicts retain their notes for debugging.
- **Verdict Extraction**: Reviewer system prompts require the reply to end with a `VERDICT: <token>` line. `_extract_verdict_from_result` first looks for that marker (case-insensitive, last occurrence wins), so verdict words appearing in checklist prose — e.g. "blocked by a missing mitigation" in a FAIL reason — cannot be misread as the decision. When no marker line exists it falls back to the first case-insensitive bare-word match of `\b(approved|revise|blocked)\b`; if neither is found, extraction logs `plan_task_verdict_extraction_failed` and defaults to `revise`.
- **Slash Command Re-trigger Guard**: After `/task:start` the command message stays as the last `role="user"` entry in the conversation (tool results use `role="tool"` and do not replace it). `_handle_task_start` and `_handle_task_resume` therefore check the runtime `PhaseComponent`; when `PhaseComponent.phase == "TASK_RUNNING"`, they return `None` immediately, letting the LLM continue task execution without re-initializing the task queue.
- **Compaction State Reset on Workflow Switch**: `_reset_compaction_state(...)` removes `CurrentCompactionSummaryComponent`, clears `ConversationArchiveComponent.archived_summaries`, and removes `RenderedSystemPromptComponent` before `/plan:start`, `/plan:resume`, or auto-loading state from `/task:start <workflow_id>`. This keeps prompt summaries aligned with the active workflow instead of a previous session.
- **Plan Template**: `templates/workflow_plan_template.md` is an annotated reference showing the exact `workflow_plan.md` format: YAML frontmatter, `## Overview` with `### Dependency Graph`, `## Tasks` section, per-task `### Task: <task_id>` + ` ```yaml ``` ` blocks, and an optional `## Appendix` AC cross-reference table. The format spec is embedded verbatim as `_WORKFLOW_PLAN_FORMAT` only into `WRITE_PLAN_SYSTEM_PROMPT` — the `plan_writer` subagent's own system prompt — so the writer has an unambiguous reference without reading a file; the `build_write_plan_prompt()` trigger message stays short (paths and steps only) since it persists in the main-agent conversation.
- **Dependency Resolution**: Tasks are executed in topological order based on their `dependencies` list.
- **Review Verdict Lifecycle**: Each phase holds at most one verdict in `review_verdicts`. Upsert semantics: non-approved verdicts replace earlier non-approved verdicts for the same phase; `approved` is terminal and cannot be overwritten. `approved` verdicts always have `notes=None`. The `plan_version` field has been removed from `ReviewVerdict`.
- **Status Lifecycle**: `status` is a pure derivation, never assigned by handlers: terminal phases yield `"completed"` (`"aborted"` when `abort_reason` is set), `TASK_READY` → `"ready"`, `TASK_BLOCKED` → `"blocked"`, a review phase awaiting its verdict → `"needs_review"`, everything else `"active"` (`phase_sync.derive_status()`, stamped by the mirror on every transition and on restore — persisted statuses survive `/plan:resume` and `/task:start` auto-load).
- **Token Prefix Caching**: This example uses `ApiFormat.OPENAI_RESPONSES` with `enable_store=True` (default config). The DashScope Responses API endpoint (`/api/v2/apps/protocols/compatible-mode/v1`) returns `usage.input_tokens_details.cached_tokens`, which `normalize_openai_usage()` maps to `cached_input_tokens`. On warm calls where the system prompt prefix is cached, `cached_input_tokens > 0` and `plan_task_llm_cache_stats` will be emitted with a non-zero `cache_hit_rate`. The DashScope Chat Completions API (`/compatible-mode/v1`) does not return cached token counts and does not support the Responses protocol — switching back to it would make cache observability unavailable.
