# RFC: An agent-drivable debug harness for the plan-and-task example

- Status: accepted
- Date: 2026-07-26
- Scope: `examples/e2e/plan_and_task/` only (no framework changes)

## 1. Problem

The `plan_and_task` e2e example is a 13-phase, review-gated, human-in-the-loop
workflow. It is our richest integration surface — and the hardest thing in the
repo for an **agent** (a Claude Code session or a subagent) to debug, because
every existing way to run it is either fire-and-forget or needs a human at a
terminal:

1. **Stdin REPL (`main.py`) is fire-and-forget.** The only agent-drivable entry
   is `printf '/cmd\n\n…' | python main.py`. The whole input script must be
   committed *before* seeing a single model reply, so an agent cannot do the one
   thing debugging requires: read the model's output, decide, then send the next
   input. There is no read→decide→act loop.

2. **`ask_question` needs a live front end.** The planner pauses mid-turn and
   blocks on a `UserQuestionRequestedEvent` future (`ask_tool.py`). In piped
   mode the stdin runtime still answers it, but from the *same* blind stdin
   stream — the agent can't see the question text before its answer is consumed.
   Non-interactive runs fail the tool fast (`has_subscribers` guard), so any
   flow that reaches the interview simply dies.

3. **No state introspection between turns.** You get the final `Conversation:`
   dump at exit and, with `DEBUG=1`, an interleaved structlog flood on stderr.
   There is no way to ask *"what phase am I in now? what's the task queue? which
   verdicts are recorded? what did the QA subagent actually return?"* at a chosen
   point mid-session without grepping the whole log.

4. **The TUI is undrivable by an agent.** `tui/` needs a real terminal, key
   events, and a screen. An agent cannot open the subagent inspector or read the
   sidebar.

5. **Nondeterminism, latency, cost.** A real-LLM run is slow, nondeterministic,
   and burns tokens; reproducing a specific bug reliably is hard, and there is no
   deterministic replay path wired to the *full* world (only the controller-level
   unit tests use `FakeModel`, never the interactive loop).

6. **Turn boundaries are implicit.** "Has the agent finished reacting and is it
   waiting for me again?" is only observable as a `You>` string on stdout. There
   is no programmatic await-until-idle.

The common root cause is that the example has exactly two front ends —
`runtime.setup_interactive_input` (stdin) and `tui/PlanTaskTuiBridge` — and
**both assume a human**. Neither is scriptable or introspectable.

## 2. Key observation: it is already a clean event/future contract

Nothing about the world needs to change. Interaction is fully event-driven and
the two existing front ends are just two implementations of one contract:

- `UserInputRequestedEvent(entity_id, prompt, input_future)` — a front end
  resolves `input_future` with the next user text. `UserInputSystem` (priority
  −15) publishes it and **awaits the future inside `world.process()`**, which is
  precisely how the runner "blocks" for input.
- `UserQuestionRequestedEvent(entity_id, questions, answer_future)` — a front end
  resolves `answer_future` with `list[QuestionAnswer] | None`.
- `ReasoningCompleteEvent` / `ErrorOccurredEvent` → re-arm `UserInputComponent`
  so the next tick requests input again.
- `Runner.run(world, max_ticks=None)` loops until a `TerminalComponent` appears.

`build_plan_task_world(model, base_dir, …)` already returns
`(world, agent_id, adapter_ref, runtime_state)` — a testable factory with no CLI
coupling. The TUI proves the pattern: build the world, attach a front end, run
`Runner.run` as a background task.

**So the fix is a third front end** — a headless, scriptable, introspectable one.

## 3. Design

### 3.1 `PlanTaskDebugSession` (the driver / third front end)

A class that wraps a world built by `build_plan_task_world` and attaches itself
as the front end, mirroring `setup_interactive_input`/`PlanTaskTuiBridge`
wiring exactly (register `UserInputSystem` at −15 and
`TerminalCleanupSystem(clear_reasons=("reasoning_complete",))` at 1; subscribe
the input/question/reasoning/error handlers). Fidelity is the point: the same
contract the real front ends use means bugs reproduce here.

Lifecycle:

- `async with PlanTaskDebugSession.build(model=…, base_dir=…) as s:` builds the
  world, attaches the front end, and starts `Runner.run(max_ticks=None)` as a
  background task.
- The driver records **every** event it needs (the same set the TUI routes:
  stream deltas, tool start/complete, delegation start/complete, phase changes,
  `LLMInvocationEvent`, compaction, errors, input/question) into a structured,
  timestamped, **turn-segmented** event log.

Core method:

```python
result: TurnResult = await s.send("/plan:start …", timeout=180)
```

`send(text)` awaits the input boundary, resolves `input_future` with `text`,
then waits until the **next** boundary — the next `UserInputRequestedEvent` for
the agent, **or** a surfaced `ask_question`, **or** terminal/error — and returns
a `TurnResult` describing exactly what happened during that turn:

- `assistant_messages`, `reasoning` (final text)
- `tool_calls`: `[{name, arguments, result, success, duration}]`
- `subagents`: `[{name, task, result, verdict, success, duration}]` (verdict
  extracted the same way `main.py` does)
- `phase_transitions`: `[{from, to, reason, forced, tick}]`
- `errors`, `llm_usage` (summed), `pending_question` (see below)
- `snapshot`: the post-turn `StateSnapshot`

### 3.2 `ask_question` handling — two policies

Because `send()` waits for the next *boundary*, an `ask_question` mid-turn must
be resolved by an **answer policy** (`policies.py`):

- `AutoAnswerPolicy` (default): pick the option marked `(recommended)` or the
  first option; free-text → a fixed canned string. Lets a whole flow run
  unattended (deterministic/batch mode).
- `ScriptedAnswerPolicy(answers)`: a queue of pre-baked answers, consumed in
  order (reproducible scenarios).
- `CallbackAnswerPolicy(fn)`: delegate to a caller-supplied function.
- **Interactive mode** (`surface_questions=True`): do *not* auto-answer. `send()`
  returns early with `TurnResult.pending_question` populated (the real question
  text/options) and the runner parked on the answer future. The caller then
  calls `await s.answer(...)`, which resolves the future and runs to the next
  boundary. **This is the capability that fixes problem #2** — answer based on
  what was actually asked.

If nothing resolves a question it is auto-dismissed (never hangs), recorded as
`dismissed`.

### 3.3 Inspection surface

- `s.snapshot() -> StateSnapshot`: phase (+ recent history), status, review
  verdicts, task queue (id/status/retry), active subagents, pending question,
  conversation length, cumulative usage, and the scratchbook artifact listing.
- `s.read_artifact(relpath)`: read any file under the workflow scratchbook
  (`plan/draft.md`, `plan/workflow_plan.md`, `state/runtime_state.json`,
  `state/events.jsonl`, `review/*.json`).
- `s.events(turn=…, kinds=…)`: filtered access to the recorded event log.

### 3.4 JSONL-over-stdio CLI (`python -m examples.e2e.plan_and_task.debug`)

A line-oriented JSON protocol so an agent can drive the session turn-by-turn
**from a shell** without writing Python — one command per stdin line, one JSON
result per stdout line:

| stdin command | effect |
|---|---|
| `{"cmd":"send","text":"/plan:start …"}` | run one turn, return `TurnResult` |
| `{"cmd":"answer","answers":[…]}` or `{"cmd":"answer","choice":1}` | resolve a surfaced `ask_question`, continue to next boundary |
| `{"cmd":"snapshot"}` | return `StateSnapshot` |
| `{"cmd":"artifact","path":"plan/draft.md"}` | return file contents |
| `{"cmd":"events","turn":N}` | return that turn's event log |
| `{"cmd":"quit"}` | terminate |

Because each `send` returns a *complete* structured turn result and then waits
for the next command, this gives true interactive read→decide→act. It degrades
to `printf`-piping for batch runs, and — driven through a persistent terminal
(e.g. the paseo terminal MCP, or `run_in_background` + a pipe) — supports live
back-and-forth. Model comes from `build_model_from_env()` (so `LLM_API_FORMAT`,
etc. all apply) or `--fake <script.json>` for deterministic replay. `--record
path.jsonl` tees the full transcript for post-hoc analysis and evidence.

### 3.5 Ergonomics

- `--base-dir <tmp>` isolates runs from the committed `scratchbook/`.
- `--max-turn-seconds` stall guard: a hung turn returns an error `TurnResult`
  instead of blocking forever (mirrors the `LLM_STREAM_READ_TIMEOUT` intent).
- `--debug` captures structlog to the record file, correlated per turn.

## 4. Why this shape

- **Faithful** — same event/future contract as the real front ends, world built
  by the real factory, same subagent/verdict/phase wiring ⇒ a bug found here is
  a real bug.
- **Deterministic-capable** — `FakeModel` drives the *entire* interactive loop,
  not just the controller, so a reproduction can be committed as a test.
- **Introspective** — phase, task queue, verdicts, artifacts, and the full event
  log are first-class, not stderr archaeology.
- **Agent-native** — one turn per exchange over JSONL; no terminal, no TUI, no
  human.
- **Zero framework risk** — lives entirely under the example; imports no
  `textual`; changes no library code.

## 5. File layout

```
examples/e2e/plan_and_task/debug/
├── __init__.py     # exports: PlanTaskDebugSession, TurnResult, StateSnapshot, policies
├── session.py      # driver front end + TurnResult + StateSnapshot + event recorder
├── policies.py     # AnswerPolicy: Auto / Scripted / Callback
├── cli.py          # JSONL command loop
├── __main__.py     # entrypoint
└── README.md       # protocol + copy-paste recipes
```

Tests: `tests/integration/test_plan_and_task_debug_harness.py` (FakeModel,
deterministic) and a live-gated smoke in `tests/live/`.

## 6. Non-goals

- No changes to the framework or to `main.py`/`tui/` behavior.
- Not a replacement for the TUI; it is the machine/agent counterpart.
- Not a general framework debugger — it is plan-and-task-shaped (knows the
  `ask_question` event, the phase graph, and the scratchbook layout).
