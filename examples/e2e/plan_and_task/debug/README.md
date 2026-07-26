# Plan-and-task debug harness

A headless, scriptable, introspectable **third front end** for the plan-and-task
world — alongside the stdin REPL (`main.py`) and the Textual TUI (`tui/`), but
drivable by an agent or a test instead of a human. It reuses
`build_plan_task_world` verbatim and attaches the same input/question/reasoning
wiring the real front ends use, so a bug seen here is a real bug.

Design/rationale: `docs/rfc/2026-07-26-plan-task-debug-harness.md`.

## Why

The example is otherwise only drivable by piping a *pre-committed* input script
into `main.py` (no read→decide→act loop), `ask_question` needs a live front end,
there is no state introspection between turns, and the TUI needs a terminal.
This harness fixes all four: **one turn per exchange, full introspection, no
terminal**.

## Library API

```python
from examples.e2e.plan_and_task.debug import PlanTaskDebugSession

async with await PlanTaskDebugSession.build(model=model, base_dir=tmp) as s:
    r = await s.send("/plan:start Build a CLI todo app")   # -> TurnResult
    print(r.snapshot.phase)                                # "DRAFT_INTERVIEW"
    print(r.tool_calls, r.subagents, r.phase_transitions)
    print(s.read_artifact("plan/draft.md"))
```

- `send(text)` runs exactly one turn and returns a `TurnResult`
  (`assistant_messages`, `tool_calls`, `subagents` + extracted verdicts,
  `phase_transitions`, `questions_asked`, `errors`, `usage`, `pending_question`,
  `snapshot`, `note`).
- With `surface_questions=True`, a mid-turn `ask_question` makes `send` return
  early with `kind="question"` and `pending_question` set; call
  `await s.answer([1])` (1-based option index / free text / dismiss) to continue.
- Without it, an `AnswerPolicy` auto-answers (`AutoAnswerPolicy` picks the
  `(recommended)`/first option) so a whole flow runs unattended.
- `snapshot()`, `read_artifact(relpath)`, `events(turn=…, kinds=…)` for
  introspection. `finished` / `runner_exception` expose terminal state.

Deterministic replay: pass a `FakeModel` as `model`.

## JSONL CLI

Drive it turn-by-turn from a shell (one JSON command per stdin line, one JSON
result per stdout line):

```bash
# deterministic replay
python -m examples.e2e.plan_and_task.debug --fake script.json --base-dir /tmp/pt

# live model (same LLM_* env as main.py)
LLM_API_FORMAT=openai_chat_completions LLM_MODEL=gpt-5.6-sol \
  python -m examples.e2e.plan_and_task.debug --base-dir /tmp/pt --record run.jsonl
```

| stdin | effect |
|---|---|
| `{"cmd":"send","text":"/plan:start …"}` | run one turn → `TurnResult` |
| `{"cmd":"answer","answers":[1]}` | resolve a surfaced `ask_question` |
| `{"cmd":"answer"}` | dismiss the question |
| `{"cmd":"snapshot"}` | `StateSnapshot` |
| `{"cmd":"artifact","path":"plan/draft.md"}` | file contents |
| `{"cmd":"events","turn":0}` | that turn's event log |
| `{"cmd":"quit"}` | terminate |

Flags: `--fake`, `--base-dir`, `--auto-answer` (batch), `--no-tool-sink` (the
tool-results sink is ON by default, matching `main.py`), `--record FILE.jsonl`,
`--max-turn-seconds`.

For true live back-and-forth, run the CLI in a persistent terminal (e.g. the
paseo terminal MCP or a `run_in_background` pipe) and send one command per turn.
For batch/deterministic runs, `printf '…\n…\n' | python -m …debug --fake …`.

### `--fake` script format

```json
{"responses": [
  {"content": "text", "tool_calls": [{"id":"c1","name":"ask_question","arguments": {…}}]},
  "plain assistant text"
]}
```

Responses are returned in order (a top-level list also works). Note `/plan:start`
consumes one response for LLM slug derivation before the turn's reasoning.
