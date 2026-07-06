# Phase Graphs

Explicit, auditable phase transitions for agents, driven by a command-based
model: the graph is pure data, transitions are function calls.

## Overview

A phase graph gives an agent a declared set of phases, the allowed transitions
between them, and the per-phase harness surface (system prompt, tool
allowlist, review gates, restart policy) — all in one validated data
structure. There is **no polling system**: nothing evaluates conditions per
tick. Code that knows a transition should happen calls `advance()` (or
`force()`, or `record_approval()`), the transition is validated against the
graph, committed atomically, audited, and published as an event.

```
build_graph(...)  ──►  PhaseGraph (immutable, structure-hashed)
        │
bind_phase_graph(world, eid, graph)
        │
        ▼
PhaseComponent (serialized truth)  +  PhaseDefinitionComponent (runtime graph)
        │
await advance(world, eid, "REVIEW", reason=...)     ─┐
await record_approval(world, eid, "approved")        ├─►  validate → commit →
await force(world, eid, "DRAFT", reason=...)        ─┘    effects → history →
                                                          PhaseChangedEvent
        │
SystemPromptRenderSystem resolves ${_phase_prompt} for the committed phase
```

Design goals (each maps to a mechanism below):

| Goal | Mechanism |
|---|---|
| Strong per-phase harness | per-phase `prompts` + `tools` applied atomically on entry |
| Gated review | first-class `ApprovalGate` verdict routing + audit ledger |
| No silent failure modes | unbound/half-bound/unknown-phase all raise typed errors |
| Restart correctness | idempotent `bind_phase_graph` that never resets progress; declarative `on_resume` |
| Prompt-cache stability | content-hashed provider fingerprint; `_phase_prompt` in the volatile tail |
| Auditability | bounded transition history + `PhaseChangedEvent` → Langfuse `phase.transition` |

## Quick Start

A complete two-phase agent where a tool call drives the transition (this is
the shape of `examples/workflow_agent.py`):

```python
from ecs_agent.components import ConversationComponent, LLMComponent
from ecs_agent.phases import PhaseSpec, advance, bind_phase_graph, build_graph
from ecs_agent.prompts.contracts import PromptTemplateSource, SystemPromptConfigSpec
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem

GRAPH = build_graph(
    "writing-flow",
    initial="DRAFT",
    phases=[
        PhaseSpec(
            phase_id="DRAFT",
            prompts={"assistant": "You draft. Call `submit` when ready."},
            to=("REVIEW",),
        ),
        PhaseSpec(
            phase_id="REVIEW",
            prompts={"assistant": "You review critically."},
            terminal=True,
        ),
    ],
)

world.add_component(agent, LLMComponent(model=model))
world.add_component(agent, ConversationComponent(messages=[...]))
world.add_component(
    agent,
    SystemPromptConfigSpec(
        template_source=PromptTemplateSource(inline="${_phase_prompt}")
    ),
)

# Tool handlers transition directly — async handlers are awaited by
# ToolExecutionSystem:
@tool(name="submit", description="Submit the draft for review.")
async def submit() -> str:
    await advance(world, agent, "REVIEW", reason="tool:submit")
    return "Submitted."

await bind_phase_graph(world, agent, GRAPH, agent_key="assistant")

world.register_system(SystemPromptRenderSystem(priority=-20), priority=-20)
world.register_system(ReasoningSystem(priority=0), priority=0)
world.register_system(ToolExecutionSystem(priority=5), priority=5)
```

There is no phase system to register and no priority contract to respect —
the only ordering that matters is the natural one: a transition committed
before `SystemPromptRenderSystem` runs is visible to that render (same tick).

## Core Components

Import from `ecs_agent` (top level). The verb API lives in `ecs_agent.phases`.

### PhaseComponent

The single runtime source of truth for an entity's position. **Serialized.**

| Field | Type | Meaning |
|---|---|---|
| `graph_id` | `str` | Which graph this entity is bound to |
| `phase` | `str` | Current phase id |
| `graph_hash` | `str` | Structure hash captured at bind time (drift detection) |
| `agent_key` | `str` (default `"main"`) | Which prompt binding this entity uses |
| `entered_at_tick` | `int` | `RunnerStateComponent.current_tick` at the last commit (stall observability) |
| `history` | `list[dict]` | Bounded audit trail (see [Transition History](#transition-history)) |

### PhaseDefinitionComponent

Holds the bound `PhaseGraph` at runtime. **Never serialized** — the graph may
reference `Path` prompts and is cheap to rebuild; re-attach it after a restore
with `bind_phase_graph()`. Using a restored entity before re-binding raises
`PhaseIntegrityError` from every API call and from prompt rendering — loudly,
by design.

### PhaseApprovalsComponent

Audit ledger of verdicts recorded through `record_approval()`. **Serialized.**
Each record is a plain dict: `{"phase", "verdict", "notes", "decided_at"}`.

## Authoring a Graph

### PhaseSpec

```python
PhaseSpec(
    phase_id="DRAFT_QA_REVIEW",
    prompts={"main": QA_PROMPT},              # agent_key -> str | Path
    to=("WRITE_PLAN", "DRAFT_INTERVIEW"),     # adjacency for advance()
    tools=("record_verdict",),                 # optional per-phase allowlist
    approval=ApprovalGate(                     # optional gated review
        verdicts={"approved": "WRITE_PLAN", "revise": None, "blocked": None}
    ),
    on_resume=None,                            # optional restart demotion target
    terminal=False,
)
```

| Field | Rules |
|---|---|
| `phase_id` | non-empty, unique within the graph |
| `prompts` | `agent_key -> str \| Path`; `Path` files are read at render time; `bind_phase_graph` requires its `agent_key` to have a prompt in **every** phase |
| `to` | tuple of target phase ids; no duplicates; every target must exist |
| `tools` | `None` = no declaration (see [Tool Allowlists](#per-phase-tool-allowlists)); tuple = allowlist applied on entry |
| `approval` | verdict → target mapping; every non-`None` target must be in this phase's `to` |
| `on_resume` | phase to demote to when re-binding after a restore; must exist in the graph |
| `terminal` | terminal phases must not declare `to` or `approval`; non-terminal phases must declare at least one target |

### build_graph()

```python
graph = build_graph("plan-task", initial="IDLE", phases=[...])
```

Validates everything above plus: non-empty `graph_id`, at least one phase,
`initial` exists. Returns a `PhaseGraph` with:

- `phases_by_id` — read-only mapping (`MappingProxyType`) over a private copy.
  `ApprovalGate.verdicts` and `PhaseSpec.prompts` are wrapped the same way, so
  the validated graph is **deeply immutable**: mutation attempts raise
  `TypeError`, and mutating the dicts you passed in after building cannot
  affect the graph or its hash.
- `structure_hash` — see [Structural Hash](#structural-hash--graph-evolution).
- `manages_tools` — `True` when any phase declares `tools`.

Authoring conventions that work well in practice (see
`examples/e2e/plan_and_task/phase_graph.py` for a 13-phase production-shaped
graph): share prompt constants between phases via plain Python variables (this
is what keeps the prompt cache stable across transitions), and keep verdict
routing in gates rather than in handler code.

## The Transition API

All mutating calls are `async` (they publish on the event bus). Read helpers
are sync. Import from `ecs_agent.phases`.

```python
component = await bind_phase_graph(world, eid, graph, agent_key="main")
component = await advance(world, eid, "REVIEW", reason="draft submitted")
component = await force(world, eid, "DRAFT", reason="admin rollback")
phase     = await record_approval(world, eid, "approved", notes="lgtm")

allowed_targets(world, eid)   # frozenset[str] reachable via advance()
is_terminal(world, eid)       # bool
latest_verdicts(world, eid)   # dict[phase -> most recent verdict]
```

| Call | Validates | On success |
|---|---|---|
| `bind_phase_graph` | `agent_key` has a prompt in every phase; on re-bind: same `graph_id`, restored phase still exists | fresh entity → `PhaseComponent` at `initial` + effects; restored entity → definition re-attached, effects re-applied, `on_resume` demotion executed (see [Checkpoint & Resume](#checkpoint--resume)) |
| `advance` | entity fully bound; current phase not terminal; target in current phase's `to` | commit (see below) |
| `force` | entity fully bound; target exists in the graph (adjacency and terminality are **not** checked — administrative recovery) | commit with `forced=True` |
| `record_approval` | current phase declares an `ApprovalGate`; verdict is one of its keys | append ledger record; if the verdict maps to a target, `advance()` to it with reason `approval:<verdict>`; returns the resulting phase |

A **commit** is atomic from the caller's perspective: update
`PhaseComponent.phase` and `entered_at_tick`, append a history entry, apply
the target phase's effects (tool allowlist), publish `PhaseChangedEvent`.
Nothing is partially applied on a validation error — the errors below are
raised before any mutation.

### Errors

All inherit `PhaseError(ValueError)`:

| Error | Raised when |
|---|---|
| `PhaseIntegrityError` | entity has no graph bound, or is half-bound (`PhaseComponent` restored but `bind_phase_graph` not called) — raised by every API call **and** by prompt rendering |
| `InvalidPhaseTransitionError` | `advance()` to a non-adjacent target or from a terminal phase; the message lists the allowed targets |
| `PhaseGraphMismatchError` | re-bind with a different `graph_id`, or the restored phase no longer exists in the graph |
| `PhaseError` | invalid verdict, no gate on the current phase, unknown phase passed to `force()`, `agent_key` missing a prompt |

## Approval Gates (Gated Review)

`ApprovalGate` makes review checkpoints first-class instead of hand-coded
handler logic. The gate on a phase declares what each verdict does:

```python
approval=ApprovalGate(verdicts={
    "approved": "WRITE_PLAN",   # advance to WRITE_PLAN
    "revise":   None,           # record the verdict, stay put
    "blocked":  None,
})
```

`record_approval(world, eid, verdict, *, notes=None, decided_at=None)`:

1. Validates the current phase has a gate and the verdict is declared.
2. Appends `{"phase", "verdict", "notes", "decided_at"}` to
   `PhaseApprovalsComponent` (created on first use; `decided_at` defaults to
   UTC now).
3. Auto-advances when the verdict maps to a target.

Because the routing lives in graph data, resume reconciliation can **replay
the gate** instead of duplicating the rule: `resume_phase_graph()` does this
for you — pass the persisted verdicts via `approvals=` and the current phase's
gate is replayed with reason `approval_replay:<verdict>`, without calling
`record_approval()` again (no duplicate ledger entries).
`examples/e2e/plan_and_task/main.py::resume_workflow` is the reference caller.

## Per-Phase Tool Allowlists

Ownership contract, keyed on `PhaseGraph.manages_tools`:

- **Some phase declares `tools`** (`manages_tools=True`): the graph owns
  `PermissionComponent.allowed_tools`. Entering a declaring phase sets it to
  that list; entering a phase with **no** declaration clears it to `[]` —
  which means *unrestricted* under `PermissionSystem` semantics (an empty
  allowlist allows everything).
- **No phase declares `tools`** (`manages_tools=False`): the graph never
  touches `PermissionComponent` — it remains entirely yours.
- `denied_tools` is never touched in any case.

Enforcement is the existing `PermissionSystem` (see
[permissions.md](permissions.md)); the phase graph only writes the policy on
entry. Register `PermissionSystem` before `ToolExecutionSystem` as usual.

## Prompt Integration

Put `${_phase_prompt}` in the system prompt template. The
`PhasePromptPlaceholderProvider` (provider id `phase_prompt`) is picked up
automatically by the render pipeline for any entity with a `PhaseComponent` —
no registration needed. It resolves the current phase's prompt for the bound
`agent_key`; `Path`-sourced prompts are read at render time (unreadable files
raise `ValueError`).

Cache stability: the provider fingerprint is
`{agent_key}|hash:sha256(resolved_text)` — the phase id is deliberately **not**
part of it, so phases sharing identical prompt text keep an identical
fingerprint and transitions between them never invalidate the rendered
prompt. `_phase_prompt` is registered in `_VOLATILE_PLACEHOLDER_KEYS`, so it
renders in the volatile tail and never breaks the cache-stable prefix (see
the Anthropic prompt caching notes in [models.md](../models.md) and the
README).

Fingerprint states: `"disabled"` (entity has no `PhaseComponent`),
`"unbound"` (no prompt resolvable), else the content hash.

## Events & Observability

Every committed transition publishes:

```python
PhaseChangedEvent(
    entity_id=...,
    graph_id="plan-task",
    from_phase="TASK_READY",
    to_phase="TASK_RUNNING",
    reason="task:init",       # free-form; conventions: "tool:<name>",
                              # "approval:<verdict>", "approval_replay:<verdict>",
                              # "on_resume", "plan:start"
    forced=False,             # True for force() and on_resume demotions
    tick=42,
)
```

With Langfuse observability installed, each event maps to a `phase.transition`
record (input: `graph_id`/`from_phase`; output: `to_phase`/`reason`/`forced`).
Transitions that fire before the first user turn are buffered and re-parented
under the interactive turn root, like all other pre-turn telemetry (see
[langfuse.md](langfuse.md)).

Only committed transitions emit events — there is no per-tick noise.

## Checkpoint & Resume

`PhaseComponent` and `PhaseApprovalsComponent` serialize with the world;
`PhaseDefinitionComponent` (the graph) does not. After restoring, bind again
with the same graph:

```python
world = WorldSerializer.from_dict(data, providers=..., tool_handlers=...)
await bind_phase_graph(world, eid, GRAPH, agent_key="main")   # idempotent
```

The re-bind contract:

- **Progress is never reset.** The restored phase is preserved; `bind` only
  re-attaches the definition and re-applies the current phase's effects
  (regression-tested in `tests/test_phase_checkpoint.py`).
- **Structural drift is detected.** If the graph's structure hash changed but
  the restored phase still exists, the stored hash is updated with a warning
  and execution continues. If the restored phase was removed from the graph,
  `PhaseGraphMismatchError` is raised — migrate the persisted state or
  `force()` it to a valid phase.
- **`on_resume` is applied declaratively.** If the restored phase declares
  `on_resume`, the entity is demoted to that phase as an audited forced
  transition with reason `"on_resume"` (e.g. `TASK_RUNNING` →
  `TASK_BLOCKED`: work that was in flight when the process died must be
  re-queued, not assumed running).
- **Forgetting to re-bind fails loudly.** The first API call or prompt render
  on a half-bound entity raises `PhaseIntegrityError` with instructions.

State persisted outside the world serializer (e.g. plan-and-task's
`runtime_state.json`) uses `resume_phase_graph(world, entity_id, graph,
phase=..., graph_hash=..., approvals=...)`: it overwrites `PhaseComponent`
with the persisted phase, re-binds (validation, drift handling, `on_resume`
apply exactly as above), replays the current phase's approval gate against
the persisted verdicts, and returns a `ResumeReport` with `demoted_from` and
`replayed` (`examples/e2e/plan_and_task/main.py::resume_workflow` is the
reference caller). In-memory transition history restarts per restore.

## Transition History

Each commit appends one plain-dict entry to `PhaseComponent.history`:

```python
{"from": "DRAFT", "to": "REVIEW", "reason": "tool:submit", "forced": False, "tick": 7}
```

The list is bounded to `HISTORY_LIMIT` (100) — old entries are dropped so
checkpoints stay small. The durable, unbounded audit trail is the structured
log (`phase_transition` events) and Langfuse; the in-component history is for
quick inspection and tests.

## Trigger Script Integration

Slash-command handlers (see [prompt-normalization.md](prompt-normalization.md))
can transition directly, and the render in the same tick observes the new
phase:

```python
async def handle_go(world: World, entity_id: EntityId, text: str) -> str | None:
    await advance(world, entity_id, "REVIEW", reason="trigger:/go")
    return None

UserPromptConfigComponent(
    triggers=[TriggerSpec(pattern="/go", match_mode="prefix",
                          action="script", content="go")],
    script_handlers={"go": handle_go},
)
```

`UserPromptNormalizationSystem` runs the handler; by the time
`SystemPromptRenderSystem` renders (any later priority), `${_phase_prompt}`
resolves to the REVIEW prompt. Covered end-to-end by
`tests/integration/test_phase_agent_flow.py`.

## Structural Hash & Graph Evolution

`structure_hash` is a SHA-256 over the canonical JSON of: `graph_id`,
`initial`, and per-phase `{id, to, tools, approval mapping, on_resume,
terminal}` — order-insensitive.

**Prompt text and paths are deliberately excluded**: editing prompts never
invalidates restored state and never triggers drift warnings. Structural
edits (adding/removing phases or edges, changing gates/tools/resume policy)
change the hash and are handled by the re-bind contract above. Renaming a
phase looks like a removal to the restored state — that is intentional; the
owner must migrate persisted phases or `force()` past the rename.

## Examples & Tests

| Where | What |
|---|---|
| `examples/workflow_agent.py` | Minimal two-phase agent; tools call `advance()`; dual-mode (FakeModel offline / real LLM) |
| `examples/e2e/plan_and_task/phase_graph.py` | 13-phase graph with three `ApprovalGate`s and `on_resume` |
| `examples/e2e/plan_and_task/controller.py` | Verdict handling, gate replay on resume, finalize hop-walk with an import-time consistency guard |
| `examples/e2e/plan_and_task/phase_sync.py` | The single-write-path mirror between `PhaseComponent` and example-land persisted state |
| `tests/test_phase_contracts.py` | Validation rules, structural hash, deep immutability |
| `tests/test_phase_api.py` | bind/advance/force semantics, history bounds, tool ownership |
| `tests/test_phase_checkpoint.py` | The restore contract (progress preserved, loud half-bind, drift, on_resume) |
| `tests/test_phase_approvals.py` | Gate routing and the verdict ledger |
| `tests/test_phase_prompt_integration.py` | Render pipeline + fingerprint stability |
| `tests/test_phase_observability.py` | Langfuse `phase.transition` mapping |
| `tests/integration/test_phase_agent_flow.py` | Trigger script → advance → same-tick render |
| `tests/live/test_phases_live.py` | Real-LLM prompt swap (env-gated on `LLM_API_KEY`) |
