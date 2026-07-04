# Phase Graphs

Explicit, auditable phase transitions for agents. Replaces the gate-polling
workflow DSL (removed in 2026-07) with a command-driven model: the graph is
pure data, transitions are function calls.

## Model

- `PhaseComponent` — the single serialized source of truth: `graph_id`, `phase`,
  `graph_hash`, `agent_key`, `entered_at_tick`, bounded `history`.
- `PhaseGraph` — validated pure data built by `build_graph()`: per-phase prompts,
  adjacency (`to`), optional tool allowlist, optional `ApprovalGate`, optional
  `on_resume` demotion, `terminal` flag. Never serialized; re-bound at startup.
- `PhaseDefinitionComponent` — holds the bound graph at runtime (skipped by the
  serializer). Using a restored entity before re-binding raises
  `PhaseIntegrityError` — loudly, by design.

## Authoring

```python
from ecs_agent.phases import ApprovalGate, PhaseSpec, bind_phase_graph, build_graph

GRAPH = build_graph(
    "writing",
    initial="DRAFT",
    phases=[
        PhaseSpec(
            phase_id="DRAFT",
            prompts={"main": "You are a technical writer. Draft the document."},
            to=("REVIEW",),
            tools=("submit_draft",),
        ),
        PhaseSpec(
            phase_id="REVIEW",
            prompts={"main": "You are a critical reviewer."},
            to=("DRAFT", "DONE"),
            approval=ApprovalGate(
                verdicts={"approved": "DONE", "revise": "DRAFT", "blocked": None}
            ),
        ),
        PhaseSpec(phase_id="DONE", prompts={"main": "Workflow complete."}, terminal=True),
    ],
)

await bind_phase_graph(world, entity_id, GRAPH, agent_key="main")
```

## Transitions

```python
from ecs_agent.phases import advance, force, record_approval

await advance(world, eid, "REVIEW", reason="draft submitted")   # validates adjacency
await record_approval(world, eid, "approved", notes="lgtm")     # gate routes to DONE
await force(world, eid, "DRAFT", reason="admin rollback")       # audited bypass
```

- `advance()` raises `InvalidPhaseTransitionError` on non-adjacent targets or
  terminal sources; nothing is partially applied.
- Every commit appends to `PhaseComponent.history` (bounded to 100), applies the
  target phase's effects (prompt binding; `PermissionComponent.allowed_tools`
  when `tools` is declared), and publishes `PhaseChangedEvent` (traced in
  Langfuse as `phase.transition`).
- There is no polling system and no system-ordering requirement.

## Prompts

Put `${_phase_prompt}` in the system prompt template; the
`PhasePromptPlaceholderProvider` resolves the current phase's prompt for the
bound `agent_key`. The provider fingerprint hashes the resolved text — phases
sharing identical prompt text keep a stable fingerprint, so transitions between
them never invalidate the cache-stable prefix. `_phase_prompt` renders in the
volatile tail (see the Anthropic prompt caching notes in the README).

## Checkpoint & Resume

`PhaseComponent` and `PhaseApprovalsComponent` serialize; the graph does not.
After restoring, call `bind_phase_graph()` again with the same graph:

- Progress is **never reset** — the restored phase is preserved.
- Structural drift: if the restored phase still exists, the stored hash is
  updated with a warning; if it was removed, `PhaseGraphMismatchError` is raised.
- The restored phase's `on_resume` policy is applied (e.g. demote
  `TASK_RUNNING` → `TASK_BLOCKED`), audited as a forced transition.
- Forgetting to re-bind fails loudly on first API call or prompt render.

## Tool allowlists

When a phase declares `tools=(...)`, entering it sets
`PermissionComponent.allowed_tools` (enforced by the existing
`PermissionSystem`); `denied_tools` is never touched. If any phase declares
`tools`, treat `allowed_tools` as owned by the phase graph.
