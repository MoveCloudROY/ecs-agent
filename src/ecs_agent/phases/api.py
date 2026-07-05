"""Explicit phase-transition API: bind, advance, force, record_approval."""

from __future__ import annotations

import datetime

from ecs_agent.components import (
    PermissionComponent,
    PhaseApprovalsComponent,
    PhaseComponent,
    PhaseDefinitionComponent,
    RunnerStateComponent,
)
from ecs_agent.core import World
from ecs_agent.logging import get_logger
from ecs_agent.phases.contracts import PhaseGraph, PhaseSpec
from ecs_agent.types import EntityId, PhaseChangedEvent

logger = get_logger(__name__)

HISTORY_LIMIT = 100


class PhaseError(ValueError):
    """Base error for phase API misuse."""


class PhaseIntegrityError(PhaseError):
    """Entity is unbound or half-bound (component present without definition)."""


class PhaseGraphMismatchError(PhaseError):
    """Restored phase state is incompatible with the graph being bound."""


class InvalidPhaseTransitionError(PhaseError):
    """Requested transition violates the graph's adjacency or terminality rules."""


async def bind_phase_graph(
    world: World,
    entity_id: EntityId,
    graph: PhaseGraph,
    *,
    agent_key: str = "main",
) -> PhaseComponent:
    """Idempotently bind a graph to an entity.

    Fresh entity: creates PhaseComponent at the graph's initial phase and applies
    that phase's effects. Restored entity (PhaseComponent already present, e.g.
    after checkpoint load): validates compatibility, re-attaches the definition,
    re-applies the current phase's effects, then applies its on_resume policy.
    Never resets progress.
    """
    for phase_spec in graph.phases_by_id.values():
        if agent_key not in phase_spec.prompts:
            raise PhaseError(
                f"agent_key {agent_key!r} has no prompt in phase {phase_spec.phase_id!r}"
            )

    existing = world.get_component(entity_id, PhaseComponent)
    if existing is None:
        component = PhaseComponent(
            graph_id=graph.graph_id,
            phase=graph.initial,
            graph_hash=graph.structure_hash,
            agent_key=agent_key,
            entered_at_tick=_current_tick(world),
        )
        world.add_component(entity_id, component)
        world.add_component(entity_id, PhaseDefinitionComponent(graph=graph))
        _apply_phase_effects(world, entity_id, graph, graph.phases_by_id[graph.initial])
        logger.info(
            "phase_graph_bound",
            entity_id=int(entity_id),
            graph_id=graph.graph_id,
            phase=component.phase,
        )
        return component

    if existing.graph_id != graph.graph_id:
        raise PhaseGraphMismatchError(
            f"entity is bound to graph {existing.graph_id!r}; "
            f"refusing to bind {graph.graph_id!r}"
        )
    if existing.phase not in graph.phases_by_id:
        raise PhaseGraphMismatchError(
            f"restored phase {existing.phase!r} no longer exists in graph "
            f"{graph.graph_id!r}; migrate the persisted state or force() it to a valid phase"
        )
    if existing.graph_hash != graph.structure_hash:
        logger.warning(
            "phase_graph_structure_changed",
            entity_id=int(entity_id),
            graph_id=graph.graph_id,
            phase=existing.phase,
        )
        existing.graph_hash = graph.structure_hash
    existing.agent_key = agent_key
    world.add_component(entity_id, PhaseDefinitionComponent(graph=graph))
    _apply_phase_effects(world, entity_id, graph, graph.phases_by_id[existing.phase])

    spec = graph.phases_by_id[existing.phase]
    if spec.on_resume is not None and spec.on_resume != existing.phase:
        await _commit(
            world,
            entity_id,
            existing,
            graph,
            to_phase=spec.on_resume,
            reason="on_resume",
            forced=True,
        )
    logger.info(
        "phase_graph_rebound",
        entity_id=int(entity_id),
        graph_id=graph.graph_id,
        phase=existing.phase,
    )
    return existing


async def advance(
    world: World, entity_id: EntityId, to_phase: str, *, reason: str
) -> PhaseComponent:
    """Commit a graph-validated transition from the current phase to to_phase."""
    component, graph = _require_bound(world, entity_id)
    spec = graph.phases_by_id[component.phase]
    if spec.terminal:
        raise InvalidPhaseTransitionError(
            f"cannot advance from terminal phase {component.phase!r} "
            "(use force() for administrative recovery)"
        )
    if to_phase not in spec.to:
        raise InvalidPhaseTransitionError(
            f"invalid transition {component.phase!r} -> {to_phase!r}; "
            f"allowed: {sorted(spec.to)}"
        )
    await _commit(
        world, entity_id, component, graph, to_phase=to_phase, reason=reason, forced=False
    )
    return component


async def force(
    world: World, entity_id: EntityId, to_phase: str, *, reason: str
) -> PhaseComponent:
    """Administrative transition bypassing adjacency (audited as forced=True).

    Still requires the entity to be fully bound and to_phase to exist in the graph.
    """
    component, graph = _require_bound(world, entity_id)
    if to_phase not in graph.phases_by_id:
        raise PhaseError(f"unknown phase {to_phase!r} in graph {graph.graph_id!r}")
    await _commit(
        world, entity_id, component, graph, to_phase=to_phase, reason=reason, forced=True
    )
    return component


async def record_approval(
    world: World,
    entity_id: EntityId,
    verdict: str,
    *,
    notes: str | None = None,
    decided_at: str | None = None,
) -> str:
    """Record a review verdict for the current phase and auto-advance per its gate.

    Returns the phase after any auto-advance.
    """
    component, graph = _require_bound(world, entity_id)
    gate = graph.phases_by_id[component.phase].approval
    if gate is None:
        raise PhaseError(f"phase {component.phase!r} declares no approval gate")
    if verdict not in gate.verdicts:
        raise PhaseError(
            f"invalid verdict {verdict!r} for phase {component.phase!r}; "
            f"allowed: {sorted(gate.verdicts)}"
        )

    ledger = world.get_component(entity_id, PhaseApprovalsComponent)
    if ledger is None:
        ledger = PhaseApprovalsComponent()
        world.add_component(entity_id, ledger)
    ledger.records.append(
        {
            "phase": component.phase,
            "verdict": verdict,
            "notes": notes,
            "decided_at": decided_at or _utcnow_isoformat(),
        }
    )

    target = gate.verdicts[verdict]
    if target is not None:
        await advance(world, entity_id, target, reason=f"approval:{verdict}")
    return component.phase


def allowed_targets(world: World, entity_id: EntityId) -> frozenset[str]:
    """Return the phases reachable from the current phase via advance()."""
    component, graph = _require_bound(world, entity_id)
    return frozenset(graph.phases_by_id[component.phase].to)


def is_terminal(world: World, entity_id: EntityId) -> bool:
    """Return True when the entity's current phase is terminal."""
    component, graph = _require_bound(world, entity_id)
    return graph.phases_by_id[component.phase].terminal


def latest_verdicts(world: World, entity_id: EntityId) -> dict[str, str]:
    """Return the most recent verdict recorded for each phase."""
    ledger = world.get_component(entity_id, PhaseApprovalsComponent)
    if ledger is None:
        return {}
    verdicts: dict[str, str] = {}
    for record in ledger.records:
        verdicts[str(record["phase"])] = str(record["verdict"])
    return verdicts


def _require_bound(
    world: World, entity_id: EntityId
) -> tuple[PhaseComponent, PhaseGraph]:
    component = world.get_component(entity_id, PhaseComponent)
    definition = world.get_component(entity_id, PhaseDefinitionComponent)
    if component is None and definition is None:
        raise PhaseIntegrityError(f"entity {int(entity_id)} has no phase graph bound")
    if component is None or definition is None:
        missing = "PhaseComponent" if component is None else "PhaseDefinitionComponent"
        raise PhaseIntegrityError(
            f"entity {int(entity_id)} is half-bound: {missing} is missing. "
            "After restoring a checkpoint, call bind_phase_graph() before using the phase API."
        )
    return component, definition.graph


async def _commit(
    world: World,
    entity_id: EntityId,
    component: PhaseComponent,
    graph: PhaseGraph,
    *,
    to_phase: str,
    reason: str,
    forced: bool,
) -> None:
    tick = _current_tick(world)
    from_phase = component.phase
    component.phase = to_phase
    component.entered_at_tick = tick
    component.history.append(
        {"from": from_phase, "to": to_phase, "reason": reason, "forced": forced, "tick": tick}
    )
    del component.history[:-HISTORY_LIMIT]
    _apply_phase_effects(world, entity_id, graph, graph.phases_by_id[to_phase])
    logger.info(
        "phase_transition",
        entity_id=int(entity_id),
        graph_id=graph.graph_id,
        from_phase=from_phase,
        to_phase=to_phase,
        reason=reason,
        forced=forced,
    )
    await world.event_bus.publish(
        PhaseChangedEvent(
            entity_id=entity_id,
            graph_id=graph.graph_id,
            from_phase=from_phase,
            to_phase=to_phase,
            reason=reason,
            forced=forced,
            tick=tick,
        )
    )


def _apply_phase_effects(
    world: World, entity_id: EntityId, graph: PhaseGraph, spec: PhaseSpec
) -> None:
    if not graph.manages_tools:
        return
    permissions = world.get_component(entity_id, PermissionComponent)
    if spec.tools is None:
        # The graph owns allowed_tools: a phase with no declaration is
        # unrestricted (empty allowlist == allow-all under PermissionSystem).
        if permissions is not None:
            permissions.allowed_tools = []
        return
    if permissions is None:
        permissions = PermissionComponent()
        world.add_component(entity_id, permissions)
    permissions.allowed_tools = list(spec.tools)


def _current_tick(world: World) -> int:
    runner_states = world.query(RunnerStateComponent)
    if runner_states:
        _, (runner_state,) = runner_states[0]
        return int(runner_state.current_tick)
    return 0


def _utcnow_isoformat() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()
