"""Delegation execution mechanism for subagents.

``DelegationExecutor`` (Task 5 of the subagent package refactor) owns the mechanism
of running one delegation against an already-assembled child world: running the
child ``Runner`` loop, extracting the terminal result, bridging child streaming
events onto the parent bus, and installing parent observability on the child world.

Orchestration (child-entity creation, event publishing, stub sync) stays on
``SubagentSystem`` so its ``_assemble_child_world`` seam remains monkeypatchable.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from ecs_agent.components import ConversationComponent
from ecs_agent.core.runner import Runner
from ecs_agent.core.world import World
from ecs_agent.observability.context import current_run_id, current_trace_id
from ecs_agent.observability.install import install_observability
from ecs_agent.types import (
    EntityId,
    Message,
    StreamContentDeltaEvent,
    StreamEndEvent,
    StreamReasoningDeltaEvent,
    StreamStartEvent,
    SubagentConfig,
    SubagentStreamDeltaEvent,
    SubagentStreamEndEvent,
    SubagentStreamStartEvent,
)


def active_observability_context(
    world: World,
) -> tuple[str | None, str | None, str | None]:
    """Return active trace, run, and root observation IDs for this world."""
    active_run_id = current_run_id()
    active_trace_id = current_trace_id()
    active_parent_observation_id: str | None = None
    parent_subscriber = getattr(
        world,
        "_ecs_agent_observability_subscriber",
        None,
    )
    trace_states = getattr(parent_subscriber, "trace_states", None)
    if isinstance(active_run_id, str) and isinstance(trace_states, dict):
        trace_state = trace_states.get(active_run_id)
        trace_state_id = getattr(trace_state, "trace_id", None)
        trace_state_observation_id = getattr(trace_state, "observation_id", None)
        if isinstance(trace_state_id, str):
            active_trace_id = trace_state_id
        if isinstance(trace_state_observation_id, str):
            active_parent_observation_id = trace_state_observation_id
    return (active_trace_id, active_run_id, active_parent_observation_id)


class DelegationExecutor:
    """Runs a delegation against an assembled child world and extracts the result."""

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _iso_timestamp(self, timestamp: float) -> str:
        return (
            datetime.fromtimestamp(timestamp, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    async def run_delegation(
        self,
        child_world: World,
        child_entity: EntityId,
        task: str,
        config: SubagentConfig,
    ) -> str:
        """Execute child world delegation run and return extracted result."""
        child_world.add_component(
            child_entity,
            ConversationComponent(messages=[Message(role="user", content=task)]),
        )
        runner = Runner()
        await runner.run(
            child_world,
            max_ticks=config.max_ticks,
            trace_id=getattr(
                child_world,
                "_ecs_agent_trace_id",
                current_trace_id(),
            ),
            run_id=getattr(
                child_world,
                "_ecs_agent_run_id",
                current_run_id(),
            ),
            parent_observation_id=getattr(
                child_world,
                "_ecs_agent_parent_observation_id",
                None,
            ),
            emit_root_trace=False,
        )
        return self._extract_delegation_result(child_world, child_entity)

    def _extract_delegation_result(
        self, child_world: World, child_entity: EntityId
    ) -> str:
        """Extract terminal delegation result from child conversation."""
        child_conv = child_world.get_component(child_entity, ConversationComponent)
        if child_conv is None:
            return "Error: No conversation found"

        for message in reversed(child_conv.messages):
            if message.role == "assistant":
                return message.content
        return "Error: No assistant message found in subagent conversation"

    def install_child_observability(
        self,
        *,
        parent_world: World,
        child_world: World,
        trace_id: str | None = None,
        run_id: str | None = None,
        parent_observation_id: str,
    ) -> None:
        """Install parent observability sink on a child world when available."""
        parent_sink = getattr(parent_world, "_ecs_agent_observability_sink", None)
        if parent_sink is None:
            return
        parent_config = getattr(
            parent_world,
            "_ecs_agent_observability_config",
            None,
        )
        install_observability(child_world, parent_sink, config=parent_config)
        active_trace_id, active_run_id, _ = active_observability_context(parent_world)
        if trace_id is not None:
            active_trace_id = trace_id
        if run_id is not None:
            active_run_id = run_id
        if active_trace_id is not None:
            setattr(child_world, "_ecs_agent_trace_id", active_trace_id)
        if active_run_id is not None:
            setattr(child_world, "_ecs_agent_run_id", active_run_id)
        setattr(
            child_world,
            "_ecs_agent_parent_observation_id",
            parent_observation_id,
        )

    def bridge_subagent_stream_events(
        self,
        *,
        parent_world: World,
        child_world: World,
        parent_entity_id: EntityId,
        session_id: str,
        category: str,
        child_world_name: str,
    ) -> Any:
        seq = 0

        def next_seq() -> int:
            nonlocal seq
            current = seq
            seq += 1
            return current

        def publish_translated_event(event: object) -> None:
            asyncio.create_task(parent_world.event_bus.publish(event))

        async def on_start(event: StreamStartEvent) -> None:
            publish_translated_event(
                SubagentStreamStartEvent(
                    session_id=session_id,
                    parent_entity_id=parent_entity_id,
                    category=category,
                    child_world_name=child_world_name,
                    seq=next_seq(),
                    timestamp=self._iso_timestamp(event.timestamp),
                )
            )

        async def on_reasoning_delta(event: StreamReasoningDeltaEvent) -> None:
            publish_translated_event(
                SubagentStreamDeltaEvent(
                    session_id=session_id,
                    parent_entity_id=parent_entity_id,
                    category=category,
                    child_world_name=child_world_name,
                    seq=next_seq(),
                    timestamp=self._utc_now_iso(),
                    delta="",
                    reasoning_delta=event.reasoning_delta,
                )
            )

        async def on_content_delta(event: StreamContentDeltaEvent) -> None:
            publish_translated_event(
                SubagentStreamDeltaEvent(
                    session_id=session_id,
                    parent_entity_id=parent_entity_id,
                    category=category,
                    child_world_name=child_world_name,
                    seq=next_seq(),
                    timestamp=self._utc_now_iso(),
                    delta=event.delta,
                )
            )

        async def on_end(event: StreamEndEvent) -> None:
            publish_translated_event(
                SubagentStreamEndEvent(
                    session_id=session_id,
                    parent_entity_id=parent_entity_id,
                    category=category,
                    child_world_name=child_world_name,
                    seq=next_seq(),
                    timestamp=self._iso_timestamp(event.timestamp),
                )
            )

        child_world.event_bus.subscribe(StreamStartEvent, on_start)
        child_world.event_bus.subscribe(StreamReasoningDeltaEvent, on_reasoning_delta)
        child_world.event_bus.subscribe(StreamContentDeltaEvent, on_content_delta)
        child_world.event_bus.subscribe(StreamEndEvent, on_end)

        def cleanup() -> None:
            child_world.event_bus.unsubscribe(StreamStartEvent, on_start)
            child_world.event_bus.unsubscribe(
                StreamReasoningDeltaEvent,
                on_reasoning_delta,
            )
            child_world.event_bus.unsubscribe(StreamContentDeltaEvent, on_content_delta)
            child_world.event_bus.unsubscribe(StreamEndEvent, on_end)

        return cleanup


__all__ = ["DelegationExecutor", "active_observability_context"]
