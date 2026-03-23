"""Collect one-shot context entries for prompt injection."""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable
import uuid

from ecs_agent.components import (
    ContextEntry,
    PromptContextQueueComponent,
    UserPromptConfigComponent,
    ToolResultsComponent,
)
from ecs_agent.core.world import World
from ecs_agent.types import (
    DelegationCompletedEvent,
    EntityId,
    ToolExecutionCompletedEvent,
)

CONTEXT_ENTRY_DELIMITER = "\n\n---\n\n"
CONTEXT_POOL_OVERFLOW_SOURCE = "context_pool:overflow"
CONTEXT_POOL_OVERFLOW_FOOTER_PREFIX = "context_pool_overflow"
TOOL_CONTEXT_PRIORITY = 30
SUBAGENT_CONTEXT_PRIORITY = 20
STRUCTURED_OUTPUT_CONTEXT_PRIORITY = 10
OVERFLOW_FOOTER_PRIORITY = -1


class PromptContextCollectorSystem:
    def __init__(
        self,
        priority: int = 0,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.priority = priority
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self._subscribed_world_id: int | None = None
        self._tool_events: dict[EntityId, list[ToolExecutionCompletedEvent]] = (
            defaultdict(list)
        )
        self._delegation_events: dict[EntityId, list[DelegationCompletedEvent]] = (
            defaultdict(list)
        )
        self._seen_structured_output_ids: dict[EntityId, set[str]] = defaultdict(set)

    async def process(self, world: World) -> None:
        self._ensure_subscribed(world)

        for entity_id, components in world.query(
            UserPromptConfigComponent,
            PromptContextQueueComponent,
        ):
            config, queue = components
            if not config.enable_context_pool:
                continue

            next_registration_order = (
                max((entry.registration_order for entry in queue.entries), default=-1)
                + 1
            )

            for priority, source, content in self._collect_entries(world, entity_id):
                queue.entries.append(
                    ContextEntry(
                        entry_id=uuid.uuid4().hex,
                        priority=priority,
                        source_label=source,
                        content=content,
                        registration_order=next_registration_order,
                    )
                )
                next_registration_order += 1

            sorted_items = sorted(
                [
                    item
                    for item in queue.entries
                    if item.source_label != CONTEXT_POOL_OVERFLOW_SOURCE
                ],
                key=lambda item: (-item.priority, item.registration_order),
            )
            queue.entries = self._truncate(sorted_items, config.context_pool_max_chars)

    def _ensure_subscribed(self, world: World) -> None:
        world_id = id(world)
        if self._subscribed_world_id == world_id:
            return

        world.event_bus.subscribe(
            ToolExecutionCompletedEvent, self._on_tool_execution_completed
        )
        world.event_bus.subscribe(
            DelegationCompletedEvent, self._on_delegation_completed
        )
        self._subscribed_world_id = world_id

    async def _on_tool_execution_completed(
        self, event: ToolExecutionCompletedEvent
    ) -> None:
        self._tool_events[event.entity_id].append(event)

    async def _on_delegation_completed(self, event: DelegationCompletedEvent) -> None:
        self._delegation_events[event.entity_id].append(event)

    def _collect_entries(
        self, world: World, entity_id: EntityId
    ) -> list[tuple[int, str, str]]:
        entries: list[tuple[int, str, str]] = []

        tool_events = self._tool_events.pop(entity_id, [])
        for event in tool_events:
            status = "success" if event.success else "error"
            error = "" if event.success else event.result
            entries.append(
                (
                    TOOL_CONTEXT_PRIORITY,
                    f"tool:{event.tool_call_id}",
                    self._normalize(
                        source=f"tool:{event.tool_call_id}",
                        status=status,
                        result=event.result,
                        error=error,
                    ),
                )
            )

        delegation_events = self._delegation_events.pop(entity_id, [])
        for delegation_event in delegation_events:
            status = "success" if delegation_event.success else "error"
            entries.append(
                (
                    SUBAGENT_CONTEXT_PRIORITY,
                    f"subagent:{delegation_event.subagent_name}",
                    self._normalize(
                        source=f"subagent:{delegation_event.subagent_name}",
                        status=status,
                        result=delegation_event.result,
                        error=delegation_event.error or "",
                    ),
                )
            )

        results_component = world.get_component(entity_id, ToolResultsComponent)
        if results_component is not None:
            seen_ids = self._seen_structured_output_ids[entity_id]
            for result_id, result in sorted(results_component.results.items()):
                if result_id in seen_ids:
                    continue
                seen_ids.add(result_id)
                entries.append(
                    (
                        STRUCTURED_OUTPUT_CONTEXT_PRIORITY,
                        f"structured_output:{result_id}",
                        self._normalize(
                            source=f"structured_output:{result_id}",
                            status="success",
                            result=result,
                            error="",
                        ),
                    )
                )

        return entries

    def _normalize(self, source: str, status: str, result: str, error: str) -> str:
        timestamp = self._now_provider().isoformat().replace("+00:00", "Z")
        return "\n".join(
            [
                f"source: {source}",
                f"status: {status}",
                f"result: {result}",
                f"error: {error}",
                f"timestamp: {timestamp}",
            ]
        )

    def _truncate(
        self,
        items: list[ContextEntry],
        max_chars: int,
    ) -> list[ContextEntry]:
        if max_chars <= 0:
            return []

        kept = list(items)
        dropped_count = 0
        while kept and self._rendered_length(kept) > max_chars:
            kept.pop()
            dropped_count += 1

        if dropped_count == 0:
            return kept

        while True:
            footer = self._overflow_footer(dropped_count)
            footer_item = ContextEntry(
                entry_id=uuid.uuid4().hex,
                priority=OVERFLOW_FOOTER_PRIORITY,
                source_label=CONTEXT_POOL_OVERFLOW_SOURCE,
                content=footer,
                registration_order=0,
            )
            with_footer = [*kept, footer_item]
            if self._rendered_length(with_footer) <= max_chars:
                return with_footer
            if not kept:
                return []
            kept.pop()
            dropped_count += 1

    def _overflow_footer(self, dropped_count: int) -> str:
        return f"{CONTEXT_POOL_OVERFLOW_FOOTER_PREFIX} dropped_entries={dropped_count}"

    def _rendered_length(self, items: list[ContextEntry]) -> int:
        if not items:
            return 0
        return sum(len(item.content) for item in items) + len(
            CONTEXT_ENTRY_DELIMITER
        ) * (len(items) - 1)


__all__ = [
    "CONTEXT_ENTRY_DELIMITER",
    "CONTEXT_POOL_OVERFLOW_FOOTER_PREFIX",
    "CONTEXT_POOL_OVERFLOW_SOURCE",
    "OVERFLOW_FOOTER_PRIORITY",
    "PromptContextCollectorSystem",
    "STRUCTURED_OUTPUT_CONTEXT_PRIORITY",
    "SUBAGENT_CONTEXT_PRIORITY",
    "TOOL_CONTEXT_PRIORITY",
]
