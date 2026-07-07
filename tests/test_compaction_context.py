"""Tests for the compaction context provider seam (prompts.compaction_context)."""

from __future__ import annotations

from typing import get_args

from ecs_agent.components import (
    SubagentNotificationQueueComponent,
    SubagentSessionTableComponent,
)
from ecs_agent.core import World
from ecs_agent.prompts.compaction_context import (
    DEFAULT_COMPACTION_CONTEXT_PROVIDERS,
    SubagentCompactionContextProvider,
    render_compaction_context_blocks,
)
from ecs_agent.types import (
    COMPLETED_SUBAGENT_STATUSES,
    FAILED_SUBAGENT_STATUSES,
    PENDING_SUBAGENT_STATUSES,
    EntityId,
    SubagentLifecycleStatus,
    SubagentNotificationRecord,
    SubagentSessionRecord,
)


def _session(
    session_id: str,
    *,
    status: str,
    prompt: str = "Investigate",
) -> SubagentSessionRecord:
    return SubagentSessionRecord(
        session_id=session_id,
        category="research",
        prompt=prompt,
        parent_entity_id=EntityId(1),
        created_at="2026-04-01T10:00:00Z",
        updated_at="2026-04-01T10:05:00Z",
        status=status,
    )


def test_subagent_provider_classifies_and_formats_sessions() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        SubagentSessionTableComponent(
            sessions={
                "sess-running": _session("sess-running", status="running"),
                "sess-succeeded": _session(
                    "sess-succeeded", status="succeeded", prompt="Summarize"
                ),
                "sess-cancelled": _session(
                    "sess-cancelled", status="cancelled", prompt="Abort"
                ),
            }
        ),
    )
    world.add_component(
        entity_id,
        SubagentNotificationQueueComponent(
            notifications=[
                SubagentNotificationRecord(
                    notification_id="sess-succeeded:succeeded",
                    session_id="sess-succeeded",
                    parent_entity_id=1,
                    terminal_status="succeeded",
                    summary="cached summary",
                    error=None,
                    created_at="2026-04-01T10:06:00Z",
                    delivered_at=None,
                )
            ]
        ),
    )

    block = SubagentCompactionContextProvider().render_compaction_context(
        world, entity_id
    )

    assert block == (
        "Subagent session state:\n"
        "Pending: sess-running\n"
        "Completed (cancelled): sess-cancelled\n"
        "Completed (succeeded): sess-succeeded\n"
        "sess-succeeded: notification status=succeeded delivered=no "
        'summary="cached summary"'
    )


def test_subagent_provider_returns_none_without_components() -> None:
    world = World()
    entity_id = world.create_entity()

    provider = SubagentCompactionContextProvider()

    assert provider.render_compaction_context(world, entity_id) is None


def test_subagent_provider_returns_none_when_state_is_empty() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, SubagentSessionTableComponent(sessions={}))
    world.add_component(
        entity_id, SubagentNotificationQueueComponent(notifications=[])
    )

    provider = SubagentCompactionContextProvider()

    assert provider.render_compaction_context(world, entity_id) is None


def test_status_classification_sets_cover_all_lifecycle_statuses() -> None:
    """Guard against drift: a newly added SubagentLifecycleStatus value must be
    classified as pending or completed, or it silently disappears from
    compaction context."""
    all_statuses = set(get_args(SubagentLifecycleStatus))

    assert PENDING_SUBAGENT_STATUSES | COMPLETED_SUBAGENT_STATUSES == all_statuses
    assert not PENDING_SUBAGENT_STATUSES & COMPLETED_SUBAGENT_STATUSES
    assert FAILED_SUBAGENT_STATUSES <= COMPLETED_SUBAGENT_STATUSES


def test_render_blocks_joins_providers_in_order_and_skips_empty() -> None:
    class _StaticProvider:
        def __init__(self, provider_id: str, block: str | None) -> None:
            self.provider_id = provider_id
            self._block = block

        def render_compaction_context(
            self, world: World, entity_id: EntityId
        ) -> str | None:
            return self._block

    world = World()
    entity_id = world.create_entity()
    providers = [
        _StaticProvider("first", "first block"),
        _StaticProvider("silent", None),
        _StaticProvider("second", "second block"),
    ]

    joined = render_compaction_context_blocks(world, entity_id, providers)

    assert joined == "first block\n\nsecond block"


def test_render_blocks_returns_none_when_all_providers_are_silent() -> None:
    world = World()
    entity_id = world.create_entity()

    assert render_compaction_context_blocks(world, entity_id, []) is None


def test_default_providers_include_subagent_sessions_and_todo_list() -> None:
    provider_ids = [
        provider.provider_id for provider in DEFAULT_COMPACTION_CONTEXT_PROVIDERS
    ]

    assert provider_ids == ["subagent_sessions", "todo_list"]
