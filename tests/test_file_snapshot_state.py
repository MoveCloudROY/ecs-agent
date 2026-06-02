from __future__ import annotations

from pathlib import Path

from ecs_agent.components import ToolRuntimeStateComponent
from ecs_agent.core import World
from ecs_agent.tools.builtins.file_snapshot import (
    FileSnapshotState,
    current_file_snapshot_state,
)
from ecs_agent.tools.context import ToolExecutionContext, use_tool_context


def test_file_snapshot_state_keeps_multiple_snapshots_per_file(tmp_path: Path) -> None:
    component = ToolRuntimeStateComponent()
    state = FileSnapshotState(component)
    target = tmp_path / "same.txt"

    state.record_read(
        file_path="same.txt",
        target=target,
        content="one",
        offset=1,
        limit=1,
    )
    state.record_read(
        file_path="same.txt",
        target=target,
        content="one\ntwo\nthree\nfour",
        offset=4,
        limit=1,
    )

    assert [snapshot.content for snapshot in state.snapshots_for(target)] == [
        "one",
        "four",
    ]


def test_file_snapshot_state_constructor_shares_component_histories(
    tmp_path: Path,
) -> None:
    component = ToolRuntimeStateComponent()
    first = FileSnapshotState(component)
    second = FileSnapshotState(component)
    target = tmp_path / "same.txt"

    first.record_read(
        file_path="same.txt",
        target=target,
        content="one",
        offset=1,
        limit=1,
    )

    assert [snapshot.content for snapshot in second.snapshots_for(target)] == [
        "one",
    ]


def test_current_file_snapshot_state_reuses_component_state() -> None:
    world = World()
    entity_id = world.create_entity()
    context = ToolExecutionContext(world=world, entity_id=entity_id, tool_name="read_file")

    with use_tool_context(context):
        first = current_file_snapshot_state()
        second = current_file_snapshot_state()

    component = world.get_component(entity_id, ToolRuntimeStateComponent)
    assert component is not None
    namespace = component.namespaces[FileSnapshotState.namespace_name]
    assert first is second
    assert namespace.values[FileSnapshotState.store_key] is first
    assert namespace.version == 1


def test_file_snapshot_state_finds_unique_line_anchor_across_recent_snapshots(
    tmp_path: Path,
) -> None:
    component = ToolRuntimeStateComponent()
    state = FileSnapshotState(component)
    target = tmp_path / "same.txt"
    state.record_read(
        file_path="same.txt",
        target=target,
        content="one",
        offset=1,
        limit=1,
    )
    state.record_read(
        file_path="same.txt",
        target=target,
        content="one\ntwo\nthree\nfour",
        offset=4,
        limit=1,
    )

    anchor = state.find_anchor(target, 1)

    assert anchor.line_number == 1
    assert anchor.content == "one"


def test_file_snapshot_state_prefers_latest_matching_line_anchor(
    tmp_path: Path,
) -> None:
    component = ToolRuntimeStateComponent()
    state = FileSnapshotState(component)
    target = tmp_path / "same.txt"
    state.record_read(
        file_path="same.txt",
        target=target,
        content="repeat\none",
        offset=1,
        limit=1,
    )
    state.record_read(
        file_path="same.txt",
        target=target,
        content="updated\ntwo",
        offset=1,
        limit=1,
    )

    anchor = state.find_anchor(target, 1)

    assert anchor.line_number == 1
    assert anchor.content == "updated"
