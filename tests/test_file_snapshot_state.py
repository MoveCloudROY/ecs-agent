from __future__ import annotations

from ecs_agent.components import ToolRuntimeStateComponent
from ecs_agent.tools.builtins.file_snapshot import FileSnapshotState


def test_file_snapshot_state_keeps_multiple_snapshots_per_file() -> None:
    component = ToolRuntimeStateComponent()
    state = FileSnapshotState(component)

    state.record_read(
        path="same.txt",
        content="one",
        digest="digest-1",
        offset=1,
        line_count=1,
    )
    state.record_read(
        path="same.txt",
        content="four",
        digest="digest-2",
        offset=4,
        line_count=1,
    )

    assert [snapshot.content for snapshot in state.snapshots_for("same.txt")] == [
        "one",
        "four",
    ]


def test_file_snapshot_state_finds_unique_line_anchor_across_recent_snapshots() -> None:
    component = ToolRuntimeStateComponent()
    state = FileSnapshotState(component)
    state.record_read(
        path="same.txt",
        content="one",
        digest="digest-1",
        offset=1,
        line_count=1,
    )
    state.record_read(
        path="same.txt",
        content="four",
        digest="digest-2",
        offset=4,
        line_count=1,
    )

    anchor = state.find_anchor("same.txt", 1)

    assert anchor.line_number == 1
    assert anchor.content == "one"


def test_file_snapshot_state_rejects_ambiguous_line_anchor() -> None:
    component = ToolRuntimeStateComponent()
    state = FileSnapshotState(component)
    state.record_read(
        path="same.txt",
        content="repeat",
        digest="digest-1",
        offset=1,
        line_count=1,
    )
    state.record_read(
        path="same.txt",
        content="repeat",
        digest="digest-2",
        offset=1,
        line_count=1,
    )

    try:
        state.find_anchor("same.txt", 1)
    except ValueError as exc:
        assert "line is not unique in multiple read_file snapshots" in str(exc)
    else:
        raise AssertionError("ambiguous line anchor should be rejected")
