"""Tests for context management component types."""

import pytest

from ecs_agent.components import (
    StreamingComponent,
    CheckpointComponent,
    CompactionConfigComponent,
    ConversationArchiveComponent,
    RunnerStateComponent,
)


def test_streaming_component_instantiation_default() -> None:
    """Test StreamingComponent instantiation with default enabled=False."""
    component = StreamingComponent()
    assert component.enabled is False


def test_streaming_component_instantiation_enabled() -> None:
    """Test StreamingComponent instantiation with enabled=True."""
    component = StreamingComponent(enabled=True)
    assert component.enabled is True


def test_streaming_component_is_dataclass() -> None:
    """Test StreamingComponent is a dataclass with slots."""
    component = StreamingComponent()
    assert hasattr(component, "__slots__")


def test_checkpoint_component_instantiation_default() -> None:
    """Test CheckpointComponent instantiation with defaults."""
    component = CheckpointComponent()
    assert component.snapshots == []
    assert component.max_snapshots == 10


def test_checkpoint_component_instantiation_with_values() -> None:
    """Test CheckpointComponent instantiation with custom values."""
    snapshots = [{"key": "value"}]
    component = CheckpointComponent(snapshots=snapshots, max_snapshots=20)
    assert component.snapshots == snapshots
    assert component.max_snapshots == 20


def test_checkpoint_component_default_factory_independence() -> None:
    """Test CheckpointComponent snapshot lists are independent."""
    comp1 = CheckpointComponent()
    comp2 = CheckpointComponent()

    comp1.snapshots.append({"data": 1})
    assert comp2.snapshots == []
    assert comp1.snapshots != comp2.snapshots


def test_checkpoint_component_is_dataclass() -> None:
    """Test CheckpointComponent is a dataclass with slots."""
    component = CheckpointComponent()
    assert hasattr(component, "__slots__")


def test_compaction_config_component_instantiation() -> None:
    """Test CompactionConfigComponent instantiation with required fields."""
    component = CompactionConfigComponent(
        threshold_tokens=5000, summary_model="gpt-3.5-turbo"
    )
    assert component.threshold_tokens == 5000
    assert component.summary_model == "gpt-3.5-turbo"


def test_compaction_config_component_with_different_values() -> None:
    """Test CompactionConfigComponent with various threshold values."""
    comp_low = CompactionConfigComponent(threshold_tokens=1000, summary_model="gpt-4")
    comp_high = CompactionConfigComponent(
        threshold_tokens=100000, summary_model="gpt-4-turbo"
    )

    assert comp_low.threshold_tokens == 1000
    assert comp_high.threshold_tokens == 100000
    assert comp_low.summary_model != comp_high.summary_model


def test_compaction_config_component_is_dataclass() -> None:
    """Test CompactionConfigComponent is a dataclass with slots."""
    component = CompactionConfigComponent(
        threshold_tokens=5000, summary_model="gpt-3.5-turbo"
    )
    assert hasattr(component, "__slots__")


def test_conversation_archive_component_instantiation_default() -> None:
    """Test ConversationArchiveComponent instantiation with defaults."""
    component = ConversationArchiveComponent()
    assert component.archived_summaries == []


def test_conversation_archive_component_instantiation_with_values() -> None:
    """Test ConversationArchiveComponent instantiation with summaries."""
    summaries = ["Summary 1", "Summary 2"]
    component = ConversationArchiveComponent(archived_summaries=summaries)
    assert component.archived_summaries == summaries


def test_conversation_archive_component_default_factory_independence() -> None:
    """Test ConversationArchiveComponent summary lists are independent."""
    comp1 = ConversationArchiveComponent()
    comp2 = ConversationArchiveComponent()

    comp1.archived_summaries.append("Summary A")
    assert comp2.archived_summaries == []
    assert comp1.archived_summaries != comp2.archived_summaries


def test_conversation_archive_component_is_dataclass() -> None:
    """Test ConversationArchiveComponent is a dataclass with slots."""
    component = ConversationArchiveComponent()
    assert hasattr(component, "__slots__")


def test_runner_state_component_instantiation() -> None:
    """Test RunnerStateComponent instantiation with required and default fields."""
    component = RunnerStateComponent(current_tick=5)
    assert component.current_tick == 5
    assert component.is_paused is False
    assert component.checkpoint_path is None


def test_runner_state_component_with_all_fields() -> None:
    """Test RunnerStateComponent instantiation with all fields."""
    component = RunnerStateComponent(
        current_tick=10,
        is_paused=True,
        checkpoint_path="/path/to/checkpoint",
    )
    assert component.current_tick == 10
    assert component.is_paused is True
    assert component.checkpoint_path == "/path/to/checkpoint"


def test_runner_state_component_checkpoint_path_optional() -> None:
    """Test RunnerStateComponent checkpoint_path is optional."""
    comp_none = RunnerStateComponent(current_tick=1, checkpoint_path=None)
    comp_with_path = RunnerStateComponent(current_tick=1, checkpoint_path="/some/path")

    assert comp_none.checkpoint_path is None
    assert comp_with_path.checkpoint_path == "/some/path"


def test_runner_state_component_is_dataclass() -> None:
    """Test RunnerStateComponent is a dataclass with slots."""
    component = RunnerStateComponent(current_tick=0)
    assert hasattr(component, "__slots__")


def test_all_context_components_are_slots_dataclasses() -> None:
    """Test all 5 components are dataclass(slots=True)."""
    streaming = StreamingComponent()
    checkpoint = CheckpointComponent()
    compaction = CompactionConfigComponent(
        threshold_tokens=5000, summary_model="gpt-3.5-turbo"
    )
    archive = ConversationArchiveComponent()
    runner_state = RunnerStateComponent(current_tick=0)

    for comp in [streaming, checkpoint, compaction, archive, runner_state]:
        assert hasattr(comp, "__slots__"), (
            f"{comp.__class__.__name__} missing __slots__"
        )


def test_checkpoint_component_can_hold_multiple_snapshots() -> None:
    """Test CheckpointComponent can accumulate multiple snapshots."""
    component = CheckpointComponent()

    for i in range(5):
        component.snapshots.append({"tick": i, "data": f"snapshot_{i}"})

    assert len(component.snapshots) == 5
    assert component.snapshots[-1] == {"tick": 4, "data": "snapshot_4"}


def test_conversation_archive_can_hold_multiple_summaries() -> None:
    """Test ConversationArchiveComponent can accumulate multiple summaries."""
    component = ConversationArchiveComponent()

    summaries = [f"Summary {i}" for i in range(3)]
    for summary in summaries:
        component.archived_summaries.append(summary)

    assert len(component.archived_summaries) == 3
    assert component.archived_summaries == summaries


def test_streaming_component_repr() -> None:
    """Test StreamingComponent has valid string representation."""
    component = StreamingComponent(enabled=True)
    repr_str = repr(component)
    assert "StreamingComponent" in repr_str or "enabled" in repr_str


def test_checkpoint_component_repr() -> None:
    """Test CheckpointComponent has valid string representation."""
    component = CheckpointComponent()
    repr_str = repr(component)
    assert "CheckpointComponent" in repr_str or "snapshots" in repr_str


def test_runner_state_zero_tick() -> None:
    """Test RunnerStateComponent with zero current_tick."""
    component = RunnerStateComponent(current_tick=0)
    assert component.current_tick == 0
    assert component.is_paused is False


def test_runner_state_large_tick_count() -> None:
    """Test RunnerStateComponent with large tick counts."""
    component = RunnerStateComponent(current_tick=1000000)
    assert component.current_tick == 1000000
