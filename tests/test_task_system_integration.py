"""Integration tests for task system persistence."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ecs_agent.components import TaskComponent
from ecs_agent.scratchbook.service import ScratchbookService
from ecs_agent.task.persistence import (
    TaskEventLogTamperError,
    TaskPersistenceService,
    compute_task_snapshot_hash,
)
from ecs_agent.types import (
    EntityId,
    TaskBlockedEvent,
    TaskCompletedEvent,
    TaskCreatedEvent,
    TaskFailedEvent,
    TaskStateChangedEvent,
    TaskStatus,
)


@pytest.fixture
def tmp_scratchbook(tmp_path: Path) -> Path:
    """Create temporary scratchbook directory."""
    scratchbook_root = tmp_path / ".scratchbook"
    scratchbook_root.mkdir(parents=True, exist_ok=True)
    return scratchbook_root


@pytest.fixture
def scratchbook_service(tmp_scratchbook: Path) -> ScratchbookService:
    """Create scratchbook service instance."""
    return ScratchbookService(tmp_scratchbook)


@pytest.fixture
def persistence_service(
    scratchbook_service: ScratchbookService,
) -> TaskPersistenceService:
    """Create task persistence service instance."""
    return TaskPersistenceService(scratchbook_service)


@pytest.fixture
def sample_task_component() -> TaskComponent:
    """Create sample task component."""
    return TaskComponent(
        task_id="task-001",
        description="Test task",
        expected_output="Test output",
        assigned_agent=EntityId(42),
        tools=["tool1", "tool2"],
        context_dependencies=["dep1"],
        status=TaskStatus.READY,
        priority=5,
        output_schema={"type": "object"},
    )


class TestTaskSnapshotPersistence:
    """Tests for mutable task snapshot persistence."""

    def test_persist_and_read_task_snapshot_roundtrip(
        self,
        persistence_service: TaskPersistenceService,
        sample_task_component: TaskComponent,
    ) -> None:
        """Test snapshot can be persisted and read back accurately."""
        task_id = "task-001"

        # Persist snapshot
        ref = persistence_service.persist_task_snapshot(task_id, sample_task_component)

        # Verify ref metadata
        assert ref.artifact_id == f"{task_id}_snapshot"
        assert ref.category == "tasks/snapshots"
        assert len(ref.content_hash) == 64  # SHA256 hex digest

        # Read snapshot back
        snapshot = persistence_service.read_task_snapshot(task_id)
        assert snapshot is not None
        assert snapshot["task_id"] == task_id
        assert snapshot["description"] == "Test task"
        assert snapshot["status"] == "ready"  # Enum converted to string
        assert snapshot["assigned_agent"] == 42  # EntityId as int

    def test_persist_snapshot_updates_existing(
        self,
        persistence_service: TaskPersistenceService,
        sample_task_component: TaskComponent,
    ) -> None:
        """Test snapshot update overwrites previous state."""
        task_id = "task-001"

        # Persist initial snapshot
        ref1 = persistence_service.persist_task_snapshot(task_id, sample_task_component)

        # Update task component
        sample_task_component.status = TaskStatus.RUNNING

        # Persist updated snapshot
        ref2 = persistence_service.persist_task_snapshot(task_id, sample_task_component)

        # Hash should change
        assert ref1.content_hash != ref2.content_hash

        # Read latest snapshot
        snapshot = persistence_service.read_task_snapshot(task_id)
        assert snapshot is not None
        assert snapshot["status"] == "running"

    def test_read_missing_snapshot_returns_none(
        self,
        persistence_service: TaskPersistenceService,
    ) -> None:
        """Test reading non-existent snapshot returns None."""
        snapshot = persistence_service.read_task_snapshot("nonexistent-task")
        assert snapshot is None

    def test_snapshot_hash_deterministic(
        self,
        sample_task_component: TaskComponent,
    ) -> None:
        """Test snapshot hash is deterministic for same content."""
        from ecs_agent.task.persistence import TaskPersistenceService

        service = TaskPersistenceService(ScratchbookService(Path(tempfile.mkdtemp())))

        snapshot1 = service._serialize_task_component(sample_task_component)
        snapshot2 = service._serialize_task_component(sample_task_component)

        hash1 = compute_task_snapshot_hash(snapshot1)
        hash2 = compute_task_snapshot_hash(snapshot2)

        assert hash1 == hash2


class TestTaskEventLogPersistence:
    """Tests for immutable task event log persistence."""

    def test_append_and_read_task_events_roundtrip(
        self,
        persistence_service: TaskPersistenceService,
    ) -> None:
        """Test events can be appended and read back in order."""
        task_id = "task-002"
        entity_id = EntityId(100)

        # Append multiple events
        event1 = TaskCreatedEvent(
            entity_id=entity_id,
            task_id=task_id,
            description="Test task",
        )
        event2 = TaskStateChangedEvent(
            entity_id=entity_id,
            task_id=task_id,
            old_status=TaskStatus.PENDING,
            new_status=TaskStatus.READY,
        )
        event3 = TaskCompletedEvent(
            entity_id=entity_id,
            task_id=task_id,
            result="Success",
        )

        persistence_service.append_task_event(task_id, event1)
        persistence_service.append_task_event(task_id, event2)
        persistence_service.append_task_event(task_id, event3)

        # Read events back
        events = persistence_service.read_task_events(task_id)

        assert len(events) == 3
        assert events[0]["_event_type"] == "TaskCreatedEvent"
        assert events[0]["description"] == "Test task"
        assert events[1]["_event_type"] == "TaskStateChangedEvent"
        assert events[1]["old_status"] == "pending"
        assert events[1]["new_status"] == "ready"
        assert events[2]["_event_type"] == "TaskCompletedEvent"
        assert events[2]["result"] == "Success"

    def test_read_missing_event_log_returns_empty_list(
        self,
        persistence_service: TaskPersistenceService,
    ) -> None:
        """Test reading non-existent event log returns empty list."""
        events = persistence_service.read_task_events("nonexistent-task")
        assert events == []

    def test_event_log_is_append_only(
        self,
        persistence_service: TaskPersistenceService,
    ) -> None:
        """Test events are appended, not overwritten."""
        task_id = "task-003"
        entity_id = EntityId(200)

        event1 = TaskCreatedEvent(
            entity_id=entity_id,
            task_id=task_id,
            description="First event",
        )
        event2 = TaskStateChangedEvent(
            entity_id=entity_id,
            task_id=task_id,
            old_status=TaskStatus.PENDING,
            new_status=TaskStatus.READY,
        )

        persistence_service.append_task_event(task_id, event1)
        events_after_first = persistence_service.read_task_events(task_id)
        assert len(events_after_first) == 1

        persistence_service.append_task_event(task_id, event2)
        events_after_second = persistence_service.read_task_events(task_id)
        assert len(events_after_second) == 2

        # First event still present
        assert events_after_second[0]["_event_type"] == "TaskCreatedEvent"
        assert events_after_second[1]["_event_type"] == "TaskStateChangedEvent"

    def test_event_serialization_includes_metadata(
        self,
        persistence_service: TaskPersistenceService,
    ) -> None:
        """Test event serialization includes type and timestamp."""
        task_id = "task-004"
        entity_id = EntityId(300)

        event = TaskFailedEvent(
            entity_id=entity_id,
            task_id=task_id,
            error="Test error",
            retry_count=2,
        )

        persistence_service.append_task_event(task_id, event)
        events = persistence_service.read_task_events(task_id)

        assert len(events) == 1
        assert "_event_type" in events[0]
        assert "_timestamp" in events[0]
        assert events[0]["_event_type"] == "TaskFailedEvent"
        assert isinstance(events[0]["_timestamp"], float)


class TestEventLogTamperDetection:
    """Tests for event log tamper detection."""

    def test_corrupted_event_log_raises_tamper_error(
        self,
        persistence_service: TaskPersistenceService,
        tmp_scratchbook: Path,
    ) -> None:
        """Test corrupted event log raises TaskEventLogTamperError."""
        task_id = "task-005"
        entity_id = EntityId(400)

        # Append valid event
        event = TaskCreatedEvent(
            entity_id=entity_id,
            task_id=task_id,
            description="Valid event",
        )
        persistence_service.append_task_event(task_id, event)

        # Manually corrupt event log
        log_path = tmp_scratchbook / "tasks/events" / f"{task_id}_events.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write("CORRUPTED LINE\n")

        # Reading should raise tamper error
        with pytest.raises(TaskEventLogTamperError):
            persistence_service.read_task_events(task_id)

    def test_verify_intact_event_log(
        self,
        persistence_service: TaskPersistenceService,
    ) -> None:
        """Test verify_event_log_integrity returns True for valid log."""
        task_id = "task-006"
        entity_id = EntityId(500)

        # Append valid events
        event1 = TaskCreatedEvent(
            entity_id=entity_id,
            task_id=task_id,
            description="Event 1",
        )
        event2 = TaskStateChangedEvent(
            entity_id=entity_id,
            task_id=task_id,
            old_status=TaskStatus.PENDING,
            new_status=TaskStatus.READY,
        )

        persistence_service.append_task_event(task_id, event1)
        persistence_service.append_task_event(task_id, event2)

        # Verify integrity
        assert persistence_service.verify_event_log_integrity(task_id) is True

    def test_verify_tampered_event_log(
        self,
        persistence_service: TaskPersistenceService,
        tmp_scratchbook: Path,
    ) -> None:
        """Test verify_event_log_integrity returns False for tampered log."""
        task_id = "task-007"
        entity_id = EntityId(600)

        # Append valid event
        event = TaskCreatedEvent(
            entity_id=entity_id,
            task_id=task_id,
            description="Valid event",
        )
        persistence_service.append_task_event(task_id, event)

        # Manually tamper with event log
        log_path = tmp_scratchbook / "tasks/events" / f"{task_id}_events.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write("{broken json}\n")

        # Verify should return False (and raise internally)
        with pytest.raises(TaskEventLogTamperError):
            persistence_service.verify_event_log_integrity(task_id)


class TestStateAndHistoryPersistence:
    """Integration tests for combined snapshot and event log persistence."""

    def test_snapshot_and_event_log_both_queryable(
        self,
        persistence_service: TaskPersistenceService,
        sample_task_component: TaskComponent,
    ) -> None:
        """Test both snapshot and event log can be persisted and queried."""
        task_id = "task-008"
        entity_id = EntityId(700)

        # Persist snapshot
        snapshot_ref = persistence_service.persist_task_snapshot(
            task_id, sample_task_component
        )
        assert snapshot_ref is not None

        # Append events
        event1 = TaskCreatedEvent(
            entity_id=entity_id,
            task_id=task_id,
            description="Created",
        )
        event2 = TaskStateChangedEvent(
            entity_id=entity_id,
            task_id=task_id,
            old_status=TaskStatus.PENDING,
            new_status=TaskStatus.READY,
        )

        persistence_service.append_task_event(task_id, event1)
        persistence_service.append_task_event(task_id, event2)

        # Query snapshot
        snapshot = persistence_service.read_task_snapshot(task_id)
        assert snapshot is not None
        assert snapshot["status"] == "ready"

        # Query event log
        events = persistence_service.read_task_events(task_id)
        assert len(events) == 2
        assert events[0]["_event_type"] == "TaskCreatedEvent"
        assert events[1]["_event_type"] == "TaskStateChangedEvent"

    def test_snapshot_mutable_event_log_immutable(
        self,
        persistence_service: TaskPersistenceService,
        sample_task_component: TaskComponent,
    ) -> None:
        """Test snapshot can be updated but event log remains append-only."""
        task_id = "task-009"
        entity_id = EntityId(800)

        # Persist initial snapshot
        sample_task_component.status = TaskStatus.PENDING
        persistence_service.persist_task_snapshot(task_id, sample_task_component)

        # Append initial event
        event1 = TaskCreatedEvent(
            entity_id=entity_id,
            task_id=task_id,
            description="Created",
        )
        persistence_service.append_task_event(task_id, event1)

        # Update snapshot
        sample_task_component.status = TaskStatus.RUNNING
        persistence_service.persist_task_snapshot(task_id, sample_task_component)

        # Append transition event
        event2 = TaskStateChangedEvent(
            entity_id=entity_id,
            task_id=task_id,
            old_status=TaskStatus.PENDING,
            new_status=TaskStatus.RUNNING,
        )
        persistence_service.append_task_event(task_id, event2)

        # Snapshot reflects latest state
        snapshot = persistence_service.read_task_snapshot(task_id)
        assert snapshot is not None
        assert snapshot["status"] == "running"

        # Event log contains full history
        events = persistence_service.read_task_events(task_id)
        assert len(events) == 2
        assert events[0]["_event_type"] == "TaskCreatedEvent"
        assert events[1]["_event_type"] == "TaskStateChangedEvent"

    def test_event_log_provides_full_transition_trail(
        self,
        persistence_service: TaskPersistenceService,
    ) -> None:
        """Test event log provides complete audit trail of transitions."""
        task_id = "task-010"
        entity_id = EntityId(900)

        # Simulate full task lifecycle
        events_to_append = [
            TaskCreatedEvent(
                entity_id=entity_id,
                task_id=task_id,
                description="Task created",
            ),
            TaskStateChangedEvent(
                entity_id=entity_id,
                task_id=task_id,
                old_status=TaskStatus.PENDING,
                new_status=TaskStatus.READY,
            ),
            TaskStateChangedEvent(
                entity_id=entity_id,
                task_id=task_id,
                old_status=TaskStatus.READY,
                new_status=TaskStatus.RUNNING,
            ),
            TaskStateChangedEvent(
                entity_id=entity_id,
                task_id=task_id,
                old_status=TaskStatus.RUNNING,
                new_status=TaskStatus.FAILED,
            ),
            TaskFailedEvent(
                entity_id=entity_id,
                task_id=task_id,
                error="Simulated failure",
                retry_count=1,
            ),
        ]

        for event in events_to_append:
            persistence_service.append_task_event(task_id, event)

        # Read event log
        events = persistence_service.read_task_events(task_id)

        # Verify full history is present
        assert len(events) == 5
        assert events[0]["_event_type"] == "TaskCreatedEvent"
        assert events[1]["_event_type"] == "TaskStateChangedEvent"
        assert events[1]["new_status"] == "ready"
        assert events[2]["new_status"] == "running"
        assert events[3]["new_status"] == "failed"
        assert events[4]["_event_type"] == "TaskFailedEvent"

        # Events are in chronological order
        timestamps = [e["_timestamp"] for e in events]
        assert timestamps == sorted(timestamps)


class TestTaskLifecycleEventsStructured:
    """Tests for task lifecycle events with structured logging and correlation metadata."""

    def test_events_task_ready_event_emitted(self, capsys):
        """Test TaskReadyEvent emitted when task transitions to READY status."""
        from ecs_agent.logging import configure_logging, get_logger
        from ecs_agent.types import TaskReadyEvent
        import json

        configure_logging(json_output=True, level="INFO")
        logger = get_logger("test_task_lifecycle")
        
        # Simulate TaskReadyEvent emission
        event = TaskReadyEvent(
            entity_id=EntityId(1),
            task_id="task-001",
            dependencies_resolved=["task-dep-1", "task-dep-2"],
            correlation_id="corr-123"
        )
        logger.info(
            "task_ready",
            entity_id=event.entity_id,
            task_id=event.task_id,
            dependencies_resolved=event.dependencies_resolved,
            correlation_id=event.correlation_id
        )
        
        captured = capsys.readouterr()
        events = [json.loads(line) for line in captured.out.strip().split("\n") if line.strip()]
        
        ready_events = [e for e in events if e.get("event") == "task_ready"]
        assert len(ready_events) >= 1
        event_data = ready_events[0]
        assert event_data.get("task_id") == "task-001"
        assert event_data.get("entity_id") == 1
        assert event_data.get("correlation_id") == "corr-123"
        assert "timestamp" in event_data

    def test_events_task_running_event_emitted(self, capsys):
        """Test TaskRunningEvent emitted when task transitions to RUNNING status."""
        from ecs_agent.logging import configure_logging, get_logger
        from ecs_agent.types import TaskRunningEvent
        import json

        configure_logging(json_output=True, level="INFO")
        logger = get_logger("test_task_lifecycle")
        
        event = TaskRunningEvent(
            entity_id=EntityId(1),
            task_id="task-001",
            backend="fetch",
            assigned_agent="agent-wave-1",
            correlation_id="corr-123"
        )
        logger.info(
            "task_running",
            entity_id=event.entity_id,
            task_id=event.task_id,
            backend=event.backend,
            assigned_agent=event.assigned_agent,
            correlation_id=event.correlation_id
        )
        
        captured = capsys.readouterr()
        events = [json.loads(line) for line in captured.out.strip().split("\n") if line.strip()]
        
        running_events = [e for e in events if e.get("event") == "task_running"]
        assert len(running_events) >= 1
        event_data = running_events[0]
        assert event_data.get("task_id") == "task-001"
        assert event_data.get("backend") == "fetch"
        assert event_data.get("assigned_agent") == "agent-wave-1"

    def test_events_task_completed_event_emitted(self, capsys):
        """Test TaskCompletedWithMetadataEvent emitted with duration and results."""
        from ecs_agent.logging import configure_logging, get_logger
        from ecs_agent.types import TaskCompletedWithMetadataEvent
        import json

        configure_logging(json_output=True, level="INFO")
        logger = get_logger("test_task_lifecycle")
        
        event = TaskCompletedWithMetadataEvent(
            entity_id=EntityId(1),
            task_id="task-001",
            result_refs=["artifact-result-1", "artifact-result-2"],
            duration_ms=123.45,
            correlation_id="corr-123"
        )
        logger.info(
            "task_completed",
            entity_id=event.entity_id,
            task_id=event.task_id,
            result_refs=event.result_refs,
            duration_ms=event.duration_ms,
            correlation_id=event.correlation_id
        )
        
        captured = capsys.readouterr()
        events = [json.loads(line) for line in captured.out.strip().split("\n") if line.strip()]
        
        completed_events = [e for e in events if e.get("event") == "task_completed"]
        assert len(completed_events) >= 1
        event_data = completed_events[0]
        assert event_data.get("task_id") == "task-001"
        assert event_data.get("duration_ms") == 123.45
        assert "result_refs" in event_data
        assert len(event_data.get("result_refs", [])) == 2

    def test_failure_event_reason_include_context(self, capsys):
        """Test failure event reason includes backend and dependency context."""
        from ecs_agent.logging import configure_logging, get_logger
        from ecs_agent.types import TaskFailedWithReasonEvent
        import json

        configure_logging(json_output=True, level="ERROR")
        logger = get_logger("test_task_lifecycle")
        
        event = TaskFailedWithReasonEvent(
            entity_id=EntityId(1),
            task_id="task-fail-1",
            error_reason="Dependency task-dep-1 failed: upstream error",
            exception_details="RuntimeError: Provider unavailable",
            correlation_id="corr-fail-1"
        )
        logger.error(
            "task_failed",
            entity_id=event.entity_id,
            task_id=event.task_id,
            reason=event.error_reason,
            exception=event.exception_details,
            correlation_id=event.correlation_id
        )
        
        captured = capsys.readouterr()
        events = [json.loads(line) for line in captured.out.strip().split("\n") if line.strip()]
        
        failed_events = [e for e in events if e.get("event") == "task_failed"]
        assert len(failed_events) >= 1
        event_data = failed_events[0]
        assert event_data.get("task_id") == "task-fail-1"
        assert "reason" in event_data
        assert "Dependency task-dep-1 failed" in event_data.get("reason", "")
        assert "exception" in event_data

    def test_events_correlation_id_consistency(self, capsys):
        """Test correlation_id remains consistent across lifecycle."""
        from ecs_agent.logging import configure_logging, get_logger
        import json

        configure_logging(json_output=True, level="INFO")
        logger = get_logger("test_task_lifecycle")
        
        corr_id = "corr-consistent-123"
        
        logger.info("task_ready", entity_id=1, task_id="t-corr", correlation_id=corr_id)
        logger.info("task_running", entity_id=1, task_id="t-corr", backend="fetch", correlation_id=corr_id)
        logger.info("task_completed", entity_id=1, task_id="t-corr", duration_ms=50, correlation_id=corr_id)
        
        captured = capsys.readouterr()
        events = [json.loads(line) for line in captured.out.strip().split("\n") if line.strip()]
        
        lifecycle_events = [e for e in events if e.get("event") in ["task_ready", "task_running", "task_completed"]]
        assert len(lifecycle_events) >= 3
        
        # All should have same correlation_id
        for event in lifecycle_events:
            assert event.get("correlation_id") == corr_id

    def test_events_task_blocked_event_emitted(self, capsys):
        """Test TaskBlockedUpdatedEvent emitted with blocking reasons."""
        from ecs_agent.logging import configure_logging, get_logger
        from ecs_agent.types import TaskBlockedUpdatedEvent
        import json

        configure_logging(json_output=True, level="INFO")
        logger = get_logger("test_task_lifecycle")
        
        event = TaskBlockedUpdatedEvent(
            entity_id=EntityId(1),
            task_id="task-blocked-1",
            blocking_reasons=["Waiting for task-dep-1 to complete"],
            upstream_failures=["task-dep-upstream-failed"],
            correlation_id="corr-blocked-1"
        )
        logger.info(
            "task_blocked",
            entity_id=event.entity_id,
            task_id=event.task_id,
            blocking_reasons=event.blocking_reasons,
            upstream_failures=event.upstream_failures,
            correlation_id=event.correlation_id
        )
        
        captured = capsys.readouterr()
        events = [json.loads(line) for line in captured.out.strip().split("\n") if line.strip()]
        
        blocked_events = [e for e in events if e.get("event") == "task_blocked"]
        assert len(blocked_events) >= 1
        event_data = blocked_events[0]
        assert event_data.get("task_id") == "task-blocked-1"
        assert "blocking_reasons" in event_data
        assert "upstream_failures" in event_data

    def test_events_task_unblocked_event_emitted(self, capsys):
        """Test TaskUnblockedEvent emitted when task transitions from BLOCKED."""
        from ecs_agent.logging import configure_logging, get_logger
        from ecs_agent.types import TaskUnblockedEvent
        import json

        configure_logging(json_output=True, level="INFO")
        logger = get_logger("test_task_lifecycle")
        
        event = TaskUnblockedEvent(
            entity_id=EntityId(1),
            task_id="task-unblocked-1",
            unblock_reason="dependency_resolved",
            manual_override=False,
            correlation_id="corr-unblocked-1"
        )
        logger.info(
            "task_unblocked",
            entity_id=event.entity_id,
            task_id=event.task_id,
            unblock_reason=event.unblock_reason,
            manual_override=event.manual_override,
            correlation_id=event.correlation_id
        )
        
        captured = capsys.readouterr()
        events = [json.loads(line) for line in captured.out.strip().split("\n") if line.strip()]
        
        unblocked_events = [e for e in events if e.get("event") == "task_unblocked"]
        assert len(unblocked_events) >= 1
        event_data = unblocked_events[0]
        assert event_data.get("task_id") == "task-unblocked-1"
        assert event_data.get("unblock_reason") == "dependency_resolved"
        assert event_data.get("manual_override") is False



# ==============================================================================
# Context Resolution Tests (Task 16)
# ==============================================================================


class TestContextInjectionHappyPath:
    """Tests for successful context resolution from scratchbook refs."""

    def test_context_injection_resolves_tool_results(
        self, tmp_scratchbook: Path
    ) -> None:
        """Task receives tool result data through refs."""
        from ecs_agent.task import ContextResolver, ResolvedContext

        service = ScratchbookService(tmp_scratchbook)
        resolver = ContextResolver(service=service)

        # Write tool result artifact
        tool_result_data = {
            "stable_id": "tool-result-call-001",
            "tool_call_id": "call-001",
            "tool_name": "bash",
            "result": "Command executed successfully",
            "timestamp": "2026-03-07T12:00:00Z",
        }
        service.write_artifact(
            artifact_id="tool-result-call-001",
            category="tool_results",
            data=tool_result_data,
        )

        # Create task with context dependency
        task = TaskComponent(
            task_id="task-ctx-002",
            description="Process bash output",
            expected_output="Analysis complete",
            assigned_agent=None,
            tools=["analyze"],
            context_dependencies=["tool_results/tool-result-call-001"],
            status=TaskStatus.READY,
        )

        # Resolve context
        result = resolver.resolve_context(task)

        # Assert successful resolution
        assert isinstance(result, ResolvedContext)
        assert result.task_id == "task-ctx-002"
        assert len(result.missing_refs) == 0
        assert "tool_results/tool-result-call-001" in result.resolved_data
        assert (
            result.resolved_data["tool_results/tool-result-call-001"]
            == tool_result_data
        )

    def test_context_injection_resolves_plan_snapshots(
        self, tmp_scratchbook: Path
    ) -> None:
        """Task receives plan snapshot data through refs."""
        from ecs_agent.task import ContextResolver, ResolvedContext

        service = ScratchbookService(tmp_scratchbook)
        resolver = ContextResolver(service=service)

        # Write plan snapshot artifact
        plan_snapshot_data = {
            "entity_id": 1,
            "step_index": 2,
            "step_description": "Analyze data",
            "current_step": 3,
            "completed": False,
        }
        service.write_artifact(
            artifact_id="plan-snapshot-1-step-2",
            category="planning",
            data=plan_snapshot_data,
        )

        # Create task with context dependency
        task = TaskComponent(
            task_id="task-ctx-003",
            description="Review analysis",
            expected_output="Review complete",
            assigned_agent=None,
            tools=["review"],
            context_dependencies=["planning/plan-snapshot-1-step-2"],
            status=TaskStatus.READY,
        )

        # Resolve context
        result = resolver.resolve_context(task)

        # Assert successful resolution
        assert isinstance(result, ResolvedContext)
        assert result.task_id == "task-ctx-003"
        assert len(result.missing_refs) == 0
        assert "planning/plan-snapshot-1-step-2" in result.resolved_data
        assert (
            result.resolved_data["planning/plan-snapshot-1-step-2"]
            == plan_snapshot_data
        )

    def test_context_injection_resolves_multiple_refs(
        self, tmp_scratchbook: Path
    ) -> None:
        """Task receives multiple upstream artifacts."""
        from ecs_agent.task import ContextResolver, ResolvedContext

        service = ScratchbookService(tmp_scratchbook)
        resolver = ContextResolver(service=service)

        # Write multiple artifacts
        tool_result = {"tool_call_id": "call-001", "result": "Output 1"}
        plan_snapshot = {"entity_id": 1, "step_index": 1}
        replan_delta = {"entity_id": 1, "replanned_at_step": 2}

        service.write_artifact(
            artifact_id="tool-result-call-001",
            category="tool_results",
            data=tool_result,
        )
        service.write_artifact(
            artifact_id="plan-snapshot-1-step-1",
            category="planning",
            data=plan_snapshot,
        )
        service.write_artifact(
            artifact_id="replan-delta-1-step-2",
            category="replanning",
            data=replan_delta,
        )

        # Create task with multiple dependencies
        task = TaskComponent(
            task_id="task-ctx-004",
            description="Aggregate results",
            expected_output="Summary",
            assigned_agent=None,
            tools=["aggregate"],
            context_dependencies=[
                "tool_results/tool-result-call-001",
                "planning/plan-snapshot-1-step-1",
                "replanning/replan-delta-1-step-2",
            ],
            status=TaskStatus.READY,
        )

        # Resolve context
        result = resolver.resolve_context(task)

        # Assert all refs resolved
        assert isinstance(result, ResolvedContext)
        assert len(result.missing_refs) == 0
        assert len(result.resolved_data) == 3
        assert (
            result.resolved_data["tool_results/tool-result-call-001"]
            == tool_result
        )
        assert (
            result.resolved_data["planning/plan-snapshot-1-step-1"]
            == plan_snapshot
        )
        assert (
            result.resolved_data["replanning/replan-delta-1-step-2"]
            == replan_delta
        )

    def test_context_injected_into_snapshot(self, tmp_scratchbook: Path) -> None:
        """Resolved context is injected into execution snapshot."""
        from ecs_agent.task import ContextResolver, ResolvedContext

        service = ScratchbookService(tmp_scratchbook)
        resolver = ContextResolver(service=service)

        # Create resolved context
        resolved = ResolvedContext(
            task_id="task-ctx-006",
            resolved_data={
                "tool_results/tool-result-call-001": {
                    "result": "Output"
                },
                "planning/plan-snapshot-1-step-1": {"step_index": 1},
            },
            missing_refs=(),
        )

        # Base snapshot
        snapshot = {"var1": "value1", "var2": "value2"}

        # Inject context
        enhanced = resolver.inject_context_into_snapshot(snapshot, resolved)

        # Assert context injected
        assert "context" in enhanced
        assert enhanced["context"] == resolved.resolved_data
        assert enhanced["var1"] == "value1"
        assert enhanced["var2"] == "value2"


class TestContextInjectionFailurePath:
    """Tests for context resolution failures and BLOCKED transitions."""

    def test_missing_ref_returns_error(self, tmp_scratchbook: Path) -> None:
        """Missing upstream ref returns ContextResolutionError."""
        from ecs_agent.task import ContextResolver, ContextResolutionError

        service = ScratchbookService(tmp_scratchbook)
        resolver = ContextResolver(service=service)

        # Create task with missing dependency
        task = TaskComponent(
            task_id="task-ctx-007",
            description="Use missing data",
            expected_output="Done",
            assigned_agent=None,
            tools=["process"],
            context_dependencies=["tool_results/missing-artifact"],
            status=TaskStatus.READY,
        )

        # Resolve context
        result = resolver.resolve_context(task)

        # Assert error returned
        assert isinstance(result, ContextResolutionError)
        assert result.task_id == "task-ctx-007"
        assert len(result.missing_refs) == 1
        assert "tool_results/missing-artifact" in result.missing_refs
        assert "missing dependencies" in result.reason

    def test_multiple_missing_refs_returns_error(
        self, tmp_scratchbook: Path
    ) -> None:
        """Multiple missing refs all included in error."""
        from ecs_agent.task import ContextResolver, ContextResolutionError

        service = ScratchbookService(tmp_scratchbook)
        resolver = ContextResolver(service=service)

        # Create task with multiple missing dependencies
        task = TaskComponent(
            task_id="task-ctx-008",
            description="Use missing data",
            expected_output="Done",
            assigned_agent=None,
            tools=["process"],
            context_dependencies=[
                "tool_results/missing-1",
                "planning/missing-2",
                "replanning/missing-3",
            ],
            status=TaskStatus.READY,
        )

        # Resolve context
        result = resolver.resolve_context(task)

        # Assert all missing refs in error
        assert isinstance(result, ContextResolutionError)
        assert len(result.missing_refs) == 3
        assert "tool_results/missing-1" in result.missing_refs
        assert "planning/missing-2" in result.missing_refs
        assert "replanning/missing-3" in result.missing_refs

    def test_partial_missing_refs_returns_error(self, tmp_scratchbook: Path) -> None:
        """Some refs resolved, some missing = error."""
        from ecs_agent.task import ContextResolver, ContextResolutionError

        service = ScratchbookService(tmp_scratchbook)
        resolver = ContextResolver(service=service)

        # Write one artifact
        service.write_artifact(
            artifact_id="tool-result-call-001",
            category="tool_results",
            data={"result": "Output"},
        )

        # Create task with one valid, one missing dependency
        task = TaskComponent(
            task_id="task-ctx-009",
            description="Use partial data",
            expected_output="Done",
            assigned_agent=None,
            tools=["process"],
            context_dependencies=[
                "tool_results/tool-result-call-001",  # Exists
                "planning/missing-artifact",  # Missing
            ],
            status=TaskStatus.READY,
        )

        # Resolve context
        result = resolver.resolve_context(task)

        # Assert error returned (even though one ref resolved)
        assert isinstance(result, ContextResolutionError)
        assert len(result.missing_refs) == 1
        assert "planning/missing-artifact" in result.missing_refs


class TestReplanGrandfathering:
    """Tests for replan grandfathering policy."""

    def test_running_task_grandfathered(self, tmp_scratchbook: Path) -> None:
        """Running tasks keep existing context (grandfathered)."""
        from ecs_agent.task import ContextResolver, ResolvedContext

        service = ScratchbookService(tmp_scratchbook)
        resolver = ContextResolver(service=service)

        # Create running task with dependencies
        task = TaskComponent(
            task_id="task-ctx-010",
            description="Running task",
            expected_output="Done",
            assigned_agent=None,
            tools=["process"],
            context_dependencies=["tool_results/some-artifact"],
            status=TaskStatus.RUNNING,
        )

        # Resolve context with running_task_ids
        result = resolver.resolve_context(
            task, running_task_ids={"task-ctx-010"}
        )

        # Assert grandfathered (no resolution attempted)
        assert isinstance(result, ResolvedContext)
        assert result.task_id == "task-ctx-010"
        assert len(result.resolved_data) == 0  # Grandfathered, no fetch
        assert len(result.missing_refs) == 0

    def test_pending_task_applies_replan_updates(
        self, tmp_scratchbook: Path
    ) -> None:
        """Pending tasks apply replan updates (not grandfathered)."""
        from ecs_agent.task import ContextResolver, ResolvedContext

        service = ScratchbookService(tmp_scratchbook)
        resolver = ContextResolver(service=service)

        # Write replan delta artifact
        replan_delta = {
            "entity_id": 1,
            "replanned_at_step": 2,
            "old_steps": ["Step 1", "Step 2"],
            "new_steps": ["Step 1", "Step 2 revised"],
        }
        service.write_artifact(
            artifact_id="replan-delta-1-step-2",
            category="replanning",
            data=replan_delta,
        )

        # Create pending task with replan dependency
        task = TaskComponent(
            task_id="task-ctx-011",
            description="Pending task",
            expected_output="Done",
            assigned_agent=None,
            tools=["process"],
            context_dependencies=["replanning/replan-delta-1-step-2"],
            status=TaskStatus.PENDING,
        )

        # Resolve context (no running_task_ids, so not grandfathered)
        result = resolver.resolve_context(task)

        # Assert replan delta resolved
        assert isinstance(result, ResolvedContext)
        assert len(result.resolved_data) == 1
        assert (
            result.resolved_data["replanning/replan-delta-1-step-2"]
            == replan_delta
        )


# ==============================================================================
# End-to-End Integration Tests (Task 17)
# ==============================================================================


class TestEndToEndOrchestration:
    """Tests for complete fetch->dispatch->persist orchestration loop."""

    async def test_complete_workflow_happy_path(
        self, tmp_scratchbook: Path
    ) -> None:
        """Task flows through entire lifecycle: ready -> running -> completed."""
        from ecs_agent.task import (
            TaskExecutor,
            TaskFetchingUnit,
            WavePlanner,
            analyze_task_dependencies,
        )
        from ecs_agent.core import World
        from ecs_agent.components import (
            LLMComponent,
            ConversationComponent,
        )

        service = ScratchbookService(tmp_scratchbook)
        persistence = TaskPersistenceService(service)
        executor = TaskExecutor()
        fetching_unit = TaskFetchingUnit()
        wave_planner = WavePlanner()

        # Create world and entity with required components
        world = World()
        entity_id = world.create_entity()

        # Add LLMComponent and ConversationComponent for local execution
        from ecs_agent.providers import FakeProvider
        from ecs_agent.types import CompletionResult, Message

        provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant", content="Task completed successfully"
                    )
                )
            ]
        )
        world.add_component(
            entity_id,
            LLMComponent(provider=provider, model="fake", system_prompt=""),
        )
        world.add_component(entity_id, ConversationComponent(messages=[]))

        # Add ToolRegistryComponent for local execution
        from ecs_agent.components import ToolRegistryComponent

        world.add_component(
            entity_id, ToolRegistryComponent(tools={}, handlers={})
        )

        # Create simple task
        task = TaskComponent(
            task_id="task-e2e-001",
            description="Execute test task",
            expected_output="Success",
            assigned_agent=entity_id,
            tools=[],
            context_dependencies=[],
            status=TaskStatus.READY,
            priority=5,
        )

        # Step 1: Analyze dependencies
        analysis = analyze_task_dependencies([task])
        assert len(analysis.ready_task_ids) == 1

        # Step 2: Compute waves
        wave_plan = wave_planner.compute_waves(analysis)
        assert len(wave_plan.waves) == 1

        # Step 3: Generate dispatch requests
        snapshot = {"timestamp": "2026-03-07T12:00:00Z"}
        requests = fetching_unit.generate_dispatch_requests(
            wave_plan=wave_plan,
            tasks=[task],
            snapshot=snapshot,
            writer_id="task_fetching_unit",
        )
        assert len(requests) == 1
        request = requests[0]

        # Step 4: Execute request
        result = await executor.execute_dispatch_request(world, entity_id, request)
        assert result.success is True
        assert result.backend_type == "local"

        # Step 5: Persist snapshot
        task.status = TaskStatus.COMPLETED
        ref = persistence.persist_task_snapshot(task.task_id, task)
        assert ref is not None

        # Step 6: Append lifecycle event
        from ecs_agent.types import TaskCompletedEvent

        event = TaskCompletedEvent(
            entity_id=entity_id,
            task_id=task.task_id,
            result=result.result_content,
        )
        persistence.append_task_event(task.task_id, event)

        # Step 7: Verify persistence
        snapshot_data = persistence.read_task_snapshot(task.task_id)
        assert snapshot_data is not None
        assert snapshot_data["status"] == "completed"

        events = persistence.read_task_events(task.task_id)
        assert len(events) == 1
        assert events[0]["_event_type"] == "TaskCompletedEvent"

    async def test_mixed_backend_execution(
        self, tmp_scratchbook: Path
    ) -> None:
        """Tasks execute via both local tools and subagent delegation."""
        from ecs_agent.task import TaskExecutor, ExecutionResult
        from ecs_agent.core import World
        from ecs_agent.components import (
            SubagentRegistryComponent,
            LLMComponent,
            ConversationComponent,
        )

        world = World()
        entity_id = world.create_entity()

        # Add LLMComponent for local execution
        from ecs_agent.providers import FakeProvider
        from ecs_agent.types import CompletionResult, Message

        provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(role="assistant", content="Local result")
                )
            ]
        )
        world.add_component(
            entity_id,
            LLMComponent(provider=provider, model="fake", system_prompt=""),
        )
        world.add_component(entity_id, ConversationComponent(messages=[]))

        # Test 1: Local execution backend
        from ecs_agent.task import DispatchRequest

        local_request = DispatchRequest(
            task_id="task-local-001",
            wave_number=0,
            sequence_number=0,
            description="Local task",
            expected_output="Local output",
            assigned_agent=entity_id,
            tools=(),
            context_dependencies=(),
            priority=5,
        )

        executor = TaskExecutor()
        local_result = await executor.execute_dispatch_request(
            world, entity_id, local_request
        )
        assert isinstance(local_result, ExecutionResult)
        assert local_result.backend_type == "local"

        # Test 2: Subagent backend (with missing registry - should fail gracefully)
        subagent_request = DispatchRequest(
            task_id="task-subagent-001",
            wave_number=0,
            sequence_number=1,
            description="Subagent task",
            expected_output="Subagent output",
            assigned_agent="researcher",
            tools=(),
            context_dependencies=(),
            priority=5,
        )

        subagent_result = await executor.execute_dispatch_request(
            world, entity_id, subagent_request
        )
        assert isinstance(subagent_result, ExecutionResult)
        assert subagent_result.backend_type == "subagent"
        assert subagent_result.success is False  # Missing registry
        assert "SubagentRegistryComponent" in subagent_result.result_content


class TestManualUnblockFlow:
    """Tests for manual unblock workflow with blocked-until-manual policy."""

    async def test_blocked_task_requires_manual_unblock(
        self, tmp_scratchbook: Path
    ) -> None:
        """Task blocked by dependency failure requires manual action to unblock."""
        from ecs_agent.task import (
            TaskState,
            TransitionRequest,
            block_task_due_to_upstream_failure,
            manual_unblock_task,
            transition_task_state,
        )

        # Create task state
        state = TaskState(
            task_id="task-blocked-001",
            status=TaskStatus.PENDING,
            retry_count=0,
            max_retries=3,
        )

        # Simulate upstream failure blocking this task
        blocked_state = block_task_due_to_upstream_failure(
            state, dependency_task_id="task-dep-001"
        )
        assert blocked_state.status == TaskStatus.BLOCKED
        assert blocked_state.blocked_until_manual is True
        assert "task-dep-001" in blocked_state.blocked_reason

        # Attempt automatic transition to READY should fail
        from ecs_agent.task import TaskStateTransitionError

        with pytest.raises(TaskStateTransitionError) as exc_info:
            transition_task_state(
                blocked_state,
                TransitionRequest(target_status=TaskStatus.READY, manual_action=False),
            )
        assert "manual action required" in str(exc_info.value)

        # Manual unblock should succeed
        unblocked_state = manual_unblock_task(
            blocked_state, reason="User reviewed and approved"
        )
        assert unblocked_state.status == TaskStatus.READY
        assert unblocked_state.blocked_until_manual is False
        assert unblocked_state.blocked_reason is None

    async def test_manual_unblock_persistence_roundtrip(
        self, tmp_scratchbook: Path
    ) -> None:
        """Manual unblock event is persisted and queryable."""
        from ecs_agent.types import TaskUnblockedEvent

        service = ScratchbookService(tmp_scratchbook)
        persistence = TaskPersistenceService(service)

        task_id = "task-unblock-001"
        entity_id = EntityId(500)

        # Append blocked event
        blocked_event = TaskBlockedEvent(
            entity_id=entity_id,
            task_id=task_id,
            reason="Upstream dependency failed",
        )
        persistence.append_task_event(task_id, blocked_event)

        # Append unblock event
        unblock_event = TaskUnblockedEvent(
            entity_id=entity_id,
            task_id=task_id,
            unblock_reason="Manual approval",
            manual_override=True,
        )
        persistence.append_task_event(task_id, unblock_event)

        # Verify event log
        events = persistence.read_task_events(task_id)
        assert len(events) == 2
        assert events[0]["_event_type"] == "TaskBlockedEvent"
        assert events[1]["_event_type"] == "TaskUnblockedEvent"
        assert events[1]["manual_override"] is True


class TestDependencySequencing:
    """Tests for deterministic dependency-driven task ordering."""

    async def test_deterministic_topological_ordering(
        self, tmp_scratchbook: Path
    ) -> None:
        """Tasks execute in deterministic dependency order."""
        from ecs_agent.task import (
            analyze_task_dependencies,
            WavePlanner,
            TaskFetchingUnit,
        )

        # Create task chain: task-1 -> task-2 -> task-3
        task1 = TaskComponent(
            task_id="task-1",
            description="First task",
            expected_output="Output 1",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            status=TaskStatus.READY,
            priority=5,
        )

        task2 = TaskComponent(
            task_id="task-2",
            description="Second task",
            expected_output="Output 2",
            assigned_agent=None,
            tools=[],
            context_dependencies=["task-1"],
            status=TaskStatus.PENDING,
            priority=5,
        )

        task3 = TaskComponent(
            task_id="task-3",
            description="Third task",
            expected_output="Output 3",
            assigned_agent=None,
            tools=[],
            context_dependencies=["task-2"],
            status=TaskStatus.PENDING,
            priority=5,
        )

        # Analyze dependencies
        analysis = analyze_task_dependencies([task1, task2, task3])

        # Only task-1 is ready (no dependencies)
        assert len(analysis.ready_task_ids) == 1
        assert analysis.ready_task_ids[0] == "task-1"

        # task-2 and task-3 are blocked
        assert len(analysis.blocked_task_ids) == 2
        assert "task-2" in analysis.blocked_task_ids
        assert "task-3" in analysis.blocked_task_ids

        # Verify topological order
        assert analysis.topological_order == ("task-1", "task-2", "task-3")

        # Complete task-1, then reanalyze
        task1.status = TaskStatus.COMPLETED
        analysis2 = analyze_task_dependencies([task1, task2, task3])

        # Now task-2 should be ready (task-1 completed, so both show as ready)
        assert len(analysis2.ready_task_ids) == 2  # task-1 and task-2
        assert "task-2" in analysis2.ready_task_ids

        # Complete task-2, then reanalyze
        task2.status = TaskStatus.COMPLETED
        analysis3 = analyze_task_dependencies([task1, task2, task3])

        # Now task-3 should be ready (all previous tasks completed)
        assert len(analysis3.ready_task_ids) == 3  # task-1, task-2, and task-3
        assert "task-3" in analysis3.ready_task_ids

    async def test_priority_based_dispatch_ordering(
        self, tmp_scratchbook: Path
    ) -> None:
        """Ready tasks dispatch in deterministic (priority DESC, task_id ASC) order."""
        from ecs_agent.task import (
            analyze_task_dependencies,
            WavePlanner,
            TaskFetchingUnit,
        )

        # Create multiple ready tasks with different priorities
        task_low = TaskComponent(
            task_id="task-low",
            description="Low priority",
            expected_output="Output",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            status=TaskStatus.READY,
            priority=1,
        )

        task_high = TaskComponent(
            task_id="task-high",
            description="High priority",
            expected_output="Output",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            status=TaskStatus.READY,
            priority=10,
        )

        task_medium = TaskComponent(
            task_id="task-medium",
            description="Medium priority",
            expected_output="Output",
            assigned_agent=None,
            tools=[],
            context_dependencies=[],
            status=TaskStatus.READY,
            priority=5,
        )

        # Analyze and plan
        analysis = analyze_task_dependencies([task_low, task_high, task_medium])
        wave_planner = WavePlanner()
        wave_plan = wave_planner.compute_waves(analysis)

        # All tasks are ready
        assert len(analysis.ready_task_ids) == 3

        # Generate dispatch requests
        fetching_unit = TaskFetchingUnit()
        snapshot = {"timestamp": "2026-03-07T12:00:00Z"}
        requests = fetching_unit.generate_dispatch_requests(
            wave_plan=wave_plan,
            tasks=[task_low, task_high, task_medium],
            snapshot=snapshot,
            writer_id="task_fetching_unit",
        )

        # Verify dispatch order: high -> medium -> low (by priority DESC)
        assert len(requests) == 3
        assert requests[0].task_id == "task-high"  # Priority 10
        assert requests[1].task_id == "task-medium"  # Priority 5
        assert requests[2].task_id == "task-low"  # Priority 1

        # Verify sequence numbers are sequential
        assert requests[0].sequence_number == 0
        assert requests[1].sequence_number == 1
        assert requests[2].sequence_number == 2
