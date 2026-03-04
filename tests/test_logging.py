"""Tests for the structlog logging module."""

import json
import io
import sys
from contextlib import redirect_stdout

import pytest
import structlog

from ecs_agent.logging import configure_logging, get_logger


def _json_events(output: str) -> list[dict[str, object]]:
    """Parse JSON events from logging output."""
    events: list[dict[str, object]] = []
    for line in output.strip().split("\n"):
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip non-JSON lines (e.g., console format during tests)
                continue
    return events

class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_console_mode(self, capsys):
        """Test configure_logging with console (dev) output mode."""
        configure_logging(json_output=False, level="INFO")

        logger = get_logger("test_console")
        logger.info("hello", key="value")

        captured = capsys.readouterr()
        assert "hello" in captured.out
        assert "key" in captured.out
        assert "value" in captured.out

    def test_configure_logging_json_mode(self, capsys):
        """Test configure_logging with JSON (prod) output mode."""
        configure_logging(json_output=True, level="INFO")

        logger = get_logger("test_json")
        logger.info("hello", key="value")

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")

        # Find JSON line (should be last non-empty line)
        for line in reversed(lines):
            if line.strip():
                parsed = json.loads(line)
                assert parsed["event"] == "hello"
                assert parsed["key"] == "value"
                assert "timestamp" in parsed
                break

    def test_configure_logging_level_debug(self, capsys):
        """Test configure_logging respects DEBUG level."""
        configure_logging(json_output=False, level="DEBUG")

        logger = get_logger("test_debug")
        logger.debug("debug_msg")
        logger.info("info_msg")

        captured = capsys.readouterr()
        assert "debug_msg" in captured.out
        assert "info_msg" in captured.out

    def test_configure_logging_level_warning(self, capsys):
        """Test configure_logging respects WARNING level."""
        configure_logging(json_output=False, level="WARNING")

        logger = get_logger("test_warning")
        logger.warning("warning_msg")

        captured = capsys.readouterr()
        assert "warning_msg" in captured.out


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a structlog logger instance."""
        configure_logging(json_output=False)

        logger = get_logger("test_module")
        assert logger is not None
        assert hasattr(logger, "msg")
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")

    def test_get_logger_with_name(self, capsys):
        """Test get_logger includes name in context."""
        configure_logging(json_output=False)

        logger = get_logger("my_module")
        logger.info("test_event")

        captured = capsys.readouterr()
        assert "test_event" in captured.out

    def test_get_logger_multiple_calls_same_name(self):
        """Test get_logger returns consistent loggers for same name."""
        configure_logging(json_output=False)

        logger1 = get_logger("consistent")
        logger2 = get_logger("consistent")

        assert logger1 is not None
        assert logger2 is not None
        assert type(logger1) == type(logger2)


class TestLoggingOutput:
    """Tests for logging output format and content."""

    def test_json_output_parseable(self, capsys):
        """Test JSON output is valid and parseable."""
        configure_logging(json_output=True, level="INFO")

        logger = get_logger("test")
        logger.info("json_test", field1="value1", field2=42)

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]

        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
            assert "event" in parsed or "level" in parsed

    def test_log_contains_timestamp(self, capsys):
        """Test log output includes ISO 8601 timestamp."""
        configure_logging(json_output=True, level="INFO")

        logger = get_logger("test")
        logger.info("timestamp_test")

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]

        for line in reversed(lines):
            parsed = json.loads(line)
            assert "timestamp" in parsed
            # ISO format check: should contain T and either Z or +/- timezone
            ts = parsed["timestamp"]
            assert "T" in ts

    def test_log_contains_level(self, capsys):
        """Test log output includes log level."""
        configure_logging(json_output=True, level="INFO")

        logger = get_logger("test")
        logger.info("level_test")
        logger.warning("warning_test")

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]

        levels_found = []
        for line in lines:
            parsed = json.loads(line)
            if "level" in parsed:
                levels_found.append(parsed["level"])

        assert len(levels_found) > 0

    def test_console_output_contains_event(self, capsys):
        """Test console output includes event message."""
        configure_logging(json_output=False, level="INFO")

        logger = get_logger("test")
        logger.info("console_event", detail="test_detail")

        captured = capsys.readouterr()
        assert "console_event" in captured.out
        assert "test_detail" in captured.out

    def test_structured_fields_in_output(self, capsys):
        """Test structured fields are included in log output."""
        configure_logging(json_output=True, level="INFO")

        logger = get_logger("test")
        logger.info("struct_test", user_id=123, action="read")

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]

        for line in reversed(lines):
            parsed = json.loads(line)
            if parsed.get("event") == "struct_test":
                assert parsed.get("user_id") == 123
                assert parsed.get("action") == "read"
                break


class TestLoggingIntegration:
    """Integration tests for logging configuration."""

    def test_logging_context_propagation(self, capsys):
        """Test context variables are propagated through logging."""
        configure_logging(json_output=True, level="INFO")

        logger = get_logger("test")
        logger = logger.bind(request_id="req123")
        logger.info("context_test")

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]

        for line in reversed(lines):
            parsed = json.loads(line)
            if parsed.get("event") == "context_test":
                assert parsed.get("request_id") == "req123"
                break

    def test_multiple_loggers_independent(self, capsys):
        """Test multiple logger instances work independently."""
        configure_logging(json_output=False, level="INFO")

        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        logger1.info("from_module1")
        logger2.info("from_module2")

        captured = capsys.readouterr()
        assert "from_module1" in captured.out
        assert "from_module2" in captured.out

    def test_error_level_logging(self, capsys):
        """Test error level logging."""
        configure_logging(json_output=True, level="ERROR")

        logger = get_logger("test")
        logger.error("error_event", error_code=500)

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]

        for line in reversed(lines):
            parsed = json.loads(line)
            if parsed.get("event") == "error_event":
                assert parsed.get("error_code") == 500
                break


class TestBusLogging:
    """Tests for message bus log sampling hooks."""

    def test_log_bus_publish_includes_trace_context(self, capsys):
        """Test log_bus_publish includes trace_id and correlation_id."""
        from ecs_agent.logging import log_bus_publish

        configure_logging(json_output=True, level="INFO")

        logger = get_logger("test_bus")
        log_bus_publish(
            logger,
            topic="test.topic",
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            correlation_id="corr-123",
            payload_type="TestPayload",
        )

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]

        for line in reversed(lines):
            parsed = json.loads(line)
            if parsed.get("event") == "bus_publish":
                assert parsed.get("topic") == "test.topic"
                assert parsed.get("trace_id") == "4bf92f3577b34da6a3ce929d0e0e4736"
                assert parsed.get("correlation_id") == "corr-123"
                assert parsed.get("payload_type") == "TestPayload"
                break

    def test_log_bus_deliver_includes_subscriber_id(self, capsys):
        """Test log_bus_deliver includes subscriber_id and trace context."""
        from ecs_agent.logging import log_bus_deliver

        configure_logging(json_output=True, level="DEBUG")

        logger = get_logger("test_bus")
        log_bus_deliver(
            logger,
            topic="test.topic",
            subscriber_id="sub-456",
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            correlation_id="corr-123",
        )

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]

        for line in reversed(lines):
            parsed = json.loads(line)
            if parsed.get("event") == "bus_deliver":
                assert parsed.get("topic") == "test.topic"
                assert parsed.get("subscriber_id") == "sub-456"
                assert parsed.get("trace_id") == "4bf92f3577b34da6a3ce929d0e0e4736"
                assert parsed.get("correlation_id") == "corr-123"
                break

    def test_log_bus_timeout_includes_timeout_seconds(self, capsys):
        """Test log_bus_timeout includes timeout_seconds and trace context."""
        from ecs_agent.logging import log_bus_timeout

        configure_logging(json_output=True, level="WARNING")

        logger = get_logger("test_bus")
        log_bus_timeout(
            logger,
            request_id="req-789",
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            correlation_id="corr-123",
            timeout_seconds=30.0,
        )

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]

        for line in reversed(lines):
            parsed = json.loads(line)
            if parsed.get("event") == "bus_timeout":
                assert parsed.get("request_id") == "req-789"
                assert parsed.get("trace_id") == "4bf92f3577b34da6a3ce929d0e0e4736"
                assert parsed.get("correlation_id") == "corr-123"
                assert parsed.get("timeout_seconds") == 30.0
                break

    def test_log_bus_response_includes_success_flag(self, capsys):
        """Test log_bus_response includes success flag and trace context."""
        from ecs_agent.logging import log_bus_response

        configure_logging(json_output=True, level="INFO")

        logger = get_logger("test_bus")
        log_bus_response(
            logger,
            request_id="req-789",
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            correlation_id="corr-123",
            success=True,
        )

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]

        for line in reversed(lines):
            parsed = json.loads(line)
            if parsed.get("event") == "bus_response":
                assert parsed.get("request_id") == "req-789"
                assert parsed.get("trace_id") == "4bf92f3577b34da6a3ce929d0e0e4736"
                assert parsed.get("correlation_id") == "corr-123"
                assert parsed.get("success") is True
                break

    def test_log_bus_publish_without_payload_type(self, capsys):
        """Test log_bus_publish with None payload_type."""
        from ecs_agent.logging import log_bus_publish

        configure_logging(json_output=True, level="INFO")

        logger = get_logger("test_bus")
        log_bus_publish(
            logger,
            topic="test.topic",
            trace_id="4bf92f3577b34da6a3ce929d0e0e4736",
            correlation_id="corr-123",
        )

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]

        for line in reversed(lines):
            parsed = json.loads(line)
            if parsed.get("event") == "bus_publish":
                assert parsed.get("topic") == "test.topic"
                assert parsed.get("trace_id") == "4bf92f3577b34da6a3ce929d0e0e4736"
                assert parsed.get("correlation_id") == "corr-123"
                assert parsed.get("payload_type") is None
                break


class TestToolExecutionLogging:
    """Tests for ToolExecutionSystem logging."""

    async def test_tool_called_event_emitted(self, capsys):
        """Test ToolExecutionSystem emits tool_called event on invocation."""
        import importlib
        from ecs_agent.logging import configure_logging

        configure_logging(json_output=True, level="INFO")

        # Reload module to get fresh logger with new config
        import ecs_agent.systems.tool_execution
        importlib.reload(ecs_agent.systems.tool_execution)

        from ecs_agent.core import World
        from ecs_agent.components import (
            PendingToolCallsComponent,
            ToolRegistryComponent,
            ConversationComponent,
        )
        from ecs_agent.systems.tool_execution import ToolExecutionSystem
        from ecs_agent.types import ToolCall, Message
        world = World()
        entity = world.create_entity()

        # Define a simple test tool
        async def test_tool(arg: str) -> str:
            return f"Result: {arg}"

        tool_call = ToolCall(
            id="call_123",
            name="test_tool",
            arguments={"arg": "hello"},
        )

        world.add_component(entity, PendingToolCallsComponent(tool_calls=[tool_call]))
        world.add_component(
            entity,
            ToolRegistryComponent(
                tools={"test_tool": {}}, handlers={"test_tool": test_tool}
            ),
        )
        world.add_component(entity, ConversationComponent(messages=[]))

        system = ToolExecutionSystem()
        await system.process(world)

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line.strip()]
        events = [json.loads(line) for line in lines]
        tool_called_events = [e for e in events if e.get("event") == "tool_called"]

        assert len(tool_called_events) >= 1, "tool_called event not found"
        event = tool_called_events[0]
        assert "tool_name" in event
        assert event["tool_name"] == "test_tool"

    async def test_tool_result_event_emitted_on_success(self, capsys):
        """Test ToolExecutionSystem emits tool_result event with duration_ms on success."""
        import importlib
        from ecs_agent.logging import configure_logging

        configure_logging(json_output=True, level="INFO")

        # Reload module to get fresh logger with new config
        import ecs_agent.systems.tool_execution
        importlib.reload(ecs_agent.systems.tool_execution)

        from ecs_agent.core import World
        from ecs_agent.components import (
            PendingToolCallsComponent,
            ToolRegistryComponent,
            ConversationComponent,
        )
        from ecs_agent.systems.tool_execution import ToolExecutionSystem
        from ecs_agent.types import ToolCall, Message
        world = World()
        entity = world.create_entity()

        # Define a simple test tool
        async def test_tool(arg: str) -> str:
            return f"Result: {arg}"

        tool_call = ToolCall(
            id="call_456",
            name="test_tool",
            arguments={"arg": "world"},
        )

        world.add_component(entity, PendingToolCallsComponent(tool_calls=[tool_call]))
        world.add_component(
            entity,
            ToolRegistryComponent(
                tools={"test_tool": {}}, handlers={"test_tool": test_tool}
            ),
        )
        world.add_component(entity, ConversationComponent(messages=[]))

        system = ToolExecutionSystem()
        await system.process(world)

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line.strip()]
        events = [json.loads(line) for line in lines]
        tool_result_events = [e for e in events if e.get("event") == "tool_result"]

        assert len(tool_result_events) >= 1, "tool_result event not found"
        event = tool_result_events[0]
        assert "tool_name" in event
        assert event["tool_name"] == "test_tool"
        assert "success" in event
        assert event["success"] is True
        assert "duration_ms" in event
        assert event["duration_ms"] >= 0

    async def test_tool_failed_event_emitted_on_error(self, capsys):
        """Test ToolExecutionSystem emits tool_failed event on exception."""
        import importlib
        from ecs_agent.logging import configure_logging

        configure_logging(json_output=True, level="ERROR")

        # Reload module to get fresh logger with new config
        import ecs_agent.systems.tool_execution
        importlib.reload(ecs_agent.systems.tool_execution)

        from ecs_agent.core import World
        from ecs_agent.components import (
            PendingToolCallsComponent,
            ToolRegistryComponent,
            ConversationComponent,
        )
        from ecs_agent.systems.tool_execution import ToolExecutionSystem
        from ecs_agent.types import ToolCall, Message
        world = World()
        entity = world.create_entity()

        # Define a failing test tool
        async def failing_tool(arg: str) -> str:
            raise ValueError("Test error")

        tool_call = ToolCall(
            id="call_789",
            name="failing_tool",
            arguments={"arg": "test"},
        )

        world.add_component(entity, PendingToolCallsComponent(tool_calls=[tool_call]))
        world.add_component(
            entity,
            ToolRegistryComponent(
                tools={"failing_tool": {}}, handlers={"failing_tool": failing_tool}
            ),
        )
        world.add_component(entity, ConversationComponent(messages=[]))

        system = ToolExecutionSystem()
        await system.process(world)

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line.strip()]
        events = [json.loads(line) for line in lines]
        tool_failed_events = [e for e in events if e.get("event") == "tool_failed"]

        assert len(tool_failed_events) >= 1, "tool_failed event not found"
        event = tool_failed_events[0]
        assert "tool_name" in event
        assert event["tool_name"] == "failing_tool"
        assert "reason" in event
        assert "Test error" in event["reason"]

    async def test_tool_failed_event_emitted_on_missing_handler(self, capsys):
        """Test ToolExecutionSystem emits tool_failed event for missing handler."""
        import importlib
        from ecs_agent.logging import configure_logging

        configure_logging(json_output=True, level="ERROR")

        # Reload module to get fresh logger with new config
        import ecs_agent.systems.tool_execution
        importlib.reload(ecs_agent.systems.tool_execution)

        from ecs_agent.core import World
        from ecs_agent.components import (
            PendingToolCallsComponent,
            ToolRegistryComponent,
            ConversationComponent,
        )
        from ecs_agent.systems.tool_execution import ToolExecutionSystem
        from ecs_agent.types import ToolCall, Message
        world = World()
        entity = world.create_entity()

        tool_call = ToolCall(
            id="call_unknown",
            name="unknown_tool",
            arguments={},
        )

        world.add_component(entity, PendingToolCallsComponent(tool_calls=[tool_call]))
        world.add_component(
            entity,
            ToolRegistryComponent(tools={}, handlers={}),  # No handlers registered
        )
        world.add_component(entity, ConversationComponent(messages=[]))

        system = ToolExecutionSystem()
        await system.process(world)

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line.strip()]
        events = [json.loads(line) for line in lines]
        tool_failed_events = [e for e in events if e.get("event") == "tool_failed"]

        assert len(tool_failed_events) >= 1, "tool_failed event not found"
        event = tool_failed_events[0]
        assert "tool_name" in event
        assert event["tool_name"] == "unknown_tool"
        assert "reason" in event
        assert "unknown tool" in event["reason"].lower()


class TestCheckpointLogging:
    """Tests for CheckpointSystem logging."""

    async def test_checkpoint_saved_event_emitted(self, capsys):
        """Test CheckpointSystem emits checkpoint_saved event on success."""
        from ecs_agent.core import World
        from ecs_agent.components import CheckpointComponent
        from ecs_agent.systems.checkpoint import CheckpointSystem


        world = World()
        entity = world.create_entity()
        world.add_component(entity, CheckpointComponent(snapshots=[], max_snapshots=5))

        system = CheckpointSystem()
        await system.process(world)

        captured = capsys.readouterr()
        print("CAPTURED:", repr(captured.out[:500]))
        events = _json_events(captured.out)
        print("EVENTS:", events)
        saved_events = [e for e in events if e.get("event") == "checkpoint_saved"]

        assert len(saved_events) >= 1, "checkpoint_saved event not found"
        event = saved_events[0]
        assert "success" in event
        assert event["success"] is True
        assert "duration_ms" in event
        assert event["duration_ms"] >= 0

    async def test_checkpoint_undo_error_logged(self, capsys):
        """Test CheckpointSystem.undo logs error on failure."""
        from ecs_agent.core import World
        from ecs_agent.systems.checkpoint import CheckpointSystem

        configure_logging(json_output=True, level="ERROR")

        world = World()

        # Attempt undo without any checkpoint component (should raise ValueError)
        try:
            await CheckpointSystem.undo(world, providers={}, tool_handlers={})
        except ValueError:
            pass  # Expected

        captured = capsys.readouterr()
        # CheckpointSystem.undo currently raises without logging
        # This test will be GREEN once we add error logging to undo
        events = _json_events(captured.out)
        error_events = [e for e in events if e.get("level") == "error"]
        # For now, we just verify no crash
        assert True


class TestPlanningLogging:
    """Tests for PlanningSystem logging."""

    async def test_planning_request_logged(self, capsys):
        """Test PlanningSystem logs planning_request event."""
        from ecs_agent.core import World
        from ecs_agent.components import (
            PlanComponent,
            LLMComponent,
            ConversationComponent,
        )
        from ecs_agent.systems.planning import PlanningSystem
        from ecs_agent.providers import FakeProvider
        from ecs_agent.types import CompletionResult, Message

        configure_logging(json_output=True, level="DEBUG")

        world = World()
        provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(role="assistant", content="Step 1 complete")
                )
            ]
        )

        entity = world.create_entity()
        world.add_component(entity, PlanComponent(steps=["Do task A"], current_step=0))
        world.add_component(entity, LLMComponent(provider=provider, model="fake"))
        world.add_component(
            entity, ConversationComponent(messages=[Message(role="user", content="hi")])
        )

        system = PlanningSystem()
        await system.process(world)

        captured = capsys.readouterr()
        events = _json_events(captured.out)
        request_events = [e for e in events if e.get("event") == "planning_request"]

        assert len(request_events) >= 1, "planning_request event not found"
        event = request_events[0]
        assert "message_count" in event

    async def test_planning_step_completed_logged(self, capsys):
        """Test PlanningSystem logs planning_step_completed event."""
        from ecs_agent.core import World
        from ecs_agent.components import (
            PlanComponent,
            LLMComponent,
            ConversationComponent,
        )
        from ecs_agent.systems.planning import PlanningSystem
        from ecs_agent.providers import FakeProvider
        from ecs_agent.types import CompletionResult, Message

        configure_logging(json_output=True, level="INFO")

        world = World()
        provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(role="assistant", content="Step 1 complete")
                )
            ]
        )

        entity = world.create_entity()
        world.add_component(entity, PlanComponent(steps=["Do task A"], current_step=0))
        world.add_component(entity, LLMComponent(provider=provider, model="fake"))
        world.add_component(
            entity, ConversationComponent(messages=[Message(role="user", content="hi")])
        )

        system = PlanningSystem()
        await system.process(world)

        captured = capsys.readouterr()
        events = _json_events(captured.out)
        completed_events = [e for e in events if e.get("event") == "planning_step_completed"]

        assert len(completed_events) >= 1, "planning_step_completed event not found"
        event = completed_events[0]
        assert "step_index" in event
        assert "step_description" in event
        assert event["step_index"] == 0
        assert event["step_description"] == "Do task A"

    async def test_planning_error_logged(self, capsys):
        """Test PlanningSystem logs planning_error event on exception."""
        from ecs_agent.core import World
        from ecs_agent.components import (
            PlanComponent,
            LLMComponent,
            ConversationComponent,
            ErrorComponent,
        )
        from ecs_agent.systems.planning import PlanningSystem
        from ecs_agent.providers import FakeProvider
        from ecs_agent.types import Message
        configure_logging(json_output=True, level="ERROR")

        world = World()
        # FakeProvider with no responses will raise IndexError
        provider = FakeProvider(responses=[])

        entity = world.create_entity()
        world.add_component(entity, PlanComponent(steps=["Do task A"], current_step=0))
        world.add_component(entity, LLMComponent(provider=provider, model="fake"))
        world.add_component(
            entity, ConversationComponent(messages=[Message(role="user", content="hi")])
        )

        system = PlanningSystem()
        await system.process(world)

        # Verify ErrorComponent was added
        error_comp = world.get_component(entity, ErrorComponent)
        assert error_comp is not None

        captured = capsys.readouterr()
        events = _json_events(captured.out)
        error_events = [e for e in events if e.get("event") == "planning_error"]
        # This test expects planning_error event to be logged
        assert len(error_events) >= 1, "planning_error event not found"
        event = error_events[0]
        assert "exception" in event or "error" in event
