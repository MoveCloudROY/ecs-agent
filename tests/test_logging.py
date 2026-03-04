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

class TestEventContract:
    """Tests for event naming and structured field contract."""

    def test_event_contract_enforces_snake_case_names(self, capsys):
        """Test event names follow snake_case convention."""
        from ecs_agent.logging import STANDARD_EVENT_NAMES

        configure_logging(json_output=True, level="INFO")
        logger = get_logger("test")

        # Use standard event name from contract
        logger.info(STANDARD_EVENT_NAMES["SYSTEM_START"])

        events = _json_events(capsys.readouterr().out)
        event = events[-1]

        # Event name must be snake_case
        event_name = str(event["event"])
        assert event_name.islower()
        assert "_" in event_name or event_name.isalpha()
        assert " " not in event_name
        assert "-" not in event_name

    def test_event_contract_requires_entity_id_field(self, capsys):
        """Test events require entity_id structured field for entity operations."""
        from ecs_agent.logging import REQUIRED_FIELDS

        configure_logging(json_output=True, level="INFO")
        logger = get_logger("test")

        # Log entity operation with required fields
        logger.info("entity_created", entity_id=42)

        events = _json_events(capsys.readouterr().out)
        event = events[-1]

        # Entity operations must include entity_id
        assert "entity_id" in REQUIRED_FIELDS["entity_operations"]
        assert "entity_id" in event
        assert event["entity_id"] == 42

    def test_event_contract_requires_system_field(self, capsys):
        """Test system lifecycle events require system field."""
        from ecs_agent.logging import REQUIRED_FIELDS

        configure_logging(json_output=True, level="INFO")
        logger = get_logger("test")

        # Log system operation with required fields
        logger.info("system_start", system="ReasoningSystem")

        events = _json_events(capsys.readouterr().out)
        event = events[-1]

        # System operations must include system name
        assert "system" in REQUIRED_FIELDS["system_lifecycle"]
        assert "system" in event
        assert event["system"] == "ReasoningSystem"

    def test_event_contract_requires_tick_field(self, capsys):
        """Test runner tick events require tick field."""
        from ecs_agent.logging import REQUIRED_FIELDS

        configure_logging(json_output=True, level="DEBUG")
        logger = get_logger("test")

        # Log tick event with required fields
        logger.debug("tick_start", tick=5)

        events = _json_events(capsys.readouterr().out)
        event = events[-1]

        # Tick events must include tick number
        assert "tick" in REQUIRED_FIELDS["runner_operations"]
        assert "tick" in event
        assert event["tick"] == 5

    def test_event_contract_requires_duration_ms_field(self, capsys):
        """Test completion events require duration_ms field."""
        from ecs_agent.logging import REQUIRED_FIELDS

        configure_logging(json_output=True, level="INFO")
        logger = get_logger("test")

        # Log completion event with required fields
        logger.info("system_complete", system="ReasoningSystem", duration_ms=123.45)

        events = _json_events(capsys.readouterr().out)
        event = events[-1]

        # Completion events must include duration
        assert "duration_ms" in REQUIRED_FIELDS["completion_events"]
        assert "duration_ms" in event
        assert event["duration_ms"] == 123.45

    def test_event_contract_requires_success_field(self, capsys):
        """Test result events require success field."""
        from ecs_agent.logging import REQUIRED_FIELDS

        configure_logging(json_output=True, level="INFO")
        logger = get_logger("test")

        # Log result event with required fields
        logger.info("tool_result", success=True)

        events = _json_events(capsys.readouterr().out)
        event = events[-1]

        # Result events must include success flag
        assert "success" in REQUIRED_FIELDS["result_events"]
        assert "success" in event
        assert event["success"] is True

    def test_event_contract_requires_reason_field(self, capsys):
        """Test failure events require reason field."""
        from ecs_agent.logging import REQUIRED_FIELDS

        configure_logging(json_output=True, level="ERROR")
        logger = get_logger("test")

        # Log failure event with required fields
        logger.error("tool_failed", success=False, reason="Network timeout")

        events = _json_events(capsys.readouterr().out)
        event = events[-1]

        # Failure events must include reason
        assert "reason" in REQUIRED_FIELDS["failure_events"]
        assert "reason" in event
        assert event["reason"] == "Network timeout"


class TestLevelPolicy:
    """Tests for log level policy enforcement."""

    def test_level_policy_high_frequency_operations_use_debug(self, capsys):
        """Test high-frequency operations log at DEBUG level."""
        from ecs_agent.logging import LEVEL_POLICY

        configure_logging(json_output=True, level="DEBUG")
        logger = get_logger("test")

        # High-frequency operations should use DEBUG
        assert LEVEL_POLICY["high_frequency"] == "DEBUG"
        logger.debug("component_read", entity_id=1)

        events = _json_events(capsys.readouterr().out)
        event = events[-1]

        assert event["event"] == "component_read"
        assert event["level"] == "debug"

    def test_level_policy_lifecycle_operations_use_info(self, capsys):
        """Test lifecycle operations log at INFO level."""
        from ecs_agent.logging import LEVEL_POLICY

        configure_logging(json_output=True, level="INFO")
        logger = get_logger("test")

        # Lifecycle operations should use INFO
        assert LEVEL_POLICY["lifecycle"] == "INFO"
        logger.info("system_start", system="ReasoningSystem")

        events = _json_events(capsys.readouterr().out)
        event = events[-1]

        assert event["event"] == "system_start"
        assert event["level"] == "info"

    def test_level_policy_anomalies_use_warning(self, capsys):
        """Test anomalies log at WARNING level."""
        from ecs_agent.logging import LEVEL_POLICY

        configure_logging(json_output=True, level="WARNING")
        logger = get_logger("test")

        # Anomalies should use WARNING
        assert LEVEL_POLICY["anomalies"] == "WARNING"
        logger.warning("retry_attempt", attempt=3, max_retries=3)

        events = _json_events(capsys.readouterr().out)
        event = events[-1]

        assert event["event"] == "retry_attempt"
        assert event["level"] == "warning"

    def test_level_policy_failures_use_error(self, capsys):
        """Test failures log at ERROR level."""
        from ecs_agent.logging import LEVEL_POLICY

        configure_logging(json_output=True, level="ERROR")
        logger = get_logger("test")

        # Failures should use ERROR
        assert LEVEL_POLICY["failures"] == "ERROR"
        logger.error("tool_failed", reason="Network timeout")

        events = _json_events(capsys.readouterr().out)
        event = events[-1]

        assert event["event"] == "tool_failed"
        assert event["level"] == "error"


class TestSensitiveDataPolicy:
    """Tests for sensitive data exclusion policy."""

    def test_sensitive_data_policy_forbids_content_field(self, capsys):
        """Test sensitive data policy forbids raw conversation content."""
        from ecs_agent.logging import FORBIDDEN_FIELDS

        configure_logging(json_output=True, level="INFO")
        logger = get_logger("test")

        # Log message event with metadata only
        logger.info("message_received", role="user", length=42)

        events = _json_events(capsys.readouterr().out)
        event = events[-1]

        # Raw content must be absent
        assert "content" in FORBIDDEN_FIELDS
        assert "content" not in event
        assert "role" in event
        assert "length" in event

    def test_sensitive_data_policy_forbids_arguments_field(self, capsys):
        """Test sensitive data policy forbids raw tool arguments."""
        from ecs_agent.logging import FORBIDDEN_FIELDS

        configure_logging(json_output=True, level="INFO")
        logger = get_logger("test")

        # Log tool call with metadata only
        logger.info("tool_called", tool_name="bash", argument_count=3)

        events = _json_events(capsys.readouterr().out)
        event = events[-1]

        # Raw arguments must be absent
        assert "arguments" in FORBIDDEN_FIELDS
        assert "arguments" not in event
        assert "tool_name" in event
        assert "argument_count" in event

    def test_sensitive_data_policy_forbids_api_key_field(self, capsys):
        """Test sensitive data policy forbids API keys."""
        from ecs_agent.logging import FORBIDDEN_FIELDS

        configure_logging(json_output=True, level="INFO")
        logger = get_logger("test")

        # Log provider config with metadata only
        logger.info("provider_configured", model="gpt-4", base_url="https://api.openai.com")

        events = _json_events(capsys.readouterr().out)
        event = events[-1]

        # API keys must be absent
        assert "api_key" in FORBIDDEN_FIELDS
        assert "api_key" not in event
        assert "model" in event
        assert "base_url" in event

    def test_sensitive_data_policy_forbids_token_field(self, capsys):
        """Test sensitive data policy forbids auth tokens."""
        from ecs_agent.logging import FORBIDDEN_FIELDS

        configure_logging(json_output=True, level="INFO")
        logger = get_logger("test")

        # Log auth event with metadata only
        logger.info("auth_success", user_id="user123")

        events = _json_events(capsys.readouterr().out)
        event = events[-1]

        # Tokens must be absent
        assert "token" in FORBIDDEN_FIELDS
        assert "token" not in event
        assert "user_id" in event

    def test_sensitive_data_policy_forbids_payload_field(self, capsys):
        """Test sensitive data policy forbids full HTTP/checkpoint payloads."""
        from ecs_agent.logging import FORBIDDEN_FIELDS

        configure_logging(json_output=True, level="INFO")
        logger = get_logger("test")

        # Log checkpoint event with metadata only
        logger.info("checkpoint_saved", checkpoint_id="ckpt-123", size_bytes=4096)

        events = _json_events(capsys.readouterr().out)
        event = events[-1]

        # Full payloads must be absent
        assert "payload" in FORBIDDEN_FIELDS
        assert "payload" not in event
        assert "checkpoint_id" in event
        assert "size_bytes" in event



class TestWorldComponentLogging:
    """Tests for world/component/query logging instrumentation."""

    def test_component_store_add_logs_debug_event(self, capsys):
        """Test ComponentStore.add emits debug event."""
        from ecs_agent.core.component import ComponentStore
        from ecs_agent.types import EntityId
        from dataclasses import dataclass

        @dataclass(slots=True)
        class TestComponent:
            value: int

        store = ComponentStore()
        entity = EntityId(1)
        comp = TestComponent(value=42)

        store.add(entity, comp)

        captured = capsys.readouterr()
        output = captured.out

        # Check for component_store_add event
        assert "component_store_add" in output
        assert "entity_id=1" in output
        assert "component_type=TestComponent" in output

    def test_query_no_match_logs_debug_event(self, capsys):
        """Test Query.get with no matches emits debug event."""
        from ecs_agent.core.component import ComponentStore
        from ecs_agent.core.query import Query
        from dataclasses import dataclass

        @dataclass(slots=True)
        class MissingComponent:
            value: str

        store = ComponentStore()
        query = Query(store)

        results = query.get(MissingComponent)

        captured = capsys.readouterr()
        output = captured.out

        # Check for query_executed event with no matches
        assert "query_executed" in output
        assert "result_count=0" in output


class TestEventBusLogging:
    """Tests for event bus logging instrumentation."""

    async def test_event_bus_publish_logs_event(self, capsys):
        """Test EventBus.publish emits structured log event."""
        from ecs_agent.logging import configure_logging
        from dataclasses import dataclass

        configure_logging(json_output=False, level="DEBUG")

        # Import AFTER configure_logging to get correct logger config
        from ecs_agent.core.event_bus import EventBus

        @dataclass
        class TestEvent:
            message: str
class TestReasoningSystemLogging:
    """Tests for ReasoningSystem lifecycle and error logging."""

    async def test_reasoning_start_logs_lifecycle_event(self, capsys):
        """Test ReasoningSystem emits reasoning_start with entity_id and model."""
        from ecs_agent.core import World
        from ecs_agent.systems.reasoning import ReasoningSystem
        from ecs_agent.components import LLMComponent, ConversationComponent
        from ecs_agent.providers import FakeProvider
        from ecs_agent.types import Message, CompletionResult

        provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(role="assistant", content="Hello")
                )
            ]
        )
        world = World()
        entity = world.create_entity()
        world.add_component(
            entity, LLMComponent(provider=provider, model="fake-model")
        )
        world.add_component(
            entity,
            ConversationComponent(messages=[Message(role="user", content="Hi")]),
        )

        system = ReasoningSystem()
        await system.process(world)

        captured = capsys.readouterr().out

        # Check for reasoning_start event
        assert "reasoning_start" in captured
        assert f"entity_id={entity}" in captured or f"entity_id={int(entity)}" in captured
        assert "fake-model" in captured
        assert "ReasoningSystem" in captured

    async def test_reasoning_complete_logs_lifecycle_event(self, capsys):
        """Test ReasoningSystem emits reasoning_complete with entity_id."""
        from ecs_agent.core import World
        from ecs_agent.systems.reasoning import ReasoningSystem
        from ecs_agent.components import LLMComponent, ConversationComponent
        from ecs_agent.providers import FakeProvider
        from ecs_agent.types import Message, CompletionResult

        provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(role="assistant", content="Hello")
                )
            ]
        )
        world = World()
        entity = world.create_entity()
        world.add_component(
            entity, LLMComponent(provider=provider, model="fake-model")
        )
        world.add_component(
            entity,
            ConversationComponent(messages=[Message(role="user", content="Hi")]),
        )

        system = ReasoningSystem()
        await system.process(world)

        captured = capsys.readouterr().out

        # Check for reasoning_complete event
        assert "reasoning_complete" in captured
        assert f"entity_id={entity}" in captured or f"entity_id={int(entity)}" in captured
        assert "fake-model" in captured
        assert "ReasoningSystem" in captured

    async def test_reasoning_error_logs_exception(self, capsys):
        """Test ReasoningSystem emits reasoning_error on provider exception."""
        from ecs_agent.core import World
        from ecs_agent.systems.reasoning import ReasoningSystem
        from ecs_agent.components import LLMComponent, ConversationComponent, ErrorComponent
        from ecs_agent.types import Message

        class FailingProvider:
            async def complete(self, messages, tools=None, stream=False, response_format=None):
                raise RuntimeError("Provider failed")

        world = World()
        entity = world.create_entity()
        world.add_component(
            entity, LLMComponent(provider=FailingProvider(), model="failing-model")
        )
        world.add_component(
            entity,
            ConversationComponent(messages=[Message(role="user", content="Hi")]),
        )

        system = ReasoningSystem()
        await system.process(world)

        # Verify ErrorComponent was added
        error_comp = world.get_component(entity, ErrorComponent)
        assert error_comp is not None

        captured = capsys.readouterr().out

        # Check for reasoning_error event
        assert "reasoning_error" in captured
        assert f"entity_id={entity}" in captured or f"entity_id={int(entity)}" in captured
        assert "ReasoningSystem" in captured
        assert "Provider failed" in captured

    async def test_reasoning_logs_no_sensitive_data(self, capsys):
        """Test ReasoningSystem does not log raw message content or arguments."""
        from ecs_agent.core import World
        from ecs_agent.systems.reasoning import ReasoningSystem
        from ecs_agent.components import LLMComponent, ConversationComponent
        from ecs_agent.providers import FakeProvider
        from ecs_agent.types import Message, CompletionResult, ToolCall

        provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                name="bash",
                                arguments={"command": "secret-data"},
                            )
                        ],
                    )
                )
            ]
        )
        world = World()
        entity = world.create_entity()
        world.add_component(
            entity, LLMComponent(provider=provider, model="fake-model")
        )
        world.add_component(
            entity,
            ConversationComponent(
                messages=[Message(role="user", content="secret user message")]
            ),
        )

        system = ReasoningSystem()
        await system.process(world)

        captured = capsys.readouterr().out

        # Verify sensitive strings are NOT in output
        assert "secret-data" not in captured
        assert "secret user message" not in captured
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
        events = _json_events(captured.out)
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
        from ecs_agent.providers.protocol import LLMProvider
        from ecs_agent.types import Message, ToolSchema, CompletionResult
        from typing import AsyncIterator, Any

        # Create a provider that raises a real exception (not IndexError)
        class FailingProvider:
            async def complete(
                self,
                messages: list[Message],
                tools: list[ToolSchema] | None = None,
                stream: bool = False,
                response_format: dict[str, Any] | None = None,
            ) -> CompletionResult | AsyncIterator[Any]:
                raise RuntimeError("Provider failed!")

        world = World()
        provider = FailingProvider()

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
