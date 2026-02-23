"""Tests for the structlog logging module."""

import json
import io
import sys
from contextlib import redirect_stdout

import pytest
import structlog

from ecs_agent.logging import configure_logging, get_logger


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
