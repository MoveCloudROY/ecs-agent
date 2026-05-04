import json
import logging

from ecs_agent.logging import configure_logging, get_logger


def _captured_log_output(captured: object) -> str:
    return str(getattr(captured, "err", "") or getattr(captured, "out", ""))


def _json_events(output: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in output.strip().split("\n"):
        if line.strip():
            events.append(json.loads(line))
    return events


def test_configure_logging_includes_caller_info_in_json(capsys) -> None:
    configure_logging(json_output=True, level="INFO")

    logger = get_logger("enhanced.caller")
    logger.info("caller_event")

    events = _json_events(_captured_log_output(capsys.readouterr()))
    caller_event = events[-1]

    assert caller_event["event"] == "caller_event"
    assert str(caller_event["caller_file"]).endswith("test_enhanced_logging.py")
    assert (
        caller_event["caller_function"]
        == "test_configure_logging_includes_caller_info_in_json"
    )
    assert int(caller_event["caller_line"]) > 0


def test_configure_logging_formats_exceptions_with_traceback(capsys) -> None:
    configure_logging(json_output=True, level="INFO")

    logger = get_logger("enhanced.exception")
    try:
        raise RuntimeError("kaboom")
    except RuntimeError:
        logger.exception("exception_event")

    events = _json_events(_captured_log_output(capsys.readouterr()))
    exception_event = events[-1]

    assert exception_event["event"] == "exception_event"
    exception_text = str(exception_event["exception"])
    assert "Traceback (most recent call last)" in exception_text
    assert "RuntimeError: kaboom" in exception_text


def test_configure_logging_filters_by_module_level(capsys) -> None:
    configure_logging(
        json_output=True,
        level="DEBUG",
        module_levels={"ecs_agent.providers": "WARNING"},
    )

    model_logger = get_logger("ecs_agent.providers.openai_model")
    system_logger = get_logger("ecs_agent.systems.reasoning")

    model_logger.debug("provider_debug_hidden")
    model_logger.warning("provider_warning_visible")
    system_logger.debug("system_debug_visible")

    events = _json_events(_captured_log_output(capsys.readouterr()))
    names = [str(event["event"]) for event in events]

    assert "provider_debug_hidden" not in names
    assert "provider_warning_visible" in names
    assert "system_debug_visible" in names


def test_configure_logging_bridges_stdlib_logging(capsys) -> None:
    configure_logging(json_output=True, level="INFO")

    logging.getLogger("stdlib_test").warning("stdlib_message")

    events = _json_events(_captured_log_output(capsys.readouterr()))
    stdlib_event = events[-1]

    assert stdlib_event["event"] == "stdlib_message"
    assert stdlib_event["level"] == "warning"
    assert "timestamp" in stdlib_event


def test_configure_logging_filters_foreign_stdlib_by_base_level(capsys) -> None:
    configure_logging(json_output=True, level="ERROR")

    stdlib_logger = logging.getLogger("foreign.base_level_test")
    stdlib_logger.setLevel(logging.DEBUG)
    stdlib_logger.warning("foreign_warning_hidden")
    stdlib_logger.error("foreign_error_visible")

    events = _json_events(_captured_log_output(capsys.readouterr()))
    names = [str(event["event"]) for event in events]

    assert "foreign_warning_hidden" not in names
    assert "foreign_error_visible" in names


def test_configure_logging_filters_foreign_stdlib_by_module_level(capsys) -> None:
    configure_logging(
        json_output=True,
        level="DEBUG",
        module_levels={"httpx": "WARNING"},
    )

    stdlib_logger = logging.getLogger("httpx.foreign_module_test")
    stdlib_logger.setLevel(logging.DEBUG)
    stdlib_logger.debug("httpx_debug_hidden")
    stdlib_logger.warning("httpx_warning_visible")

    events = _json_events(_captured_log_output(capsys.readouterr()))
    names = [str(event["event"]) for event in events]

    assert "httpx_debug_hidden" not in names
    assert "httpx_warning_visible" in names


def test_configure_logging_disables_console_colors(capsys) -> None:
    configure_logging(json_output=False, level="INFO", colors=False)

    logger = get_logger("enhanced.colors")
    logger.info("colorless_event")

    output = _captured_log_output(capsys.readouterr())
    assert "colorless_event" in output
    assert "\x1b[" not in output
