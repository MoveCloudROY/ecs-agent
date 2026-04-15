"""Closed slash-command grammar for the plan-and-task example."""

from dataclasses import dataclass


@dataclass(slots=True)
class Command:
    """Represents a parsed slash command with its name, raw text, and arguments."""

    name: str
    raw: str
    args: list[str]


_COMMANDS_WITH_ARGS = {
    "/plan:start",
    "/task:start",
    "/task:resume",
    "/task:replan",
}

_COMMANDS_WITHOUT_ARGS = {
    "/plan:status",
    "/plan:finalize",
    "/task:status",
    "/task:abort",
}

_SUPPORTED_COMMANDS = _COMMANDS_WITH_ARGS | _COMMANDS_WITHOUT_ARGS


def parse_command(text: str) -> Command:
    normalized = text.strip()
    if not normalized:
        raise ValueError("Command input cannot be empty.")

    parts = normalized.split()
    name = parts[0]
    args = parts[1:]

    if name not in _SUPPORTED_COMMANDS:
        raise ValueError(f"Unsupported command: {name}")

    if name in _COMMANDS_WITHOUT_ARGS and args:
        raise ValueError(f"Command does not accept arguments: {name}")

    return Command(name=name, raw=normalized, args=args)
