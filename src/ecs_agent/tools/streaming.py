"""Shared aggregation of provider tool-call deltas."""

import json
from typing import Any

from ecs_agent.types import ToolCall


def merge_stream_tool_calls(
    buffers: dict[str, dict[str, Any]],
    delta_tool_calls: list[ToolCall] | None,
) -> None:
    """Accumulate tool fragments, keeping provider identities intact."""
    if not delta_tool_calls:
        return

    for tool_call in delta_tool_calls:
        if tool_call.id:
            tool_call_id = tool_call.id
        elif buffers:
            # Continuation fragments from providers that omit the id
            # extend the most recently started tool call.
            tool_call_id = next(reversed(buffers))
        else:
            tool_call_id = ""
        current = buffers.setdefault(
            tool_call_id,
            {
                "id": tool_call_id,
                "name": "",
                "arguments_buffer": "",
                "arguments": None,
            },
        )

        if tool_call.name:
            current["name"] = tool_call.name

        partial = tool_call.arguments.get("_partial")
        if isinstance(partial, str):
            current["arguments_buffer"] += partial
        else:
            # A fully parsed dict supersedes any partial snapshots
            # buffered earlier for the same call.
            current["arguments"] = tool_call.arguments
            current["arguments_buffer"] = ""


def finalize_tool_calls(
    buffers: dict[str, dict[str, Any]],
) -> list[ToolCall] | None:
    """Parse accumulated arguments without treating partial JSON as executable."""
    if not buffers:
        return None

    completed: list[ToolCall] = []
    for buffered in buffers.values():
        parsed_arguments: dict[str, Any]
        arguments_buffer = buffered["arguments_buffer"]
        arguments = buffered["arguments"]

        if isinstance(arguments, dict):
            # A complete parsed dict always wins over the fragment buffer.
            parsed_arguments = arguments
        elif arguments_buffer:
            try:
                loaded_arguments = json.loads(arguments_buffer)
            except json.JSONDecodeError:
                loaded_arguments = {"_partial": arguments_buffer}

            if isinstance(loaded_arguments, dict):
                parsed_arguments = loaded_arguments
            else:
                parsed_arguments = {"_partial": arguments_buffer}
        else:
            parsed_arguments = {}

        completed.append(
            ToolCall(
                id=buffered["id"],
                name=buffered["name"],
                arguments=parsed_arguments,
            )
        )

    return completed
