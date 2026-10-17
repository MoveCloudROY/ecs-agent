"""Validate model tool calls before admitting them to execution."""

from collections.abc import Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing import Registry
from referencing.exceptions import Unresolvable

from ecs_agent.types import ToolCall, ToolSchema


def tool_call_error(
    calls: Sequence[ToolCall], tools: Sequence[ToolSchema] | None,
) -> str | None:
    """Return a diagnostic for an invalid batch, without executing any tool.

    Parameters use JSON Schema Draft 2020-12, including local references.
    Remote schema retrieval is disabled; schemas must be self-contained.
    """
    schemas = {tool.name: tool for tool in tools or []}
    seen: set[str] = set()
    for call in calls:
        if not call.id or not call.name or call.id in seen:
            return "Invalid or duplicate tool call identity"
        seen.add(call.id)
        if not isinstance(call.arguments, dict) or "_partial" in call.arguments:
            return f"Incomplete arguments for tool '{call.name}'"
        schema = schemas.get(call.name)
        if schema is None:
            return f"Unknown tool '{call.name}'"
        try:
            Draft202012Validator.check_schema(schema.parameters)
            validator = Draft202012Validator(schema.parameters, registry=Registry())
            failure = next(validator.iter_errors(call.arguments), None)
        except (SchemaError, Unresolvable) as exc:
            return f"Invalid schema for tool '{call.name}': {type(exc).__name__}"
        if failure is not None:
            path = "/".join(str(part) for part in failure.absolute_path) or "<root>"
            return f"Invalid arguments for tool '{call.name}' at {path}: {failure.validator}"
    return None
