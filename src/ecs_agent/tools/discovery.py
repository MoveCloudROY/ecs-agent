"""Tool discovery helpers for setup-time registration."""

import asyncio
import inspect
from functools import partial
from types import ModuleType
from typing import Any, Awaitable, Callable, cast

from ecs_agent.types import ToolSchema

_TOOL_REGISTRY: dict[str, ToolSchema] = {}


def _map_parameter_type(annotation: Any) -> str:
    if annotation in ("str", "builtins.str"):
        return "string"
    if annotation in ("int", "builtins.int"):
        return "integer"
    if annotation in ("float", "builtins.float"):
        return "number"
    if annotation in ("bool", "builtins.bool"):
        return "boolean"

    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    return "string"


def _build_parameters_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    signature = inspect.signature(fn)
    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []

    for name, parameter in signature.parameters.items():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue

        properties[name] = {"type": _map_parameter_type(parameter.annotation)}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _create_async_handler(fn: Callable[..., Any]) -> Callable[..., Awaitable[str]]:
    if inspect.iscoroutinefunction(fn):

        async def async_handler(**kwargs: Any) -> str:
            result = await cast(Callable[..., Awaitable[str]], fn)(**kwargs)
            return str(result)

        return async_handler

    async def sync_wrapper(**kwargs: Any) -> str:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, partial(fn, **kwargs))
        return str(result)

    return sync_wrapper


def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for marking functions as tools."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__
        tool_description = (
            description if description is not None else (fn.__doc__ or "")
        )
        schema = ToolSchema(
            name=tool_name,
            description=tool_description,
            parameters=_build_parameters_schema(fn),
        )
        handler = _create_async_handler(fn)

        setattr(fn, "_tool_schema", schema)
        setattr(fn, "_tool_handler", handler)
        _TOOL_REGISTRY[tool_name] = schema
        return fn

    return decorator


def scan_module(
    module: ModuleType,
) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
    """Scan a module for @tool-decorated functions."""

    discovered: dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]] = {}

    for attr_name in dir(module):
        candidate = getattr(module, attr_name)
        if not hasattr(candidate, "_tool_schema"):
            continue

        schema = cast(ToolSchema, getattr(candidate, "_tool_schema"))
        handler = cast(
            Callable[..., Awaitable[str]], getattr(candidate, "_tool_handler")
        )

        if schema.name in discovered:
            raise ValueError(f"Duplicate tool name: {schema.name}")

        discovered[schema.name] = (schema, handler)

    return discovered
