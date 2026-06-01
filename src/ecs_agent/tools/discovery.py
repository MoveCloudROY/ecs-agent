"""Tool discovery helpers for setup-time registration."""

import asyncio
import inspect
from functools import partial
from types import NoneType, UnionType, ModuleType
from typing import Annotated, Any, Awaitable, Callable, Union, cast, get_args, get_origin

from ecs_agent.types import ToolSchema

_TOOL_REGISTRY: dict[str, ToolSchema] = {}


def _map_parameter_type(annotation: Any) -> str | list[str]:
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]

    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        mapped_types = [_map_parameter_type(arg) for arg in get_args(annotation)]
        flattened: list[str] = []
        for mapped in mapped_types:
            if isinstance(mapped, list):
                flattened.extend(mapped)
            else:
                flattened.append(mapped)
        return list(dict.fromkeys(flattened))

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
    if annotation is None or annotation is NoneType:
        return "null"
    return "string"


def _extract_param_description(annotation: Any) -> str | None:
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        for arg in args[1:]:
            if isinstance(arg, str):
                return arg
    return None


def _build_parameters_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    try:
        hints = _get_type_hints_safe(fn)
    except Exception:
        hints = {}

    signature = inspect.signature(fn)
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for name, parameter in signature.parameters.items():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue

        annotation = hints.get(name, parameter.annotation)
        param_schema: dict[str, Any] = {"type": _map_parameter_type(annotation)}

        description = _extract_param_description(annotation)
        if description:
            param_schema["description"] = description

        properties[name] = param_schema
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _get_type_hints_safe(fn: Callable[..., Any]) -> dict[str, Any]:
    import typing

    return typing.get_type_hints(fn, include_extras=True)


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
