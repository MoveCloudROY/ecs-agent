"""Tool discovery helpers for setup-time registration."""

import asyncio
import inspect
import json
from copy import deepcopy
from functools import partial, wraps
from types import NoneType, UnionType, ModuleType, SimpleNamespace
from typing import Annotated, Any, Awaitable, Callable, Literal, TypeVar, Union, cast, get_args, get_origin, get_type_hints

from ecs_agent.logging import get_logger
from ecs_agent.tools.definitions import ToolDefinition
from ecs_agent.types import ToolSchema

logger = get_logger(__name__)
_TOOL_REGISTRY: dict[str, ToolSchema] = {}
F = TypeVar("F", bound=Callable[..., Any])


def _parameter_schema(annotation: Any, path: str) -> dict[str, Any]:
    """Map supported JSON-native annotations without dropping constraints."""
    if get_origin(annotation) is Annotated:
        base, *metadata = get_args(annotation)
        schema = _parameter_schema(base, path)
        description = next((item for item in metadata if isinstance(item, str)), None)
        if description is not None:
            schema["description"] = description
        return schema

    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        branches = [_parameter_schema(arg, path) for arg in get_args(annotation)]
        if all(set(branch) == {"type"} and isinstance(branch["type"], str) for branch in branches):
            return {"type": list(dict.fromkeys(branch["type"] for branch in branches))}
        return {"anyOf": branches}

    if origin is list:
        if len(get_args(annotation)) != 1:
            raise ValueError(f"{path}: list requires an item annotation")
        return {"type": "array", "items": _parameter_schema(get_args(annotation)[0], f"{path}[]")}
    if origin is dict:
        if len(get_args(annotation)) != 2:
            raise ValueError(f"{path}: dict requires key and value annotations")
        key, value = get_args(annotation)
        if key is not str:
            raise ValueError(f"{path}: dict keys must be str, got {key!r}")
        return {"type": "object", "additionalProperties": _parameter_schema(value, f"{path}.*")}
    if origin is Literal:
        values = list(get_args(annotation))
        if not values or any(type(value) not in (str, int, float, bool, NoneType) for value in values):
            raise ValueError(f"{path}: Literal values must be JSON primitives, got {annotation!r}")
        try:
            json.dumps(values, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: Literal values must be finite JSON primitives") from exc
        types = list(dict.fromkeys(_parameter_schema(type(value), path)["type"] for value in values))
        return {"type": types[0] if len(types) == 1 else types, "enum": values}

    for python_type, json_type in ((str, "string"), (int, "integer"), (float, "number"), (bool, "boolean")):
        if annotation is python_type:
            return {"type": json_type}
    if annotation is None or annotation is NoneType:
        return {"type": "null"}
    raise ValueError(f"{path}: unsupported annotation {annotation!r}; use a JSON-native annotation or an explicit ToolDefinition adapter")


def _build_parameters_schema(fn: Callable[..., Any], tool_name: str) -> dict[str, Any]:
    signature = inspect.signature(fn)
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for name, parameter in signature.parameters.items():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise ValueError(f"Tool '{tool_name}' parameter '{name}': unsupported parameter kind {parameter.kind.description}")

        # Resolve each input independently, so diagnostics identify the parameter
        # and an unrelated return annotation cannot invalidate an input schema.
        holder = SimpleNamespace(__annotations__={name: parameter.annotation})
        try:
            annotation = get_type_hints(holder, globalns=getattr(fn, "__globals__", {}), include_extras=True)[name]
        except (NameError, TypeError, SyntaxError, AttributeError) as exc:
            raise ValueError(f"Tool '{tool_name}' parameter '{name}': cannot resolve annotation {parameter.annotation!r}: {exc}") from exc
        properties[name] = _parameter_schema(annotation, f"Tool '{tool_name}' parameter '{name}'")
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _create_async_handler(fn: Callable[..., Any]) -> Callable[..., Awaitable[str]]:
    signature = inspect.signature(fn)

    def bind(kwargs: dict[str, Any]) -> None:
        try:
            signature.bind(**kwargs)
        except TypeError as exc:
            logger.error("tool_argument_binding_error", tool=fn.__name__, exception=str(exc))
            raise

    if inspect.iscoroutinefunction(fn):

        @wraps(fn)
        async def async_handler(**kwargs: Any) -> str:
            bind(kwargs)
            result = await cast(Callable[..., Awaitable[str]], fn)(**kwargs)
            return result

        return async_handler

    @wraps(fn)
    async def sync_wrapper(**kwargs: Any) -> str:
        bind(kwargs)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, partial(fn, **kwargs))
        return cast(str, result)

    return sync_wrapper


def tool(
    name: str | None = None,
    description: str | None = None,
    concurrency_safe: bool = False,
    inline_result: bool = False,
) -> Callable[[F], F]:
    """Describe a JSON-native tool while preserving the original function.

    Unsupported inputs fail at decoration time. Use ToolDefinition for manual
    schemas and handlers that adapt JSON values to custom Python objects.
    """
    def decorator(fn: F) -> F:
        tool_name = fn.__name__ if name is None else name
        tool_description = (
            description if description is not None else (fn.__doc__ or "")
        )
        try:
            parameters = _build_parameters_schema(fn, tool_name)
        except ValueError as exc:
            logger.error("invalid_tool_definition", tool=tool_name, exception=str(exc))
            raise
        schema = ToolSchema(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            concurrency_safe=concurrency_safe,
            inline_result=inline_result,
        )
        handler = _create_async_handler(fn)

        setattr(fn, "_tool_schema", schema)
        setattr(fn, "_tool_handler", handler)
        _TOOL_REGISTRY[tool_name] = schema
        return fn

    return decorator


def get_tool_definition(fn: Callable[..., Any]) -> ToolDefinition:
    """Read a decorated function's definition without private attribute access."""
    schema = getattr(fn, "_tool_schema", None)
    handler = getattr(fn, "_tool_handler", None)
    if not isinstance(schema, ToolSchema) or not callable(handler):
        logger.error("undecorated_tool", function=getattr(fn, "__name__", repr(fn)))
        raise ValueError("Expected a @tool() decorated function or an explicit ToolDefinition")
    return ToolDefinition(deepcopy(schema), handler)


def scan_module(
    module: ModuleType,
) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
    """Discover decorated functions; the returned mapping can be registered directly."""
    discovered: dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]] = {}

    for attr_name in dir(module):
        candidate = getattr(module, attr_name)
        if not hasattr(candidate, "_tool_schema"):
            continue

        definition = get_tool_definition(candidate)
        schema, handler = definition.schema, definition.handler

        if schema.name in discovered:
            logger.error("duplicate_tool_name", tool=schema.name)
            raise ValueError(f"Duplicate tool name: {schema.name}")

        discovered[schema.name] = (schema, handler)

    return discovered
