"""Validated, atomic registration of tools on an existing agent entity."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from ecs_agent.logging import get_logger
from ecs_agent.tools.definitions import ToolDefinition
from ecs_agent.tools.discovery import get_tool_definition
from ecs_agent.types import EntityId, ToolSchema

if TYPE_CHECKING:
    from ecs_agent.components import ToolRegistryComponent
    from ecs_agent.core.world import World

logger = get_logger(__name__)
ToolInputs = (
    Iterable[ToolDefinition | Callable[..., Any]]
    | Mapping[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]
)


def _validate_definition(definition: ToolDefinition) -> None:
    schema = definition.schema
    if not schema.name or any(character.isspace() for character in schema.name):
        raise ValueError(f"Invalid tool name: {schema.name!r}")
    if not callable(definition.handler):
        raise ValueError(f"Tool '{schema.name}': handler must be callable")
    try:
        json.dumps(schema.parameters, allow_nan=False)
        Draft202012Validator.check_schema(schema.parameters)
    except (SchemaError, TypeError, ValueError) as exc:
        raise ValueError(f"Tool '{schema.name}': invalid JSON parameter schema: {exc}") from exc
    if "type" in schema.parameters and schema.parameters["type"] != "object":
        raise ValueError(f"Tool '{schema.name}': parameters must describe an object")
    try:
        signature = inspect.signature(definition.handler)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Tool '{schema.name}': handler signature cannot be inspected; provide a Python adapter") from exc
    parameters = signature.parameters
    for parameter in parameters.values():
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY and parameter.default is inspect.Parameter.empty:
            raise ValueError(f"Tool '{schema.name}' parameter '{parameter.name}': required positional-only input cannot be supplied by JSON keywords")
    # Arbitrary composed schemas require an explicit adapter. Validate their
    # structure, but do not pretend to prove every possible signature binding.
    if any(keyword in schema.parameters for keyword in ("$ref", "anyOf", "oneOf", "allOf")):
        return
    properties = schema.parameters.get("properties", {})
    accepts_extra = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    keyword_parameters = {
        name: parameter for name, parameter in parameters.items()
        if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    if not accepts_extra:
        unknown = sorted(set(properties) - set(keyword_parameters))
        if unknown:
            raise ValueError(f"Tool '{schema.name}': schema parameters not accepted by handler: {unknown}")
    required = set(schema.parameters.get("required", []))
    missing = sorted(
        name for name, parameter in keyword_parameters.items()
        if parameter.default is inspect.Parameter.empty and name not in required
    )
    if missing:
        raise ValueError(f"Tool '{schema.name}': schema must require handler parameters: {missing}")


def register_tools(world: World, entity: EntityId, tools: ToolInputs) -> ToolRegistryComponent:
    """Validate a batch, then merge its definitions into the entity registry.

    Args:
        world: World containing the agent entity.
        entity: Existing entity whose tools should be extended.
        tools: Decorated functions, explicit definitions, or scan_module's mapping.

    Returns:
        The existing or newly installed ToolRegistryComponent.

    Raises:
        ValueError: A definition is invalid, duplicated, or conflicts with the
            existing registry. No registry changes are made on validation failure.
    """
    from ecs_agent.components import ToolRegistryComponent

    try:
        if not world.has_entity(entity):
            raise ValueError(f"Cannot register tools on missing entity {entity}")
        definitions: list[ToolDefinition] = []
        if isinstance(tools, Mapping):
            for name, (schema, handler) in tools.items():
                if name != schema.name:
                    raise ValueError(f"Tool mapping key '{name}' disagrees with schema name '{schema.name}'")
                definitions.append(ToolDefinition(deepcopy(schema), handler))
        else:
            for item in tools:
                definitions.append(
                    ToolDefinition(deepcopy(item.schema), item.handler)
                    if isinstance(item, ToolDefinition) else get_tool_definition(item)
                )

        registry = world.get_component(entity, ToolRegistryComponent)
        if registry is not None and set(registry.tools) != set(registry.handlers):
            raise ValueError("Existing tool registry has mismatched schema and handler names")
        seen = set(registry.tools) if registry is not None else set()
        for definition in definitions:
            _validate_definition(definition)
            name = definition.schema.name
            if name in seen:
                raise ValueError(f"Duplicate tool name: {name}")
            seen.add(name)
    except (TypeError, ValueError) as exc:
        logger.error("tool_registration_failed", entity_id=int(entity), exception=str(exc))
        raise

    # No awaits or user callbacks between validation and committing the batch.
    if registry is None:
        registry = ToolRegistryComponent(tools={}, handlers={})
        world.add_component(entity, registry)
    registry.tools.update({definition.schema.name: definition.schema for definition in definitions})
    registry.handlers.update({definition.schema.name: definition.handler for definition in definitions})
    return registry


def register_framework_tools(
    world: World, entity: EntityId, definitions: list[ToolDefinition], *,
    owner: object, rebind: bool = False,
) -> None:
    """Install owned framework slots through the same batch validator.

    Repeating the same installation preserves handler identity. Rebinding is
    explicit for a restored World; ordinary installs reject foreign owners.
    """
    from ecs_agent.components import ToolRegistryComponent
    from ecs_agent.core.world import World

    registry = world.get_component(entity, ToolRegistryComponent)
    updates = []
    for definition in definitions:
        name = definition.schema.name
        existing = registry.handlers.get(name) if registry is not None else None
        if registry is not None and name in registry.tools:
            owned = getattr(existing, "_ecs_framework_owner", None) == owner
            if owned and registry.tools[name] == definition.schema:
                continue
            if not owned and not rebind:
                raise ValueError(f"Framework tool name collision: {name}")
            if owned and existing is not None:
                definition = ToolDefinition(definition.schema, existing)
        updates.append(definition)
    if not updates:
        return
    names = {definition.schema.name for definition in updates}
    probe = World()
    probe_entity = probe.create_entity()
    retained = []
    if registry is not None:
        if not rebind and set(registry.tools) != set(registry.handlers):
            raise ValueError("Existing tool registry has mismatched schema and handler names")
        for name, schema in registry.tools.items():
            if name not in names:
                handler = registry.handlers.get(name)
                if handler is None:
                    if rebind:
                        continue
                    raise ValueError(f"Restore the tool handler for '{name}' before binding framework tools")
                retained.append(ToolDefinition(schema, handler))
    validated = register_tools(probe, probe_entity, [*retained, *updates])
    for definition in updates:
        setattr(definition.handler, "_ecs_framework_owner", owner)
    if registry is None:
        world.add_component(entity, validated)
    else:
        registry.tools.update(validated.tools)
        registry.handlers.update(validated.handlers)
