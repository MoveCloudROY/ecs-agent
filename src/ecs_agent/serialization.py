from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ecs_agent.components import (
    CheckpointComponent,
    CollaborationComponent,
    CompactionConfigComponent,
    ConversationArchiveComponent,
    ConversationComponent,
    EmbeddingComponent,
    ErrorComponent,
    KVStoreComponent,
    LLMComponent,
    OwnerComponent,
    PendingToolCallsComponent,
    PlanComponent,
    PlanSearchComponent,
    RAGTriggerComponent,
    RunnerStateComponent,
    SandboxConfigComponent,
    StreamingComponent,
    SystemPromptComponent,
    TerminalComponent,
    ToolApprovalComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
    VectorStoreComponent,
)
from ecs_agent.core.world import World
from ecs_agent.types import ApprovalPolicy, EntityId, Message, ToolCall, ToolSchema

NON_SERIALIZABLE_PLACEHOLDER = "<non-serializable>"

COMPONENT_REGISTRY: dict[str, type[Any]] = {
    LLMComponent.__name__: LLMComponent,
    ConversationComponent.__name__: ConversationComponent,
    PlanComponent.__name__: PlanComponent,
    ToolRegistryComponent.__name__: ToolRegistryComponent,
    PendingToolCallsComponent.__name__: PendingToolCallsComponent,
    ToolResultsComponent.__name__: ToolResultsComponent,
    KVStoreComponent.__name__: KVStoreComponent,
    CollaborationComponent.__name__: CollaborationComponent,
    OwnerComponent.__name__: OwnerComponent,
    ErrorComponent.__name__: ErrorComponent,
    TerminalComponent.__name__: TerminalComponent,
    SystemPromptComponent.__name__: SystemPromptComponent,
    ToolApprovalComponent.__name__: ToolApprovalComponent,
    SandboxConfigComponent.__name__: SandboxConfigComponent,
    PlanSearchComponent.__name__: PlanSearchComponent,
    RAGTriggerComponent.__name__: RAGTriggerComponent,
    EmbeddingComponent.__name__: EmbeddingComponent,
    VectorStoreComponent.__name__: VectorStoreComponent,
    StreamingComponent.__name__: StreamingComponent,
    CheckpointComponent.__name__: CheckpointComponent,
    CompactionConfigComponent.__name__: CompactionConfigComponent,
    ConversationArchiveComponent.__name__: ConversationArchiveComponent,
    RunnerStateComponent.__name__: RunnerStateComponent,
}


class WorldSerializer:
    @staticmethod
    def to_dict(world: World) -> dict[str, Any]:
        entities: dict[str, dict[str, Any]] = {}
        component_store = world._components._components

        entity_ids: set[EntityId] = set()
        for entity_map in component_store.values():
            entity_ids.update(entity_map.keys())

        for entity_id in sorted(entity_ids):
            serialized_components: dict[str, Any] = {}
            for component_type, entity_map in component_store.items():
                component = entity_map.get(entity_id)
                if component is None:
                    continue
                serialized_components[component_type.__name__] = (
                    WorldSerializer._serialize_component(component)
                )
            entities[str(int(entity_id))] = serialized_components

        next_entity_id = world._entity_gen._counter + 1
        return {
            "next_entity_id": next_entity_id,
            "entities": entities,
        }

    @staticmethod
    def from_dict(
        data: dict[str, Any],
        providers: dict[str, Any],
        tool_handlers: dict[str, Any],
    ) -> World:
        world = World()

        entities_data = data.get("entities", {})
        for entity_id_str, serialized_components in entities_data.items():
            entity_id = EntityId(int(entity_id_str))
            for component_name, component_data in serialized_components.items():
                component_type = COMPONENT_REGISTRY.get(component_name)
                if component_type is None:
                    continue

                normalized_data = WorldSerializer._normalize_component_data(
                    component_name,
                    component_data,
                    providers,
                    tool_handlers,
                )
                world.add_component(entity_id, component_type(**normalized_data))

        next_entity_id = int(data.get("next_entity_id", 1))
        world._entity_gen._counter = max(0, next_entity_id - 1)
        return world

    @staticmethod
    def save(world: World, path: Path) -> None:
        path.write_text(
            json.dumps(WorldSerializer.to_dict(world), indent=2), encoding="utf-8"
        )

    @staticmethod
    def load(
        path: Path,
        providers: dict[str, Any],
        tool_handlers: dict[str, Any],
    ) -> World:
        data = json.loads(path.read_text(encoding="utf-8"))
        return WorldSerializer.from_dict(
            data, providers=providers, tool_handlers=tool_handlers
        )

    @staticmethod
    def _serialize_component(component: Any) -> dict[str, Any]:
        serialized = asdict(component)

        if isinstance(component, LLMComponent):
            serialized["provider"] = NON_SERIALIZABLE_PLACEHOLDER

        if isinstance(component, ToolRegistryComponent):
            serialized["handlers"] = NON_SERIALIZABLE_PLACEHOLDER

        if isinstance(component, EmbeddingComponent):
            serialized["provider"] = NON_SERIALIZABLE_PLACEHOLDER

        if isinstance(component, VectorStoreComponent):
            serialized["store"] = NON_SERIALIZABLE_PLACEHOLDER

        return serialized

    @staticmethod
    def _normalize_component_data(
        component_name: str,
        component_data: dict[str, Any],
        providers: dict[str, Any],
        tool_handlers: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_data = dict(component_data)

        if component_name == ConversationComponent.__name__:
            normalized_data["messages"] = [
                WorldSerializer._message_from_dict(msg)
                for msg in normalized_data.get("messages", [])
            ]

        if component_name == PendingToolCallsComponent.__name__:
            normalized_data["tool_calls"] = [
                ToolCall(**tool_call)
                for tool_call in normalized_data.get("tool_calls", [])
            ]

        if component_name == ToolRegistryComponent.__name__:
            normalized_data["tools"] = {
                name: ToolSchema(**schema)
                for name, schema in normalized_data.get("tools", {}).items()
            }
            handlers_value = normalized_data.get("handlers")
            if handlers_value == NON_SERIALIZABLE_PLACEHOLDER:
                normalized_data["handlers"] = tool_handlers

        if component_name == CollaborationComponent.__name__:
            normalized_data["peers"] = [
                EntityId(int(peer)) for peer in normalized_data.get("peers", [])
            ]
            normalized_data["inbox"] = [
                (EntityId(int(sender)), WorldSerializer._message_from_dict(message))
                for sender, message in normalized_data.get("inbox", [])
            ]

        if component_name == OwnerComponent.__name__:
            normalized_data["owner_id"] = EntityId(int(normalized_data["owner_id"]))

        if component_name == ToolApprovalComponent.__name__:
            policy_value = normalized_data.get("policy")
            if isinstance(policy_value, str):
                normalized_data["policy"] = ApprovalPolicy(policy_value)

        if component_name == LLMComponent.__name__:
            provider_value = normalized_data.get("provider")
            if provider_value == NON_SERIALIZABLE_PLACEHOLDER:
                model = normalized_data.get("model")
                # Ensure model is a string for dict lookup
                model_str: str = model if isinstance(model, str) else "default"
                provider = providers.get(model_str, providers.get("default"))
                if provider is None:
                    raise ValueError(
                        f"No provider configured for model '{model}' and no default provider found"
                    )
                normalized_data["provider"] = provider

        return normalized_data

    @staticmethod
    def _message_from_dict(data: dict[str, Any]) -> Message:
        tool_calls_data = data.get("tool_calls")
        tool_calls = None
        if tool_calls_data is not None:
            tool_calls = [ToolCall(**tool_call) for tool_call in tool_calls_data]

        return Message(
            role=data["role"],
            content=data["content"],
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
        )
