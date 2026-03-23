from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ecs_agent.components import (
    CheckpointComponent,
    CompactionConfigComponent,
    ConversationArchiveComponent,
    ConversationComponent,
    ConversationTreeComponent,
    EmbeddingComponent,
    ErrorComponent,
    KVStoreComponent,
    LLMComponent,
    MessageBusConfigComponent,
    MessageBusConversationComponent,
    MessageBusSubscriptionComponent,
    OneShotContextPoolComponent,
    OwnerComponent,
    PendingToolCallsComponent,
    PlanComponent,
    PlanSearchComponent,
    UserPromptConfigComponent,
    RenderedSystemPromptComponent,
    RenderedUserPromptComponent,
    RAGTriggerComponent,
    RunnerStateComponent,
    ResponsesAPIStateComponent,
    SandboxConfigComponent,
    ScratchbookIndexComponent,
    ScratchbookRefComponent,
    StreamingComponent,
    SystemPromptComponent,
    SubagentRegistryComponent,
    SubagentSessionTableComponent,
    TaskComponent,
    TerminalComponent,
    ToolApprovalComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
    TurnStateComponent,
    VectorStoreComponent,
)
from ecs_agent.core.world import World
from ecs_agent.prompts.contracts import (
    PlaceholderSpec,
    SystemPromptConfigSpec,
    PromptTemplateSource,
)
from ecs_agent.types import ApprovalPolicy, EntityId, Message, ToolCall, ToolSchema

NON_SERIALIZABLE_PLACEHOLDER = "<non-serializable>"

COMPONENT_REGISTRY: dict[str, type[Any]] = {
    LLMComponent.__name__: LLMComponent,
    ConversationComponent.__name__: ConversationComponent,
    ConversationTreeComponent.__name__: ConversationTreeComponent,
    PlanComponent.__name__: PlanComponent,
    ToolRegistryComponent.__name__: ToolRegistryComponent,
    PendingToolCallsComponent.__name__: PendingToolCallsComponent,
    ToolResultsComponent.__name__: ToolResultsComponent,
    KVStoreComponent.__name__: KVStoreComponent,
    OwnerComponent.__name__: OwnerComponent,
    ErrorComponent.__name__: ErrorComponent,
    TerminalComponent.__name__: TerminalComponent,
    SystemPromptComponent.__name__: SystemPromptComponent,
    ToolApprovalComponent.__name__: ToolApprovalComponent,
    SandboxConfigComponent.__name__: SandboxConfigComponent,
    PlanSearchComponent.__name__: PlanSearchComponent,
    RAGTriggerComponent.__name__: RAGTriggerComponent,
    EmbeddingComponent.__name__: EmbeddingComponent,
    ResponsesAPIStateComponent.__name__: ResponsesAPIStateComponent,
    VectorStoreComponent.__name__: VectorStoreComponent,
    StreamingComponent.__name__: StreamingComponent,
    SubagentRegistryComponent.__name__: SubagentRegistryComponent,
    SubagentSessionTableComponent.__name__: SubagentSessionTableComponent,
    CheckpointComponent.__name__: CheckpointComponent,
    CompactionConfigComponent.__name__: CompactionConfigComponent,
    ConversationArchiveComponent.__name__: ConversationArchiveComponent,
    RunnerStateComponent.__name__: RunnerStateComponent,
    MessageBusConfigComponent.__name__: MessageBusConfigComponent,
    MessageBusSubscriptionComponent.__name__: MessageBusSubscriptionComponent,
    MessageBusConversationComponent.__name__: MessageBusConversationComponent,
    ScratchbookIndexComponent.__name__: ScratchbookIndexComponent,
    ScratchbookRefComponent.__name__: ScratchbookRefComponent,
    TaskComponent.__name__: TaskComponent,
    UserPromptConfigComponent.__name__: UserPromptConfigComponent,
    SystemPromptConfigSpec.__name__: SystemPromptConfigSpec,
    RenderedSystemPromptComponent.__name__: RenderedSystemPromptComponent,
    RenderedUserPromptComponent.__name__: RenderedUserPromptComponent,
    OneShotContextPoolComponent.__name__: OneShotContextPoolComponent,
    TurnStateComponent.__name__: TurnStateComponent,
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

        # Serialize entity registry
        entity_registry = {
            name: int(entity_id) for name, entity_id in world._entity_registry.items()
        }
        entity_tags = {
            tag: sorted([int(eid) for eid in entity_ids])
            for tag, entity_ids in world._entity_tags.items()
        }

        return {
            "next_entity_id": next_entity_id,
            "entities": entities,
            "_entity_registry": entity_registry,
            "_entity_tags": entity_tags,
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

        # Restore entity registry (backward compatible)
        entity_registry_data = data.get("_entity_registry", {})
        world._entity_registry = {
            name: EntityId(int(eid)) for name, eid in entity_registry_data.items()
        }

        entity_tags_data = data.get("_entity_tags", {})
        world._entity_tags = {
            tag: set(EntityId(int(eid)) for eid in eids)
            for tag, eids in entity_tags_data.items()
        }

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
        """Serialize component to dict, handling special types.

        SERIALIZATION BOUNDARY:
        - SERIALIZED: Config (timeouts, buffer sizes), subscriptions, conversation history
        - NOT SERIALIZED: Runtime queues, pending futures, in-flight requests
        """
        serialized = asdict(component)

        if isinstance(component, LLMComponent):
            serialized["provider"] = NON_SERIALIZABLE_PLACEHOLDER

        if isinstance(component, ToolRegistryComponent):
            serialized["handlers"] = NON_SERIALIZABLE_PLACEHOLDER

        if isinstance(component, EmbeddingComponent):
            serialized["provider"] = NON_SERIALIZABLE_PLACEHOLDER

        if isinstance(component, VectorStoreComponent):
            serialized["store"] = NON_SERIALIZABLE_PLACEHOLDER

        # MessageBusSubscriptionComponent: convert set[str] to list[str]
        if isinstance(component, MessageBusSubscriptionComponent):
            subscriptions_dict = {}
            for topic, entity_ids in serialized.get("subscriptions", {}).items():
                # Convert set of entity IDs (as strings) to list
                subscriptions_dict[topic] = sorted(list(entity_ids))
            serialized["subscriptions"] = subscriptions_dict

        # MessageBusConversationComponent: convert EntityId to int
        if isinstance(component, MessageBusConversationComponent):
            serialized["entity_id"] = int(serialized["entity_id"])

        # TaskComponent: convert EntityId assigned_agent to int for JSON serialization
        if isinstance(component, TaskComponent):
            # Convert assigned_agent if it's an EntityId (which is an int at runtime)
            assigned_agent = serialized.get("assigned_agent")
            if (
                assigned_agent is not None
                and isinstance(assigned_agent, int)
                and not isinstance(assigned_agent, bool)
            ):
                # It's an EntityId (NewType over int), keep as int for JSON
                pass
            # If string or None, leave as-is (JSON-serializable)

            # Convert TaskStatus enum to string
            status = serialized.get("status")
            if status is not None and hasattr(status, "value"):  # Enum
                serialized["status"] = status.value

        # ScratchbookIndexComponent: convert ScratchbookRef objects to dicts
        if isinstance(component, ScratchbookIndexComponent):
            artifacts_dict = {}
            for artifact_id, artifact_ref in serialized.get("artifacts", {}).items():
                # If it's a ScratchbookRef dataclass, convert to dict
                if hasattr(artifact_ref, "__dataclass_fields__"):
                    artifacts_dict[artifact_id] = asdict(artifact_ref)
                else:
                    # Already a dict
                    artifacts_dict[artifact_id] = artifact_ref
            serialized["artifacts"] = artifacts_dict

        # SubagentSessionTableComponent: convert SubagentSessionRecord objects to dicts with EntityId → int
        if isinstance(component, SubagentSessionTableComponent):
            sessions_dict = {}
            for session_id, session_record in serialized.get("sessions", {}).items():
                # If it's a SubagentSessionRecord dataclass, convert to dict
                if hasattr(session_record, "__dataclass_fields__"):
                    session_dict = asdict(session_record)
                    # Convert parent_entity_id (EntityId) to int for JSON serialization
                    parent_entity_id = session_dict.get("parent_entity_id")
                    if isinstance(parent_entity_id, int):
                        session_dict["parent_entity_id"] = int(parent_entity_id)
                    sessions_dict[session_id] = session_dict
                else:
                    # Already a dict
                    sessions_dict[session_id] = session_record
            serialized["sessions"] = sessions_dict
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

        if component_name == OwnerComponent.__name__:
            normalized_data["owner_id"] = EntityId(int(normalized_data["owner_id"]))

        if component_name == ToolApprovalComponent.__name__:
            policy_value = normalized_data.get("policy")
            if isinstance(policy_value, str):
                normalized_data["policy"] = ApprovalPolicy(policy_value)

        if component_name == UserPromptConfigComponent.__name__:
            allowed_fields = {
                "triggers",
                "enable_context_pool",
                "context_pool_max_chars",
            }
            unknown_fields = sorted(set(normalized_data.keys()) - allowed_fields)
            if unknown_fields:
                raise ValueError(
                    "UserPromptConfigComponent contains unsupported fields: "
                    f"{', '.join(unknown_fields)}"
                )

        if component_name == SystemPromptConfigSpec.__name__:
            template_source_data = normalized_data.get("template_source")
            if isinstance(template_source_data, dict):
                normalized_data["template_source"] = PromptTemplateSource(
                    **template_source_data
                )

            placeholders_data = normalized_data.get("placeholders", [])
            normalized_placeholders: list[PlaceholderSpec] = []
            for placeholder in placeholders_data:
                if isinstance(placeholder, dict):
                    normalized_placeholders.append(PlaceholderSpec(**placeholder))
                else:
                    normalized_placeholders.append(placeholder)
            normalized_data["placeholders"] = normalized_placeholders

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

        # MessageBusSubscriptionComponent: convert list[str] back to set[str]
        if component_name == MessageBusSubscriptionComponent.__name__:
            subscriptions_dict = {}
            for topic, entity_ids in normalized_data.get("subscriptions", {}).items():
                # Convert list back to set
                subscriptions_dict[topic] = set(entity_ids)
            normalized_data["subscriptions"] = subscriptions_dict

        # MessageBusConversationComponent: convert int to EntityId, reconstruct Messages
        if component_name == MessageBusConversationComponent.__name__:
            normalized_data["entity_id"] = EntityId(int(normalized_data["entity_id"]))
            normalized_data["messages"] = [
                WorldSerializer._message_from_dict(msg)
                for msg in normalized_data.get("messages", [])
            ]

        # SubagentRegistryComponent: reconstruct SubagentConfig and InheritancePolicy
        if component_name == SubagentRegistryComponent.__name__:
            from ecs_agent.types import SubagentConfig, InheritancePolicy

            subagents_dict = {}
            for name, config_data in normalized_data.get("subagents", {}).items():
                # Reconstruct InheritancePolicy from dict
                policy_data = config_data.get("inheritance_policy", {})
                inheritance_policy = InheritancePolicy(**policy_data)

                # Reconstruct SubagentConfig with InheritancePolicy
                # Provider needs to be resolved (currently dict or placeholder)
                provider_value = config_data.get("provider")
                if provider_value == NON_SERIALIZABLE_PLACEHOLDER:
                    # Try to get provider from providers dict
                    model = config_data.get("model")
                    model_str = model if isinstance(model, str) else "default"
                    provider = providers.get(model_str, providers.get("default"))
                    if provider is None:
                        raise ValueError(
                            f"No provider configured for subagent '{name}' model '{model}'"
                        )
                else:
                    provider = provider_value

                subagent_config = SubagentConfig(
                    name=config_data["name"],
                    provider=provider,
                    model=config_data["model"],
                    system_prompt=config_data.get("system_prompt", ""),
                    skills=config_data.get("skills", []),
                    max_ticks=config_data.get("max_ticks", 10),
                    inheritance_policy=inheritance_policy,
                )
                subagents_dict[name] = subagent_config
            normalized_data["subagents"] = subagents_dict

        # TaskComponent: convert assigned_agent int to EntityId if needed
        if component_name == TaskComponent.__name__:
            assigned_agent_value = normalized_data.get("assigned_agent")
            if isinstance(assigned_agent_value, int):
                # EntityId stored as int, reconstruct it
                normalized_data["assigned_agent"] = EntityId(assigned_agent_value)
            # If string or None, leave as-is

            # Convert status string to TaskStatus enum if needed
            status_value = normalized_data.get("status")
            if isinstance(status_value, str):
                from ecs_agent.types import TaskStatus

                normalized_data["status"] = TaskStatus(status_value)

        # ScratchbookRefComponent: no special handling needed (all fields are primitives)
        # ScratchbookRefComponent fields (artifact_id, category, content_hash, timestamp) are all strings

        # ScratchbookIndexComponent: reconstruct ScratchbookRef objects in artifacts dict
        if component_name == ScratchbookIndexComponent.__name__:
            from ecs_agent.types import ScratchbookRef

            artifacts_dict = {}
            for artifact_id, artifact_data in normalized_data.get(
                "artifacts", {}
            ).items():
                # If artifact_data is a dict, reconstruct it as ScratchbookRef
                if isinstance(artifact_data, dict):
                    artifacts_dict[artifact_id] = ScratchbookRef(**artifact_data)
                else:
                    # Already a ScratchbookRef object
                    artifacts_dict[artifact_id] = artifact_data
            normalized_data["artifacts"] = artifacts_dict

        # SubagentSessionTableComponent: reconstruct SubagentSessionRecord objects in sessions dict
        if component_name == SubagentSessionTableComponent.__name__:
            from ecs_agent.types import SubagentSessionRecord

            sessions_dict = {}
            for session_id, session_data in normalized_data.get("sessions", {}).items():
                # If session_data is a dict, reconstruct it as SubagentSessionRecord
                if isinstance(session_data, dict):
                    # Convert parent_entity_id if it's an int
                    parent_entity_id_value = session_data.get("parent_entity_id")
                    if isinstance(parent_entity_id_value, int):
                        session_data["parent_entity_id"] = EntityId(
                            parent_entity_id_value
                        )
                    sessions_dict[session_id] = SubagentSessionRecord(**session_data)
                else:
                    # Already a SubagentSessionRecord object
                    sessions_dict[session_id] = session_data
            normalized_data["sessions"] = sessions_dict

        if component_name == SystemPromptComponent.__name__:
            from ecs_agent.prompts.contracts import PromptSectionSpec

            sections_list = []
            for section_data in normalized_data.get("sections", []):
                if isinstance(section_data, dict):
                    sections_list.append(PromptSectionSpec(**section_data))
                else:
                    sections_list.append(section_data)
            normalized_data["sections"] = sections_list

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
