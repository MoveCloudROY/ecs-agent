from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ecs_agent.components import (
    CheckpointComponent,
    CompactionConfigComponent,
    ContextBudgetConfig,
    ContextCacheComponent,
    ContextEntry,
    CurrentCompactionSummaryComponent,
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
    OwnerComponent,
    PendingToolCallsComponent,
    PlanComponent,
    PlanSearchComponent,
    UserPromptConfigComponent,
    RenderedSystemPromptComponent,
    RenderedUserPromptComponent,
    PromptContextQueueComponent,
    PromptContextReservationComponent,
    RAGTriggerComponent,
    RunnerStateComponent,
    ResponsesAPIStateComponent,
    SandboxConfigComponent,
    ScratchbookIndexComponent,
    ScratchbookRefComponent,
    StreamingComponent,
    SubagentNotificationQueueComponent,
    SystemPromptComponent,
    SubagentRegistryComponent,
    SubagentSessionTableComponent,
    SubagentWaitComponent,
    TerminalComponent,
    ToolApprovalComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
    VectorStoreComponent,
)
from ecs_agent.core.world import World
from ecs_agent.prompts.contracts import (
    PlaceholderSpec,
    SystemPromptConfigSpec,
    PromptTemplateSource,
    TriggerSpec,
)
from ecs_agent.types import (
    ApprovalPolicy,
    CachedToolResultRef,
    EntityId,
    FileRefPart,
    ImageUrlPart,
    Message,
    MessagePart,
    SubagentNotificationRecord,
    SubagentSessionRecord,
    ToolCall,
    ToolSchema,
)

NON_SERIALIZABLE_PLACEHOLDER = "<non-serializable>"
LEGACY_COMPACTION_SUMMARY_PREFIX = "Previous conversation summary: "

EPHEMERAL_COMPONENT_TYPES: tuple[type[Any], ...] = ()

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
    SubagentNotificationQueueComponent.__name__: SubagentNotificationQueueComponent,
    SubagentRegistryComponent.__name__: SubagentRegistryComponent,
    SubagentSessionTableComponent.__name__: SubagentSessionTableComponent,
    SubagentWaitComponent.__name__: SubagentWaitComponent,
    CheckpointComponent.__name__: CheckpointComponent,
    ContextBudgetConfig.__name__: ContextBudgetConfig,
    CompactionConfigComponent.__name__: CompactionConfigComponent,
    ContextCacheComponent.__name__: ContextCacheComponent,
    CurrentCompactionSummaryComponent.__name__: CurrentCompactionSummaryComponent,
    ConversationArchiveComponent.__name__: ConversationArchiveComponent,
    RunnerStateComponent.__name__: RunnerStateComponent,
    MessageBusConfigComponent.__name__: MessageBusConfigComponent,
    MessageBusSubscriptionComponent.__name__: MessageBusSubscriptionComponent,
    MessageBusConversationComponent.__name__: MessageBusConversationComponent,
    ScratchbookIndexComponent.__name__: ScratchbookIndexComponent,
    ScratchbookRefComponent.__name__: ScratchbookRefComponent,
    UserPromptConfigComponent.__name__: UserPromptConfigComponent,
    SystemPromptConfigSpec.__name__: SystemPromptConfigSpec,
    RenderedSystemPromptComponent.__name__: RenderedSystemPromptComponent,
    RenderedUserPromptComponent.__name__: RenderedUserPromptComponent,
    ContextEntry.__name__: ContextEntry,
    PromptContextQueueComponent.__name__: PromptContextQueueComponent,
    PromptContextReservationComponent.__name__: PromptContextReservationComponent,
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
                if isinstance(component, EPHEMERAL_COMPONENT_TYPES):
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
            "world_name": world._name,
        }

    @staticmethod
    def from_dict(
        data: dict[str, Any],
        providers: dict[str, Any],
        tool_handlers: dict[str, Any],
    ) -> World:
        world = World(name=data.get("world_name"))

        entities_data = data.get("entities", {})
        for entity_id_str, serialized_components in entities_data.items():
            entity_id = EntityId(int(entity_id_str))
            for component_name, component_data in serialized_components.items():
                component_type = COMPONENT_REGISTRY.get(component_name)
                if component_type is None:
                    continue

                current_summary: str | None = None
                if component_name == ConversationComponent.__name__:
                    current_summary = (
                        WorldSerializer._current_compaction_summary_from_messages(
                            [
                                WorldSerializer._message_from_dict(message_data)
                                for message_data in component_data.get("messages", [])
                            ]
                        )
                    )

                normalized_data = WorldSerializer._normalize_component_data(
                    component_name,
                    component_data,
                    providers,
                    tool_handlers,
                )
                world.add_component(entity_id, component_type(**normalized_data))
                if current_summary is not None:
                    world.add_component(
                        entity_id,
                        CurrentCompactionSummaryComponent(summary=current_summary),
                    )

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
        if isinstance(component, SubagentWaitComponent):
            return {
                "session_ids": component.session_ids,
                "timeout": component.timeout,
                "future": None,
                "started_at": component.started_at,
            }

        if isinstance(component, SubagentNotificationQueueComponent):
            return {
                "notifications": [
                    asdict(notification) for notification in component.notifications
                ]
            }

        if isinstance(component, ContextCacheComponent):
            return {
                "cached_tool_results": [
                    {
                        "tool_call_id": ref.tool_call_id,
                        "artifact_path": ref.artifact_path,
                        "summary": ref.summary,
                        "original_content": ref.original_content,
                    }
                    for ref in component.cached_tool_results
                ]
            }

        serialized = asdict(component)

        if isinstance(component, LLMComponent):
            serialized["model"] = getattr(component.model, "model_id", "default")
            serialized["pending_model"] = None

        if isinstance(component, ToolRegistryComponent):
            serialized["handlers"] = NON_SERIALIZABLE_PLACEHOLDER

        if isinstance(component, UserPromptConfigComponent):
            serialized["script_handlers"] = NON_SERIALIZABLE_PLACEHOLDER

        if isinstance(component, EmbeddingComponent):
            serialized["provider"] = NON_SERIALIZABLE_PLACEHOLDER

        if isinstance(component, VectorStoreComponent):
            serialized["store"] = NON_SERIALIZABLE_PLACEHOLDER

        if isinstance(component, SubagentRegistryComponent):
            subagents_serialized: dict[str, Any] = {}
            for name, cfg in component.subagents.items():
                subagents_serialized[name] = {
                    "name": cfg.name,
                    "model": getattr(cfg.model, "model_id", "default"),
                    "description": cfg.description,
                    "system_prompt": cfg.system_prompt,
                    "skills": list(cfg.skills),
                    "max_ticks": cfg.max_ticks,
                    "inheritance_policy": {
                        "inherit_tools": cfg.inheritance_policy.inherit_tools,
                        "inherit_system_prompt": cfg.inheritance_policy.inherit_system_prompt,
                    },
                    "provider": NON_SERIALIZABLE_PLACEHOLDER,
                }
            serialized["subagents"] = subagents_serialized

        if isinstance(component, ConversationComponent):
            serialized["messages"] = [
                WorldSerializer._message_to_dict(message)
                for message in component.messages
            ]

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
            serialized["messages"] = [
                WorldSerializer._message_to_dict(message)
                for message in component.messages
            ]

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
            deserialized_messages = [
                WorldSerializer._message_from_dict(msg)
                for msg in normalized_data.get("messages", [])
            ]
            normalized_data["messages"] = (
                WorldSerializer._strip_legacy_compaction_messages(deserialized_messages)
            )

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
                "script_handlers",
                "enable_context_pool",
                "context_pool_max_chars",
            }
            unknown_fields = sorted(set(normalized_data.keys()) - allowed_fields)
            if unknown_fields:
                raise ValueError(
                    "UserPromptConfigComponent contains unsupported fields: "
                    f"{', '.join(unknown_fields)}"
                )
            # Reconstruct TriggerSpec objects from serialized dicts
            triggers_data = normalized_data.get("triggers", [])
            normalized_data["triggers"] = [
                TriggerSpec(**trigger_data)
                if isinstance(trigger_data, dict)
                else trigger_data
                for trigger_data in triggers_data
            ]
            script_handlers_value = normalized_data.get("script_handlers")
            if script_handlers_value == NON_SERIALIZABLE_PLACEHOLDER:
                normalized_data["script_handlers"] = {}

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
            model_str_or_obj = normalized_data.get("model")
            if not callable(getattr(model_str_or_obj, "complete", None)):
                # model field holds a model_id string (serialized state)
                model_str: str = model_str_or_obj if isinstance(model_str_or_obj, str) else "default"
                model_obj = providers.get(model_str, providers.get("default"))
                if model_obj is None:
                    raise ValueError(
                        f"No model configured for model_id '{model_str}' and no default model found"
                    )
                normalized_data["model"] = model_obj
            # Drop legacy "provider" key if present in saved state
            normalized_data.pop("provider", None)

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
                    # Try to get model from models dict
                    model_key = config_data.get("model")
                    model_str = model_key if isinstance(model_key, str) else "default"
                    model = providers.get(model_str, providers.get("default"))
                    if model is None:
                        raise ValueError(
                            f"No model configured for subagent '{name}' model '{model_str}'"
                        )
                else:
                    model = provider_value

                subagent_config = SubagentConfig(
                    name=config_data["name"],
                    description=config_data.get("description", ""),
                    model=model,
                    system_prompt=config_data.get("system_prompt", ""),
                    skills=config_data.get("skills", []),
                    max_ticks=config_data.get("max_ticks", 10),
                    inheritance_policy=inheritance_policy,
                )
                subagents_dict[name] = subagent_config
            normalized_data["subagents"] = subagents_dict

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

        if component_name == SubagentNotificationQueueComponent.__name__:
            notifications = []
            for notification_data in normalized_data.get("notifications", []):
                if isinstance(notification_data, dict):
                    notifications.append(
                        SubagentNotificationRecord(**notification_data)
                    )
                else:
                    notifications.append(notification_data)
            normalized_data["notifications"] = notifications

        if component_name == SubagentWaitComponent.__name__:
            normalized_data["future"] = None

        if component_name == PromptContextQueueComponent.__name__:
            entries_data = normalized_data.get("entries", [])
            normalized_data["entries"] = [
                ContextEntry(**entry_data)
                if isinstance(entry_data, dict)
                else entry_data
                for entry_data in entries_data
            ]

        if component_name == PromptContextReservationComponent.__name__:
            entries_data = normalized_data.get("reserved_entries", [])
            normalized_data["reserved_entries"] = [
                ContextEntry(**entry_data)
                if isinstance(entry_data, dict)
                else entry_data
                for entry_data in entries_data
            ]

        if component_name == ContextCacheComponent.__name__:
            cached_tool_results_data = normalized_data.get("cached_tool_results", [])
            normalized_data["cached_tool_results"] = [
                CachedToolResultRef(**cached_tool_result_data)
                if isinstance(cached_tool_result_data, dict)
                else cached_tool_result_data
                for cached_tool_result_data in cached_tool_results_data
            ]

        return normalized_data

    @staticmethod
    def _message_from_dict(data: dict[str, Any]) -> Message:
        tool_calls_data = data.get("tool_calls")
        tool_calls = None
        if tool_calls_data is not None:
            tool_calls = [ToolCall(**tool_call) for tool_call in tool_calls_data]

        parts_data = data.get("parts")
        parts = None
        if parts_data is not None:
            parts = [
                WorldSerializer._message_part_from_dict(part) for part in parts_data
            ]

        return Message(
            role=data["role"],
            content=data.get("content", ""),
            parts=parts,
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            compaction_metadata=data.get("compaction_metadata"),
        )

    @staticmethod
    def _message_to_dict(message: Message) -> dict[str, Any]:
        serialized: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
            "tool_calls": None,
            "tool_call_id": message.tool_call_id,
        }

        if message.parts is not None:
            serialized["parts"] = [
                WorldSerializer._message_part_to_dict(part) for part in message.parts
            ]

        if message.tool_calls is not None:
            serialized["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                }
                for tool_call in message.tool_calls
            ]

        if message.compaction_metadata is not None:
            serialized["compaction_metadata"] = message.compaction_metadata

        return serialized

    @staticmethod
    def _strip_legacy_compaction_messages(messages: list[Message]) -> list[Message]:
        return [
            message
            for message in messages
            if not WorldSerializer._is_legacy_compaction_message(message)
        ]

    @staticmethod
    def _current_compaction_summary_from_messages(
        messages: list[Message],
    ) -> str | None:
        current_summary: str | None = None
        for message in messages:
            if not WorldSerializer._is_legacy_compaction_message(message):
                continue
            current_summary = message.content.removeprefix(
                LEGACY_COMPACTION_SUMMARY_PREFIX
            )
        return current_summary

    @staticmethod
    def _is_legacy_compaction_message(message: Message) -> bool:
        return message.role == "compaction" and message.content.startswith(
            LEGACY_COMPACTION_SUMMARY_PREFIX
        )

    @staticmethod
    def _message_part_to_dict(part: MessagePart) -> dict[str, Any]:

        if isinstance(part, ImageUrlPart):
            return {
                "type": "image_url",
                "url": part.url,
                "detail": part.detail,
            }
        if isinstance(part, FileRefPart):
            return {
                "type": "file_ref",
                "file_id": part.file_id,
                "filename": part.filename,
            }
        raise ValueError(f"Unsupported message part type: {type(part).__name__}")

    @staticmethod
    def _message_part_from_dict(data: dict[str, Any]) -> MessagePart:
        part_type = data.get("type")

        if part_type == "image_url":
            return ImageUrlPart(url=data["url"], detail=data.get("detail"))
        if part_type == "file_ref":
            return FileRefPart(file_id=data["file_id"], filename=data.get("filename"))
        raise ValueError(f"Unsupported message part discriminator: {part_type}")
