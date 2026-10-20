"""World serialization for save and load."""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, cast

from ecs_agent.components import (
    AgentTurnComponent,
    CheckpointComponent,
    ChildStubComponent,
    CompactionConfigComponent,
    ContextTrimConfig,
    ContextCacheComponent,
    ContextEntry,
    CurrentCompactionSummaryComponent,
    ConversationArchiveComponent,
    ConversationComponent,
    SessionRefComponent,
    SessionPolicyComponent,
    ConversationTreeComponent,
    EmbeddingComponent,
    EntityRegistryComponent,
    ErrorComponent,
    InterruptionComponent,
    KVStoreComponent,
    LLMComponent,
    MessageBusConfigComponent,
    MessageBusConversationComponent,
    MessageBusSubscriptionComponent,
    OwnerComponent,
    PendingToolCallsComponent,
    PermissionComponent,
    PhaseApprovalsComponent,
    PhaseComponent,
    PhaseDefinitionComponent,
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
    SkillComponent,
    SkillMetadata,
    StreamingComponent,
    SubagentNotificationQueueComponent,
    SystemPromptComponent,
    SubagentRegistryComponent,
    SubagentSessionTableComponent,
    SubagentWaitComponent,
    TerminalComponent,
    TodoItem,
    TodoListComponent,
    TokenUsageComponent,
    ToolApprovalComponent,
    ToolExecutionConfigComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
    ToolRuntimeStateComponent,
    ToolStateNamespace,
    UserInputComponent,
    VectorStoreComponent,
    WorkspaceBindingComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
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
    FinishReason,
    ImageUrlPart,
    Message,
    MessagePart,
    ProviderOutput,
    SubagentNotificationRecord,
    SubagentSessionRecord,
    ToolCall,
    ToolSchema,
)
from ecs_agent.outcomes import GenerationStatus

logger = get_logger(__name__)

NON_SERIALIZABLE_PLACEHOLDER = "<non-serializable>"
LEGACY_COMPACTION_SUMMARY_PREFIX = "Previous conversation summary: "

# Bumped whenever the checkpoint payload shape changes incompatibly.
# Loading a payload with a NEWER version than this fails loudly instead of
# silently dropping data it does not understand.
CHECKPOINT_SCHEMA_VERSION = 4

# Runtime signals that are intentionally NOT serialized. Every component
# dataclass must appear in exactly one of COMPONENT_REGISTRY,
# EPHEMERAL_COMPONENT_TYPES, or NON_COMPONENT_VALUE_TYPES — enforced by
# tests/test_serialization.py::test_component_registry_and_ephemeral_partition_cover_all_components.
EPHEMERAL_COMPONENT_TYPES: tuple[type[Any], ...] = (
    # Transient interrupt signal; a resumed world must not re-interrupt.
    InterruptionComponent,
    # Internal tool runtime state with arbitrary object values; tools
    # rebuild it lazily (e.g. re-read before edit) after resume.
    ToolRuntimeStateComponent,
    # Definitions are re-bound at resume time (bind_phase_graph).
    PhaseDefinitionComponent,
)

# Dataclasses in components.definitions that are nested value objects,
# never attached to entities directly.
NON_COMPONENT_VALUE_TYPES: tuple[type[Any], ...] = (
    SkillMetadata,
    ToolStateNamespace,
    TodoItem,
)

COMPONENT_REGISTRY: dict[str, type[Any]] = {
    AgentTurnComponent.__name__: AgentTurnComponent,
    LLMComponent.__name__: LLMComponent,
    ConversationComponent.__name__: ConversationComponent,
    SessionRefComponent.__name__: SessionRefComponent,
    SessionPolicyComponent.__name__: SessionPolicyComponent,
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
    ToolExecutionConfigComponent.__name__: ToolExecutionConfigComponent,
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
    ContextTrimConfig.__name__: ContextTrimConfig,
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
    PhaseComponent.__name__: PhaseComponent,
    PhaseApprovalsComponent.__name__: PhaseApprovalsComponent,
    TodoListComponent.__name__: TodoListComponent,
    ChildStubComponent.__name__: ChildStubComponent,
    EntityRegistryComponent.__name__: EntityRegistryComponent,
    PermissionComponent.__name__: PermissionComponent,
    SkillComponent.__name__: SkillComponent,
    TokenUsageComponent.__name__: TokenUsageComponent,
    UserInputComponent.__name__: UserInputComponent,
    WorkspaceBindingComponent.__name__: WorkspaceBindingComponent,
}


class WorldSerializer:
    @staticmethod
    def to_dict(
        world: World,
        *,
        exclude_component_types: tuple[type[Any], ...] = (),
    ) -> dict[str, Any]:
        """Serialize the world into a JSON-compatible payload.

        Args:
            world: World instance to serialize.
            exclude_component_types: Component types omitted from this
                payload only (e.g. TerminalComponent for resume-friendly
                checkpoints) — the live world is never mutated.
        """
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
                if exclude_component_types and isinstance(
                    component, exclude_component_types
                ):
                    continue
                serialized_components[component_type.__name__] = (
                    WorldSerializer._serialize_component(component)
                )
            entities[str(entity_id)] = serialized_components

        next_entity_id = world._entity_gen._counter + 1

        # All known entities, including component-less ones — has_entity()
        # must keep answering correctly after a restore.
        all_entity_ids = set(world._entity_ids) | entity_ids

        # Serialize entity registry
        entity_registry = {
            name: entity_id for name, entity_id in world._entity_registry.items()
        }
        entity_tags = {
            tag: sorted([eid for eid in entity_ids])
            for tag, entity_ids in world._entity_tags.items()
        }

        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "next_entity_id": next_entity_id,
            "entities": entities,
            "entity_ids": sorted(int(eid) for eid in all_entity_ids),
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
        """Rebuild a World from a serialized payload.

        Raises:
            ValueError: If the payload's schema_version is newer than this
                build supports (loading it would silently drop data).
        """
        version = int(data.get("schema_version", 1))
        if version > CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Checkpoint schema_version {version} is newer than the "
                f"supported version {CHECKPOINT_SCHEMA_VERSION}; refusing to "
                "load it partially"
            )

        world = World(name=data.get("world_name"))

        entities_data = data.get("entities", {})
        for entity_id_str, serialized_components in entities_data.items():
            entity_id = EntityId(int(entity_id_str))
            for component_name, component_data in serialized_components.items():
                component_type = COMPONENT_REGISTRY.get(component_name)
                if component_type is None:
                    logger.warning(
                        "serialization_unknown_component",
                        component=component_name,
                        entity_id=int(entity_id),
                    )
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
                if version < 2 and component_name == PermissionComponent.__name__:
                    # Version 1 interpreted a missing/empty allowlist as unrestricted.
                    if not normalized_data.get("allowed_tools"):
                        normalized_data["allowed_tools"] = None
                world.add_component(entity_id, component_type(**normalized_data))
                if current_summary is not None:
                    world.add_component(
                        entity_id,
                        CurrentCompactionSummaryComponent(summary=current_summary),
                    )

        # A canonical config can be rendered again against restored bindings.
        # Never let a persisted derived body outrank its current inputs.
        for entity, _ in world.query(SystemPromptConfigSpec):
            world.remove_component(entity, RenderedSystemPromptComponent)

        for entity, _ in world.query(LLMComponent):
            turn = world.get_component(entity, AgentTurnComponent)
            if version < 3 and turn is None and not world.has_component(entity, ChildStubComponent):
                world.add_component(entity, AgentTurnComponent(
                    turn_id="", status="legacy",
                    reason="Legacy checkpoint has no turn boundary; explicitly adopt context before managed continuation",
                ))
            elif turn is not None:
                if version < 4:
                    # Older managed checkpoints had no script execution state.
                    # A matching accepted render proves completion; otherwise a
                    # script may already have changed external state and must
                    # not be inferred safe to replay from a paused turn alone.
                    config = world.get_component(entity, UserPromptConfigComponent)
                    rendered = world.get_component(entity, RenderedUserPromptComponent)
                    conversation = world.get_component(entity, ConversationComponent)
                    accepted_render = False
                    if rendered is not None and conversation is not None:
                        index = rendered.source_message_index
                        if index is not None and 0 <= index < len(conversation.messages):
                            source = conversation.messages[index]
                            accepted_render = (
                                source.role == "user"
                                and source.turn_id == turn.turn_id
                                and hashlib.sha256(source.content.encode("utf-8")).hexdigest() == rendered.source_fingerprint
                            )
                    if accepted_render or turn.status in {"completed", "abandoned"}:
                        turn.input_normalization_state = "completed"
                    elif config is not None and any(spec.action == "script" for spec in config.triggers):
                        turn.input_normalization_state = "running"
                if turn.input_normalization_state == "running":
                    turn.status = "blocked"
                    turn.reason = "Input normalization completion is unconfirmed"
                elif turn.unconfirmed_tool_calls:
                    turn.status = "blocked"
                    turn.reason = "Tool completion is unconfirmed: " + ", ".join(turn.unconfirmed_tool_calls)
                elif turn.status == "running":
                    terminal = world.get_component(entity, TerminalComponent)
                    if terminal is not None:
                        from ecs_agent.core.lifecycle import finish_turn
                        from ecs_agent.core.run_result import snapshot_run_result

                        finish_turn(world, snapshot_run_result(
                            world, run_id=None, reason="terminal_component", ticks=0, target_entity=entity,
                        ))
                    else:
                        turn.status = "paused"
                        turn.reason = "checkpoint_restore"

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

        # Restore entity tracking. Older payloads without "entity_ids" fall
        # back to every id observable in components and the registry.
        restored_entity_ids = {
            EntityId(int(eid)) for eid in data.get("entity_ids", [])
        }
        restored_entity_ids.update(
            EntityId(int(eid)) for eid in entities_data.keys()
        )
        restored_entity_ids.update(world._entity_registry.values())
        world._entity_ids = restored_entity_ids

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
        if isinstance(component, SystemPromptConfigSpec):
            from ecs_agent.prompts.bindings import prompt_config_to_dict
            return prompt_config_to_dict(component)

        if isinstance(component, SubagentWaitComponent):
            return {
                "session_ids": component.session_ids,
                "timeout": component.timeout,
                "future": None,
                "started_at": component.started_at,
                "resolved_session_ids": component.resolved_session_ids,
                "auto_restart_budget": component.auto_restart_budget,
                "restart_counts": dict(component.restart_counts),
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

        if isinstance(component, UserInputComponent):
            # Built by hand: asdict() would deepcopy the pending Future.
            return {
                "prompt": component.prompt,
                "future": None,
                "timeout": component.timeout,
                "result": component.result,
            }

        if isinstance(component, EntityRegistryComponent):
            # Built by hand: metadata may carry live objects (e.g. a provider
            # registry) that asdict()'s deepcopy or json.dumps would choke on.
            return {
                "entity_id": int(component.entity_id),
                "name": component.name,
                "tags": sorted(component.tags),
                "metadata": {
                    key: WorldSerializer._json_safe_or_placeholder(value)
                    for key, value in component.metadata.items()
                },
            }

        if isinstance(component, AgentTurnComponent):
            return {
                "turn_id": component.turn_id,
                "status": component.status,
                "reason": component.reason,
                "last_message": (
                    WorldSerializer._message_to_dict(component.last_message)
                    if component.last_message is not None else None
                ),
                "unconfirmed_tool_calls": list(component.unconfirmed_tool_calls),
                "input_normalization_state": component.input_normalization_state,
            }

        if isinstance(component, ConversationComponent):
            # Built by hand: asdict() would deep-copy the entire message
            # history only for _message_to_dict to convert it all over again
            # — the double full-history conversion tax on every snapshot.
            return {
                "messages": [
                    WorldSerializer._message_to_dict(message)
                    for message in component.messages
                ],
                "max_messages": component.max_messages,
                "history_id": component.history_id,
                "revision": component.revision,
            }

        if isinstance(component, MessageBusConversationComponent):
            # Same single-pass treatment as ConversationComponent.
            return {
                "entity_id": int(component.entity_id),
                "messages": [
                    WorldSerializer._message_to_dict(message)
                    for message in component.messages
                ],
                "max_messages": component.max_messages,
            }

        if isinstance(component, LLMComponent):
            # SDK clients contain locks and live transports; never deepcopy
            # them just to replace them with an external provider reference.
            return {
                "model": getattr(component.model, "model_id", "default"),
                "system_prompt": component.system_prompt,
                "pending_model": None,
            }

        if isinstance(component, ToolRegistryComponent):
            return {
                "tools": {name: asdict(schema) for name, schema in component.tools.items()},
                "handlers": NON_SERIALIZABLE_PLACEHOLDER,
            }
        if isinstance(component, SkillComponent):
            return {"skills": {
                name: {**asdict(metadata), "skill_dir_path": str(metadata.skill_dir_path) if metadata.skill_dir_path is not None else None}
                for name, metadata in component.skills.items()
            }}
        if isinstance(component, UserPromptConfigComponent):
            return {
                "triggers": [asdict(trigger) for trigger in component.triggers],
                "enable_context_pool": component.enable_context_pool,
                "context_pool_max_chars": component.context_pool_max_chars,
                "script_handlers": NON_SERIALIZABLE_PLACEHOLDER,
            }
        if isinstance(component, EmbeddingComponent):
            return {"provider": NON_SERIALIZABLE_PLACEHOLDER, "dimension": component.dimension}
        if isinstance(component, VectorStoreComponent):
            return {"store": NON_SERIALIZABLE_PLACEHOLDER}

        serialized = {} if isinstance(component, SubagentRegistryComponent) else asdict(component)

        if isinstance(component, ToolApprovalComponent):
            serialized["policy"] = component.policy.value

        if isinstance(component, WorkspaceBindingComponent):
            serialized["workspace_root"] = str(component.workspace_root)

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
                    "runtime_profile": cfg.runtime_profile,
                    "tools": cfg.tools,
                    "prompt_config": WorldSerializer._serialize_component(cfg.prompt_config) if cfg.prompt_config is not None else None,
                    "triggers": [asdict(trigger) for trigger in cfg.triggers],
                    "local_skills": [
                        {"name": skill.name, "path": str(path) if (path := getattr(skill, "skill_dir_path", None)) is not None else None}
                        for skill in cfg.local_skills
                    ],
                    "tool_definitions": [asdict(definition.schema) for definition in cfg.tool_definitions],
                    "subagents": WorldSerializer._serialize_component(SubagentRegistryComponent(subagents=cfg.subagents))["subagents"] if cfg.subagents else {},
                    "metadata": cfg.metadata,
                    "authoring": cfg.authoring,
                    "inheritance_policy": asdict(cfg.inheritance_policy),
                    "provider": NON_SERIALIZABLE_PLACEHOLDER,
                }
            serialized["subagents"] = subagents_serialized
            serialized["free_subagent_config"] = {
                "enabled": component.free_subagent_config.enabled,
                "system_prompt_template": component.free_subagent_config.system_prompt_template,
                "skills": list(component.free_subagent_config.skills),
                "max_ticks": component.free_subagent_config.max_ticks,
                "inheritance_policy": asdict(
                    component.free_subagent_config.inheritance_policy
                ),
            }

        # MessageBusSubscriptionComponent: convert set[str] to list[str]
        if isinstance(component, MessageBusSubscriptionComponent):
            subscriptions_dict = {}
            for topic, entity_ids in serialized.get("subscriptions", {}).items():
                # Convert set of entity IDs (as strings) to list
                subscriptions_dict[topic] = sorted(list(entity_ids))
            serialized["subscriptions"] = subscriptions_dict

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
    def _json_safe_or_placeholder(value: Any) -> Any:
        """Return the value if JSON-encodable, else the placeholder marker."""
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            return NON_SERIALIZABLE_PLACEHOLDER
        return value

    @staticmethod
    def _normalize_component_data(
        component_name: str,
        component_data: dict[str, Any],
        providers: dict[str, Any],
        tool_handlers: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_data = dict(component_data)

        if component_name == AgentTurnComponent.__name__:
            if normalized_data.get("input_normalization_state", "pending") not in {"pending", "running", "completed"}:
                raise ValueError("Invalid input normalization state in checkpoint")
            if normalized_data.get("status", "paused") not in {"running", "paused", "completed", "failed", "blocked", "legacy", "abandoned"}:
                raise ValueError("Invalid agent turn state in checkpoint")
            if normalized_data.get("last_message") is not None:
                normalized_data["last_message"] = WorldSerializer._message_from_dict(normalized_data["last_message"])

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

        if component_name == TodoListComponent.__name__:
            normalized_data["items"] = [
                TodoItem(**item) if isinstance(item, dict) else item
                for item in normalized_data.get("items", [])
            ]

        if component_name == ToolRegistryComponent.__name__:
            normalized_data["tools"] = {
                name: ToolSchema(**schema)
                for name, schema in normalized_data.get("tools", {}).items()
            }
            handlers_value = normalized_data.get("handlers")
            if handlers_value == NON_SERIALIZABLE_PLACEHOLDER:
                normalized_data["handlers"] = {name: tool_handlers[name] for name in normalized_data["tools"] if name in tool_handlers}

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
            from ecs_agent.prompts.bindings import prompt_config_from_dict
            config = prompt_config_from_dict(normalized_data)
            return {"template_source": config.template_source, "placeholders": config.placeholders, "revision": config.revision, "source_path": config.source_path}

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
            from ecs_agent.types import FreeSubagentConfig, SubagentConfig, InheritancePolicy

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
                    runtime_profile=config_data.get("runtime_profile"),
                    tools=config_data.get("tools"),
                )
                prompt_data = config_data.get("prompt_config")
                if prompt_data is not None:
                    subagent_config.prompt_config = SystemPromptConfigSpec(**WorldSerializer._normalize_component_data(
                        SystemPromptConfigSpec.__name__, prompt_data, providers, tool_handlers,
                    ))
                subagent_config.triggers = [TriggerSpec(**trigger) for trigger in config_data.get("triggers", [])]
                subagent_config.metadata = dict(config_data.get("metadata", {}))
                subagent_config.authoring = config_data.get("authoring", False)
                if config_data.get("subagents"):
                    subagent_config.subagents = WorldSerializer._normalize_component_data(
                        SubagentRegistryComponent.__name__, {"subagents": config_data["subagents"]}, providers, tool_handlers,
                    )["subagents"]
                from ecs_agent.skills import catalog as skill_catalog
                from ecs_agent.skills.skill import Skill
                from ecs_agent.skills.script_skill import ScriptSkill
                from ecs_agent.tools.definitions import ToolDefinition

                for reference in config_data.get("local_skills", []):
                    if reference.get("path") is not None:
                        skill = Skill(skill_path=Path(reference["path"]) / "SKILL.md")
                        if not skill.valid or skill.name != reference["name"]:
                            raise ValueError(f"Local skill '{reference['name']}' no longer matches its checkpoint reference")
                        # File skills expose readonly name/description; the
                        # legacy protocol declares these attributes writable.
                        subagent_config.local_skills.append(cast(ScriptSkill, skill))
                    else:
                        descriptor = skill_catalog.lookup(reference["name"])
                        if descriptor is None:
                            raise ValueError(f"Register a skill descriptor for '{reference['name']}' before restoring this child")
                        bound_skill = descriptor.materialize()
                        if not isinstance(bound_skill, ScriptSkill):
                            raise ValueError(f"Invalid skill binding: {reference['name']}")
                        subagent_config.local_skills.append(bound_skill)
                for schema_data in config_data.get("tool_definitions", []):
                    schema = ToolSchema(**schema_data)
                    handler = tool_handlers.get(f"{name}:{schema.name}", tool_handlers.get(schema.name))
                    if handler is None:
                        raise ValueError(f"Missing tool handler for subagent '{name}': {schema.name}")
                    subagent_config.tool_definitions.append(ToolDefinition(schema, handler))
                subagents_dict[name] = subagent_config
            normalized_data["subagents"] = subagents_dict
            free_config_data = normalized_data.get("free_subagent_config", {})
            if isinstance(free_config_data, FreeSubagentConfig):
                free_config = free_config_data
            else:
                if not isinstance(free_config_data, dict):
                    free_config_data = {}
                free_policy_data = free_config_data.get("inheritance_policy", {})
                if isinstance(free_policy_data, InheritancePolicy):
                    free_policy = free_policy_data
                elif isinstance(free_policy_data, dict):
                    free_policy = InheritancePolicy(**free_policy_data)
                else:
                    free_policy = InheritancePolicy()
                free_config = FreeSubagentConfig(
                    enabled=bool(free_config_data.get("enabled", False)),
                    system_prompt_template=free_config_data.get(
                        "system_prompt_template",
                        FreeSubagentConfig().system_prompt_template,
                    ),
                    skills=list(free_config_data.get("skills", [])),
                    max_ticks=free_config_data.get("max_ticks"),
                    inheritance_policy=free_policy,
                )
            normalized_data["free_subagent_config"] = free_config

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
                    normalized_session_data = dict(session_data)
                    # Convert parent_entity_id if it's an int
                    parent_entity_id_value = normalized_session_data.get(
                        "parent_entity_id"
                    )
                    if isinstance(parent_entity_id_value, int):
                        normalized_session_data["parent_entity_id"] = EntityId(
                            parent_entity_id_value
                        )
                    allowed_fields = {
                        field.name for field in fields(SubagentSessionRecord)
                    }
                    sessions_dict[session_id] = SubagentSessionRecord(
                        **{
                            key: value
                            for key, value in normalized_session_data.items()
                            if key in allowed_fields
                        }
                    )
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

        if component_name == UserInputComponent.__name__:
            normalized_data["future"] = None

        if component_name == WorkspaceBindingComponent.__name__:
            workspace_root = normalized_data.get("workspace_root")
            if isinstance(workspace_root, str):
                normalized_data["workspace_root"] = Path(workspace_root)

        if component_name == EntityRegistryComponent.__name__:
            normalized_data["entity_id"] = EntityId(
                int(normalized_data["entity_id"])
            )
            normalized_data["tags"] = set(normalized_data.get("tags", []))

        if component_name == SkillComponent.__name__:
            normalized_data["skills"] = {
                name: SkillMetadata(**metadata)
                if isinstance(metadata, dict)
                else metadata
                for name, metadata in normalized_data.get("skills", {}).items()
            }

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
            reasoning_content=data.get("reasoning_content"),
            reasoning_signature=data.get("reasoning_signature"),
            compaction_metadata=data.get("compaction_metadata"),
            generation_status=(
                GenerationStatus(data["generation_status"])
                if data.get("generation_status") is not None else None
            ),
            finish_reason=(
                FinishReason(data["finish_reason"])
                if data.get("finish_reason") is not None else None
            ),
            provider_output=(
                ProviderOutput(**data["provider_output"])
                if data.get("provider_output") is not None else None
            ),
            cache_control=data.get("cache_control", False),
            turn_id=data.get("turn_id"),
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

        if message.reasoning_content is not None:
            serialized["reasoning_content"] = message.reasoning_content

        if message.reasoning_signature is not None:
            serialized["reasoning_signature"] = message.reasoning_signature

        if message.generation_status is not None:
            serialized["generation_status"] = str(message.generation_status)
        if message.finish_reason is not None:
            serialized["finish_reason"] = str(message.finish_reason)
        if message.provider_output is not None:
            serialized["provider_output"] = asdict(message.provider_output)
        if message.cache_control:
            serialized["cache_control"] = True
        if message.turn_id is not None:
            serialized["turn_id"] = message.turn_id

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
