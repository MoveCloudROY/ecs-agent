"""Explicit World installation, writer release, and checkpoint rebinding."""

from dataclasses import replace

from ecs_agent.components import (
    ChildStubComponent, ConversationComponent, ConversationTreeComponent, PermissionComponent, SessionPolicyComponent,
    SessionRefComponent, ToolRegistryComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.sessions import tools as session_tools
from ecs_agent.sessions.service import SessionService
from ecs_agent.tools.discovery import scan_module
from ecs_agent.types import EntityId

logger = get_logger(__name__)


def install_session_recording(
    world: World, entity_id: EntityId, service: SessionService, *, workspace_id: str,
    agent_name: str, parent_session_id: str | None = None,
    policy: SessionPolicyComponent | None = None, restore: bool = False,
    register_tools: bool = True, source_session_id: str | None = None,
    session_id: str | None = None,
) -> SessionRefComponent:
    """Bind recording and optionally register discovery/search tools.

    Install built-in read_file first when register_tools is True. Permissions
    must already permit all three tools. Use restore=True for a checkpoint;
    release its previous live writer before rebinding. Recording-only child
    installation does not grant or register additional model tools.
    """
    if not world.has_entity(entity_id):
        raise ValueError("Session entity does not exist in this World")
    if world.session_service is not None and world.session_service is not service:
        raise ValueError("World already uses another SessionService")
    effective_policy = policy or world.get_component(entity_id, SessionPolicyComponent) or SessionPolicyComponent()
    if effective_policy.scope not in {"self", "tree", "descendants"}:
        raise ValueError("Unsupported session policy scope")
    discovered = scan_module(session_tools)
    registry = world.get_component(entity_id, ToolRegistryComponent)
    if register_tools:
        handler = registry.handlers.get("read_file") if registry else None
        if registry is None or "read_file" not in registry.tools or not getattr(handler, "supports_session_resources", False):
            raise ValueError("Install a session-capable read_file before session tools; custom tools are not replaced")
        permissions = world.get_component(entity_id, PermissionComponent)
        for name in ("read_file", *discovered):
            if permissions and (name in permissions.denied_tools or (permissions.allowed_tools and name not in permissions.allowed_tools)):
                raise PermissionError(f"Session installation requires existing permission for {name}")
        for name, (_, tool_handler) in discovered.items():
            if name in registry.handlers and registry.handlers[name] is not tool_handler:
                raise ValueError(f"Session tool conflicts with existing handler: {name}")

    ref = world.get_component(entity_id, SessionRefComponent)
    new_identity = ref is None and session_id is None
    if session_id is not None and ref is not None:
        raise ValueError("Cannot replace an existing session binding")
    if ref is not None and entity_id not in world._session_writer_tokens and not restore:
        raise ValueError("Existing session identity requires explicit restore=True")
    if ref is not None and entity_id in world._session_writer_tokens:
        world.add_component(entity_id, replace(effective_policy, shared_session_ids=list(effective_policy.shared_session_ids)))
        if register_tools:
            assert registry is not None
            for name, (schema, tool_handler) in discovered.items():
                registry.tools[name] = schema
                registry.handlers[name] = tool_handler
        return ref
    if ref is not None:
        info = service.store.get_session(ref.session_id)
        if info.workspace_id != workspace_id:
            raise ValueError("Restored session workspace does not match host binding")
        if ref.head_record_id is not None and service.store.get_record(ref.head_record_id).session_id != ref.session_id:
            raise ValueError("Restored head belongs to another session")
        token = service.claim_writer(ref.session_id)
        # Work on a copy so a failed append leaves the checkpoint unchanged.
        ref = replace(ref)
        try:
            service.append(ref, "branch", {"reason": "checkpoint_restore", "saved_head": ref.head_record_id}, writer_id=token, branch=True)
        except Exception as exc:
            service.store.release_writer(ref.session_id, token)
            logger.error("session_restore_failed", session_id=ref.session_id, exception=str(exc))
            raise
    elif session_id is not None:
        # The host can reserve an identity while a background job is queued.
        info = service.store.get_session(session_id)
        if info.workspace_id != workspace_id or info.parent_session_id != parent_session_id:
            raise ValueError("Reserved session does not match host workspace/parent")
        ref = SessionRefComponent(info.session_id, info.head_record_id)
        token = service.claim_writer(ref.session_id)
    else:
        info = service.create_session(agent_name=agent_name, workspace_id=workspace_id, parent_session_id=parent_session_id, source_session_id=source_session_id)
        ref = SessionRefComponent(info.session_id)
        token = service.claim_writer(ref.session_id)
        conversation = world.get_component(entity_id, ConversationComponent)
        try:
            if restore:
                service.append(ref, "import", {"source": "legacy_checkpoint", "history_incomplete": True, "missing_history": "Messages absent from the checkpoint cannot be recovered"}, writer_id=token)
            tree = world.get_component(entity_id, ConversationTreeComponent)
            if conversation and (tree is None or not tree.messages):
                for index, message in enumerate(conversation.messages):
                    service.append_message(
                        ref, message, writer_id=token, history_id=conversation.history_id,
                        revision=conversation.revision, metadata={"source": "checkpoint" if restore else "initial_context", "index": index},
                    )
        except Exception as exc:
            service.store.release_writer(ref.session_id, token)
            logger.error("session_import_failed", session_id=ref.session_id, exception=str(exc))
            raise
    world.session_service = service
    world._session_writer_tokens[entity_id] = token
    world.add_component(entity_id, ref)
    world.add_component(entity_id, replace(effective_policy, shared_session_ids=list(effective_policy.shared_session_ids)))
    tree = world.get_component(entity_id, ConversationTreeComponent)
    if tree is not None:
        from ecs_agent.sessions.tree import bind_conversation_tree

        bind_conversation_tree(world, entity_id, tree, select=new_identity, incomplete=restore and new_identity)
    if register_tools:
        assert registry is not None
        for name, (schema, tool_handler) in discovered.items():
            registry.tools[name] = schema
            registry.handlers[name] = tool_handler
    return ref


def release_session_writer(world: World, entity_id: EntityId) -> None:
    """Release one writer without closing the service shared with child Worlds."""
    ref = world.get_component(entity_id, SessionRefComponent)
    token = world._session_writer_tokens.get(entity_id)
    if ref is not None and token is not None and world.session_service is not None:
        world.session_service.store.release_writer(ref.session_id, token)
        del world._session_writer_tokens[entity_id]


def prepare_checkpoint_sessions(world: World, restored: World) -> None:
    """Prepare restored identities before an in-place checkpoint swaps components.

    Existing writer claims stay owned by the host. Each saved head explicitly
    starts a new branch; no normal append may bypass head conflict detection.
    """
    service = world.session_service
    if service is None:
        return
    existing = {
        ref.session_id: world._session_writer_tokens[entity]
        for entity, (ref,) in world.query(SessionRefComponent)
        if entity in world._session_writer_tokens
    }
    acquired: list[tuple[str, str]] = []
    try:
        for entity, (ref,) in restored.query(SessionRefComponent):
            if restored.has_component(entity, ChildStubComponent):
                continue
            token = existing.get(ref.session_id)
            if token is None:
                token = service.claim_writer(ref.session_id)
                acquired.append((ref.session_id, token))
            service.append(ref, "branch", {"reason": "checkpoint_undo", "saved_head": ref.head_record_id}, writer_id=token, branch=True)
            restored._session_writer_tokens[entity] = token
        restored.session_service = service
        for entity, _ in restored.query(ConversationComponent):
            if restored.has_component(entity, SessionRefComponent) or restored.has_component(entity, ChildStubComponent):
                continue
            old = world.get_component(entity, SessionRefComponent)
            if old is None:
                continue
            info = service.store.get_session(old.session_id)
            install_session_recording(
                restored, entity, service, workspace_id=info.workspace_id, agent_name=info.agent_name,
                parent_session_id=info.parent_session_id, restore=True, register_tools=False,
                policy=world.get_component(entity, SessionPolicyComponent),
            )
    except Exception as exc:
        for session_id, token in acquired:
            service.store.release_writer(session_id, token)
        logger.error("session_checkpoint_prepare_failed", exception=str(exc))
        raise
    retained = set(restored._session_writer_tokens.values())
    for session_id, token in existing.items():
        if token not in retained:
            service.store.release_writer(session_id, token)
