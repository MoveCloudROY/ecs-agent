"""World binding, resource-aware file tools, and history recording."""

import json
import asyncio

import pytest

from ecs_agent.components import (
    ConversationComponent, PermissionComponent, ResponsesAPIStateComponent,
    SessionPolicyComponent, SessionRefComponent, ToolRegistryComponent,
)
from ecs_agent.core import World
from ecs_agent.history import append_conversation_message, replace_conversation_history
from ecs_agent.scratchbook.artifact_registry import ArtifactRegistry
from ecs_agent.serialization import WorldSerializer
from ecs_agent.sessions import InMemorySessionStore, SessionService, install_session_recording, release_session_writer
from ecs_agent.tools.builtins import BuiltinToolsSkill
from ecs_agent.tools.context import ToolExecutionContext, use_tool_context
from ecs_agent.types import Message


def make_world(tmp_path, service, *, name="main", parent=None, install=True):
    world = World(name=name)
    entity = world.create_entity()
    world.add_component(entity, ConversationComponent(messages=[Message(role="user", content=f"task {name}")]))
    tools = BuiltinToolsSkill().bind_workspace(str(tmp_path)).tools()
    world.add_component(entity, ToolRegistryComponent(
        tools={name: schema for name, (schema, _) in tools.items()},
        handlers={name: handler for name, (_, handler) in tools.items()},
    ))
    if install:
        install_session_recording(world, entity, service, workspace_id="project", agent_name=name, parent_session_id=parent)
    return world, entity


@pytest.fixture
def service(tmp_path):
    result = SessionService(InMemorySessionStore(), artifact_registry=ArtifactRegistry(tmp_path))
    yield result
    result.close()


async def test_inherited_read_handler_uses_actual_child_identity(tmp_path, service):
    parent, parent_id = make_world(tmp_path, service)
    parent_ref = parent.get_component(parent_id, SessionRefComponent)
    child, child_id = make_world(tmp_path, service, name="child", parent=parent_ref.session_id)
    registry = parent.get_component(parent_id, ToolRegistryComponent)
    child_ref = child.get_component(child_id, SessionRefComponent)
    child_uri = service.transcript_uri(service.store.get_session(child_ref.session_id))
    parent_uri = service.transcript_uri(service.store.get_session(parent_ref.session_id))
    child.add_component(child_id, SessionPolicyComponent(scope="self"))
    # Deliberately invoke the parent's closures in the child's execution context.
    with use_tool_context(ToolExecutionContext(child, child_id, "read_file")):
        assert "task child" in await registry.handlers["read_file"](file_path=child_uri)
        with pytest.raises(PermissionError):
            await registry.handlers["read_file"](file_path=parent_uri)
        page = json.loads(await registry.handlers["sessions_list"]())
        assert [item["session_id"] for item in page["items"]] == [child_ref.session_id]
    with pytest.raises(RuntimeError, match="context|Context"):
        await registry.handlers["read_file"](file_path=child_uri)


async def test_file_reads_and_edit_snapshots_survive_session_reads(tmp_path, service):
    world, entity = make_world(tmp_path, service)
    registry = world.get_component(entity, ToolRegistryComponent)
    ref = world.get_component(entity, SessionRefComponent)
    uri = service.transcript_uri(service.store.get_session(ref.session_id))
    (tmp_path / "hello.txt").write_text("hello\nworld\n")
    with use_tool_context(ToolExecutionContext(world, entity, "read_file")):
        assert await registry.handlers["read_file"](file_path="hello.txt", offset=2, limit=1) == "world"
        await registry.handlers["read_file"](file_path=uri)
        await registry.handlers["write_file"](file_path="hello.txt", content="updated")
        with pytest.raises(ValueError, match="URI|read.only|resource"):
            await registry.handlers["write_file"](file_path=uri, content="bad")
        with pytest.raises(ValueError, match="URI|read.only|resource"):
            await registry.handlers["edit_file"](file_path=uri, op="replace", pos=1, content="bad")
        with pytest.raises(ValueError, match="scheme|URI"):
            await registry.handlers["read_file"](file_path="https://example.com/file")
    assert (tmp_path / "hello.txt").read_text() == "updated"
    assert not (tmp_path / "session:").exists()


async def test_uninstalled_session_resources_fail_but_files_work(tmp_path, service):
    world, entity = make_world(tmp_path, service, install=False)
    registry = world.get_component(entity, ToolRegistryComponent)
    (tmp_path / "normal").write_text("content")
    with use_tool_context(ToolExecutionContext(world, entity, "read_file")):
        assert await registry.handlers["read_file"](file_path="normal") == "content"
        with pytest.raises(RuntimeError, match="enabled|installed"):
            await registry.handlers["read_file"](file_path="session://sess_x/records/rec_x?v=1")


def test_install_refuses_custom_read_or_missing_permission_without_mutation(tmp_path, service):
    world, entity = make_world(tmp_path, service, install=False)
    registry = world.get_component(entity, ToolRegistryComponent)

    async def custom(**kwargs):
        return "custom"

    original = registry.handlers["read_file"]
    registry.handlers["read_file"] = custom
    with pytest.raises(ValueError, match="read_file"):
        install_session_recording(world, entity, service, workspace_id="project", agent_name="main")
    assert world.get_component(entity, SessionRefComponent) is None
    assert registry.handlers["read_file"] is custom
    registry.handlers["read_file"] = original
    world.add_component(entity, PermissionComponent(allowed_tools=["read_file"]))
    with pytest.raises(PermissionError):
        install_session_recording(world, entity, service, workspace_id="project", agent_name="main")
    assert world.get_component(entity, PermissionComponent).allowed_tools == ["read_file"]


def test_history_compaction_reset_and_reinstall_preserve_originals(tmp_path, service):
    world, entity = make_world(tmp_path, service)
    ref = world.get_component(entity, SessionRefComponent)
    original_id = ref.session_id
    install_session_recording(world, entity, service, workspace_id="project", agent_name="main")
    append_conversation_message(world, entity, Message(role="assistant", content="unique evidence"))
    replace_conversation_history(world, entity, [Message(role="user", content="summary")], reason="compaction_summary")
    assert len(service.search(original_id, SessionPolicyComponent(), "unique evidence").items) == 1
    assert len([r for r in service.store.records(original_id) if r.payload.get("content") == "task main"]) == 1
    assert world.get_component(entity, ConversationComponent).revision == 1
    replace_conversation_history(world, entity, [Message(role="user", content="next task")], reason="workflow_reset", new_conversation=True)
    new_ref = world.get_component(entity, SessionRefComponent)
    assert new_ref.session_id != original_id
    assert service.store.get_session(new_ref.session_id).root_session_id == new_ref.session_id
    assert world.get_component(entity, ConversationComponent).messages[0].content == "next task"


def test_checkpoint_rebind_requires_explicit_restore_and_preserves_saved_path(tmp_path, service):
    world, entity = make_world(tmp_path, service)
    saved_ref = world.get_component(entity, SessionRefComponent)
    saved_head = saved_ref.head_record_id
    saved = WorldSerializer.to_dict(world)
    append_conversation_message(world, entity, Message(role="assistant", content="future branch"))
    future_head = saved_ref.head_record_id
    registry = world.get_component(entity, ToolRegistryComponent)
    restored = WorldSerializer.from_dict(saved, providers={}, tool_handlers=registry.handlers)
    with pytest.raises(ValueError, match="restore"):
        install_session_recording(restored, entity, service, workspace_id="project", agent_name="main")
    with pytest.raises(RuntimeError, match="writer"):
        install_session_recording(restored, entity, service, workspace_id="project", agent_name="main", restore=True)
    release_session_writer(world, entity)
    install_session_recording(restored, entity, service, workspace_id="project", agent_name="main", restore=True)
    append_conversation_message(restored, entity, Message(role="assistant", content="restored branch"))
    new_ref = restored.get_component(entity, SessionRefComponent)
    current = list(service.store.path(new_ref.session_id, new_ref.head_record_id))
    assert saved_head in [item.record_id for item in current]
    assert future_head not in [item.record_id for item in current]
    assert service.store.get_record(future_head).payload["content"] == "future branch"


def test_record_failure_does_not_mutate_working_context(tmp_path, service, monkeypatch):
    world, entity = make_world(tmp_path, service)
    ref = world.get_component(entity, SessionRefComponent)
    before_head = ref.head_record_id
    conversation = world.get_component(entity, ConversationComponent)
    before = list(conversation.messages)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(service.store, "append", fail)
    with pytest.raises(OSError, match="disk full"):
        append_conversation_message(world, entity, Message(role="assistant", content="not committed"))
    assert conversation.messages == before
    assert ref.head_record_id == before_head
    with pytest.raises(OSError):
        replace_conversation_history(world, entity, [], reason="compaction_trim")
    assert conversation.messages == before


def test_uncertain_commit_is_confirmed_before_advancing_memory(tmp_path, service, monkeypatch):
    world, entity = make_world(tmp_path, service)
    original = service.store.append

    def committed_but_ack_failed(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("commit acknowledgement lost")

    monkeypatch.setattr(service.store, "append", committed_but_ack_failed)
    append_conversation_message(world, entity, Message(role="assistant", content="committed once"))
    ref = world.get_component(entity, SessionRefComponent)
    assert service.store.get_record(ref.head_record_id).payload["content"] == "committed once"
    assert world.get_component(entity, ConversationComponent).messages[-1].content == "committed once"
    assert sum(record.payload.get("content") == "committed once" for record in service.store.records(ref.session_id)) == 1


@pytest.mark.parametrize("when", ["before_commit", "after_commit"])
async def test_requested_cancellation_cannot_split_commit_and_context_update(tmp_path, service, monkeypatch, when):
    world, entity = make_world(tmp_path, service)
    original = service.store.append

    def cancel_during_commit(*args, **kwargs):
        if when == "before_commit":
            asyncio.current_task().cancel()
        record = original(*args, **kwargs)
        if when == "after_commit":
            asyncio.current_task().cancel()
        return record

    monkeypatch.setattr(service.store, "append", cancel_during_commit)

    async def operation():
        append_conversation_message(world, entity, Message(role="assistant", content="coordinated"))
        await asyncio.sleep(0)

    task = asyncio.create_task(operation())
    with pytest.raises(asyncio.CancelledError):
        await task
    ref = world.get_component(entity, SessionRefComponent)
    assert service.store.get_record(ref.head_record_id).payload["content"] == "coordinated"
    assert world.get_component(entity, ConversationComponent).messages[-1].content == "coordinated"


def test_recording_only_install_can_later_enable_tools_without_reimport(tmp_path, service):
    world, entity = make_world(tmp_path, service, install=False)
    ref = install_session_recording(world, entity, service, workspace_id="project", agent_name="main", register_tools=False)
    count = len(list(service.store.records(ref.session_id)))
    install_session_recording(world, entity, service, workspace_id="project", agent_name="main", policy=SessionPolicyComponent(scope="self"))
    assert "sessions_list" in world.get_component(entity, ToolRegistryComponent).tools
    assert world.get_component(entity, SessionPolicyComponent).scope == "self"
    assert len(list(service.store.records(ref.session_id))) == count
