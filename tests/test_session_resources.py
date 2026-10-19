"""Authorization, stable discovery/search, and lossless session resource pages."""

import json

import pytest

from ecs_agent.components import SessionPolicyComponent, SessionRefComponent
from ecs_agent.scratchbook.artifact_registry import ArtifactRegistry
from ecs_agent.sessions import InMemorySessionStore, SessionService, SQLiteSessionStore
from ecs_agent.sessions.resources import read_session_resource
from ecs_agent.types import Message, ProviderOutput


@pytest.fixture(params=["memory", "sqlite"])
def service(request, tmp_path):
    store = InMemorySessionStore() if request.param == "memory" else SQLiteSessionStore(tmp_path / "sessions.sqlite3")
    result = SessionService(store, artifact_registry=ArtifactRegistry(tmp_path))
    yield result
    result.close()


def writer(service, name="main", parent=None, workspace="project"):
    info = service.create_session(agent_name=name, workspace_id=workspace, parent_session_id=parent)
    ref = SessionRefComponent(info.session_id)
    token = service.claim_writer(info.session_id)
    return ref, token


def write(service, ref, token, text, role="user"):
    return service.append_message(ref, Message(role=role, content=text), writer_id=token)


def read(service, caller, uri, policy=None, **kwargs):
    return read_session_resource(service, caller.session_id, policy or SessionPolicyComponent(), uri, **kwargs)


def unpack(page):
    header, body = page.split("\n---\n", 1)
    return json.loads(header), body


def test_tree_permissions_and_explicit_sharing_apply_to_all_entrypoints(service):
    root, rt = writer(service)
    coder, ct = writer(service, "coder", root.session_id)
    reviewer, vt = writer(service, "reviewer", root.session_id)
    outsider, ot = writer(service, "outsider")
    foreign, ft = writer(service, "foreign", workspace="other")
    record = write(service, coder, ct, "独有测试证据")
    for ref, token in ((root, rt), (reviewer, vt), (outsider, ot), (foreign, ft)):
        write(service, ref, token, "hello")
    policy = SessionPolicyComponent()
    assert len(service.list_sessions(reviewer.session_id, policy).items) == 3
    hit = service.search(reviewer.session_id, policy, "独有测试证据").items[0]
    assert hit["record_id"] == record.record_id
    assert "独有测试证据" in read(service, reviewer, hit["read_uri"])
    with pytest.raises(PermissionError):
        read(service, outsider, hit["read_uri"])
    shared = SessionPolicyComponent(scope="self", shared_session_ids=[coder.session_id, foreign.session_id])
    assert "独有测试证据" in read(service, outsider, hit["read_uri"], shared)
    assert {row["session_id"] for row in service.list_sessions(outsider.session_id, shared).items} == {outsider.session_id, coder.session_id}
    with pytest.raises(PermissionError):
        read(service, reviewer, hit["read_uri"], SessionPolicyComponent(scope="self"))


def test_descendants_excludes_parent_and_sibling(service):
    root, _ = writer(service)
    child, _ = writer(service, "child", root.session_id)
    sibling, _ = writer(service, "sibling", root.session_id)
    grandchild, _ = writer(service, "grandchild", child.session_id)
    page = service.list_sessions(child.session_id, SessionPolicyComponent(scope="descendants"))
    assert {i["session_id"] for i in page.items} == {child.session_id, grandchild.session_id}


def test_catalog_cursor_pins_heads_and_filters_then_rechecks_policy(service):
    root, rt = writer(service)
    child, ct = writer(service, "coder", root.session_id)
    write(service, root, rt, "root")
    old = write(service, child, ct, "old")
    policy = SessionPolicyComponent()
    page = service.list_sessions(root.session_id, policy, limit=1)
    assert page.next_cursor
    write(service, child, ct, "new")
    writer(service, "later", root.session_id)
    continuation = service.list_sessions(root.session_id, policy, cursor=page.next_cursor, limit=1)
    assert continuation.next_cursor is None
    assert old.record_id in continuation.items[0]["read_uri"]
    with pytest.raises(ValueError, match="filter"):
        service.list_sessions(root.session_id, policy, agent_name="coder", cursor=page.next_cursor)
    with pytest.raises(PermissionError):
        service.list_sessions(root.session_id, SessionPolicyComponent(scope="self"), cursor=page.next_cursor)


def test_search_cursor_pins_hits_and_matches_chinese_literal_text(service):
    ref, token = writer(service)
    for text in ("测试 abc%_", "测试 第二次", "unmatched"):
        write(service, ref, token, text)
    policy = SessionPolicyComponent()
    page = service.search(ref.session_id, policy, "测试", limit=1)
    write(service, ref, token, "测试 第三次")
    rest = service.search(ref.session_id, policy, "测试", cursor=page.next_cursor, limit=1)
    assert rest.next_cursor is None
    assert len(service.search(ref.session_id, policy, "测试").items) == 3
    assert len(service.search(ref.session_id, policy, "abc%_").items) == 1
    with pytest.raises(ValueError, match="filter"):
        service.search(ref.session_id, policy, "different", cursor=page.next_cursor)


def test_long_artifact_read_is_bounded_lossless_and_stable_during_append(service):
    ref, token = writer(service)
    text = "长输出" * 14000 + "\n最后一行\n"
    record = write(service, ref, token, text, role="tool")
    assert record.artifact_ref is not None
    hit = service.search(ref.session_id, SessionPolicyComponent(), "最后一行").items[0]
    record_text = read(service, ref, hit["read_uri"])
    assert f"/records/{record.record_id}/tool-result?v=1" in record_text
    uri = service.tool_result_uri(record)
    first = read(service, ref, uri, limit=10000)
    assert len(first) <= 12000
    write(service, ref, token, "later")
    assert first == read(service, ref, uri, limit=10000)
    offset = 1
    reconstructed = ""
    while offset is not None:
        raw = read(service, ref, uri, offset=offset, limit=10000)
        metadata, body = unpack(raw)
        assert len(raw) <= 12000
        reconstructed += "".join(json.loads(line.removeprefix("text: ")) for line in body.splitlines() if line.startswith("text: "))
        offset = metadata["next_offset"]
    assert reconstructed == text


def test_transcript_uri_freezes_path_and_offsets(service):
    ref, token = writer(service)
    write(service, ref, token, "old\n" * 200)
    uri = service.list_sessions(ref.session_id, SessionPolicyComponent()).items[0]["read_uri"]
    before = read(service, ref, uri, offset=2, limit=4, line_numbers=True)
    write(service, ref, token, "new unique")
    assert before == read(service, ref, uri, offset=2, limit=4, line_numbers=True)
    meta, body = unpack(before)
    assert meta["start_line"] == 2
    assert body.startswith("2: ")
    new_uri = service.list_sessions(ref.session_id, SessionPolicyComponent()).items[0]["read_uri"]
    assert new_uri != uri
    assert "new unique" not in read(service, ref, uri, limit=500)


@pytest.mark.parametrize("uri", [
    "session://sess_x/transcript?v=1", "session://sess_x/transcript?head=rec_x&v=2",
    "session://sess_x/records/../../etc/passwd?v=1", "https://example.org/file",
    "session://sess_x/records/rec_x?v=1&v=1", "session://user@host/records/rec_x?v=1",
    "session://sess_x/records/rec_x?v=1#fragment",
])
def test_malformed_resources_fail_without_filesystem_fallback(service, uri):
    ref, _ = writer(service)
    with pytest.raises(ValueError):
        read(service, ref, uri)


def test_uri_record_membership_missing_artifacts_and_revocation(service):
    root, rt = writer(service)
    child, ct = writer(service, "child", root.session_id)
    record = write(service, child, ct, "evidence" * 1000, role="tool")
    with pytest.raises(ValueError, match="session"):
        read(service, root, f"session://{root.session_id}/records/{record.record_id}?v=1")
    with pytest.raises(PermissionError):
        read(service, root, service.tool_result_uri(record), SessionPolicyComponent(scope="self"))
    (service.artifact_registry.root / record.artifact_ref).unlink()
    with pytest.raises(FileNotFoundError, match="artifact"):
        read(service, root, service.tool_result_uri(record))


def test_query_projection_excludes_system_and_private_credentials(service):
    ref, token = writer(service)
    write(service, ref, token, "private system instructions", role="system")
    service.append_message(ref, Message(
        role="assistant", content="api_key=secret_credential; result ok",
        provider_output=ProviderOutput(protocol="responses", origin="test", items=[{"private": "hidden replay"}]),
    ), writer_id=token)
    policy = SessionPolicyComponent()
    uri = service.list_sessions(ref.session_id, policy).items[0]["read_uri"]
    output = read(service, ref, uri)
    assert "private system instructions" not in output
    assert "hidden replay" not in output
    assert "secret_credential" not in output
    assert service.search(ref.session_id, policy, "private system instructions").items == []
    assert service.search(ref.session_id, policy, "secret_credential").items == []
