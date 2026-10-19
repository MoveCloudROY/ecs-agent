"""Shared persistence contract for session record backends."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from ecs_agent.sessions import InMemorySessionStore, SQLiteSessionStore
from ecs_agent.sessions.types import RecordDraft, SessionInfo


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    result = (
        InMemorySessionStore()
        if request.param == "memory"
        else SQLiteSessionStore(tmp_path / "sessions.sqlite3")
    )
    yield result
    result.close()


def create(store, session_id="sess_root", parent=None):
    session = SessionInfo(
        session_id=session_id, parent_session_id=parent,
        root_session_id=parent or session_id, agent_name="coder", workspace_id="project",
    )
    store.create_session(session)
    store.claim_writer(session_id, "writer")
    return session


def draft(session_id="sess_root", record_id="rec_one", parent=None, text="测试通过"):
    return RecordDraft(
        record_id=record_id, session_id=session_id, parent_record_id=parent,
        kind="message", payload={"role": "user", "content": text},
        history_id="history", revision=0,
    )


def test_append_deduplicates_exact_retry_and_rejects_identity_reuse(store):
    create(store)
    item = draft()
    first = store.append(item, writer_id="writer")
    second = store.append(item, writer_id="writer")
    assert first == second
    assert first.seq == 1
    with pytest.raises(ValueError, match="record_id"):
        store.append(replace(item, payload={"content": "different"}), writer_id="writer")
    assert len(list(store.records("sess_root"))) == 1


def test_branch_requires_explicit_fork_and_retains_both_paths(store):
    create(store)
    store.append(draft(), writer_id="writer")
    store.append(draft(record_id="rec_two", parent="rec_one"), writer_id="writer")
    with pytest.raises(ValueError, match="head"):
        store.append(draft(record_id="rec_branch", parent="rec_one"), writer_id="writer")
    branch = store.append(
        draft(record_id="rec_branch", parent="rec_one"), writer_id="writer", branch=True,
    )
    assert branch.seq == 3
    assert [r.record_id for r in store.path("sess_root", "rec_two")] == ["rec_one", "rec_two"]
    assert [r.record_id for r in store.path("sess_root", "rec_branch")] == ["rec_one", "rec_branch"]
    assert store.get_session("sess_root").head_record_id == "rec_branch"


def test_writer_exclusivity_and_record_membership(store):
    create(store)
    create(store, "sess_child", parent="sess_root")
    store.append(draft(), writer_id="writer")
    with pytest.raises(RuntimeError, match="writer"):
        store.claim_writer("sess_root", "other")
    with pytest.raises(RuntimeError, match="writer"):
        store.append(draft(record_id="rec_wrong"), writer_id="other")
    with pytest.raises(ValueError, match="session"):
        store.append(draft(session_id="sess_child", record_id="rec_foreign", parent="rec_one"), writer_id="writer")
    with pytest.raises(ValueError, match="session"):
        list(store.path("sess_child", "rec_one"))
    store.release_writer("sess_root", "writer")
    store.claim_writer("sess_root", "other")


def test_sessions_require_consistent_host_relationships(store):
    create(store)
    with pytest.raises(ValueError, match="workspace"):
        store.create_session(SessionInfo(
            session_id="sess_foreign", parent_session_id="sess_root",
            root_session_id="sess_root", workspace_id="other", agent_name="reviewer",
        ))
    with pytest.raises(ValueError, match="root"):
        store.create_session(SessionInfo(
            session_id="sess_bad", parent_session_id=None,
            root_session_id="sess_root", workspace_id="project", agent_name="reviewer",
        ))


def test_concurrent_sessions_have_independent_order(store):
    for index in range(4):
        create(store, f"sess_{index}")

    def write(index):
        parent = None
        for seq in range(20):
            item = draft(f"sess_{index}", f"rec_{index}_{seq}", parent)
            store.append(item, writer_id="writer")
            parent = item.record_id

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write, range(4)))
    for index in range(4):
        assert [r.seq for r in store.records(f"sess_{index}")] == list(range(1, 21))


def test_return_values_cannot_mutate_authoritative_records(store):
    create(store)
    original = draft()
    saved = store.append(original, writer_id="writer")
    original.payload["content"] = "tampered"
    saved.payload["content"] = "also tampered"
    assert store.get_record("rec_one").payload["content"] == "测试通过"


def test_sqlite_reopen_preserves_records_and_cursor_snapshot(tmp_path):
    path = tmp_path / "sessions.sqlite3"
    store = SQLiteSessionStore(path)
    create(store)
    store.append(draft(), writer_id="writer")
    store.save_snapshot("cursor_test", {"rows": ["rec_one"]})
    store.close()
    reopened = SQLiteSessionStore(path)
    try:
        assert reopened.get_record("rec_one").payload["content"] == "测试通过"
        assert reopened.load_snapshot("cursor_test") == {"rows": ["rec_one"]}
        reopened.claim_writer("sess_root", "new")
        assert reopened.append(draft(record_id="rec_next", parent="rec_one"), writer_id="new").seq == 2
    finally:
        reopened.close()


def test_two_sqlite_connections_cannot_claim_same_live_session(tmp_path):
    path = tmp_path / "sessions.sqlite3"
    first = SQLiteSessionStore(path)
    second = SQLiteSessionStore(path)
    try:
        create(first)
        with pytest.raises(RuntimeError, match="writer"):
            second.claim_writer("sess_root", "other")
        first.close()
        second.claim_writer("sess_root", "other")
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("operation", ["claim", "release"])
def test_sqlite_failed_claim_transaction_cannot_steal_or_strand_writer(tmp_path, operation):
    path = tmp_path / "sessions.sqlite3"
    first = SQLiteSessionStore(path)
    second = SQLiteSessionStore(path)
    first.create_session(SessionInfo(session_id="sess_root", parent_session_id=None, root_session_id="sess_root", agent_name="main", workspace_id="project"))
    connection = first._db

    class FailCommit:
        def __getattr__(self, name):
            return getattr(connection, name)

        def commit(self):
            raise OSError("commit rejected")

    try:
        if operation == "claim":
            first._db = FailCommit()
            with pytest.raises(OSError):
                first.claim_writer("sess_root", "first")
            first._db = connection
            second.claim_writer("sess_root", "second")
            with pytest.raises(RuntimeError, match="writer"):
                first.claim_writer("sess_root", "first")
        else:
            first.claim_writer("sess_root", "first")
            first._db = FailCommit()
            with pytest.raises(OSError):
                first.release_writer("sess_root", "first")
            first._db = connection
            first.close()
            second.claim_writer("sess_root", "second")
    finally:
        first._db = connection
        first.close()
        second.close()
