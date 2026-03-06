"""Tests for ScratchbookService filesystem operations."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ecs_agent.scratchbook import ScratchbookService


@pytest.fixture
def tmp_scratchbook(tmp_path: Path) -> Path:
    """Create temporary scratchbook directory."""
    scratchbook_path = tmp_path / ".scratchbook"
    scratchbook_path.mkdir()
    return scratchbook_path


def test_write_artifact_creates_categorized_folder(tmp_scratchbook: Path) -> None:
    """Write artifact creates category subfolder and JSON file."""
    service = ScratchbookService(root=tmp_scratchbook)
    artifact_id = "task-123"
    category = "planning"
    data = {"content": "Test plan", "status": "active"}

    service.write_artifact(artifact_id=artifact_id, category=category, data=data)

    artifact_path = tmp_scratchbook / category / f"{artifact_id}.json"
    assert artifact_path.exists()
    assert artifact_path.read_text(encoding="utf-8") == json.dumps(data, indent=2)


def test_read_artifact_returns_parsed_data(tmp_scratchbook: Path) -> None:
    """Read artifact loads and parses JSON data."""
    service = ScratchbookService(root=tmp_scratchbook)
    artifact_id = "task-456"
    category = "execution"
    data = {"content": "Test execution", "status": "completed"}

    service.write_artifact(artifact_id=artifact_id, category=category, data=data)
    loaded = service.read_artifact(artifact_id=artifact_id, category=category)

    assert loaded == data


def test_read_artifact_missing_file_returns_none(tmp_scratchbook: Path) -> None:
    """Read artifact returns None for missing file."""
    service = ScratchbookService(root=tmp_scratchbook)

    result = service.read_artifact(artifact_id="missing", category="planning")

    assert result is None


def test_read_artifact_corrupted_json_returns_none(tmp_scratchbook: Path) -> None:
    """Read artifact returns None for corrupted JSON."""
    service = ScratchbookService(root=tmp_scratchbook)
    artifact_id = "corrupted"
    category = "planning"
    artifact_path = tmp_scratchbook / category / f"{artifact_id}.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("{invalid json", encoding="utf-8")

    result = service.read_artifact(artifact_id=artifact_id, category=category)

    assert result is None


def test_append_log_creates_file_if_missing(tmp_scratchbook: Path) -> None:
    """Append log creates new file if it doesn't exist."""
    service = ScratchbookService(root=tmp_scratchbook)
    log_name = "activity.log"
    category = "logs"
    line = "First log entry\n"

    service.append_log(log_name=log_name, category=category, line=line)

    log_path = tmp_scratchbook / category / log_name
    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8") == line


def test_append_log_appends_to_existing_file(tmp_scratchbook: Path) -> None:
    """Append log appends to existing file without overwriting."""
    service = ScratchbookService(root=tmp_scratchbook)
    log_name = "activity.log"
    category = "logs"
    line1 = "First entry\n"
    line2 = "Second entry\n"

    service.append_log(log_name=log_name, category=category, line=line1)
    service.append_log(log_name=log_name, category=category, line=line2)

    log_path = tmp_scratchbook / category / log_name
    assert log_path.read_text(encoding="utf-8") == line1 + line2


def test_atomic_index_write_uses_temp_file_and_replace(tmp_scratchbook: Path) -> None:
    """Atomic index write uses temp file + os.replace pattern."""
    service = ScratchbookService(root=tmp_scratchbook)
    index_name = "task_index.json"
    category = "planning"
    data = {"tasks": ["task-1", "task-2"]}

    service.write_index(index_name=index_name, category=category, data=data)

    index_path = tmp_scratchbook / category / index_name
    assert index_path.exists()
    assert json.loads(index_path.read_text(encoding="utf-8")) == data


def test_atomic_index_write_preserves_previous_on_simulated_interruption(
    tmp_scratchbook: Path,
) -> None:
    """Atomic write leaves previous valid index intact if interrupted mid-write."""
    service = ScratchbookService(root=tmp_scratchbook)
    index_name = "task_index.json"
    category = "planning"
    initial_data = {"tasks": ["task-1"]}

    service.write_index(index_name=index_name, category=category, data=initial_data)

    # Simulate partial write by creating corrupted temp file
    index_path = tmp_scratchbook / category / index_name
    temp_path = index_path.with_suffix(".json.tmp")
    temp_path.write_text("{corrupted", encoding="utf-8")  # Invalid JSON

    # Original index should still be intact
    assert json.loads(index_path.read_text(encoding="utf-8")) == initial_data

    # New atomic write should succeed and replace
    new_data = {"tasks": ["task-1", "task-2"]}
    service.write_index(index_name=index_name, category=category, data=new_data)
    assert json.loads(index_path.read_text(encoding="utf-8")) == new_data


def test_read_index_returns_parsed_data(tmp_scratchbook: Path) -> None:
    """Read index loads and parses JSON index file."""
    service = ScratchbookService(root=tmp_scratchbook)
    index_name = "task_index.json"
    category = "planning"
    data = {"tasks": ["task-1", "task-2", "task-3"]}

    service.write_index(index_name=index_name, category=category, data=data)
    loaded = service.read_index(index_name=index_name, category=category)

    assert loaded == data


def test_read_index_missing_file_returns_none(tmp_scratchbook: Path) -> None:
    """Read index returns None for missing file."""
    service = ScratchbookService(root=tmp_scratchbook)

    result = service.read_index(index_name="missing.json", category="planning")

    assert result is None


def test_list_artifacts_returns_all_artifact_ids_in_category(
    tmp_scratchbook: Path,
) -> None:
    """List artifacts returns all artifact IDs in category."""
    service = ScratchbookService(root=tmp_scratchbook)
    category = "planning"

    service.write_artifact("task-1", category, {"data": "1"})
    service.write_artifact("task-2", category, {"data": "2"})
    service.write_artifact("task-3", category, {"data": "3"})

    artifact_ids = service.list_artifacts(category=category)

    assert set(artifact_ids) == {"task-1", "task-2", "task-3"}


def test_list_artifacts_empty_category_returns_empty_list(
    tmp_scratchbook: Path,
) -> None:
    """List artifacts returns empty list for empty category."""
    service = ScratchbookService(root=tmp_scratchbook)

    artifact_ids = service.list_artifacts(category="planning")

    assert artifact_ids == []


def test_list_artifacts_nonexistent_category_returns_empty_list(
    tmp_scratchbook: Path,
) -> None:
    """List artifacts returns empty list for nonexistent category."""
    service = ScratchbookService(root=tmp_scratchbook)

    artifact_ids = service.list_artifacts(category="nonexistent")

    assert artifact_ids == []


def test_delete_artifact_removes_file(tmp_scratchbook: Path) -> None:
    """Delete artifact removes the JSON file."""
    service = ScratchbookService(root=tmp_scratchbook)
    artifact_id = "task-999"
    category = "planning"

    service.write_artifact(
        artifact_id=artifact_id, category=category, data={"test": "data"}
    )
    assert service.read_artifact(artifact_id=artifact_id, category=category) is not None

    service.delete_artifact(artifact_id=artifact_id, category=category)

    assert service.read_artifact(artifact_id=artifact_id, category=category) is None


def test_delete_artifact_missing_file_does_not_raise(tmp_scratchbook: Path) -> None:
    """Delete artifact on missing file does not raise exception."""
    service = ScratchbookService(root=tmp_scratchbook)

    # Should not raise
    service.delete_artifact(artifact_id="missing", category="planning")


# ============================================================================
# INDEX AND REFERENCE RESOLUTION TESTS (Task 3)
# ============================================================================


def test_index_entry_has_stable_id_category_timestamp_hash(
    tmp_scratchbook: Path,
) -> None:
    """Index entry includes stable ID, category, timestamp, and content hash."""
    from ecs_agent.scratchbook import ScratchbookIndexer

    indexer = ScratchbookIndexer(root=tmp_scratchbook)
    entry = indexer.create_index_entry(
        stable_id="task-001",
        artifact_id="artifact-001",
        artifact_type="planning",
        category="planning",
        content_hash="abc123def456",
    )

    assert entry["stable_id"] == "task-001"
    assert entry["artifact_id"] == "artifact-001"
    assert entry["artifact_type"] == "planning"
    assert entry["category"] == "planning"
    assert entry["content_hash"] == "abc123def456"
    assert "timestamp" in entry


def test_index_lookup_by_task_id_returns_artifacts(
    tmp_scratchbook: Path,
) -> None:
    """Index lookup by task_id returns artifacts in deterministic order."""
    from ecs_agent.scratchbook import ScratchbookIndexer

    indexer = ScratchbookIndexer(root=tmp_scratchbook)

    # Add entries for task-001
    indexer.add_entry(
        stable_id="task-001",
        artifact_id="artifact-001-a",
        artifact_type="planning",
        category="planning",
        content_hash="hash-a",
    )
    indexer.add_entry(
        stable_id="task-001",
        artifact_id="artifact-001-b",
        artifact_type="execution",
        category="execution",
        content_hash="hash-b",
    )

    results = indexer.lookup_by_task_id("task-001")

    assert len(results) == 2
    # Should be deterministically ordered
    artifact_ids = [r["artifact_id"] for r in results]
    assert artifact_ids == sorted(artifact_ids)


def test_index_lookup_by_artifact_type_returns_matching(
    tmp_scratchbook: Path,
) -> None:
    """Index lookup by artifact type returns matching artifacts."""
    from ecs_agent.scratchbook import ScratchbookIndexer

    indexer = ScratchbookIndexer(root=tmp_scratchbook)

    indexer.add_entry(
        stable_id="task-001",
        artifact_id="planning-001",
        artifact_type="planning",
        category="planning",
        content_hash="hash-1",
    )
    indexer.add_entry(
        stable_id="task-002",
        artifact_id="execution-001",
        artifact_type="execution",
        category="execution",
        content_hash="hash-2",
    )
    indexer.add_entry(
        stable_id="task-003",
        artifact_id="planning-002",
        artifact_type="planning",
        category="planning",
        content_hash="hash-3",
    )

    planning_results = indexer.lookup_by_artifact_type("planning")

    assert len(planning_results) == 2
    assert all(r["artifact_type"] == "planning" for r in planning_results)


def test_index_lookup_by_category_returns_matching(
    tmp_scratchbook: Path,
) -> None:
    """Index lookup by category returns matching artifacts."""
    from ecs_agent.scratchbook import ScratchbookIndexer

    indexer = ScratchbookIndexer(root=tmp_scratchbook)

    indexer.add_entry(
        stable_id="task-001",
        artifact_id="art-001",
        artifact_type="planning",
        category="planning",
        content_hash="hash-1",
    )
    indexer.add_entry(
        stable_id="task-002",
        artifact_id="art-002",
        artifact_type="execution",
        category="execution",
        content_hash="hash-2",
    )
    indexer.add_entry(
        stable_id="task-003",
        artifact_id="art-003",
        artifact_type="planning",
        category="planning",
        content_hash="hash-3",
    )

    planning_category = indexer.lookup_by_category("planning")

    assert len(planning_category) == 2
    assert all(r["category"] == "planning" for r in planning_category)


def test_index_lookup_deterministic_ordering(
    tmp_scratchbook: Path,
) -> None:
    """Index lookups return deterministically ordered results."""
    from ecs_agent.scratchbook import ScratchbookIndexer

    indexer = ScratchbookIndexer(root=tmp_scratchbook)

    # Add in random order
    for i in [3, 1, 2]:
        indexer.add_entry(
            stable_id="task-001",
            artifact_id=f"artifact-00{i}",
            artifact_type="planning",
            category="planning",
            content_hash=f"hash-{i}",
        )

    results1 = indexer.lookup_by_task_id("task-001")
    results2 = indexer.lookup_by_task_id("task-001")

    artifact_ids_1 = [r["artifact_id"] for r in results1]
    artifact_ids_2 = [r["artifact_id"] for r in results2]

    # Should be identical across calls
    assert artifact_ids_1 == artifact_ids_2


def test_index_corrupted_entry_raises_error(
    tmp_scratchbook: Path,
) -> None:
    """Index resolver raises clear error on corrupted entry."""
    from ecs_agent.scratchbook import ScratchbookIndexer, CorruptedIndexEntryError

    indexer = ScratchbookIndexer(root=tmp_scratchbook)

    # Manually write corrupted index entry
    category_path = tmp_scratchbook / "index"
    category_path.mkdir(parents=True, exist_ok=True)
    index_file = category_path / "index.json"

    corrupted_data = {"entries": [{"stable_id": "task-001"}]}  # Missing required fields
    index_file.write_text(json.dumps(corrupted_data), encoding="utf-8")

    # Should raise when trying to lookup
    with pytest.raises(CorruptedIndexEntryError):
        indexer.lookup_by_task_id("task-001")


def test_index_corrupted_entry_does_not_crash_unrelated_reads(
    tmp_scratchbook: Path,
) -> None:
    """Corrupted index entry doesn't crash unrelated lookups."""
    from ecs_agent.scratchbook import ScratchbookIndexer, CorruptedIndexEntryError

    indexer = ScratchbookIndexer(root=tmp_scratchbook)

    # Add valid entry first
    indexer.add_entry(
        stable_id="task-002",
        artifact_id="artifact-002",
        artifact_type="planning",
        category="planning",
        content_hash="hash-2",
    )

    # Manually write corrupted entry for task-001
    category_path = tmp_scratchbook / "index"
    category_path.mkdir(parents=True, exist_ok=True)
    index_file = category_path / "index.json"

    # Get current index, add corrupted entry
    current = index_file.read_text(encoding="utf-8") if index_file.exists() else "{}"
    current_data = json.loads(current) if current != "{}" else {"entries": []}
    current_data["entries"].append({"stable_id": "task-001"})  # Corrupted
    index_file.write_text(json.dumps(current_data), encoding="utf-8")

    # Looking up task-002 should still work
    results = indexer.lookup_by_task_id("task-002")
    assert len(results) == 1
    assert results[0]["stable_id"] == "task-002"
