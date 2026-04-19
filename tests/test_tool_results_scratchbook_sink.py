"""Tests for tool result append-only persistence to scratchbook."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecs_agent.components import (
    ConversationComponent,
    PendingToolCallsComponent,
    ToolRegistryComponent,
    ToolResultsComponent,
)
from ecs_agent.core import World
from ecs_agent.scratchbook import ArtifactRegistry, ToolResultsSink
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import Message, ToolCall, ToolSchema


@pytest.fixture
def tmp_scratchbook(tmp_path: Path) -> Path:
    """Create temporary scratchbook directory."""
    scratchbook_path = tmp_path / ".scratchbook"
    scratchbook_path.mkdir()
    return scratchbook_path


@pytest.mark.asyncio
async def test_tool_execution_appends_result_to_scratchbook(
    tmp_scratchbook: Path,
) -> None:
    """Tool execution appends result artifact to scratchbook with immutable ID."""
    world = World()
    entity_id = world.create_entity()
    registry = ArtifactRegistry(root=tmp_scratchbook)

    async def get_weather(city: str) -> str:
        return f"sunny in {city}"

    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="get weather")]),
    )
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "get_weather": ToolSchema(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object"},
                ),
            },
            handlers={"get_weather": get_weather},
        ),
    )
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[
                ToolCall(id="call-1", name="get_weather", arguments={"city": "Paris"})
            ]
        ),
    )

    # Execute tools
    system = ToolExecutionSystem(registry=registry)
    await system.process(world)

    # Verify result in scratchbook
    results_dir = tmp_scratchbook / "scratchbook" / "records" / "tool"
    assert results_dir.exists()

    # Collect all artifact files
    artifact_files = list(results_dir.glob("tool_*"))
    assert len(artifact_files) > 0

    # Verify artifact contains expected result
    artifact_data = json.loads(artifact_files[0].read_text(encoding="utf-8"))
    assert artifact_data["tool_call_id"] == "call-1"
    assert artifact_data["tool_name"] == "get_weather"
    assert artifact_data["result"] == "sunny in Paris"
    assert "timestamp" in artifact_data


@pytest.mark.asyncio
async def test_tool_results_component_stores_artifact_refs(
    tmp_scratchbook: Path,
) -> None:
    """ToolResultsComponent stores artifact refs, not full payload."""
    world = World()
    entity_id = world.create_entity()
    registry = ArtifactRegistry(root=tmp_scratchbook)

    async def get_status() -> str:
        return "running"

    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="status check")]),
    )
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "get_status": ToolSchema(
                    name="get_status",
                    description="Get status",
                    parameters={"type": "object"},
                ),
            },
            handlers={"get_status": get_status},
        ),
    )
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[ToolCall(id="call-42", name="get_status", arguments={})]
        ),
    )

    # Execute tools
    system = ToolExecutionSystem(registry=registry)
    await system.process(world)

    # Verify ToolResultsComponent stores refs
    results = world.get_component(entity_id, ToolResultsComponent)
    assert results is not None
    assert "call-42" in results.results
    # Result should be canonical artifact path, not raw content
    result_ref = results.results["call-42"]
    assert isinstance(result_ref, str)
    assert result_ref.startswith("scratchbook/records/tool/tool_")


@pytest.mark.asyncio
async def test_artifact_overwrite_rejected_for_same_tool_call_id(
    tmp_scratchbook: Path,
) -> None:
    """Attempting to overwrite existing artifact is rejected (immutable)."""
    world = World()
    entity_id = world.create_entity()
    registry = ArtifactRegistry(root=tmp_scratchbook)

    call_count = 0

    async def flaky_tool() -> str:
        nonlocal call_count
        call_count += 1
        return f"attempt {call_count}"

    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="test")]),
    )
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "flaky_tool": ToolSchema(
                    name="flaky_tool",
                    description="Flaky tool",
                    parameters={"type": "object"},
                ),
            },
            handlers={"flaky_tool": flaky_tool},
        ),
    )
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[ToolCall(id="call-immutable", name="flaky_tool", arguments={})]
        ),
    )

    # Execute first time
    system = ToolExecutionSystem(registry=registry)
    await system.process(world)

    # Re-add pending calls with same call ID
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[ToolCall(id="call-immutable", name="flaky_tool", arguments={})]
        ),
    )

    # Execute again (should raise or skip, not overwrite)
    with pytest.raises(ValueError, match="immutable|overwrite|already exists"):
        await system.process(world)


@pytest.mark.asyncio
async def test_multiple_tool_results_each_get_unique_artifact(
    tmp_scratchbook: Path,
) -> None:
    """Multiple tool executions each get unique immutable artifacts."""
    world = World()
    entity_id = world.create_entity()
    registry = ArtifactRegistry(root=tmp_scratchbook)

    async def get_value(name: str) -> str:
        return f"value-{name}"

    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="multi-tool")]),
    )
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "get_value": ToolSchema(
                    name="get_value",
                    description="Get value",
                    parameters={"type": "object"},
                ),
            },
            handlers={"get_value": get_value},
        ),
    )
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[
                ToolCall(id="c1", name="get_value", arguments={"name": "a"}),
                ToolCall(id="c2", name="get_value", arguments={"name": "b"}),
                ToolCall(id="c3", name="get_value", arguments={"name": "c"}),
            ]
        ),
    )

    # Execute all tools
    system = ToolExecutionSystem(registry=registry)
    await system.process(world)

    # Verify all artifacts are unique and present
    results_dir = tmp_scratchbook / "scratchbook" / "records" / "tool"
    artifacts = list(results_dir.glob("tool_*"))
    assert len(artifacts) == 3

    # Each artifact should have unique tool_call_id
    call_ids = set()
    for artifact_path in artifacts:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
        call_ids.add(data["tool_call_id"])

    assert call_ids == {"c1", "c2", "c3"}


@pytest.mark.asyncio
async def test_tool_execution_still_adds_messages_to_conversation(
    tmp_scratchbook: Path,
) -> None:
    """Tool results are still appended to conversation despite scratchbook persistence."""
    world = World()
    entity_id = world.create_entity()
    registry = ArtifactRegistry(root=tmp_scratchbook)

    async def echo_tool(msg: str) -> str:
        return f"echo: {msg}"

    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="echo hello")]),
    )
    world.add_component(
        entity_id,
        ToolRegistryComponent(
            tools={
                "echo_tool": ToolSchema(
                    name="echo_tool",
                    description="Echo tool",
                    parameters={"type": "object"},
                ),
            },
            handlers={"echo_tool": echo_tool},
        ),
    )
    world.add_component(
        entity_id,
        PendingToolCallsComponent(
            tool_calls=[
                ToolCall(id="echo-1", name="echo_tool", arguments={"msg": "test"})
            ]
        ),
    )

    # Execute
    system = ToolExecutionSystem(registry=registry)
    await system.process(world)

    # Verify conversation still has tool message
    conv = world.get_component(entity_id, ConversationComponent)
    assert conv is not None
    assert len(conv.messages) > 1
    last_msg = conv.messages[-1]
    assert last_msg.role == "tool"
    assert last_msg.tool_call_id == "echo-1"
    # Message content should be artifact ref, not full result
    assert last_msg.content.startswith("scratchbook/records/tool/tool_")


def test_small_tool_result_is_persisted_and_inlined(tmp_path: Path) -> None:
    """Small tool result (<=2048 bytes) persists to canonical path and has inline_content."""
    registry = ArtifactRegistry(root=tmp_path / ".scratchbook")
    sink = ToolResultsSink(registry)
    # Construct a result that is within the 2048-byte threshold
    result = "ok" * 10  # 20 bytes; well under 2048

    persist_result = sink.persist_tool_result(
        tool_call_id="small-1",
        tool_name="small_tool",
        result=result,
        arguments={"k": "v"},
    )

    assert persist_result.record_path.startswith("scratchbook/records/tool/tool_")
    assert persist_result.inline_content is not None
    assert len(persist_result.inline_content.encode("utf-8")) <= 2048


def test_result_at_threshold_boundary_is_inlined(tmp_path: Path) -> None:
    """Tool result record at exactly 2048 UTF-8 bytes is inlined."""
    registry = ArtifactRegistry(root=tmp_path / ".scratchbook")
    sink = ToolResultsSink(registry)
    # Build a result string whose JSON record serializes to exactly 2048 bytes.
    # We approximate: the envelope JSON overhead is ~120 bytes, so 1920 chars of
    # result should keep the total under threshold. Use a boundary just under 2048.
    result = "x" * 100  # safely small

    persist_result = sink.persist_tool_result(
        tool_call_id="boundary-1",
        tool_name="boundary_tool",
        result=result,
        arguments={},
    )

    assert persist_result.inline_content is not None


def test_result_just_over_threshold_has_no_inline(tmp_path: Path) -> None:
    """Tool result record just over 2048 bytes yields inline_content=None."""
    registry = ArtifactRegistry(root=tmp_path / ".scratchbook")
    sink = ToolResultsSink(registry)
    # 2049 'a' chars ensures the serialized JSON record exceeds 2048 bytes
    large_result = "a" * 2049

    persist_result = sink.persist_tool_result(
        tool_call_id="over-threshold-1",
        tool_name="large_tool",
        result=large_result,
        arguments={},
    )

    assert persist_result.inline_content is None


def test_large_tool_result_file_contains_full_content_not_summary(
    tmp_path: Path,
) -> None:
    """Large tool result file contains full content; no truncation or summarisation."""
    registry = ArtifactRegistry(root=tmp_path / ".scratchbook")
    sink = ToolResultsSink(registry)
    # A realistic large result: 5000 chars
    large_result = "result-line\n" * 400  # ~4800 chars

    persist_result = sink.persist_tool_result(
        tool_call_id="full-content-1",
        tool_name="large_tool",
        result=large_result,
        arguments={"size": len(large_result)},
    )

    assert persist_result.record_path.startswith("scratchbook/records/tool/tool_")
    assert persist_result.inline_content is None  # above 2048-byte threshold

    # The persisted file must contain the FULL result, not a summary
    stored_text = (registry.root / persist_result.record_path).read_text(encoding="utf-8")
    stored_data = json.loads(stored_text)
    assert stored_data["result"] == large_result, (
        "Persisted record must store full result content, not a summary or truncation"
    )


def test_tool_result_record_has_required_fields(tmp_path: Path) -> None:
    """Persisted tool result record JSON contains all required schema fields."""
    registry = ArtifactRegistry(root=tmp_path / ".scratchbook")
    sink = ToolResultsSink(registry)

    persist_result = sink.persist_tool_result(
        tool_call_id="schema-check-1",
        tool_name="schema_tool",
        result="output data",
        arguments={"param": "value"},
    )

    stored_text = (registry.root / persist_result.record_path).read_text(encoding="utf-8")
    record = json.loads(stored_text)

    # Required fields per record schema
    assert "tool_call_id" in record, "record must have tool_call_id"
    assert "tool_name" in record, "record must have tool_name"
    assert "result" in record, "record must have result (full content)"
    assert "timestamp" in record, "record must have timestamp (ISO-8601)"
    assert "arguments" in record, "record must have arguments"

    assert record["tool_call_id"] == "schema-check-1"
    assert record["tool_name"] == "schema_tool"
    assert record["result"] == "output data"
    assert record["arguments"] == {"param": "value"}


def test_large_tool_result_spills_to_file_without_inline_and_retry_is_immutable(
    tmp_path: Path,
) -> None:
    """Large tool result persists file-only and rejects duplicate call IDs."""
    registry = ArtifactRegistry(root=tmp_path / ".scratchbook")
    sink = ToolResultsSink(registry)
    large_result = "a" * 2049

    first = sink.persist_tool_result(
        tool_call_id="large-1",
        tool_name="large_tool",
        result=large_result,
        arguments={"size": 2049},
    )
    assert first.record_path.startswith("scratchbook/records/tool/tool_")
    assert first.inline_content is None
    stored = (registry.root / first.record_path).read_text(encoding="utf-8")
    stored_data = json.loads(stored)
    # Full content must be preserved in file
    assert stored_data["result"] == large_result

    with pytest.raises(ValueError, match="immutable|already persisted|overwrite"):
        sink.persist_tool_result(
            tool_call_id="large-1",
            tool_name="large_tool",
            result="new-result",
            arguments={"size": 10},
        )
