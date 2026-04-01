from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, cast, get_type_hints

import pytest

from ecs_agent.components import ConversationComponent, UserPromptConfigComponent
from ecs_agent.core import World
from ecs_agent.prompts.contracts import TriggerSpec
from ecs_agent.systems.user_prompt_normalization_system import (
    UserPromptNormalizationSystem,
)
from ecs_agent.types import EntityId, Message

try:
    from ecs_agent.scratchbook import artifact_registry as artifact_registry_module
except ImportError as exc:
    artifact_registry_module = None
    IMPORT_ERROR: ImportError | None = exc
else:
    IMPORT_ERROR = None


UUID24_PATTERN = re.compile(r"^[0-9a-f]{24}$")


def _require_registry_symbol(name: str) -> Any:
    if IMPORT_ERROR is not None:
        pytest.fail(
            "ArtifactRegistry contract module is not available: "
            "ecs_agent.scratchbook.artifact_registry",
        )

    assert artifact_registry_module is not None
    if not hasattr(artifact_registry_module, name):
        pytest.fail(
            f"ArtifactRegistry contract symbol missing: "
            f"ecs_agent.scratchbook.artifact_registry.{name}",
        )
    return getattr(artifact_registry_module, name)


def _new_registry(tmp_path: Path) -> Any:
    registry_cls = _require_registry_symbol("ArtifactRegistry")
    return registry_cls(root=tmp_path)


def _assert_no_legacy_category_paths(path: str) -> None:
    assert "tool_results/" not in path
    assert "planning/" not in path
    assert "replanning/" not in path


def test_artifact_kind_enum_includes_required_members() -> None:
    artifact_kind = _require_registry_symbol("ArtifactKind")

    member_names = set(artifact_kind.__members__.keys())
    assert {"TOOL", "SUBAGENT", "PLAN", "BOULDER"}.issubset(member_names)


def test_artifact_descriptor_shape_matches_contract() -> None:
    descriptor_cls = _require_registry_symbol("ArtifactDescriptor")

    assert dataclasses.is_dataclass(descriptor_cls)
    assert hasattr(descriptor_cls, "__slots__")

    field_names = [field.name for field in dataclasses.fields(descriptor_cls)]
    assert field_names == [
        "artifact_id",
        "kind",
        "record_path",
        "created_at",
        "inline_content",
    ]

    hints = get_type_hints(descriptor_cls)
    assert hints["inline_content"] == str | None


def test_artifact_persist_result_shape_matches_contract() -> None:
    result_cls = _require_registry_symbol("ArtifactPersistResult")

    assert dataclasses.is_dataclass(result_cls)
    assert hasattr(result_cls, "__slots__")

    field_names = [field.name for field in dataclasses.fields(result_cls)]
    assert field_names == ["descriptor", "record_path", "inline_content"]

    hints = get_type_hints(result_cls)
    assert hints["inline_content"] == str | None


def test_tool_artifact_path_matches_records_tool_tool_uuid24(tmp_path: Path) -> None:
    artifact_kind = _require_registry_symbol("ArtifactKind")
    registry = _new_registry(tmp_path)

    result = registry.persist(kind=artifact_kind.TOOL, content="tool output")

    assert result.record_path == result.descriptor.record_path
    assert re.fullmatch(
        r"scratchbook/records/tool/tool_[0-9a-f]{24}", result.record_path
    )

    artifact_id = result.descriptor.artifact_id
    assert artifact_id.startswith("tool_")
    assert UUID24_PATTERN.fullmatch(artifact_id.replace("tool_", "", 1))
    _assert_no_legacy_category_paths(result.record_path)


def test_subagent_artifact_path_matches_records_subagent_subagent_uuid24(
    tmp_path: Path,
) -> None:
    artifact_kind = _require_registry_symbol("ArtifactKind")
    registry = _new_registry(tmp_path)

    result = registry.persist(kind=artifact_kind.SUBAGENT, content="subagent output")

    assert result.record_path == result.descriptor.record_path
    assert re.fullmatch(
        r"scratchbook/records/subagent/subagent_[0-9a-f]{24}",
        result.record_path,
    )

    artifact_id = result.descriptor.artifact_id
    assert artifact_id.startswith("subagent_")
    assert UUID24_PATTERN.fullmatch(artifact_id.replace("subagent_", "", 1))
    _assert_no_legacy_category_paths(result.record_path)


def test_inline_threshold_keeps_8192_utf8_bytes_inline(tmp_path: Path) -> None:
    artifact_kind = _require_registry_symbol("ArtifactKind")
    registry = _new_registry(tmp_path)

    content = "a" * 8192
    assert len(content.encode("utf-8")) == 8192

    result = registry.persist(kind=artifact_kind.TOOL, content=content)

    assert result.inline_content == content
    assert result.descriptor.inline_content == content


def test_inline_threshold_spills_8193_utf8_bytes_to_file_only(tmp_path: Path) -> None:
    artifact_kind = _require_registry_symbol("ArtifactKind")
    registry = _new_registry(tmp_path)

    content = "a" * 8193
    assert len(content.encode("utf-8")) == 8193

    result = registry.persist(kind=artifact_kind.TOOL, content=content)

    assert result.inline_content is None
    assert result.descriptor.inline_content is None
    file_path = tmp_path / result.record_path
    assert file_path.exists()


def test_plan_name_slugging_and_collision_resolution_are_deterministic(
    tmp_path: Path,
) -> None:
    registry = _new_registry(tmp_path)

    normalized = registry.normalize_plan_name("  Plan: Q2 / 2026 !!!  ")
    assert normalized == "plan-q2-2026"

    existing = {"plan-q2-2026", "plan-q2-2026-2"}
    resolved_1 = registry.resolve_plan_name_collision(
        normalized_plan_name=normalized,
        existing_plan_names=existing,
    )
    resolved_2 = registry.resolve_plan_name_collision(
        normalized_plan_name=normalized,
        existing_plan_names=existing,
    )

    assert resolved_1 == "plan-q2-2026-3"
    assert resolved_2 == "plan-q2-2026-3"


def test_plan_and_boulder_paths_match_canonical_layout(tmp_path: Path) -> None:
    registry = _new_registry(tmp_path)

    plan_path = registry.plan_path(plan_name="Roadmap 2026")
    boulder_path = registry.boulder_path(plan_name="Roadmap 2026")

    assert plan_path == "scratchbook/roadmap-2026/plan.md"
    assert boulder_path == "scratchbook/roadmap-2026/executes/boulder.json"
    _assert_no_legacy_category_paths(plan_path)
    _assert_no_legacy_category_paths(boulder_path)


def test_artifact_id_validation_accepts_uuid24_prefixed_ids(tmp_path: Path) -> None:
    registry = _new_registry(tmp_path)

    assert registry.is_valid_artifact_id("tool_0123456789abcdef01234567")
    assert registry.is_valid_artifact_id("subagent_0123456789abcdef01234567")


def test_artifact_id_validation_rejects_non_uuid24_or_unknown_prefix(
    tmp_path: Path,
) -> None:
    registry = _new_registry(tmp_path)

    assert not registry.is_valid_artifact_id("tool_" + "a" * 22)
    assert not registry.is_valid_artifact_id("tool_0123456789abcdef01234")
    assert not registry.is_valid_artifact_id("tool_0123456789abcdef0123456")
    assert not registry.is_valid_artifact_id("tool_0123456789abcdef01234g")
    assert not registry.is_valid_artifact_id("plan_0123456789abcdef012345")


def test_record_path_generation_is_canonical_and_relative(tmp_path: Path) -> None:
    artifact_kind = _require_registry_symbol("ArtifactKind")
    registry = _new_registry(tmp_path)

    tool_path = cast(
        str,
        registry.record_path_for(
            kind=artifact_kind.TOOL,
            artifact_id="tool_0123456789abcdef01234567",
        ),
    )
    subagent_path = cast(
        str,
        registry.record_path_for(
            kind=artifact_kind.SUBAGENT,
            artifact_id="subagent_0123456789abcdef01234567",
        ),
    )

    assert tool_path == "scratchbook/records/tool/tool_0123456789abcdef01234567"
    assert (
        subagent_path
        == "scratchbook/records/subagent/subagent_0123456789abcdef01234567"
    )
    assert not Path(tool_path).is_absolute()
    assert not Path(subagent_path).is_absolute()
    _assert_no_legacy_category_paths(tool_path)
    _assert_no_legacy_category_paths(subagent_path)


def test_plan_name_normalization_is_idempotent_and_ascii_safe(tmp_path: Path) -> None:
    registry = _new_registry(tmp_path)

    normalized_once = registry.normalize_plan_name("  _Mixed CASE__Name_  ")
    normalized_twice = registry.normalize_plan_name(normalized_once)

    assert normalized_once == "mixed-case-name"
    assert normalized_twice == "mixed-case-name"
    assert normalized_once.isascii()


def test_runtime_session_id_remains_distinct_from_subagent_artifact_id(
    tmp_path: Path,
) -> None:
    artifact_kind = _require_registry_symbol("ArtifactKind")
    registry = _new_registry(tmp_path)

    runtime_session_id = "ses_abc123"
    result = registry.persist(
        kind=artifact_kind.SUBAGENT,
        content="subagent output",
    )

    assert runtime_session_id != result.descriptor.artifact_id
    assert result.descriptor.artifact_id.startswith("subagent_")
    assert result.record_path.startswith("scratchbook/records/subagent/subagent_")


def test_background_subagent_completion_persists_full_output_to_records_subagent(
    tmp_path: Path,
) -> None:
    artifact_kind = _require_registry_symbol("ArtifactKind")
    registry = _new_registry(tmp_path)

    full_output = "y" * 9000
    result = registry.persist(
        kind=artifact_kind.SUBAGENT,
        content=full_output,
    )

    assert result.record_path.startswith("scratchbook/records/subagent/subagent_")
    persisted_file = tmp_path / result.record_path
    assert persisted_file.exists()
    assert persisted_file.read_text(encoding="utf-8") == full_output


def test_registry_descriptor_drives_context_resolution_without_legacy_categories(
    tmp_path: Path,
) -> None:
    from ecs_agent.components import TaskComponent
    from ecs_agent.scratchbook.service import ScratchbookService
    from ecs_agent.task import ContextResolver, ResolvedContext
    from ecs_agent.types import TaskStatus

    artifact_kind = _require_registry_symbol("ArtifactKind")
    registry = _new_registry(tmp_path)
    service = ScratchbookService(tmp_path)
    resolver = ContextResolver(service=service)

    result = registry.persist(
        kind=artifact_kind.TOOL,
        content='{"tool_call_id": "call-001", "result": "ok"}',
    )

    task = TaskComponent(
        task_id="task-registry-context-001",
        description="resolve registry artifact",
        expected_output="resolved",
        assigned_agent=None,
        tools=[],
        context_dependencies=[result.record_path],
        status=TaskStatus.READY,
    )

    resolved = resolver.resolve_context(task)

    assert isinstance(resolved, ResolvedContext)
    assert resolved.missing_refs == ()
    assert resolved.resolved_data[result.record_path] == {
        "tool_call_id": "call-001",
        "result": "ok",
    }


def test_plan_md_is_human_readable_projection_not_runtime_state_parser(
    tmp_path: Path,
) -> None:
    registry = _new_registry(tmp_path)

    content = (
        "# Plan: Roadmap 2026\n\n"
        "## Steps\n"
        "1. [DONE] Define scope\n"
        "2. [CURRENT] Draft implementation\n"
        "3. [ ] Verify rollout\n\n"
        "## Status\n"
        "Current step: 2\n"
        "Total steps: 3\n"
    )
    record_path = registry.write_plan(plan_name="Roadmap 2026", content=content)
    plan_file = tmp_path / record_path

    assert record_path == "scratchbook/roadmap-2026/plan.md"
    assert plan_file.exists()
    text = plan_file.read_text(encoding="utf-8")
    assert text.startswith("# Plan: Roadmap 2026")
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


def test_plan_name_slugging_is_reused_across_replanning_updates(tmp_path: Path) -> None:
    registry = _new_registry(tmp_path)

    first = registry.write_plan(
        plan_name="Roadmap 2026",
        content="# Plan: Roadmap 2026\n\n## Steps\n1. [CURRENT] Step A\n",
    )
    second = registry.write_plan(
        plan_name="Roadmap 2026",
        content="# Plan: Roadmap 2026\n\n## Steps\n1. [DONE] Step A\n2. [CURRENT] Step B\n",
    )

    assert first == "scratchbook/roadmap-2026/plan.md"
    assert second == first
    plan_file = tmp_path / second
    assert plan_file.read_text(encoding="utf-8").endswith("2. [CURRENT] Step B\n")
    _assert_no_legacy_category_paths(second)


def test_boulder_initial_schema_contains_required_fields(tmp_path: Path) -> None:
    registry = _new_registry(tmp_path)

    record_path = registry.create_boulder(
        plan_name="Roadmap 2026",
        initial_data={"world_names": ["primary"]},
    )
    boulder_file = tmp_path / record_path

    assert record_path == "scratchbook/roadmap-2026/executes/boulder.json"
    assert boulder_file.exists()

    payload = json.loads(boulder_file.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1"
    assert payload["plan_name"] == "roadmap-2026"
    assert payload["active_plan"] == "Roadmap 2026"
    assert payload["status"] in {"created", "running"}
    assert isinstance(payload["started_at"], str)
    parsed = datetime.fromisoformat(payload["started_at"].replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


@pytest.mark.asyncio
async def test_failed_tool_transition_marks_boulder_without_destroying_previous_state(
    tmp_path: Path,
) -> None:
    registry = _new_registry(tmp_path)

    record_path = registry.create_boulder(
        plan_name="Roadmap 2026",
        initial_data={"trigger_pattern": "@plan"},
    )
    await registry.update_boulder(
        plan_name="Roadmap 2026",
        updates={
            "status": "running",
            "current_step": 2,
            "last_tool_call_id": "tool-call-1",
            "last_tool_record_path": "scratchbook/records/tool/tool_0123456789abcdef01234567",
        },
    )
    await registry.update_boulder(
        plan_name="Roadmap 2026",
        updates={
            "status": "tool_failed",
            "last_error": "Error executing tool 'fetch': boom",
        },
    )

    payload = json.loads((tmp_path / record_path).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1"
    assert payload["plan_name"] == "roadmap-2026"
    assert payload["active_plan"] == "Roadmap 2026"
    assert isinstance(payload["started_at"], str)
    assert payload["status"] == "tool_failed"
    assert payload["last_error"] == "Error executing tool 'fetch': boom"
    assert payload["current_step"] == 2
    assert payload["last_tool_call_id"] == "tool-call-1"
    assert (
        payload["last_tool_record_path"]
        == "scratchbook/records/tool/tool_0123456789abcdef01234567"
    )
    assert isinstance(payload["last_updated_at"], str)


@pytest.mark.asyncio
async def test_non_plan_triggers_do_not_create_boulder(tmp_path: Path) -> None:
    async def tag_handler(
        world: World,
        entity_id: EntityId,
        user_text: str,
    ) -> str | None:
        return f"tagged: {user_text}"

    registry = _new_registry(tmp_path)
    world = World()
    entity_id = world.create_entity()
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="@tag hello")]),
    )
    world.add_component(
        entity_id,
        UserPromptConfigComponent(
            triggers=[
                TriggerSpec(
                    pattern="@tag",
                    match_mode="keyword",
                    action="script",
                    content="tag_handler",
                )
            ],
            script_handlers={"tag_handler": tag_handler},
        ),
    )

    await UserPromptNormalizationSystem(registry=registry).process(world)

    scratchbook_root = tmp_path / "scratchbook"
    if scratchbook_root.exists():
        assert list(scratchbook_root.rglob("boulder.json")) == []


@pytest.mark.asyncio
async def test_stream_capture_persists_incremental_output_without_whole_buffering(
    tmp_path: Path,
) -> None:
    artifact_kind = _require_registry_symbol("ArtifactKind")
    registry = _new_registry(tmp_path)

    chunks = ["abc", "def", "ghi", "jkl"] * 4000

    async def source() -> Any:
        for chunk in chunks:
            await asyncio.sleep(0)
            yield chunk

    descriptor = await registry.capture_stream(
        kind=artifact_kind.TOOL,
        source=source(),
    )

    persisted_file = tmp_path / descriptor.record_path
    assert persisted_file.exists()
    assert persisted_file.read_text(encoding="utf-8") == "".join(chunks)


@pytest.mark.asyncio
async def test_stream_capture_cleans_up_on_cancellation(tmp_path: Path) -> None:
    artifact_kind = _require_registry_symbol("ArtifactKind")
    registry = _new_registry(tmp_path)

    async def source() -> Any:
        yield "partial"
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await registry.capture_stream(
            kind=artifact_kind.SUBAGENT,
            source=source(),
        )

    records_root = tmp_path / "scratchbook" / "records"
    if records_root.exists():
        leaked_temps = [
            path
            for path in records_root.rglob("*")
            if path.is_file() and ".tmp" in path.name
        ]
        assert leaked_temps == []


@pytest.mark.asyncio
async def test_stream_capture_returns_registry_descriptor_with_record_path(
    tmp_path: Path,
) -> None:
    artifact_kind = _require_registry_symbol("ArtifactKind")
    registry = _new_registry(tmp_path)

    async def small_source() -> Any:
        yield "small-output"

    small_descriptor = await registry.capture_stream(
        kind=artifact_kind.TOOL,
        source=small_source(),
    )

    assert small_descriptor.artifact_id.startswith("tool_")
    assert small_descriptor.kind is artifact_kind.TOOL
    assert small_descriptor.record_path.startswith("scratchbook/records/tool/tool_")
    assert small_descriptor.inline_content == "small-output"

    async def large_source() -> Any:
        yield "z" * 8193

    large_descriptor = await registry.capture_stream(
        kind=artifact_kind.SUBAGENT,
        source=large_source(),
    )

    assert large_descriptor.artifact_id.startswith("subagent_")
    assert large_descriptor.kind is artifact_kind.SUBAGENT
    assert large_descriptor.record_path.startswith(
        "scratchbook/records/subagent/subagent_"
    )
    assert large_descriptor.inline_content is None
