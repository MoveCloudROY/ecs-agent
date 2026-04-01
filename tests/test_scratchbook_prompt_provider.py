from __future__ import annotations

import pytest

from ecs_agent.scratchbook.prompt_definition import (
    ScratchbookArtifactPromptDef,
    ScratchbookPromptConfig,
    normalize_artifact_type_id,
)


def test_artifact_type_normalization_is_deterministic_and_collision_safe() -> None:
    with pytest.raises(ValueError, match="normalize"):
        ScratchbookPromptConfig(
            overview_default_template=None,
            scratchbook_root_path="scratchbook",
            artifacts=[
                ScratchbookArtifactPromptDef(
                    artifact_type_id="Tool Output",
                    path="scratchbook/records/tool/latest.md",
                    purpose="Persist tool execution output.",
                    readonly=True,
                    read_when="Inspecting tool execution details.",
                ),
                ScratchbookArtifactPromptDef(
                    artifact_type_id="TOOL-OUTPUT",
                    path="scratchbook/records/tool/fallback.md",
                    purpose="Alternative location for tool output.",
                    readonly=True,
                    read_when="Fallback when canonical artifact is absent.",
                ),
            ],
        )


def test_invalid_or_colliding_artifact_types_raise_at_definition_time() -> None:
    with pytest.raises(ValueError, match="artifact_type_id"):
        ScratchbookArtifactPromptDef(
            artifact_type_id="   ",
            path="scratchbook/records/tool/latest.md",
            purpose="Persist tool execution output.",
            readonly=True,
            read_when="Inspecting tool execution details.",
        )

    with pytest.raises(ValueError, match="artifact_type_id"):
        ScratchbookArtifactPromptDef(
            artifact_type_id="!!!",
            path="scratchbook/records/tool/latest.md",
            purpose="Persist tool execution output.",
            readonly=True,
            read_when="Inspecting tool execution details.",
        )

    with pytest.raises(ValueError, match="normalize"):
        ScratchbookPromptConfig(
            overview_default_template=None,
            scratchbook_root_path="scratchbook",
            artifacts=[
                ScratchbookArtifactPromptDef(
                    artifact_type_id="subagent output",
                    path="scratchbook/records/subagent/latest.md",
                    purpose="Persist delegated execution output.",
                    readonly=True,
                    read_when="Inspecting child execution details.",
                ),
                ScratchbookArtifactPromptDef(
                    artifact_type_id="subagent_output",
                    path="scratchbook/records/subagent/fallback.md",
                    purpose="Alternative location for delegated output.",
                    readonly=True,
                    read_when="Fallback when canonical artifact is absent.",
                ),
            ],
        )


def test_overview_definition_supports_optional_override_template() -> None:
    without_override = ScratchbookPromptConfig(
        overview_default_template="Default overview template",
        scratchbook_root_path="scratchbook",
        artifacts=[],
    )
    with_override = ScratchbookPromptConfig(
        overview_default_template="User override template",
        scratchbook_root_path="scratchbook",
        artifacts=[],
    )

    assert without_override.overview_default_template == "Default overview template"
    assert with_override.overview_default_template == "User override template"


def test_artifact_override_template_is_optional() -> None:
    without_override = ScratchbookArtifactPromptDef(
        artifact_type_id="tool_output",
        path="scratchbook/records/tool/latest.md",
        purpose="Persist tool execution output.",
        readonly=True,
        read_when="Inspecting tool execution details.",
        default_template_override=None,
        user_override_template=None,
    )
    with_override = ScratchbookArtifactPromptDef(
        artifact_type_id="tool_output",
        path="scratchbook/records/tool/latest.md",
        purpose="Persist tool execution output.",
        readonly=True,
        read_when="Inspecting tool execution details.",
        default_template_override="Default artifact template.",
        user_override_template="User artifact template.",
    )

    assert without_override.default_template_override is None
    assert without_override.user_override_template is None
    assert with_override.default_template_override == "Default artifact template."
    assert with_override.user_override_template == "User artifact template."


def test_artifact_definition_includes_all_required_fields() -> None:
    artifact = ScratchbookArtifactPromptDef(
        artifact_type_id="tool_output",
        path="scratchbook/records/tool/latest.md",
        purpose="Persist tool execution output.",
        readonly=True,
        read_when="Inspecting tool execution details.",
    )

    assert artifact.path == "scratchbook/records/tool/latest.md"
    assert artifact.purpose == "Persist tool execution output."
    assert artifact.readonly is True
    assert artifact.read_when == "Inspecting tool execution details."


def test_artifact_type_normalization_examples() -> None:
    assert normalize_artifact_type_id("Tool Output") == "tool_output"
    assert normalize_artifact_type_id("tool_output") == "tool_output"
    assert normalize_artifact_type_id("TOOL-OUTPUT") == "tool_output"
    assert normalize_artifact_type_id("  tool@@@output  ") == "tool_output"
