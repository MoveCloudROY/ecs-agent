from __future__ import annotations

import pytest

from ecs_agent.core import World
from ecs_agent.scratchbook.prompt_provider import ScratchbookPromptPlaceholderProvider
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


def test_provider_emits_approved_placeholder_surface() -> None:
    config = ScratchbookPromptConfig(
        overview_default_template="Overview for ${scratchbook_path}\n${artifact_types}",
        scratchbook_root_path="scratchbook",
        artifacts=[
            ScratchbookArtifactPromptDef(
                artifact_type_id="Tool Output",
                path="scratchbook/records/tool/latest.md",
                purpose="Tool results.",
                readonly=True,
                read_when="When checking tool behavior.",
            ),
            ScratchbookArtifactPromptDef(
                artifact_type_id="Subagent Output",
                path="scratchbook/records/subagent/latest.md",
                purpose="Delegation output.",
                readonly=True,
                read_when="When checking delegated execution.",
            ),
        ],
    )

    provider = ScratchbookPromptPlaceholderProvider(config)
    world = World()
    entity_id = world.create_entity()

    placeholders = provider.resolve_placeholders(world, entity_id)

    assert "_scratchbook_path" in placeholders
    assert "_scratchbook_artifact_types" in placeholders
    assert "_scratchbook_artifacts" in placeholders
    assert "_scratchbook_overview" in placeholders
    assert "_scratchbook_artifact_tool_output" in placeholders
    assert "_scratchbook_artifact_subagent_output" in placeholders
    assert "_scratchbook_artifact_path_tool_output" in placeholders
    assert "_scratchbook_artifact_path_subagent_output" in placeholders


def test_empty_state_outputs_use_none_contract() -> None:
    config = ScratchbookPromptConfig(
        overview_default_template="Overview for ${scratchbook_path}\n${artifact_types}",
        scratchbook_root_path="scratchbook",
        artifacts=[],
    )

    provider = ScratchbookPromptPlaceholderProvider(config)
    world = World()
    entity_id = world.create_entity()

    placeholders = provider.resolve_placeholders(world, entity_id)

    assert placeholders["_scratchbook_artifacts"] == "- none"
    assert placeholders["_scratchbook_artifact_types"] == "- none"


def test_custom_overview_and_artifact_templates_override_defaults() -> None:
    config = ScratchbookPromptConfig(
        overview_default_template="Custom overview ${scratchbook_path}",
        scratchbook_root_path="scratchbook",
        artifacts=[
            ScratchbookArtifactPromptDef(
                artifact_type_id="tool_output",
                path="scratchbook/records/tool/latest.md",
                purpose="Tool results.",
                readonly=True,
                read_when="When checking tool behavior.",
                default_template_override="Default ${artifact_path}",
                user_override_template="User ${artifact_path}",
            )
        ],
    )

    provider = ScratchbookPromptPlaceholderProvider(config)
    world = World()
    entity_id = world.create_entity()

    placeholders = provider.resolve_placeholders(world, entity_id)

    assert placeholders["_scratchbook_overview"] == "Custom overview scratchbook"
    assert placeholders["_scratchbook_artifact_tool_output"].startswith("User ")
    assert (
        "scratchbook/records/tool/latest.md"
        in placeholders["_scratchbook_artifact_tool_output"]
    )


def test_readonly_emphasis_appears_in_artifact_block() -> None:
    config = ScratchbookPromptConfig(
        overview_default_template="Overview",
        scratchbook_root_path="scratchbook",
        artifacts=[
            ScratchbookArtifactPromptDef(
                artifact_type_id="tool_output",
                path="scratchbook/records/tool/latest.md",
                purpose="Tool results.",
                readonly=True,
                read_when="When checking tool behavior.",
            )
        ],
    )

    provider = ScratchbookPromptPlaceholderProvider(config)
    world = World()
    entity_id = world.create_entity()

    placeholders = provider.resolve_placeholders(world, entity_id)
    block = placeholders["_scratchbook_artifact_tool_output"]

    assert "READONLY" in block or "⚠️" in block


def test_scratchbook_path_is_root_relative_without_trailing_slash() -> None:
    config = ScratchbookPromptConfig(
        overview_default_template="Overview",
        scratchbook_root_path="scratchbook/",
        artifacts=[],
    )

    provider = ScratchbookPromptPlaceholderProvider(config)
    world = World()
    entity_id = world.create_entity()

    placeholders = provider.resolve_placeholders(world, entity_id)

    assert placeholders["_scratchbook_path"] == "scratchbook"


def test_artifact_blocks_are_sorted_deterministically() -> None:
    config = ScratchbookPromptConfig(
        overview_default_template="Overview",
        scratchbook_root_path="scratchbook",
        artifacts=[
            ScratchbookArtifactPromptDef(
                artifact_type_id="z_type",
                path="scratchbook/z.md",
                purpose="Z",
                readonly=True,
                read_when="Z only.",
            ),
            ScratchbookArtifactPromptDef(
                artifact_type_id="a_type",
                path="scratchbook/a.md",
                purpose="A",
                readonly=True,
                read_when="A only.",
            ),
        ],
    )

    provider = ScratchbookPromptPlaceholderProvider(config)
    world = World()
    entity_id = world.create_entity()

    placeholders = provider.resolve_placeholders(world, entity_id)

    assert placeholders["_scratchbook_artifact_types"] == "- a_type\n- z_type"
    assert placeholders["_scratchbook_artifacts"].index(
        "Artifact: a_type"
    ) < placeholders["_scratchbook_artifacts"].index("Artifact: z_type")


def test_provider_fingerprint_is_stable_for_same_config() -> None:
    config = ScratchbookPromptConfig(
        overview_default_template="Overview",
        scratchbook_root_path="scratchbook",
        artifacts=[
            ScratchbookArtifactPromptDef(
                artifact_type_id="tool_output",
                path="scratchbook/records/tool/latest.md",
                purpose="Tool results.",
                readonly=True,
                read_when="When checking tool behavior.",
            )
        ],
    )

    provider_one = ScratchbookPromptPlaceholderProvider(config)
    provider_two = ScratchbookPromptPlaceholderProvider(config)
    world = World()
    entity_id = world.create_entity()

    assert provider_one.provider_fingerprint(
        world,
        entity_id,
    ) == provider_two.provider_fingerprint(world, entity_id)
