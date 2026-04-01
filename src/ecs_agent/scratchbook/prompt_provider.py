"""Scratchbook built-in placeholder provider for system prompt rendering."""

from __future__ import annotations

from string import Template

from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.scratchbook.prompt_definition import (
    ScratchbookArtifactPromptDef,
    ScratchbookPromptConfig,
)
from ecs_agent.types import EntityId

logger = get_logger(__name__)

_NONE_CONTRACT_VALUE = "- none"
_DEFAULT_ARTIFACT_TEMPLATE = "\n".join(
    [
        "Artifact: ${artifact_type_id}",
        "Path: ${artifact_path}",
        "Purpose: ${purpose}",
        "${readonly_notice}",
        "Read when: ${read_when}",
    ]
)
_DEFAULT_OVERVIEW_TEMPLATE = "\n".join(
    [
        "Scratchbook path: ${scratchbook_path}",
        "Registered artifact types:",
        "${artifact_types}",
        "Use built-in read/write/edit tools to inspect or update artifacts when allowed.",
    ]
)


class ScratchbookPromptPlaceholderProvider:
    """Config-driven provider for scratchbook prompt placeholders."""

    provider_id = "scratchbook"

    def __init__(self, config: ScratchbookPromptConfig) -> None:
        self._config = config

    def resolve_placeholders(self, world: World, entity_id: EntityId) -> dict[str, str]:
        del world, entity_id

        normalized_root_path = self._normalize_root_path(
            self._config.scratchbook_root_path
        )
        sorted_artifacts = sorted(
            self._config.artifacts, key=lambda artifact: artifact.artifact_type_id
        )

        artifact_types_value = self._render_artifact_types(sorted_artifacts)
        artifacts_by_type: dict[str, str] = {}
        artifact_paths_by_type: dict[str, str] = {}
        rendered_blocks: list[str] = []

        for artifact in sorted_artifacts:
            rendered_block = self._render_artifact_block(artifact)
            artifacts_by_type[artifact.artifact_type_id] = rendered_block
            artifact_paths_by_type[artifact.artifact_type_id] = artifact.path
            rendered_blocks.append(rendered_block)

        placeholders: dict[str, str] = {
            "_scratchbook_path": normalized_root_path,
            "_scratchbook_artifact_types": artifact_types_value,
            "_scratchbook_artifacts": (
                "\n\n".join(rendered_blocks)
                if rendered_blocks
                else _NONE_CONTRACT_VALUE
            ),
            "_scratchbook_overview": self._render_overview(
                scratchbook_path=normalized_root_path,
                artifact_types=artifact_types_value,
            ),
        }

        for artifact_type_id in sorted(artifacts_by_type):
            placeholders[f"_scratchbook_artifact_{artifact_type_id}"] = (
                artifacts_by_type[artifact_type_id]
            )
            placeholders[f"_scratchbook_artifact_path_{artifact_type_id}"] = (
                artifact_paths_by_type[artifact_type_id]
            )

        return placeholders

    def provider_fingerprint(self, world: World, entity_id: EntityId) -> str:
        del world, entity_id

        normalized_root_path = self._normalize_root_path(
            self._config.scratchbook_root_path
        )
        sorted_pairs = sorted(
            (artifact.artifact_type_id, artifact.path)
            for artifact in self._config.artifacts
        )
        serialized_pairs = ",".join(
            f"{artifact_type_id}={artifact_path}"
            for artifact_type_id, artifact_path in sorted_pairs
        )
        return f"path:{normalized_root_path}|artifacts:{serialized_pairs}"

    @staticmethod
    def _normalize_root_path(root_path: str) -> str:
        return root_path.rstrip("/")

    @staticmethod
    def _render_artifact_types(artifacts: list[ScratchbookArtifactPromptDef]) -> str:
        if not artifacts:
            return _NONE_CONTRACT_VALUE
        return "\n".join(f"- {artifact.artifact_type_id}" for artifact in artifacts)

    @staticmethod
    def _render_artifact_block(artifact: ScratchbookArtifactPromptDef) -> str:
        if artifact.user_override_template is not None:
            template_source = artifact.user_override_template
        elif artifact.default_template_override is not None:
            template_source = artifact.default_template_override
        else:
            template_source = _DEFAULT_ARTIFACT_TEMPLATE

        readonly_notice = (
            "⚠️ READONLY — do NOT modify this file"
            if artifact.readonly
            else "Writable artifact — modify only when explicitly required"
        )
        return Template(template_source).substitute(
            {
                "artifact_type_id": artifact.artifact_type_id,
                "artifact_path": artifact.path,
                "purpose": artifact.purpose,
                "readonly_notice": readonly_notice,
                "read_when": artifact.read_when,
            }
        )

    def _render_overview(self, *, scratchbook_path: str, artifact_types: str) -> str:
        overview_template = self._config.overview_default_template
        if overview_template is None:
            return _NONE_CONTRACT_VALUE

        template_source = overview_template or _DEFAULT_OVERVIEW_TEMPLATE
        return Template(template_source).substitute(
            {
                "scratchbook_path": scratchbook_path,
                "artifact_types": artifact_types,
            }
        )


__all__ = [
    "ScratchbookPromptPlaceholderProvider",
]
