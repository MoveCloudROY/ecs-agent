"""Phase built-in placeholder provider for system prompt rendering."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ecs_agent.components import PhaseComponent, PhaseDefinitionComponent
from ecs_agent.core.world import World
from ecs_agent.phases.api import PhaseGraphMismatchError, PhaseIntegrityError
from ecs_agent.types import EntityId

PHASE_PROMPT_PLACEHOLDER = "_phase_prompt"


class PhasePromptPlaceholderProvider:
    """Expose the current phase's prompt for the bound agent key as ${_phase_prompt}."""

    provider_id = "phase_prompt"

    def resolve_placeholders(self, world: World, entity_id: EntityId) -> dict[str, str]:
        resolved = self._resolve_prompt(world, entity_id)
        if resolved is None:
            return {}
        return {PHASE_PROMPT_PLACEHOLDER: resolved}

    def provider_fingerprint(self, world: World, entity_id: EntityId) -> str:
        component = world.get_component(entity_id, PhaseComponent)
        if component is None:
            return "disabled"
        resolved = self._resolve_prompt(world, entity_id)
        if resolved is None:
            return "unbound"
        digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()
        # Content-hash only (no phase id): phases sharing identical prompt text
        # keep an identical fingerprint, so transitions between them never
        # invalidate the cache-stable prefix.
        return f"{component.agent_key}|hash:{digest}"

    def _resolve_prompt(self, world: World, entity_id: EntityId) -> str | None:
        component = world.get_component(entity_id, PhaseComponent)
        if component is None:
            return None
        definition = world.get_component(entity_id, PhaseDefinitionComponent)
        if definition is None:
            raise PhaseIntegrityError(
                f"entity {int(entity_id)} has PhaseComponent but no "
                "PhaseDefinitionComponent; call bind_phase_graph() after restoring "
                "a checkpoint"
            )
        spec = definition.graph.phases_by_id.get(component.phase)
        if spec is None:
            raise PhaseGraphMismatchError(
                f"phase {component.phase!r} not found in graph {component.graph_id!r}"
            )
        prompt = spec.prompts.get(component.agent_key)
        if prompt is None:
            return None
        if isinstance(prompt, Path):
            try:
                return prompt.read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(f"unreadable phase prompt file: {prompt}") from exc
        return prompt


__all__ = ["PHASE_PROMPT_PLACEHOLDER", "PhasePromptPlaceholderProvider"]
