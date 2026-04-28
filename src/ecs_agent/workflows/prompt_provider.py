"""Workflow built-in placeholder provider for system prompt rendering."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ecs_agent.components import (
    WorkflowBindingComponent,
    WorkflowDefinitionComponent,
    WorkflowRuntimeComponent,
)
from ecs_agent.core.world import World
from ecs_agent.types import EntityId
from ecs_agent.workflows.compiler import CompiledPromptProfile, CompiledWorkflow


class WorkflowPromptPlaceholderProvider:
    """Expose the active workflow prompt profile as a built-in placeholder."""

    provider_id = "workflow_prompt"

    def resolve_placeholders(self, world: World, entity_id: EntityId) -> dict[str, str]:
        resolved = self._resolve_profile(world, entity_id)
        if resolved is None:
            return {}

        _, prompt_text = resolved
        return {"_workflow_state_prompt": prompt_text}

    def provider_fingerprint(self, world: World, entity_id: EntityId) -> str:
        state = self._workflow_state(world, entity_id)
        if state is None:
            return "disabled"

        resolved = self._resolve_profile(world, entity_id)
        if resolved is None:
            return "unbound"

        profile_id, prompt_text = resolved
        digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        return f"{state.agent_key}|{profile_id}|hash:{digest}"

    def _resolve_profile(
        self, world: World, entity_id: EntityId
    ) -> tuple[str, str] | None:
        state = self._workflow_state(world, entity_id)
        if state is None:
            return None

        state_bindings = state.compiled.bindings_by_state.get(state.current_state_id)
        if state_bindings is None:
            return None

        profile_id = state_bindings.get(state.agent_key)
        if profile_id is None:
            return None

        agent_profiles = state.compiled.profile_table.get(state.agent_key)
        if agent_profiles is None:
            return None

        profile = agent_profiles.get(profile_id)
        if profile is None:
            return None

        prompt_cache: dict[str, str] = {}
        return profile_id, self._resolve_prompt_text(profile, prompt_cache)

    def _workflow_state(
        self, world: World, entity_id: EntityId
    ) -> _WorkflowPromptState | None:
        binding = world.get_component(entity_id, WorkflowBindingComponent)
        runtime = world.get_component(entity_id, WorkflowRuntimeComponent)
        definition = world.get_component(entity_id, WorkflowDefinitionComponent)
        if binding is None or runtime is None or definition is None:
            return None

        return _WorkflowPromptState(
            agent_key=binding.agent_key,
            current_state_id=runtime.current_state_id,
            compiled=definition.compiled,
        )

    @staticmethod
    def _resolve_prompt_text(
        profile: CompiledPromptProfile, prompt_cache: dict[str, str]
    ) -> str:
        if profile.source_kind == "inline":
            assert profile.prompt_text is not None
            return profile.prompt_text
        if profile.source_kind == "path":
            assert profile.prompt_path is not None
            cached = prompt_cache.get(profile.prompt_path)
            if cached is not None:
                return cached

            try:
                prompt_text = Path(profile.prompt_path).read_text(encoding="utf-8")
            except OSError as exc:
                raise ValueError(
                    f"unreadable workflow prompt file: {profile.prompt_path}"
                ) from exc

            prompt_cache[profile.prompt_path] = prompt_text
            return prompt_text
        if profile.source_kind == "callable":
            assert profile.prompt_factory is not None
            return profile.prompt_factory()
        raise ValueError(f"Unknown source_kind: {profile.source_kind}")


@dataclass(frozen=True, slots=True)
class _WorkflowPromptState:
    agent_key: str
    current_state_id: str
    compiled: CompiledWorkflow


__all__ = ["WorkflowPromptPlaceholderProvider"]
