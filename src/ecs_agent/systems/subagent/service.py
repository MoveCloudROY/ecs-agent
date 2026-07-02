"""Subagent config resolution, result payloads, and artifact persistence (Task 8).

``SubagentService`` owns the pure, ECS-independent operations behind subagent
delegation: resolving a ``SubagentConfig`` for a request (registry lookup, free-form
fallback, RetryModel wrapping), validating tool parameters, normalizing skills,
resolving timeouts, building the JSON payloads returned by the control tools, and
persisting results to the artifact registry.

``SubagentSystem`` keeps thin methods delegating here so white-box tests that call
``system._resolve_subagent_config`` / ``_validate_subagent_params`` /
``_normalize_load_skills`` / ``_session_payload`` still work.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from ecs_agent.components import SubagentRegistryComponent
from ecs_agent.providers.fake_model import FakeModel
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.providers.retry_model import RetryModel
from ecs_agent.scratchbook.artifact_registry import ArtifactKind, ArtifactRegistry
from ecs_agent.types import RetryConfig, SubagentConfig, SubagentSessionRecord


class SubagentService:
    """Config resolution, result-payload rendering, and artifact persistence."""

    def __init__(
        self,
        *,
        default_timeout: float | None = None,
        registry: ArtifactRegistry | None = None,
    ) -> None:
        self._default_timeout = default_timeout
        self._registry = registry

    # --- artifact persistence ----------------------------------------------

    def persist_result(self, result: str) -> tuple[str, str, str | None] | None:
        if self._registry is None:
            return None

        persist_result = self._registry.persist(
            kind=ArtifactKind.SUBAGENT,
            content=result,
        )
        return (
            persist_result.descriptor.artifact_id,
            persist_result.record_path,
            persist_result.inline_content,
        )

    # --- result payloads ----------------------------------------------------

    def session_inline_content(self, session: SubagentSessionRecord) -> str | None:
        if session.artifact_inline_content is not None:
            return session.artifact_inline_content

        if session.artifact_record_path is not None:
            return (
                f"Result persisted to {session.artifact_record_path}. "
                "Read that file to access the full content."
            )

        return None

    def session_payload(
        self,
        session: SubagentSessionRecord,
        *,
        status: str,
        queue_position: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": status,
            "session_id": session.session_id,
            "category": session.category,
            "lifecycle_status": session.status,
            "artifact_id": session.artifact_id,
            "record_path": session.artifact_record_path,
            "inline_content": self.session_inline_content(session),
            "error": session.error,
        }
        if queue_position is not None:
            payload["queue_position"] = queue_position

        return payload

    def summary_payload(
        self,
        session: SubagentSessionRecord,
        *,
        status: str,
        queue_position: int | None = None,
    ) -> dict[str, Any]:
        payload = self.session_payload(
            session,
            status=status,
            queue_position=queue_position,
        )
        payload["read_method"] = "summary"
        payload["inline_content"] = session.result_summary
        return payload

    def terminal_result_payload(
        self,
        session: SubagentSessionRecord,
        read_method: str,
        session_id: str,
    ) -> str:
        """Render the subagent_result JSON for a (now terminal) session.

        Preserves the historical shapes: succeeded -> status="success";
        failed/timed_out/cancelled -> status="terminal"; read_method="summary"
        returns the cached summary or a summary-unavailable error.
        """
        status_label = "success" if session.status == "succeeded" else "terminal"
        if read_method == "summary":
            if session.result_summary is None:
                return json.dumps(
                    {
                        "error": 'Summary not available for this session. Retry with read_method="full".',
                        "read_method": "summary",
                        "session_id": session_id,
                    }
                )
            return json.dumps(self.summary_payload(session, status=status_label))
        return json.dumps(self.session_payload(session, status=status_label))

    # --- timeout / config resolution ---------------------------------------

    def resolve_timeout(self, per_call_timeout: float | None) -> float | None:
        """Resolve timeout with precedence: per-call > global > None."""
        return (
            per_call_timeout if per_call_timeout is not None else self._default_timeout
        )

    def config_for_session(
        self,
        registry: SubagentRegistryComponent,
        session: SubagentSessionRecord,
    ) -> SubagentConfig:
        base_config = self.resolve_subagent_config(registry, session.category)
        if session.load_skills == base_config.skills:
            return base_config

        return replace(base_config, skills=list(session.load_skills))

    def wrap_retry_model_if_needed(self, model: LLMModel) -> LLMModel:
        """Wrap model with RetryModel if not already wrapped.

        Returns RetryModel-wrapped model, or original if already wrapped or FakeModel.
        """
        if isinstance(model, RetryModel):
            return model

        # Skip FakeModel (deterministic tests)
        if isinstance(model, FakeModel):
            return model

        return RetryModel(model=model, retry_config=RetryConfig())

    def resolve_subagent_config(
        self,
        registry: SubagentRegistryComponent,
        subagent_name: str,
        *,
        parent_model: LLMModel | None = None,
    ) -> SubagentConfig:
        """Resolve and validate subagent configuration from registry."""
        config = registry.subagents.get(subagent_name)
        if config is None:
            config = self._resolve_free_subagent_config(
                registry,
                subagent_name,
                parent_model,
            )
            if config is None:
                raise ValueError(
                    f"Error: Unknown subagent '{subagent_name}'. Available subagents: {list(registry.subagents.keys())}"
                )

        # Wrap model with RetryModel by default
        wrapped_model = self.wrap_retry_model_if_needed(config.model)

        # Return config with wrapped model (use replace to preserve other fields)
        return replace(config, model=wrapped_model)

    def _resolve_free_subagent_config(
        self,
        registry: SubagentRegistryComponent,
        subagent_name: str,
        parent_model: LLMModel | None,
    ) -> SubagentConfig | None:
        free_config = registry.free_subagent_config
        if not free_config.enabled:
            return None
        if parent_model is None:
            raise ValueError(
                f"Error: Unknown subagent '{subagent_name}' cannot be created without a parent LLMComponent model"
            )
        return SubagentConfig(
            name=subagent_name,
            model=parent_model,
            description="Dynamically created free-form subagent.",
            system_prompt=free_config.system_prompt_template.replace(
                "{name}", subagent_name
            ),
            skills=list(free_config.skills),
            max_ticks=free_config.max_ticks,
            inheritance_policy=replace(free_config.inheritance_policy),
        )

    # --- parameter validation / skill normalization ------------------------

    def validate_subagent_params(
        self, category: str, prompt: str, load_skills: list[str]
    ) -> None:
        """Validate subagent invocation parameters. Raises ValueError if invalid."""
        if not category or not category.strip():
            raise ValueError("Error: category cannot be empty")
        if not prompt or not prompt.strip():
            raise ValueError("Error: prompt cannot be empty")
        if not isinstance(load_skills, list):
            raise ValueError(
                f"Error: load_skills must be a list, got {type(load_skills).__name__}"
            )

    def normalize_load_skills(
        self, config: SubagentConfig, load_skills: list[str]
    ) -> list[str]:
        """Ordered, de-duplicated merge of config.skills followed by load_skills."""
        seen: set[str] = set()
        result: list[str] = []
        for skill in config.skills + load_skills:
            if skill not in seen:
                seen.add(skill)
                result.append(skill)
        return result


__all__ = ["SubagentService"]
