"""Compatibility compilation of DSL specs into a partially assembled World."""

from collections.abc import Callable
from pathlib import Path

from ecs_agent.core.world import World
from ecs_agent.dsl.schema import AgentSpec
from ecs_agent.logging import get_logger
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.types import EntityId

logger = get_logger(__name__)


def compile_agent_specs(
    specs: dict[str, AgentSpec],
    model_factory: Callable[[str, str], LLMModel],
    *,
    source_dir: Path | None = None,
) -> tuple[EntityId, World]:
    """Install DSL components and preprocessing while retaining the partial-world contract.

    Use build_agent_from_specs for the complete runtime and an AgentHandle.
    """
    from ecs_agent.authoring.api import compile_partial_specs

    try:
        return compile_partial_specs(specs, model_factory, source_dir=source_dir)
    except Exception as exc:
        logger.error("agent_spec_compile_failed", exception=str(exc))
        raise


__all__ = ["compile_agent_specs"]
