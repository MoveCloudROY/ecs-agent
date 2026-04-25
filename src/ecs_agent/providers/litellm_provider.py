"""Backward-compat alias — use litellm_model instead."""
from ecs_agent.providers.litellm_model import LiteLLMModel as LiteLLMProvider  # noqa: F401
from ecs_agent.providers.litellm_model import HAS_LITELLM  # noqa: F401

__all__ = ["LiteLLMProvider", "HAS_LITELLM"]
