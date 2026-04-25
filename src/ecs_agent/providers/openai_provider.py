"""Backward-compat alias — use openai_model instead."""
from ecs_agent.providers.openai_model import OpenAIModel as OpenAIProvider  # noqa: F401
from ecs_agent.providers.openai_model import pydantic_to_response_format  # noqa: F401

__all__ = ["OpenAIProvider", "pydantic_to_response_format"]
