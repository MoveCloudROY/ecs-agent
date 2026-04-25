"""Backward-compat alias — use retry_model instead."""
from ecs_agent.providers.retry_model import RetryModel as RetryProvider  # noqa: F401

__all__ = ["RetryProvider"]
