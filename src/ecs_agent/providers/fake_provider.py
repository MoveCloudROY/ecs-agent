"""Backward-compat alias — use fake_model instead."""
from ecs_agent.providers.fake_model import FakeModel as FakeProvider  # noqa: F401

__all__ = ["FakeProvider"]
