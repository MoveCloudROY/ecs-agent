"""Pytest configuration and shared fixtures."""

import pytest

pytest_plugins = ["tests.support.pytest_plugin", "pytester"]


@pytest.fixture(autouse=True)
def reset_ecs_logging() -> None:
    """Reset ecs-agent logging around each test to avoid leaked handlers."""
    from ecs_agent.logging import reset_logging

    reset_logging()
    yield
    reset_logging()


@pytest.fixture(autouse=True)
def deterministic_token_counting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the CJK-aware character fallback in ``token_counting``.

    Unit tests assert exact token estimates (often via ``chars_per_token``), which
    must be deterministic regardless of whether tiktoken's BPE ranks happen to be
    cached on the machine. The real BPE path is validated by the env-gated
    real-LLM tests instead.
    """
    import ecs_agent.token_counting as token_counting

    monkeypatch.setattr(token_counting, "_encoder", None, raising=False)
    monkeypatch.setattr(token_counting, "_encoder_loaded", True, raising=False)


@pytest.fixture(autouse=True)
def reset_subagent_scheduler() -> None:
    """Reset the global subagent scheduler singleton between tests to prevent state leakage."""
    from ecs_agent.systems.subagent_runtime import reset_global_scheduler

    reset_global_scheduler()
    yield
    reset_global_scheduler()
