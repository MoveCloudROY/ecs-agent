"""Pytest configuration and shared fixtures."""

import os

import pytest


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


@pytest.fixture
def live_api_key() -> str:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        pytest.skip("LLM_API_KEY not set")
    return api_key


@pytest.fixture
def live_image_url() -> str:
    image_url = os.getenv("IMAGE_URL")
    if not image_url:
        pytest.skip("IMAGE_URL not set")
    return image_url
