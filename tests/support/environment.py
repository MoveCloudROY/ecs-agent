"""Explicit environment adapters for subprocesses launched by tests."""

from collections.abc import Mapping

from tests.support.config import LEGACY_CONFIG, LEGACY_SWITCHES, LiveSettings


def offline_environment(
    environment: Mapping[str, str], settings: LiveSettings | None = None
) -> dict[str, str]:
    """Keep normal process settings but remove service credentials/live switches."""
    sensitive = set(LEGACY_CONFIG) | set(LEGACY_SWITCHES)
    sensitive.update(
        {
            "LLM_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_LIVE_API_KEY",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_HOST",
            "LANGFUSE_BASE_URL",
            "IMAGE_URL",
            "PYTEST_ADDOPTS",
            "LLM_STREAM_READ_TIMEOUT",
            "LLM_ENABLE_STORE",
            "LLM_ENABLE_RESPONSE_CHAINING",
        }
    )
    if settings:
        sensitive.update(p.api_key_env for p in settings.profiles.values())
        sensitive.update(v for k, v in settings.langfuse.items() if k.endswith("_env"))
    return {k: v for k, v in environment.items() if k not in sensitive}


def example_environment(settings: LiveSettings) -> dict[str, str]:
    """Translate one profile to the existing example application's env contract."""
    p = settings.profile()
    environment = offline_environment(settings.environment, settings)
    environment.update(
        LLM_API_KEY=p.key(settings.environment),
        LLM_BASE_URL=p.base_url,
        LLM_MODEL=p.model,
        LLM_API_FORMAT=p.api_format.value,
        LLM_STREAM_READ_TIMEOUT=str(p.timeouts.stream_read_seconds),
        LLM_ENABLE_STORE="false",
        LLM_ENABLE_RESPONSE_CHAINING="false",
        PLAN_TASK_LANGFUSE="0",
    )
    return environment
