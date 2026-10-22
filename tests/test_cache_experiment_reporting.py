"""A zero cache hit must remain a measured result in live cache experiments."""

import pytest

from ecs_agent.types import Usage
from tests.live.test_prompt_cache_experiment_live import _require_cache_usage


@pytest.mark.parametrize("usage", [
    Usage(cached_input_tokens=0),
    Usage(cache_read_tokens=0),
    Usage(cache_creation_tokens=0),
    Usage(cached_input_tokens=1024),
])
def test_cache_experiment_does_not_skip_reported_cache_fields(usage: Usage) -> None:
    _require_cache_usage([Usage(), usage], "test-provider")


def test_declared_cache_capability_fails_when_usage_fields_are_missing() -> None:
    with pytest.raises(pytest.fail.Exception, match="no prompt-cache usage fields"):
        _require_cache_usage([Usage(prompt_tokens=2048), Usage()], "test-provider")
