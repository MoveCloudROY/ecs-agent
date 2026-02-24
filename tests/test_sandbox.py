"""Tests for sandboxed tool execution with timeout and output limits."""

import asyncio

import pytest

from ecs_agent.tools.sandbox import sandboxed_execute
from ecs_agent.types import ToolTimeoutError


async def fast_tool(value: str) -> str:
    """Returns immediately."""
    return f"result: {value}"


async def slow_tool(sleep_seconds: float) -> str:
    """Sleeps then returns."""
    await asyncio.sleep(sleep_seconds)
    return "completed"


async def large_output_tool() -> str:
    """Returns a large string."""
    return "x" * 20_000


async def error_tool() -> str:
    """Raises ValueError."""
    raise ValueError("intentional error")


async def test_sandboxed_execute_normal_execution_completes() -> None:
    """Normal tool execution completes and returns result."""
    result = await sandboxed_execute(
        handler=fast_tool,
        arguments={"value": "test"},
        timeout=5.0,
    )
    assert result == "result: test"


async def test_sandboxed_execute_timeout_raises_tool_timeout_error() -> None:
    """Slow tool with short timeout raises ToolTimeoutError."""
    start = asyncio.get_event_loop().time()
    with pytest.raises(ToolTimeoutError, match="timed out after 1.0s"):
        await sandboxed_execute(
            handler=slow_tool,
            arguments={"sleep_seconds": 5.0},
            timeout=1.0,
        )
    elapsed = asyncio.get_event_loop().time() - start
    # Should timeout quickly (< 2s), not wait 5s for sleep to complete
    assert elapsed < 2.0, f"Timeout took {elapsed}s, expected < 2s"


async def test_sandboxed_execute_output_truncation() -> None:
    """Large output is truncated to max_output_size."""
    result = await sandboxed_execute(
        handler=large_output_tool,
        arguments={},
        timeout=5.0,
        max_output_size=100,
    )
    # Result should be truncated: 100 chars + "... [truncated]"
    assert len(result) < 200
    assert result.endswith("... [truncated]")
    assert result.startswith("xxxx")


async def test_sandboxed_execute_exception_propagates() -> None:
    """Exception from handler propagates (not swallowed)."""
    with pytest.raises(ValueError, match="intentional error"):
        await sandboxed_execute(
            handler=error_tool,
            arguments={},
            timeout=5.0,
        )


async def test_sandboxed_execute_default_parameters() -> None:
    """Default timeout and max_output_size work correctly."""
    result = await sandboxed_execute(
        handler=fast_tool,
        arguments={"value": "defaults"},
    )
    assert result == "result: defaults"
