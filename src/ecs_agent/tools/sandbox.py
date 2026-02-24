"""Sandboxed tool execution with timeout and output limits."""

import asyncio
from typing import Any, Awaitable, Callable

from ecs_agent.types import ToolTimeoutError


async def sandboxed_execute(
    handler: Callable[..., Awaitable[str]],
    arguments: dict[str, Any],
    timeout: float = 30.0,
    max_output_size: int = 10_000,
) -> str:
    """Execute handler with timeout and output size limits.

    Args:
        handler: Async function to execute
        arguments: Kwargs to pass to handler
        timeout: Seconds before timeout (default 30.0)
        max_output_size: Max characters in result (default 10,000)

    Returns:
        Result string (possibly truncated)

    Raises:
        ToolTimeoutError: If execution exceeds timeout
        Other exceptions: Propagated from handler unchanged
    """
    try:
        result = await asyncio.wait_for(handler(**arguments), timeout=timeout)
    except asyncio.TimeoutError as e:
        raise ToolTimeoutError(f"Tool execution timed out after {timeout}s") from e

    # Truncate if too long
    if len(result) > max_output_size:
        result = result[:max_output_size] + "... [truncated]"

    return result
