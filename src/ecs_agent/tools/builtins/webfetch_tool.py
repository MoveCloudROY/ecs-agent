"""Built-in webfetch tool for fetching URL content."""

from __future__ import annotations

from typing import Annotated

import httpx

from ecs_agent.logging import get_logger
from ecs_agent.tools.discovery import tool

logger = get_logger(__name__)


@tool(
    description="Fetch content from a URL and return the response body as text.",
    concurrency_safe=True,
)
async def webfetch(
    url: Annotated[str, "The URL to fetch."],
    timeout: Annotated[float, "Request timeout in seconds."] = 30.0,
) -> str:
    logger.info("webfetch", url=url, timeout=timeout)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text
