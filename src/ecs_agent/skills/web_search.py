"""Web search skill using Brave Search API."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.skills.protocol import Skill
from ecs_agent.types import EntityId, ToolSchema

logger = get_logger(__name__)


class WebSearchSkill(Skill):
    """Skill providing web search via Brave Search API."""

    name = "web-search"
    description = "Web search using Brave Search API."

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize WebSearchSkill.

        Args:
            api_key: Brave API key. Falls back to BRAVE_API_KEY env var if None.

        Raises:
            ValueError: If no API key is provided and BRAVE_API_KEY is not set.
        """
        self._api_key: str = api_key or os.environ.get("BRAVE_API_KEY") or ""
        if not self._api_key:
            raise ValueError(
                "BRAVE_API_KEY must be set via constructor or environment variable"
            )
        self._client = httpx.AsyncClient(trust_env=False)

    def tools(self) -> dict[str, tuple[ToolSchema, Callable[..., Awaitable[str]]]]:
        """Return web search tool."""
        schema = ToolSchema(
            name="web_search",
            description="Search the web using Brave Search API",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of results to return (default 10)",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        )
        return {"web_search": (schema, self._web_search)}

    async def _web_search(self, query: str, count: int = 10) -> str:
        """Execute web search and return formatted results.

        Args:
            query: Search query string.
            count: Number of results to return.

        Returns:
            Formatted search results as text, or error message.
        """
        url = "https://api.search.brave.com/res/v1/web/search"
        headers: dict[str, str] = {
            "X-Subscription-Token": self._api_key,
            "Accept": "application/json",
        }
        params: dict[str, str | int] = {"q": query, "count": count}

        try:
            response = await self._client.get(url, headers=headers, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "web_search_http_error",
                status_code=exc.response.status_code,
                response_body=exc.response.text,
                exception=str(exc),
            )
            return f"Web search failed with HTTP error: {exc.response.status_code}"
        except httpx.RequestError as exc:
            logger.error(
                "web_search_network_error",
                exception_type=type(exc).__name__,
                exception=str(exc),
            )
            return f"Web search failed with network error: {type(exc).__name__}"

        response_data: dict[str, Any] = response.json()
        results = response_data.get("web", {}).get("results", [])

        if not results:
            return "No results found."

        # Format results as readable text
        formatted = []
        for idx, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url_str = result.get("url", "No URL")
            description = result.get("description", "No description")
            formatted.append(f"{idx}. {title}\n   {url_str}\n   {description}")

        return "\n\n".join(formatted)

    def system_prompt(self) -> str:
        """Return system prompt for web search skill."""
        return (
            "You have access to web search via the web_search tool. "
            "Use it to find current information, facts, or answers that require real-time data."
        )

    def install(self, world: World, entity_id: EntityId) -> None:
        """Install web search skill on entity.

        Args:
            world: World instance.
            entity_id: Entity to install skill on.
        """
        _ = world
        _ = entity_id
        logger.info("web_search_skill_install")

    def uninstall(self, world: World, entity_id: EntityId) -> None:
        """Uninstall web search skill from entity.

        Args:
            world: World instance.
            entity_id: Entity to uninstall skill from.
        """
        _ = world
        _ = entity_id
        logger.info("web_search_skill_uninstall")


__all__ = ["WebSearchSkill"]
