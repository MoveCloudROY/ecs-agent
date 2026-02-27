"""Tests for WebSearchSkill with Brave Search API integration."""

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ecs_agent.core import World
from ecs_agent.skills.web_search import WebSearchSkill
from ecs_agent.types import EntityId


@pytest.fixture
def mock_brave_response() -> dict[str, Any]:
    """Return a typical Brave Search API response."""
    return {
        "web": {
            "results": [
                {
                    "title": "Python Programming",
                    "url": "https://python.org",
                    "description": "Official Python website with docs and downloads.",
                },
                {
                    "title": "Learn Python",
                    "url": "https://learnpython.org",
                    "description": "Interactive Python tutorial for beginners.",
                },
            ]
        }
    }


async def test_web_search_skill_returns_formatted_results(
    mock_brave_response: dict[str, Any],
) -> None:
    """Test web search returns formatted text results (not raw JSON)."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = mock_brave_response
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        skill = WebSearchSkill(api_key="test-key")
        tools = skill.tools()

        assert "web_search" in tools
        schema, handler = tools["web_search"]
        assert schema.name == "web_search"
        assert callable(handler)

        result = await handler(query="python programming")

        # Should return formatted text, not JSON
        assert isinstance(result, str)
        assert "Python Programming" in result
        assert "https://python.org" in result
        assert "Official Python website" in result
        assert "Learn Python" in result
        assert "https://learnpython.org" in result

        # Verify httpx call
        mock_client.get.assert_awaited_once()
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "https://api.search.brave.com/res/v1/web/search"
        assert call_args[1]["headers"]["X-Subscription-Token"] == "test-key"
        assert call_args[1]["params"]["q"] == "python programming"


async def test_web_search_skill_missing_api_key_raises_valueerror() -> None:
    """Test that missing API key raises ValueError."""
    # Clear environment variable
    old_val = os.environ.pop("BRAVE_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="BRAVE_API_KEY"):
            WebSearchSkill()
    finally:
        if old_val is not None:
            os.environ["BRAVE_API_KEY"] = old_val


async def test_web_search_skill_uses_env_var_fallback() -> None:
    """Test that constructor falls back to BRAVE_API_KEY env var."""
    old_val = os.environ.get("BRAVE_API_KEY")
    try:
        os.environ["BRAVE_API_KEY"] = "env-test-key"
        skill = WebSearchSkill()
        assert skill._api_key == "env-test-key"
    finally:
        if old_val is not None:
            os.environ["BRAVE_API_KEY"] = old_val
        else:
            os.environ.pop("BRAVE_API_KEY", None)


async def test_web_search_skill_http_error_returns_error_string() -> None:
    """Test that HTTP errors return error string instead of raising exception."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=mock_response,
        )
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        skill = WebSearchSkill(api_key="test-key")
        tools = skill.tools()
        _, handler = tools["web_search"]

        result = await handler(query="test")

        assert isinstance(result, str)
        assert "error" in result.lower() or "failed" in result.lower()


async def test_web_search_skill_empty_results_returns_message() -> None:
    """Test that empty results return 'No results found' message."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        skill = WebSearchSkill(api_key="test-key")
        tools = skill.tools()
        _, handler = tools["web_search"]

        result = await handler(query="nonexistent query")

        assert isinstance(result, str)
        assert "no results" in result.lower()


async def test_web_search_skill_respects_count_parameter() -> None:
    """Test that count parameter is passed to API."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        skill = WebSearchSkill(api_key="test-key")
        tools = skill.tools()
        _, handler = tools["web_search"]

        await handler(query="test", count=5)

        call_args = mock_client.get.call_args
        assert call_args[1]["params"]["count"] == 5


async def test_web_search_skill_install_uninstall() -> None:
    """Test install and uninstall methods exist and are callable."""
    skill = WebSearchSkill(api_key="test-key")
    world = World()
    entity = world.create_entity()

    # Should not raise
    skill.install(world, entity)
    skill.uninstall(world, entity)


async def test_web_search_skill_has_required_attributes() -> None:
    """Test that WebSearchSkill has name and description attributes."""
    skill = WebSearchSkill(api_key="test-key")
    assert hasattr(skill, "name")
    assert hasattr(skill, "description")
    assert isinstance(skill.name, str)
    assert isinstance(skill.description, str)
    assert len(skill.name) > 0
    assert len(skill.description) > 0


async def test_web_search_skill_system_prompt() -> None:
    """Test that system_prompt method exists and returns a string."""
    skill = WebSearchSkill(api_key="test-key")
    prompt = skill.system_prompt()
    assert isinstance(prompt, str)
