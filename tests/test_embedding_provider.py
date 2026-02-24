"""Tests for embedding provider implementations."""


from __future__ import annotations
import pytest
import httpx
from unittest.mock import AsyncMock, patch

from ecs_agent.providers.embedding_protocol import EmbeddingProvider
from ecs_agent.providers.fake_embedding_provider import FakeEmbeddingProvider
from ecs_agent.providers.embedding_provider import OpenAIEmbeddingProvider


@pytest.mark.asyncio
async def test_fake_embedding_provider_returns_deterministic_vectors() -> None:
    """FakeEmbeddingProvider should return deterministic vectors for same input."""
    provider = FakeEmbeddingProvider(dimension=8)

    result1 = await provider.embed(["hello"])
    result2 = await provider.embed(["hello"])

    assert result1 == result2, "Same input should produce same vector"
    assert len(result1) == 1
    assert len(result1[0]) == 8
    assert all(isinstance(v, float) for v in result1[0])


@pytest.mark.asyncio
async def test_fake_embedding_provider_satisfies_protocol() -> None:
    """FakeEmbeddingProvider should satisfy EmbeddingProvider protocol."""
    provider = FakeEmbeddingProvider(dimension=384)
    assert isinstance(provider, EmbeddingProvider)


@pytest.mark.asyncio
async def test_fake_embedding_provider_empty_list_returns_empty() -> None:
    """FakeEmbeddingProvider.embed([]) should return []."""
    provider = FakeEmbeddingProvider(dimension=10)

    result = await provider.embed([])

    assert result == []


@pytest.mark.asyncio
async def test_fake_embedding_provider_single_text_correct_dimension() -> None:
    """FakeEmbeddingProvider should return vector of configured dimension."""
    provider = FakeEmbeddingProvider(dimension=16)

    result = await provider.embed(["hello"])

    assert len(result) == 1
    assert len(result[0]) == 16


@pytest.mark.asyncio
async def test_fake_embedding_provider_multiple_texts() -> None:
    """FakeEmbeddingProvider should return one vector per input text."""
    provider = FakeEmbeddingProvider(dimension=5)

    result = await provider.embed(["a", "b"])

    assert len(result) == 2
    assert len(result[0]) == 5
    assert len(result[1]) == 5


@pytest.mark.asyncio
async def test_fake_embedding_provider_different_texts_different_vectors() -> None:
    """Different input texts should produce different vectors."""
    provider = FakeEmbeddingProvider(dimension=8)

    result1 = await provider.embed(["hello"])
    result2 = await provider.embed(["world"])

    assert result1 != result2


# OpenAIEmbeddingProvider Tests


@pytest.mark.asyncio
async def test_openai_embedding_provider_satisfies_protocol() -> None:
    """OpenAIEmbeddingProvider should satisfy EmbeddingProvider protocol."""
    provider = OpenAIEmbeddingProvider(api_key="test-key")
    assert isinstance(provider, EmbeddingProvider)


@pytest.mark.asyncio
async def test_openai_embedding_provider_makes_correct_api_call() -> None:
    """OpenAIEmbeddingProvider should POST correct request to embeddings endpoint."""
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        base_url="https://api.example.com/v1",
        model="text-embedding-3-small",
    )

    request = httpx.Request("POST", "https://api.example.com/v1/embeddings")
    mock_response = httpx.Response(
        status_code=200,
        request=request,
        json={
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
                {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1},
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        },
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        result = await provider.embed(["hello", "world"])

        # Verify API call was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.args[0] == "https://api.example.com/v1/embeddings"
        assert call_args.kwargs["json"] == {
            "model": "text-embedding-3-small",
            "input": ["hello", "world"],
        }
        assert call_args.kwargs["headers"] == {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        }

        # Verify response parsing
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@pytest.mark.asyncio
async def test_openai_embedding_provider_parses_response_correctly() -> None:
    """OpenAIEmbeddingProvider should extract embeddings from response data."""
    provider = OpenAIEmbeddingProvider(api_key="test-key")

    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    mock_response = httpx.Response(
        status_code=200,
        request=request,
        json={
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [1.0, 2.0], "index": 0},
                {"object": "embedding", "embedding": [3.0, 4.0], "index": 1},
                {"object": "embedding", "embedding": [5.0, 6.0], "index": 2},
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        },
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        result = await provider.embed(["first", "second", "third"])

        assert len(result) == 3
        assert result[0] == [1.0, 2.0]
        assert result[1] == [3.0, 4.0]
        assert result[2] == [5.0, 6.0]


@pytest.mark.asyncio
async def test_openai_embedding_provider_handles_http_error() -> None:
    """OpenAIEmbeddingProvider should log and re-raise HTTPStatusError."""
    provider = OpenAIEmbeddingProvider(api_key="test-key")

    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    mock_response = httpx.Response(
        status_code=401,
        request=request,
        json={"error": {"message": "Invalid API key", "type": "invalid_request_error"}},
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        with pytest.raises(httpx.HTTPStatusError):
            await provider.embed(["test"])


@pytest.mark.asyncio
async def test_openai_embedding_provider_handles_request_error() -> None:
    """OpenAIEmbeddingProvider should log and re-raise RequestError."""
    provider = OpenAIEmbeddingProvider(api_key="test-key")

    with patch.object(
        provider._client, "post", new_callable=AsyncMock
    ) as mock_post:
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(httpx.ConnectError):
            await provider.embed(["test"])


@pytest.mark.asyncio
async def test_openai_embedding_provider_empty_input_returns_empty() -> None:
    """OpenAIEmbeddingProvider should return empty list for empty input without API call."""
    provider = OpenAIEmbeddingProvider(api_key="test-key")

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        result = await provider.embed([])

        # Should not make API call
        mock_post.assert_not_called()
        assert result == []


@pytest.mark.asyncio
async def test_openai_embedding_provider_default_model() -> None:
    """OpenAIEmbeddingProvider should use default model text-embedding-3-small."""
    provider = OpenAIEmbeddingProvider(api_key="test-key")

    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    mock_response = httpx.Response(
        status_code=200,
        request=request,
        json={
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1], "index": 0}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        await provider.embed(["test"])

        call_args = mock_post.call_args
        assert call_args.kwargs["json"]["model"] == "text-embedding-3-small"


@pytest.mark.asyncio
async def test_openai_embedding_provider_custom_timeouts() -> None:
    """OpenAIEmbeddingProvider should respect custom timeout settings."""
    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        connect_timeout=5.0,
        read_timeout=60.0,
    )

    assert provider._timeout.connect == 5.0
    assert provider._timeout.read == 60.0