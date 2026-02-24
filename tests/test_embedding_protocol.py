"""Tests for EmbeddingProvider protocol."""

import pytest

from ecs_agent.providers.embedding_protocol import EmbeddingProvider


class ConformingEmbedder:
    """Implements EmbeddingProvider interface without explicit inheritance."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return mock embeddings."""
        return [[0.1, 0.2, 0.3] for _ in texts]


class NonConformingEmbedder:
    """Does not implement embed() method."""

    async def some_other_method(self) -> None:
        """Wrong method signature."""
        pass


@pytest.mark.asyncio
async def test_embedding_provider_is_runtime_checkable() -> None:
    """EmbeddingProvider should be runtime checkable with isinstance."""
    # This test verifies the protocol is decorated with @runtime_checkable
    conformer = ConformingEmbedder()
    assert isinstance(conformer, EmbeddingProvider)


@pytest.mark.asyncio
async def test_conforming_class_passes_isinstance_check() -> None:
    """Class with async embed(texts: list[str]) -> list[list[float]] should pass isinstance."""
    embedder = ConformingEmbedder()
    assert isinstance(embedder, EmbeddingProvider)

    # Verify the method is callable
    result = await embedder.embed(["hello", "world"])
    assert len(result) == 2
    assert all(isinstance(vec, list) for vec in result)
    assert all(isinstance(val, float) for vec in result for val in vec)


@pytest.mark.asyncio
async def test_non_conforming_class_fails_isinstance_check() -> None:
    """Class without embed() method should fail isinstance check."""
    non_conformer = NonConformingEmbedder()
    assert not isinstance(non_conformer, EmbeddingProvider)
