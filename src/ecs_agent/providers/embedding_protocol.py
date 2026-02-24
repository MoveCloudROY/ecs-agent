"""Embedding provider protocol for vector operations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding text into vectors.

    Implementations must provide an `embed` method that converts
    a list of text strings into their vector representations.
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts into vectors.

        Args:
            texts: List of strings to embed

        Returns:
            List of vectors (each vector is a list of floats)
        """
        ...
