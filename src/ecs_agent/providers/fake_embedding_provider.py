"""Fake embedding provider for testing and deterministic workflows."""

from __future__ import annotations


class FakeEmbeddingProvider:
    """Fake embedding provider that returns deterministic vectors based on text hash.

    Used for testing and as a placeholder in demo workflows.
    """

    def __init__(self, dimension: int = 384) -> None:
        """Initialize with configured vector dimension.

        Args:
            dimension: Number of floats in each returned vector (default 384).
        """
        self._dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts into vectors deterministically.

        Args:
            texts: List of strings to embed.

        Returns:
            List of vectors (each vector is a list of floats).
            Empty input returns empty list.
        """
        if not texts:
            return []

        vectors: list[list[float]] = []
        for text in texts:
            # Use hash of text as deterministic seed
            seed = hash(text) % 1000 / 1000.0

            # Generate dimension floats deterministically from seed
            vector = []
            for i in range(self._dimension):
                # Mix seed with index to get different values for each dimension
                val = (seed + i * 0.1) % 1.0
                vector.append(val)

            vectors.append(vector)

        return vectors
