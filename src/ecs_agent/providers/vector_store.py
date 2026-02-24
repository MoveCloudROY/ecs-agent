"""Vector store protocol and in-memory cosine similarity implementation."""

from __future__ import annotations

import math
import importlib
from typing import Any, Protocol, runtime_checkable

try:
    np: Any | None = importlib.import_module("numpy")
except ImportError:
    np = None


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector storage and similarity search."""

    async def add(
        self,
        id: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a vector with optional metadata."""
        ...

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Search similar vectors sorted by descending score."""
        ...

    async def delete(self, id: str) -> None:
        """Delete vector by id; no-op when id is missing."""
        ...


class InMemoryVectorStore:
    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")

        self._dimension = dimension
        self._vectors: dict[str, list[float]] = {}
        self._metadata: dict[str, dict[str, Any] | None] = {}

    async def add(
        self,
        id: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._validate_dimension(vector)
        self._vectors[id] = list(vector)
        self._metadata[id] = metadata

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        self._validate_dimension(query_vector)
        if not self._vectors:
            return []

        scores: list[tuple[str, float]] = []
        for vector_id, vector in self._vectors.items():
            similarity = self._cosine_similarity(query_vector, vector)
            scores.append((vector_id, similarity))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    async def delete(self, id: str) -> None:
        self._vectors.pop(id, None)
        self._metadata.pop(id, None)

    def _validate_dimension(self, vector: list[float]) -> None:
        if len(vector) != self._dimension:
            raise ValueError(f"Expected dimension {self._dimension}, got {len(vector)}")

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if np is not None:
            a_array = np.array(a, dtype=float)
            b_array = np.array(b, dtype=float)
            norm_a = float(np.linalg.norm(a_array))
            norm_b = float(np.linalg.norm(b_array))
            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0

            return float(np.dot(a_array, b_array) / (norm_a * norm_b))

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)
