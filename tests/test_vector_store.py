from __future__ import annotations

import importlib
import sys

import pytest

from ecs_agent.providers.vector_store import InMemoryVectorStore, VectorStore


class ConformingVectorStore:
    async def add(
        self,
        id: str,
        vector: list[float],
        metadata: dict[str, object] | None = None,
    ) -> None:
        del id, vector, metadata

    async def search(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[tuple[str, float]]:
        del query_vector, top_k
        return []

    async def delete(self, id: str) -> None:
        del id


class NonConformingVectorStore:
    async def only_add(self, id: str, vector: list[float]) -> None:
        del id, vector


@pytest.mark.asyncio
async def test_add_search_delete_lifecycle() -> None:
    store = InMemoryVectorStore(dimension=3)
    await store.add("a", [1.0, 0.0, 0.0])
    await store.add("b", [0.0, 1.0, 0.0])
    await store.add("c", [0.9, 0.1, 0.0])

    results = await store.search([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0][0] == "a"
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][0] == "c"

    await store.delete("a")
    after_delete = await store.search([1.0, 0.0, 0.0], top_k=2)
    assert all(result_id != "a" for result_id, _ in after_delete)


@pytest.mark.asyncio
async def test_empty_store_search_returns_empty_list() -> None:
    store = InMemoryVectorStore(dimension=3)
    results = await store.search([1.0, 0.0, 0.0])
    assert results == []


@pytest.mark.asyncio
async def test_add_dimension_mismatch_raises_value_error() -> None:
    store = InMemoryVectorStore(dimension=3)

    with pytest.raises(ValueError, match="Expected dimension 3, got 2"):
        await store.add("x", [1.0, 0.0])


@pytest.mark.asyncio
async def test_zero_vector_handling_returns_zero_score() -> None:
    store = InMemoryVectorStore(dimension=3)
    await store.add("zero", [0.0, 0.0, 0.0])
    await store.add("unit", [1.0, 0.0, 0.0])

    query_zero = await store.search([0.0, 0.0, 0.0], top_k=2)
    assert query_zero[0][1] == 0.0
    assert query_zero[1][1] == 0.0

    query_unit = await store.search([1.0, 0.0, 0.0], top_k=2)
    unit_result = dict(query_unit)
    assert unit_result["zero"] == 0.0


@pytest.mark.asyncio
async def test_cosine_similarity_correctness_and_ordering() -> None:
    store = InMemoryVectorStore(dimension=2)
    await store.add("same", [1.0, 0.0])
    await store.add("diag", [1.0, 1.0])
    await store.add("opposite", [-1.0, 0.0])

    results = await store.search([1.0, 0.0], top_k=3)

    assert [result_id for result_id, _ in results] == ["same", "diag", "opposite"]
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][1] == pytest.approx(0.70710678, rel=1e-6)
    assert results[2][1] == pytest.approx(-1.0)


@pytest.mark.asyncio
async def test_delete_missing_id_is_no_op() -> None:
    store = InMemoryVectorStore(dimension=2)
    await store.add("a", [1.0, 0.0])

    await store.delete("missing")
    results = await store.search([1.0, 0.0], top_k=1)
    assert results[0][0] == "a"


@pytest.mark.asyncio
async def test_search_respects_top_k() -> None:
    store = InMemoryVectorStore(dimension=2)
    await store.add("a", [1.0, 0.0])
    await store.add("b", [0.9, 0.1])
    await store.add("c", [0.8, 0.2])

    results = await store.search([1.0, 0.0], top_k=2)
    assert len(results) == 2
    assert [result_id for result_id, _ in results] == ["a", "b"]


def test_vector_store_protocol_is_runtime_checkable() -> None:
    assert isinstance(ConformingVectorStore(), VectorStore)
    assert not isinstance(NonConformingVectorStore(), VectorStore)


@pytest.mark.asyncio
async def test_in_memory_vector_store_works_without_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "ecs_agent.providers.vector_store"
    original_module = importlib.import_module(module_name)

    monkeypatch.setitem(sys.modules, "numpy", None)
    reloaded = importlib.reload(original_module)

    try:
        store = reloaded.InMemoryVectorStore(dimension=3)
        await store.add("a", [1.0, 0.0, 0.0])
        await store.add("b", [0.0, 1.0, 0.0])
        results = await store.search([1.0, 0.0, 0.0], top_k=2)
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0)
        assert results[1][1] == pytest.approx(0.0)
    finally:
        monkeypatch.undo()
        importlib.reload(reloaded)


def test_init_with_non_positive_dimension_raises_value_error() -> None:
    with pytest.raises(ValueError, match="dimension must be greater than 0"):
        InMemoryVectorStore(dimension=0)
