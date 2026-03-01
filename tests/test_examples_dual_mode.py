from __future__ import annotations

import importlib
import os
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from ecs_agent.providers.fake_provider import FakeProvider
from ecs_agent.providers.fake_embedding_provider import FakeEmbeddingProvider
from ecs_agent.types import CompletionResult, Message

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-flash"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v3"
DEFAULT_EMBEDDING_DIMENSION = 1024


class _OpenAIProviderStub:
    def __init__(self, content: str = "stub response") -> None:
        self.complete: AsyncMock = AsyncMock(
            return_value=CompletionResult(
                message=Message(role="assistant", content=content),
            )
        )


class _EmbeddingProviderStub:
    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIMENSION) -> None:
        self._dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dimension for _ in texts]


class _VectorStoreStub:
    def __init__(self) -> None:
        self._metadata: dict[str, dict[str, str]] = {}

    async def add(
        self, doc_id: str, _vector: list[float], metadata: dict[str, str]
    ) -> None:
        self._metadata[doc_id] = metadata

    async def search(
        self, _query_vector: list[float], top_k: int = 3
    ) -> list[tuple[str, float]]:
        doc_ids = list(self._metadata.keys())[:top_k]
        return [(doc_id, 1.0) for doc_id in doc_ids]


def _completion(content: str = "fake response") -> CompletionResult:
    return CompletionResult(message=Message(role="assistant", content=content))


def _fake_provider(responses: int = 20, content: str = "fake response") -> FakeProvider:
    return FakeProvider(responses=[_completion(content) for _ in range(responses)])


def _load_example(module_name: str) -> Any:
    return importlib.import_module(f"examples.{module_name}")


def _assert_openai_defaults(openai_ctor: Any, expected_count: int) -> None:
    assert openai_ctor.call_count == expected_count
    for call in openai_ctor.call_args_list:
        assert call.kwargs["api_key"] == "test-api-key"
        assert call.kwargs["base_url"] == DEFAULT_BASE_URL
        assert call.kwargs["model"] == DEFAULT_MODEL


@pytest.mark.asyncio
class TestChatAgentDualMode:
    async def test_fake_mode(self) -> None:
        module = _load_example("chat_agent")

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "examples.chat_agent.FakeProvider", return_value=_fake_provider()
            ) as fake_ctor:
                with patch(
                    "examples.chat_agent.OpenAIProvider", create=True
                ) as openai_ctor:
                    await module.main()

        fake_ctor.assert_called_once()
        openai_ctor.assert_not_called()

    async def test_real_mode(self) -> None:
        module = _load_example("chat_agent")

        with patch.dict(os.environ, {"LLM_API_KEY": "test-api-key"}, clear=True):
            with patch(
                "examples.chat_agent.FakeProvider", side_effect=FakeProvider
            ) as fake_ctor:
                with patch(
                    "examples.chat_agent.OpenAIProvider",
                    return_value=_OpenAIProviderStub(),
                    create=True,
                ) as openai_ctor:
                    await module.main()

        fake_ctor.assert_not_called()
        _assert_openai_defaults(openai_ctor, expected_count=1)


@pytest.mark.asyncio
class TestMultiAgentDualMode:
    async def test_fake_mode(self) -> None:
        module = _load_example("multi_agent")

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "examples.multi_agent.FakeProvider",
                side_effect=[_fake_provider(), _fake_provider()],
            ) as fake_ctor:
                with patch(
                    "examples.multi_agent.OpenAIProvider", create=True
                ) as openai_ctor:
                    await module.main()

        assert fake_ctor.call_count == 2
        openai_ctor.assert_not_called()

    async def test_real_mode(self) -> None:
        module = _load_example("multi_agent")

        with patch.dict(os.environ, {"LLM_API_KEY": "test-api-key"}, clear=True):
            with patch(
                "examples.multi_agent.FakeProvider", side_effect=FakeProvider
            ) as fake_ctor:
                with patch(
                    "examples.multi_agent.OpenAIProvider",
                    side_effect=[_OpenAIProviderStub("a"), _OpenAIProviderStub("b")],
                    create=True,
                ) as openai_ctor:
                    await module.main()

        fake_ctor.assert_not_called()
        _assert_openai_defaults(openai_ctor, expected_count=2)


@pytest.mark.asyncio
class TestSkillDiscoveryAgentDualMode:
    async def test_fake_mode(self) -> None:
        module = _load_example("skill_discovery_agent")

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "examples.skill_discovery_agent.FakeProvider",
                return_value=_fake_provider(),
            ) as fake_ctor:
                with patch(
                    "examples.skill_discovery_agent.OpenAIProvider", create=True
                ) as openai_ctor:
                    await module.main()

        fake_ctor.assert_called_once()
        openai_ctor.assert_not_called()

    async def test_real_mode(self) -> None:
        module = _load_example("skill_discovery_agent")

        with patch.dict(os.environ, {"LLM_API_KEY": "test-api-key"}, clear=True):
            with patch(
                "examples.skill_discovery_agent.FakeProvider", side_effect=FakeProvider
            ) as fake_ctor:
                with patch(
                    "examples.skill_discovery_agent.OpenAIProvider",
                    return_value=_OpenAIProviderStub(),
                    create=True,
                ) as openai_ctor:
                    await module.main()

        fake_ctor.assert_not_called()
        _assert_openai_defaults(openai_ctor, expected_count=1)


@pytest.mark.asyncio
class TestPermissionAgentDualMode:
    async def test_fake_mode(self) -> None:
        module = _load_example("permission_agent")

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "examples.permission_agent.FakeProvider",
                return_value=_fake_provider(),
            ) as fake_ctor:
                with patch(
                    "examples.permission_agent.OpenAIProvider", create=True
                ) as openai_ctor:
                    await module.main()

        fake_ctor.assert_called_once()
        openai_ctor.assert_not_called()

    async def test_real_mode(self) -> None:
        module = _load_example("permission_agent")

        with patch.dict(os.environ, {"LLM_API_KEY": "test-api-key"}, clear=True):
            with patch(
                "examples.permission_agent.FakeProvider", side_effect=FakeProvider
            ) as fake_ctor:
                with patch(
                    "examples.permission_agent.OpenAIProvider",
                    return_value=_OpenAIProviderStub(),
                    create=True,
                ) as openai_ctor:
                    await module.main()

        fake_ctor.assert_not_called()
        _assert_openai_defaults(openai_ctor, expected_count=1)


@pytest.mark.asyncio
class TestSubagentDelegationDualMode:
    async def test_fake_mode(self) -> None:
        module = _load_example("subagent_delegation")

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "examples.subagent_delegation.FakeProvider",
                side_effect=[_fake_provider(), _fake_provider(), _fake_provider()],
            ) as fake_ctor:
                with patch(
                    "examples.subagent_delegation.OpenAIProvider", create=True
                ) as openai_ctor:
                    await module.main()

        assert fake_ctor.call_count == 3
        openai_ctor.assert_not_called()

    async def test_real_mode(self) -> None:
        module = _load_example("subagent_delegation")

        with patch.dict(os.environ, {"LLM_API_KEY": "test-api-key"}, clear=True):
            with patch(
                "examples.subagent_delegation.FakeProvider", side_effect=FakeProvider
            ) as fake_ctor:
                with patch(
                    "examples.subagent_delegation.OpenAIProvider",
                    side_effect=[
                        _OpenAIProviderStub("manager"),
                        _OpenAIProviderStub("subagent"),
                        _OpenAIProviderStub("summary"),
                    ],
                    create=True,
                ) as openai_ctor:
                    await module.main()

        fake_ctor.assert_not_called()
        _assert_openai_defaults(openai_ctor, expected_count=3)


@pytest.mark.asyncio
class TestTreeSearchAgentDualMode:
    async def test_fake_mode(self) -> None:
        module = _load_example("tree_search_agent")

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "examples.tree_search_agent.FakeProvider",
                return_value=_fake_provider(content="0.75"),
            ) as fake_ctor:
                with patch(
                    "examples.tree_search_agent.OpenAIProvider", create=True
                ) as openai_ctor:
                    await module.main()

        fake_ctor.assert_called_once()
        openai_ctor.assert_not_called()

    async def test_real_mode(self) -> None:
        module = _load_example("tree_search_agent")

        with patch.dict(os.environ, {"LLM_API_KEY": "test-api-key"}, clear=True):
            with patch(
                "examples.tree_search_agent.FakeProvider", side_effect=FakeProvider
            ) as fake_ctor:
                with patch(
                    "examples.tree_search_agent.OpenAIProvider",
                    return_value=_OpenAIProviderStub("0.75"),
                    create=True,
                ) as openai_ctor:
                    await module.main()

        fake_ctor.assert_not_called()
        _assert_openai_defaults(openai_ctor, expected_count=1)


@pytest.mark.asyncio
class TestContextManagementAgentDualMode:
    async def test_fake_mode(self) -> None:
        module = _load_example("context_management_agent")

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "examples.context_management_agent.FakeProvider",
                side_effect=[_fake_provider(), _fake_provider(), _fake_provider()],
            ) as fake_ctor:
                with patch(
                    "examples.context_management_agent.OpenAIProvider", create=True
                ) as openai_ctor:
                    await module.main()

        assert fake_ctor.call_count == 3
        openai_ctor.assert_not_called()

    async def test_real_mode(self) -> None:
        module = _load_example("context_management_agent")

        with patch.dict(os.environ, {"LLM_API_KEY": "test-api-key"}, clear=True):
            with patch(
                "examples.context_management_agent.FakeProvider",
                side_effect=FakeProvider,
            ) as fake_ctor:
                with patch(
                    "examples.context_management_agent.OpenAIProvider",
                    side_effect=[
                        _OpenAIProviderStub("r1"),
                        _OpenAIProviderStub("r2"),
                        _OpenAIProviderStub("r3"),
                    ],
                    create=True,
                ) as openai_ctor:
                    await module.main()

        fake_ctor.assert_not_called()
        _assert_openai_defaults(openai_ctor, expected_count=3)


@pytest.mark.asyncio
class TestRAGAgentDualMode:
    async def test_fake_mode(self) -> None:
        module = _load_example("rag_agent")

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "examples.rag_agent.FakeProvider",
                return_value=_fake_provider(),
            ) as fake_llm_ctor:
                with patch(
                    "examples.rag_agent.FakeEmbeddingProvider",
                    return_value=_EmbeddingProviderStub(dimension=8),
                ) as fake_embed_ctor:
                    with patch(
                        "examples.rag_agent.OpenAIProvider", create=True
                    ) as openai_ctor:
                        with patch(
                            "examples.rag_agent.OpenAIEmbeddingProvider", create=True
                        ) as openai_embed_ctor:
                            await module.main()

        fake_llm_ctor.assert_called_once()
        fake_embed_ctor.assert_called_once()
        openai_ctor.assert_not_called()
        openai_embed_ctor.assert_not_called()

    async def test_real_mode(self) -> None:
        module = _load_example("rag_agent")

        with patch.dict(os.environ, {"LLM_API_KEY": "test-api-key"}, clear=True):
            with patch(
                "examples.rag_agent.FakeProvider", side_effect=FakeProvider
            ) as fake_llm_ctor:
                with patch(
                    "examples.rag_agent.FakeEmbeddingProvider",
                    side_effect=FakeEmbeddingProvider,
                ) as fake_embed_ctor:
                    with patch(
                        "examples.rag_agent.OpenAIProvider",
                        return_value=_OpenAIProviderStub(),
                        create=True,
                    ) as openai_ctor:
                        with patch(
                            "examples.rag_agent.OpenAIEmbeddingProvider",
                            return_value=_EmbeddingProviderStub(),
                            create=True,
                        ) as openai_embed_ctor:
                            with patch(
                                "examples.rag_agent.InMemoryVectorStore",
                                return_value=_VectorStoreStub(),
                            ) as vector_store_ctor:
                                await module.main()

        fake_llm_ctor.assert_not_called()
        fake_embed_ctor.assert_not_called()
        _assert_openai_defaults(openai_ctor, expected_count=1)
        assert openai_embed_ctor.call_count == 1
        embed_call = openai_embed_ctor.call_args
        assert embed_call.kwargs["api_key"] == "test-api-key"
        assert embed_call.kwargs["base_url"] == DEFAULT_BASE_URL
        assert embed_call.kwargs["model"] == DEFAULT_EMBEDDING_MODEL
        vector_store_ctor.assert_called_once_with(dimension=DEFAULT_EMBEDDING_DIMENSION)
