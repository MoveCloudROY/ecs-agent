from __future__ import annotations

import importlib
import os
from io import StringIO
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from ecs_agent.providers.fake_provider import FakeProvider
from ecs_agent.providers.fake_embedding_provider import FakeEmbeddingProvider
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.systems.subagent_wait import SubagentWaitSystem
from ecs_agent.types import CompletionResult, Message
from ecs_agent.types import ToolCall

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-flash"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v3"
DEFAULT_EMBEDDING_DIMENSION = 1024


class _OpenAIProviderStub:
    model_id: str = "stub"

    def __init__(self, content: str = "stub response") -> None:
        self.complete: AsyncMock = AsyncMock(
            return_value=CompletionResult(
                message=Message(role="assistant", content=content),
            )
        )


class _SubagentDelegationOpenAIStub:
    model_id: str = "stub"

    def __init__(self) -> None:
        self._manager_turn = 0

    async def complete(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        thread_response_id: str | None = None,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult:
        _ = thread_response_id
        _ = stream
        _ = response_format

        if tools:
            return self._next_manager_response()

        prompt = messages[-1].content
        if "early quantum value" in prompt:
            return _completion(
                "Synchronous subagent answer: quantum gains will arrive first in optimization and chemistry."
            )
        if "slow background answer" in prompt:
            return _completion(
                "<subagent_background_result>\n"
                "<summary>Slow background summary for the parent.</summary>\n"
                "<full_result>Slow background answer finished after the queued job waited its turn.</full_result>\n"
                "</subagent_background_result>"
            )
        if "queued background answer" in prompt:
            return _completion(
                "<subagent_background_result>\n"
                "<summary>Queued background summary for the parent.</summary>\n"
                "<full_result>Queued background answer completed once the slow session released the slot.</full_result>\n"
                "</subagent_background_result>"
            )
        if "Stream a concise answer back to the parent" in prompt:
            return _completion(
                "Streamed background answer delivered to the parent event bus."
            )

        raise AssertionError(f"Unexpected subagent prompt: {prompt}")

    def _next_manager_response(self) -> CompletionResult:
        self._manager_turn += 1

        if self._manager_turn == 1:
            return CompletionResult(
                message=Message(
                    role="assistant",
                    content="Starting synchronous delegation.",
                    tool_calls=[
                        ToolCall(
                            id="call-sync",
                            name="subagent",
                            arguments={
                                "category": "sync-worker",
                                "prompt": "Give one sentence on where early quantum value appears.",
                                "background": False,
                            },
                        )
                    ],
                )
            )

        if self._manager_turn == 2:
            return CompletionResult(
                message=Message(
                    role="assistant",
                    content="Launching the slow background worker.",
                    tool_calls=[
                        ToolCall(
                            id="call-slow-background",
                            name="subagent",
                            arguments={
                                "category": "slow-worker",
                                "prompt": "Produce the slow background answer.",
                                "background": True,
                            },
                        )
                    ],
                )
            )

        if self._manager_turn == 3:
            return CompletionResult(
                message=Message(
                    role="assistant",
                    content="Launching the queued background worker.",
                    tool_calls=[
                        ToolCall(
                            id="call-queued-background",
                            name="subagent",
                            arguments={
                                "category": "queued-worker",
                                "prompt": "Produce the queued background answer.",
                                "background": True,
                            },
                        )
                    ],
                )
            )

        if self._manager_turn == 4:
            return CompletionResult(
                message=Message(
                    role="assistant",
                    content="Waiting for the background workers to finish.",
                    tool_calls=[
                        ToolCall(
                            id="call-background-wait",
                            name="subagent_wait",
                            arguments={
                                "session_ids": [
                                    "session-slow-worker",
                                    "session-queued-worker",
                                ]
                            },
                        )
                    ],
                )
            )

        if self._manager_turn == 5:
            return CompletionResult(
                message=Message(
                    role="assistant",
                    content="Reading the cached slow background summary.",
                    tool_calls=[
                        ToolCall(
                            id="call-slow-result",
                            name="subagent_result",
                            arguments={
                                "session_id": "session-slow-worker",
                                "read_method": "summary",
                            },
                        )
                    ],
                )
            )

        if self._manager_turn == 6:
            return CompletionResult(
                message=Message(
                    role="assistant",
                    content="Reading the full queued background result.",
                    tool_calls=[
                        ToolCall(
                            id="call-queued-result",
                            name="subagent_result",
                            arguments={
                                "session_id": "session-queued-worker",
                                "read_method": "full",
                            },
                        )
                    ],
                )
            )

        if self._manager_turn == 7:
            return CompletionResult(
                message=Message(
                    role="assistant",
                    content="Launching the streaming background worker.",
                    tool_calls=[
                        ToolCall(
                            id="call-stream-background",
                            name="subagent",
                            arguments={
                                "category": "stream-worker",
                                "prompt": "Stream a concise answer back to the parent.",
                                "background": True,
                                "stream": True,
                            },
                        )
                    ],
                )
            )

        if self._manager_turn == 8:
            return CompletionResult(
                message=Message(
                    role="assistant",
                    content="Collecting the streaming background result.",
                    tool_calls=[
                        ToolCall(
                            id="call-stream-result",
                            name="subagent_result",
                            arguments={
                                "session_id": "session-stream-worker",
                                "timeout": 5.0,
                            },
                        )
                    ],
                )
            )

        if self._manager_turn == 9:
            return _completion(
                "Delegation complete. Sync, background, and stream runs succeeded, and the background lifecycle progressed from queued to running to succeeded."
            )

        raise AssertionError(f"Unexpected manager turn: {self._manager_turn}")


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
        config = call.kwargs["config"]
        assert isinstance(config, ProviderConfig)
        assert config.api_key == "test-api-key"
        assert config.base_url == DEFAULT_BASE_URL
        assert config.api_format is ApiFormat.OPENAI_CHAT_COMPLETIONS
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
class TestScriptSkillDiscoveryAgentDualMode:
    async def test_fake_mode(self) -> None:
        module = _load_example("script_skill_discovery_agent")

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "examples.script_skill_discovery_agent.FakeProvider",
                return_value=_fake_provider(),
            ) as fake_ctor:
                with patch(
                    "examples.script_skill_discovery_agent.OpenAIProvider", create=True
                ) as openai_ctor:
                    await module.main()

        fake_ctor.assert_called_once()
        openai_ctor.assert_not_called()

    async def test_real_mode(self) -> None:
        module = _load_example("script_skill_discovery_agent")

        with patch.dict(os.environ, {"LLM_API_KEY": "test-api-key"}, clear=True):
            with patch(
                "examples.script_skill_discovery_agent.FakeProvider",
                side_effect=FakeProvider,
            ) as fake_ctor:
                with patch(
                    "examples.script_skill_discovery_agent.OpenAIProvider",
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
        import ecs_agent.systems.subagent_runtime as runtime_module

        module = _load_example("subagent_delegation")

        stdout = StringIO()

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(runtime_module, "_GLOBAL_SCHEDULER", None):
                with patch(
                    "examples.subagent_delegation.FakeProvider",
                    side_effect=FakeProvider,
                ) as fake_ctor:
                    with patch(
                        "examples.subagent_delegation.OpenAIProvider", create=True
                    ) as openai_ctor:
                        with patch("sys.stdout", stdout):
                            await module.main()

                assert fake_ctor.call_count == 5
                openai_ctor.assert_not_called()

                output = stdout.getvalue()
                world, parent_id = module._build_world(
                    parent_provider=FakeProvider(
                        responses=module._fake_parent_responses()
                    ),
                    registry=module._build_providers(
                        "", DEFAULT_BASE_URL, DEFAULT_MODEL
                    )[1],
                    model="fake-parent",
                )
                world.apply_pending_system_operations()
                system_names = [
                    entry.system.__class__.__name__
                    for entry in world._systems._systems  # type: ignore[attr-defined]
                ]
                wait_priorities = [
                    entry.priority
                    for entry in world._systems._systems  # type: ignore[attr-defined]
                    if isinstance(entry.system, SubagentWaitSystem)
                ]

                assert parent_id >= 0
                assert "SubagentWaitSystem" in system_names
                assert wait_priorities == [-5]
                assert "TOOL CALL HISTORY" in output
                assert "[Action] subagent" in output
                assert "[Action] subagent_wait" in output
                assert (
                    "[Action] subagent_result({'session_id': 'session-slow-worker', 'read_method': 'summary'})"
                    in output
                )
                assert (
                    "[Action] subagent_result({'session_id': 'session-queued-worker', 'read_method': 'full'})"
                    in output
                )
                assert "[System] Background subagent updates:" in output
                assert output.index(
                    "[System] Background subagent updates:"
                ) < output.index(
                    "[Action] subagent_result({'session_id': 'session-slow-worker', 'read_method': 'summary'})"
                )
                assert "[Result]" in output
                assert "Synchronous subagent run" in output
                assert "Background queue lifecycle" in output
                assert "Streamed background subagent" in output
                assert "sync" in output.lower()
                assert "background" in output.lower()
                assert "stream" in output.lower()
                assert "queued" in output
                assert "running" in output
                assert "succeeded" in output
                assert "Working" not in output
                assert "[Action] subagent_status" not in output

    async def test_real_mode(self) -> None:
        import ecs_agent.systems.subagent_runtime as runtime_module

        module = _load_example("subagent_delegation")

        stdout = StringIO()

        with patch.dict(os.environ, {"LLM_API_KEY": "test-api-key"}, clear=True):
            with patch.object(runtime_module, "_GLOBAL_SCHEDULER", None):
                with patch(
                    "examples.subagent_delegation.FakeProvider",
                    side_effect=FakeProvider,
                ) as fake_ctor:
                    with patch(
                        "examples.subagent_delegation.OpenAIProvider",
                        return_value=_SubagentDelegationOpenAIStub(),
                        create=True,
                    ) as openai_ctor:
                        with patch("sys.stdout", stdout):
                            await module.main()

        fake_ctor.assert_not_called()
        _assert_openai_defaults(openai_ctor, expected_count=1)

        output = stdout.getvalue()
        assert "TOOL CALL HISTORY" in output
        assert "[Action] subagent" in output
        assert "[Action] subagent_wait" in output
        assert (
            "[Action] subagent_result({'session_id': 'session-slow-worker', 'read_method': 'summary'})"
            in output
        )
        assert (
            "[Action] subagent_result({'session_id': 'session-queued-worker', 'read_method': 'full'})"
            in output
        )
        assert "[System] Background subagent updates:" in output
        assert output.index("[System] Background subagent updates:") < output.index(
            "[Action] subagent_result({'session_id': 'session-slow-worker', 'read_method': 'summary'})"
        )
        assert "[Result]" in output
        assert "Synchronous subagent run" in output
        assert "Background queue lifecycle" in output
        assert "Streamed background subagent" in output
        assert "sync" in output.lower()
        assert "background" in output.lower()
        assert "stream" in output.lower()
        assert "queued" in output
        assert "running" in output
        assert "succeeded" in output
        assert "Working" not in output
        assert "[Action] subagent_status" not in output


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
