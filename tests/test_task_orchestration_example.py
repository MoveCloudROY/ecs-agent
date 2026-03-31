from __future__ import annotations

import importlib
import os
from unittest.mock import AsyncMock, patch

import pytest

from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.types import CompletionResult, Message, ToolCall


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-flash"


class _OpenAIProviderStub:
    def __init__(self, responses: list[CompletionResult]) -> None:
        self.complete: AsyncMock = AsyncMock(side_effect=responses)


def _manager_responses() -> list[CompletionResult]:
    return [
        CompletionResult(
            message=Message(
                role="assistant",
                content="Using tool collect_constraints.",
                tool_calls=[
                    ToolCall(
                        id="tool_collect_requirements",
                        name="collect_constraints",
                        arguments={"scope": "launch readiness"},
                    )
                ],
            )
        ),
        CompletionResult(
            message=Message(
                role="assistant",
                content="Using tool synthesize_plan.",
                tool_calls=[
                    ToolCall(
                        id="tool_draft_execution_plan",
                        name="synthesize_plan",
                        arguments={"inputs": "requirements + research"},
                    )
                ],
            )
        ),
        CompletionResult(
            message=Message(
                role="assistant",
                content="Using tool write_brief.",
                tool_calls=[
                    ToolCall(
                        id="tool_publish_brief",
                        name="write_brief",
                        arguments={
                            "plan": "phased rollout",
                            "risks": "staffing and data quality",
                        },
                    )
                ],
            )
        ),
    ]


def _subagent_response(content: str) -> list[CompletionResult]:
    return [CompletionResult(message=Message(role="assistant", content=content))]


@pytest.mark.asyncio
async def test_task_orchestration_example_runs_end_to_end() -> None:
    module = importlib.import_module("examples.task_orchestration_system")

    with patch.dict(os.environ, {}, clear=True):
        report = await module.run_demo()

    assert report["waves"] == [
        ["collect_requirements", "background_research"],
        ["draft_execution_plan"],
        ["risk_review", "publish_brief"],
    ]
    assert report["backend_types"] == {
        "collect_requirements": "local",
        "background_research": "subagent",
        "draft_execution_plan": "local",
        "risk_review": "subagent",
        "publish_brief": "local",
    }
    assert set(report["completed_tasks"]) == {
        "collect_requirements",
        "background_research",
        "draft_execution_plan",
        "risk_review",
        "publish_brief",
    }
    assert report["blocked_transitions"] == {
        "draft_execution_plan": 1,
        "risk_review": 1,
        "publish_brief": 1,
    }
    assert report["snapshot_count"] == 5
    assert report["event_log_lengths"]["publish_brief"] >= 3
    assert report["serialization_roundtrip"]["restored_task_statuses"] == {
        "collect_requirements": "completed",
        "background_research": "completed",
        "draft_execution_plan": "completed",
        "risk_review": "completed",
        "publish_brief": "completed",
    }
    assert report["serialization_roundtrip"]["artifact_ids"] == [
        "background_research_result",
        "collect_requirements_result",
        "draft_execution_plan_result",
        "publish_brief_result",
        "risk_review_result",
    ]
    assert "write_brief" in report["final_brief"]


@pytest.mark.asyncio
async def test_task_orchestration_example_uses_openai_provider_in_real_mode() -> None:
    module = importlib.import_module("examples.task_orchestration_system")

    with patch.dict(os.environ, {"LLM_API_KEY": "test-api-key"}, clear=True):
        with patch(
            "examples.task_orchestration_system.FakeProvider",
            side_effect=AssertionError,
        ):
            with patch(
                "examples.task_orchestration_system.OpenAIProvider",
                side_effect=[
                    _OpenAIProviderStub(_manager_responses()),
                    _OpenAIProviderStub(
                        _subagent_response(
                            "background_research: real-mode subagent research result"
                        )
                    ),
                    _OpenAIProviderStub(
                        _subagent_response(
                            "risk_review: real-mode subagent risk result"
                        )
                    ),
                ],
                create=True,
            ) as openai_ctor:
                report = await module.run_demo()

    assert openai_ctor.call_count == 3
    for call in openai_ctor.call_args_list:
        config = call.kwargs["config"]
        assert isinstance(config, ProviderConfig)
        assert config.api_key == "test-api-key"
        assert config.base_url == DEFAULT_BASE_URL
        assert config.api_format is ApiFormat.OPENAI_CHAT_COMPLETIONS
        assert call.kwargs["model"] == DEFAULT_MODEL
    assert set(report["completed_tasks"]) == {
        "collect_requirements",
        "background_research",
        "draft_execution_plan",
        "risk_review",
        "publish_brief",
    }


REAL_API_KEY = os.getenv("LLM_API_KEY", "")


@pytest.mark.asyncio
@pytest.mark.skipif(not REAL_API_KEY, reason="LLM_API_KEY environment variable not set")
async def test_task_orchestration_example_runs_with_real_llm() -> None:
    module = importlib.import_module("examples.task_orchestration_system")

    report = await module.run_demo()

    assert len(report["waves"]) >= 1
    assert set(report["completed_tasks"]) == {
        "collect_requirements",
        "background_research",
        "draft_execution_plan",
        "risk_review",
        "publish_brief",
    }
