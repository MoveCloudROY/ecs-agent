from __future__ import annotations

import importlib
import os
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from ecs_agent.providers.fake_model import FakeModel
from ecs_agent.providers.config import ApiFormat
from ecs_agent.types import CompletionResult, Message


class _OpenAIModelStub:
    def __init__(self, content: str = "stub response") -> None:
        self.complete: AsyncMock = AsyncMock(
            return_value=CompletionResult(
                message=Message(role="assistant", content=content),
            )
        )


def _completion(content: str = "fake demo response") -> CompletionResult:
    return CompletionResult(message=Message(role="assistant", content=content))


def _fake_provider(
    responses: int = 3, content: str = "fake demo response"
) -> FakeModel:
    return FakeModel(responses=[_completion(content) for _ in range(responses)])


def _load_example(module_name: str) -> Any:
    return importlib.import_module(f"examples.{module_name}")


@pytest.mark.asyncio
async def test_prompt_normalization_demo_fake_mode_prints_rendered_markers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_example("prompt_normalization_demo")

    with patch.dict(os.environ, {}, clear=True):
        with patch(
            "examples.prompt_normalization_demo.FakeModel",
            return_value=_fake_provider(),
        ) as fake_ctor:
            with patch("examples.prompt_normalization_demo.Model") as model_ctor:
                await module.main()

    fake_ctor.assert_called_once()
    model_ctor.assert_not_called()

    stdout = capsys.readouterr().out
    assert "[mode] fake" in stdout
    assert "=== 1. System Prompt: Built-in Placeholders ===" in stdout
    assert "- read_file: Read file contents" in stdout
    assert "- code_review: Review code for issues" in stdout
    assert "- researcher" in stdout
    assert "${_installed_tools} resolved" in stdout
    assert "=== 3. System Prompt: Full Rendered Text ===" in stdout
    assert "=== 4. User Prompt: Keyword Trigger Injection ===" in stdout
    assert "[PROMPT_INJECT:@code]" in stdout
    assert "=== 5. User Prompt: Outbound Message (with Context Pool) ===" in stdout
    assert "[PROMPT_CONTEXT_POOL]" in stdout
    assert "Prioritize deterministic code-first reasoning." in stdout
    assert "source: tool:search" in stdout
    assert "source: subagent:researcher" in stdout
    assert "Please @code summarize latest findings" in stdout
    assert "user text preserved: True" in stdout
    assert "=== 7. Verification Checks ===" in stdout


@pytest.mark.asyncio
async def test_prompt_normalization_demo_real_mode_wiring_uses_env_vars() -> None:
    module = _load_example("prompt_normalization_demo")

    with patch.dict(
        os.environ,
        {
            "LLM_API_KEY": "test-api-key",
            "LLM_BASE_URL": "https://example.test/v1",
            "LLM_MODEL": "example-model",
        },
        clear=True,
    ):
        with patch(
            "examples.prompt_normalization_demo.FakeModel",
            side_effect=FakeModel,
        ) as fake_ctor:
            with patch(
                "examples.prompt_normalization_demo.Model",
                return_value=_OpenAIModelStub(),
            ) as model_ctor:
                await module.main()

    fake_ctor.assert_not_called()
    model_ctor.assert_called_once()
    call = model_ctor.call_args
    assert call.args[0] == "example-model"
    assert call.kwargs["api_key"] == "test-api-key"
    assert call.kwargs["base_url"] == "https://example.test/v1"
    assert call.kwargs["api_format"] is ApiFormat.OPENAI_CHAT_COMPLETIONS


API_KEY = os.getenv("LLM_API_KEY", "")


@pytest.mark.skipif(not API_KEY, reason="LLM_API_KEY environment variable not set")
@pytest.mark.asyncio
async def test_prompt_normalization_demo_real_mode_env_gate() -> None:
    module = _load_example("prompt_normalization_demo")

    with patch(
        "examples.prompt_normalization_demo.Model",
        return_value=_OpenAIModelStub(),
    ) as model_ctor:
        await module.main()

    model_ctor.assert_called_once()
