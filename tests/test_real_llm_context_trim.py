"""Env-gated real-LLM validation of the ISSUE-5 trim -> compaction pipeline.

With ``ANTHROPIC_API_KEY`` set, runs ``CompactionSystem`` against a real
Anthropic-format endpoint: an over-budget conversation is first trimmed
(oldest tool span dropped, permanently) and, when trimming is not enough, the
real model produces a compaction summary. Skips cleanly without a key.
"""

from __future__ import annotations

import os

import pytest

from ecs_agent.components import (
    CompactionConfigComponent,
    ContextTrimConfig,
    ConversationArchiveComponent,
    ConversationComponent,
    CurrentCompactionSummaryComponent,
    LLMComponent,
)
from ecs_agent.core import World
from ecs_agent.providers.claude_model import ClaudeModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.types import Message, ToolCall

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")


def _claude_model() -> ClaudeModel:
    config = ProviderConfig(
        provider_id="anthropic",
        base_url=ANTHROPIC_BASE_URL,
        api_key=ANTHROPIC_API_KEY,
        api_format=ApiFormat.ANTHROPIC_MESSAGES,
    )
    return ClaudeModel(config=config, model=ANTHROPIC_MODEL, max_tokens=256)


@pytest.mark.skipif(
    not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY environment variable not set"
)
@pytest.mark.asyncio
async def test_real_trim_then_summarize_pipeline() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(model=_claude_model()))
    # Oldest turn is a tool span (trimmable); the rest is essential prose that
    # cannot be trimmed, so trimming alone won't fit -> real summary is produced.
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[
                Message(role="user", content="Look up the capital of France."),
                Message(
                    role="assistant",
                    content="Using the lookup tool.",
                    tool_calls=[ToolCall(id="c1", name="lookup", arguments={})],
                ),
                Message(role="tool", content="Paris. " * 200, tool_call_id="c1"),
                Message(
                    role="assistant",
                    content="The capital of France is Paris. " * 40,
                ),
                Message(role="user", content="Now summarize our whole discussion. " * 40),
            ]
        ),
    )
    world.add_component(entity_id, ConversationArchiveComponent())
    world.add_component(
        entity_id, CompactionConfigComponent(threshold_tokens=50)
    )
    world.add_component(entity_id, ContextTrimConfig(max_tokens=50))

    await CompactionSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    # The oldest tool span was permanently dropped by the trim step.
    assert all("Paris. Paris." not in (m.content or "") for m in conversation.messages)
    # Trimming wasn't enough -> the real model produced a non-empty summary.
    summary = world.get_component(entity_id, CurrentCompactionSummaryComponent)
    assert summary is not None
    assert summary.summary.strip()
