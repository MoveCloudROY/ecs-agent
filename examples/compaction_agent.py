"""Standalone demo of conversation compaction features in ecs-agent.

This example is split into four self-contained parts:
1. Full-history compaction
2. Pre-drop-then-compact behavior for tool-heavy history
3. Custom compaction prompt templates
4. Repeated compaction with summary folding across rounds

It supports dual-mode execution:
- Without ``LLM_API_KEY``: uses ``FakeModel`` and runs deterministically
- With ``LLM_API_KEY``: uses ``Model(...)`` with a real OpenAI-compatible API

Environment variables for real-LLM mode:
    LLM_API_KEY   - API key for the OpenAI-compatible model
    LLM_BASE_URL  - Base URL (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
    LLM_MODEL     - Model name (default: qwen3.5-flash)

Usage:
    uv run python examples/compaction_agent.py
"""

import asyncio
import os

from ecs_agent.components import (
    CompactionConfigComponent,
    ContextBudgetConfig,
    ConversationArchiveComponent,
    ConversationComponent,
    LLMComponent,
)
from ecs_agent.components.definitions import CurrentCompactionSummaryComponent
from ecs_agent.core import World
from ecs_agent.prompts.message_assembly import apply_outbound_budget
from ecs_agent.providers import FakeModel, Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.systems.compaction import CompactionSystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.types import (
    CompactionCompleteEvent,
    CompletionResult,
    Message,
    ToolCall,
    Usage,
)


def _make_chat_response(text: str) -> CompletionResult:
    """Fake chat completion response."""
    return CompletionResult(
        message=Message(role="assistant", content=text),
        usage=Usage(prompt_tokens=12, completion_tokens=18, total_tokens=30),
    )


def _make_summary_response(text: str) -> CompletionResult:
    """Fake summary response returned by CompactionSystem's LLM call."""
    return CompletionResult(
        message=Message(role="assistant", content=text),
        usage=Usage(prompt_tokens=48, completion_tokens=24, total_tokens=72),
    )


def _get_model_name() -> str:
    """Return 'fake' or the configured env model name."""
    if os.environ.get("LLM_API_KEY", ""):
        return os.environ.get("LLM_MODEL", "qwen3.5-flash")
    return "fake"


def _create_real_model() -> LLMModel:
    """Build an LLMModel from environment variables."""
    return Model(
        os.environ.get("LLM_MODEL", "qwen3.5-flash"),
        base_url=os.environ.get(
            "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        api_key=os.environ["LLM_API_KEY"],
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )


def _build_long_conversation() -> list[Message]:
    """Build a 10-message ML conversation that exceeds threshold_tokens=50."""
    return [
        Message(
            role="user",
            content=(
                "What is machine learning, and why do people say it learns patterns "
                "instead of memorizing every example?"
            ),
        ),
        Message(
            role="assistant",
            content=(
                "Machine learning fits statistical patterns from examples so a model "
                "can generalize to new inputs instead of replaying raw training rows."
            ),
        ),
        Message(
            role="user",
            content=(
                "How is supervised learning different from unsupervised learning in "
                "practical projects?"
            ),
        ),
        Message(
            role="assistant",
            content=(
                "Supervised learning uses labeled targets, while unsupervised learning "
                "looks for structure such as clusters, anomalies, or latent factors."
            ),
        ),
        Message(
            role="user",
            content=(
                "Where do neural networks fit into that picture, and why are layers useful?"
            ),
        ),
        Message(
            role="assistant",
            content=(
                "Neural networks are flexible function approximators, and stacked layers "
                "let them learn progressively richer features from raw input signals."
            ),
        ),
        Message(
            role="user",
            content=(
                "Can you explain what transformers changed for language models and attention?"
            ),
        ),
        Message(
            role="assistant",
            content=(
                "Transformers replaced recurrence with attention, making long-range token "
                "interactions easier to model and training much more parallelizable."
            ),
        ),
        Message(
            role="user",
            content=(
                "Before we finish, what exactly is a token and why does token count matter?"
            ),
        ),
        Message(
            role="assistant",
            content=(
                "A token is a chunk of text the model processes, and token counts matter "
                "because they affect context limits, latency, and cost."
            ),
        ),
    ]


def _subscribe_compaction_events(world: World) -> None:
    """Subscribe a compaction event handler that prints token savings."""

    async def _on_compaction(event: CompactionCompleteEvent) -> None:
        savings = event.original_tokens - event.compacted_tokens
        pct = savings / event.original_tokens * 100 if event.original_tokens else 0.0
        print(
            "  [CompactionCompleteEvent] "
            f"{event.original_tokens} → {event.compacted_tokens} tokens "
            f"({savings} saved, {pct:.0f}% reduction)"
        )

    world.event_bus.subscribe(CompactionCompleteEvent, _on_compaction)


def _create_demo_model(summary_responses: list[str]) -> LLMModel:
    """Create a real or fake model for the demo part."""
    if os.environ.get("LLM_API_KEY", ""):
        return _create_real_model()
    return FakeModel(
        responses=[_make_summary_response(text) for text in summary_responses]
    )


def _truncate(text: str, limit: int = 120) -> str:
    """Return a readable preview string."""
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _estimate_tokens(messages: list[Message]) -> int:
    """Mirror the compaction token estimate used by the system."""
    word_count = sum(len(message.content.split()) for message in messages)
    return int((word_count * 1.3) + 0.999999)


def _register_compaction_world(world: World) -> None:
    """Register the systems needed for one compaction tick."""
    world.register_system(CompactionSystem(), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)


def _build_predrop_messages() -> list[Message]:
    """Build tool-heavy history for the pre-drop compaction demo."""
    return [
        Message(
            role="user",
            content="Estimate this week's transformer training cost and keep only the durable conclusions.",
        ),
        Message(
            role="assistant",
            content="Calling gpu_pricing_lookup for raw accelerator pricing.",
            tool_calls=[
                ToolCall(
                    id="tc_001",
                    name="gpu_pricing_lookup",
                    arguments={"region": "us-east", "gpu": "a100"},
                )
            ],
        ),
        Message(
            role="tool",
            content=(
                "Raw pricing table: hourly A100 rates across three vendors, checkpoint storage fees, "
                "network egress estimates, reserved capacity notes, and burst pricing exceptions."
            ),
            tool_call_id="tc_001",
        ),
        Message(
            role="assistant",
            content="Calling throughput_lookup for token throughput benchmarks.",
            tool_calls=[
                ToolCall(
                    id="tc_002",
                    name="throughput_lookup",
                    arguments={"model": "medium-transformer"},
                )
            ],
        ),
        Message(
            role="tool",
            content=(
                "Throughput benchmark: tokens per second across prompt lengths, batch sizes, mixed precision "
                "settings, and context-window sizes for the same hardware class."
            ),
            tool_call_id="tc_002",
        ),
        Message(
            role="user",
            content="Now give me the conclusion only.",
        ),
        Message(
            role="assistant",
            content="I will keep the cost guidance and drop the bulky raw tool output.",
        ),
    ]


def _build_round_messages(round_number: int) -> list[Message]:
    """Build four verbose messages that still exceed the compaction threshold."""
    return [
        Message(
            role="user",
            content=(
                f"Round {round_number}: explain how embeddings convert words or concepts into dense numeric vectors that preserve semantic similarity."
            ),
        ),
        Message(
            role="assistant",
            content=(
                f"Round {round_number}: embeddings map related concepts nearby in vector space so downstream retrieval and ranking use meaning instead of exact phrasing alone."
            ),
        ),
        Message(
            role="user",
            content=(
                f"Round {round_number}: also connect embeddings to transformers, token windows, and why retrieval systems often summarize before storing context."
            ),
        ),
        Message(
            role="assistant",
            content=(
                f"Round {round_number}: transformers produce token representations, retrieval systems index them, and summarization keeps high-value facts while shrinking repeated history."
            ),
        ),
    ]


async def part_1_full_history() -> None:
    print("\n" + "=" * 70)
    print("PART 1: FULL HISTORY")
    print("=" * 70)

    world = World()
    model = _create_demo_model(
        [
            "Summary: The conversation covered ML basics, supervised versus unsupervised learning, neural networks, transformers, and why token counts matter."
        ]
    )
    agent_id = world.create_entity()
    messages = _build_long_conversation()

    world.add_component(
        agent_id,
        LLMComponent(
            model=model,
            system_prompt="You are a helpful assistant.",
        ),
    )
    world.add_component(agent_id, ConversationComponent(messages=list(messages)))
    world.add_component(
        agent_id,
        CompactionConfigComponent(
            threshold_tokens=50,
            compaction_method="full_history",
        ),
    )
    _subscribe_compaction_events(world)
    _register_compaction_world(world)

    print(
        f"Before compaction: {len(messages)} messages, "
        f"about {_estimate_tokens(messages)} estimated tokens"
    )
    await world.process()

    conv = world.get_component(agent_id, ConversationComponent)
    archive = world.get_component(agent_id, ConversationArchiveComponent)
    current_summary = world.get_component(agent_id, CurrentCompactionSummaryComponent)

    print(
        f"After compaction: {len(conv.messages) if conv else 0} messages remain in live history"
    )
    if archive and archive.archived_summaries:
        print(f"Archived summary[0]: {_truncate(archive.archived_summaries[0])}")
    if current_summary is not None:
        print(f"Current summary: {_truncate(current_summary.summary)}")
        print(
            "This summary becomes the XML block injected into the system prompt on "
            "the next SystemPromptRenderSystem render."
        )


async def part_2_predrop_then_compact() -> None:
    print("\n" + "=" * 70)
    print("PART 2: PREDROP THEN COMPACT")
    print("=" * 70)

    world = World()
    model = _create_demo_model(
        [
            "Summary: The tool-heavy conversation narrowed down training-cost analysis and kept the decision-relevant guidance while dropping bulky tool transcripts first."
        ]
    )
    agent_id = world.create_entity()
    messages = _build_predrop_messages()
    budgeted_view = apply_outbound_budget(
        list(messages),
        system_prompt="",
        context_entries=[],
        config=ContextBudgetConfig(
            max_tokens=50,
            prune_tool_results=True,
            prune_reasoning=False,
            overflow_behavior="truncate",
        ),
    )

    world.add_component(
        agent_id,
        LLMComponent(
            model=model,
            system_prompt="You are a helpful assistant.",
        ),
    )
    world.add_component(agent_id, ConversationComponent(messages=list(messages)))
    world.add_component(
        agent_id,
        CompactionConfigComponent(
            threshold_tokens=50,
            compaction_method="predrop_then_compact",
        ),
    )
    _subscribe_compaction_events(world)
    _register_compaction_world(world)

    original_tool_count = sum(1 for message in messages if message.role == "tool")
    budgeted_tool_count = sum(1 for message in budgeted_view if message.role == "tool")
    print(
        f"Before compaction: {len(messages)} messages, {original_tool_count} tool messages, "
        f"about {_estimate_tokens(messages)} estimated tokens"
    )
    print(
        f"Budgeted view used for summarization: {len(budgeted_view)} messages, "
        f"{budgeted_tool_count} tool messages"
    )
    await world.process()

    conv = world.get_component(agent_id, ConversationComponent)
    archive = world.get_component(agent_id, ConversationArchiveComponent)
    print(
        f"After compaction: {len(conv.messages) if conv else 0} live messages remain; "
        "the summary was generated from the pre-dropped view, not the full raw history."
    )
    if archive and archive.archived_summaries:
        print(f"Archived summary[0]: {_truncate(archive.archived_summaries[0])}")


async def part_3_custom_prompt() -> None:
    print("\n" + "=" * 70)
    print("PART 3: CUSTOM PROMPT TEMPLATE")
    print("=" * 70)

    world = World()
    model = _create_demo_model(
        [
            "- ML learns patterns from examples.\n- Neural networks and transformers were compared as layered architectures for representation learning.\n- Token counts matter for context limits, cost, and latency."
        ]
    )
    agent_id = world.create_entity()
    messages = _build_long_conversation()

    world.add_component(
        agent_id,
        LLMComponent(
            model=model,
            system_prompt="You are a helpful assistant.",
        ),
    )
    world.add_component(agent_id, ConversationComponent(messages=list(messages)))
    world.add_component(
        agent_id,
        CompactionConfigComponent(
            threshold_tokens=50,
            compaction_method="full_history",
            compaction_prompt_template=(
                "You are a technical historian. Summarize the conversation as 3 concise bullet points."
            ),
        ),
    )
    _subscribe_compaction_events(world)
    _register_compaction_world(world)

    print(
        f"Before compaction: {len(messages)} messages, "
        f"about {_estimate_tokens(messages)} estimated tokens"
    )
    await world.process()

    conv = world.get_component(agent_id, ConversationComponent)
    current_summary = world.get_component(agent_id, CurrentCompactionSummaryComponent)
    print(
        f"After compaction: {len(conv.messages) if conv else 0} messages remain in live history"
    )
    if current_summary is not None:
        print("Custom-template summary:")
        print(current_summary.summary)


async def part_4_repeated_compaction() -> None:
    print("\n" + "=" * 70)
    print("PART 4: REPEATED COMPACTION")
    print("=" * 70)

    world = World()
    model = (
        FakeModel(
            responses=[
                _make_summary_response("Round 1 summary: user asked about ML basics."),
                _make_summary_response(
                    "Round 2 summary: (incorporating Round 1) further discussed transformers and tokens."
                ),
                _make_summary_response(
                    "Round 3 summary: (incorporating Round 2) completed the ML overview including embeddings."
                ),
            ]
        )
        if not os.environ.get("LLM_API_KEY")
        else _create_real_model()
    )
    agent_id = world.create_entity()
    starting_messages = _build_long_conversation()

    world.add_component(
        agent_id,
        LLMComponent(
            model=model,
            system_prompt="You are a helpful assistant.",
        ),
    )
    world.add_component(
        agent_id,
        ConversationComponent(messages=list(starting_messages)),
    )
    world.add_component(
        agent_id,
        CompactionConfigComponent(
            threshold_tokens=50,
            compaction_method="full_history",
        ),
    )
    _subscribe_compaction_events(world)
    _register_compaction_world(world)

    print(
        "Starting state already exceeds the threshold: "
        f"{len(starting_messages)} messages, {_estimate_tokens(starting_messages)} estimated tokens"
    )

    # After the first round, CompactionSystem prefixes the next summary request with
    # "Previous summary:" so later rounds fold the earlier summary into the new one.
    for round_number in range(1, 4):
        conversation = world.get_component(agent_id, ConversationComponent)
        if conversation is None:
            raise RuntimeError(
                "ConversationComponent missing during repeated compaction demo"
            )
        conversation.messages.extend(_build_round_messages(round_number))
        print(
            f"Round {round_number} before compaction: {len(conversation.messages)} messages, "
            f"{_estimate_tokens(conversation.messages)} estimated tokens"
        )
        await world.process()

        archive = world.get_component(agent_id, ConversationArchiveComponent)
        if archive is None or not archive.archived_summaries:
            raise RuntimeError("Compaction did not produce an archived summary")
        print(f"Round {round_number} archive size: {len(archive.archived_summaries)}")
        print(f"Latest summary: {_truncate(archive.archived_summaries[-1])}")


async def main() -> None:
    print("=" * 70)
    print("COMPACTION DEMO")
    print("Demonstrating full-history compaction, pre-drop compaction, custom prompts,")
    print("and repeated summary folding with FakeModel/OpenAI dual mode.")
    print("=" * 70)

    if os.environ.get("LLM_API_KEY", ""):
        print(f"Using model: {_get_model_name()}")
        print(
            "Base URL: "
            f"{os.environ.get('LLM_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')}"
        )
    else:
        print("No LLM_API_KEY provided. Using FakeModel for deterministic output.")
        print("Set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL to run against a real API.")

    await part_1_full_history()
    await part_2_predrop_then_compact()
    await part_3_custom_prompt()
    await part_4_repeated_compaction()

    print("\n" + "=" * 70)
    print("COMPACTION DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
