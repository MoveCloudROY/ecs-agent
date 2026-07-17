"""Live prompt-cache experiments for both API families (env-gated).

Replays the exact multi-turn agentic sequences the framework produces (captured
via ``tests.cache_audit.harness`` with production systems) against real
endpoints, recording per-call cache usage:

- Anthropic Messages: ``cache_creation_input_tokens`` / ``cache_read_input_tokens``
- OpenAI Chat/Responses: ``prompt_tokens_details.cached_tokens`` /
  ``input_tokens_details.cached_tokens``

Gating: OpenAI scenarios need ``LLM_API_KEY`` (+ ``LLM_BASE_URL`` /
``LLM_RESPONSES_BASE_URL`` / ``LLM_MODEL``); Anthropic scenarios need
``ANTHROPIC_API_KEY`` (+ ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_MODEL``).

OpenAI-side note: some gateways commit prompt-cache writes asynchronously
(observed ~1-2 min), so warm calls poll with ``LLM_CACHE_WRITE_LAG_SECONDS``
budget (default 150s, interval 30s) before concluding "miss". Run with ``-s``
to see the per-call experiment tables.
"""

from __future__ import annotations

import asyncio
import os

import httpx
import pytest

from ecs_agent.accounting.normalization import compute_cache_stats
from ecs_agent.providers.claude_model import ClaudeModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.types import CompletionResult, Message, ToolCall, ToolSchema, Usage

from tests.cache_audit.harness import (
    CapturedCall,
    build_audit_world,
    echo_tool_schema,
    run_turn,
    scripted_two_turn_responses,
)
from tests.live.api_format import live_openai_base_url, live_openai_model

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

# Async cache-write budget for OpenAI-family gateways.
CACHE_WRITE_LAG_BUDGET = float(os.getenv("LLM_CACHE_WRITE_LAG_SECONDS", "150"))
CACHE_POLL_INTERVAL = float(os.getenv("LLM_CACHE_POLL_INTERVAL_SECONDS", "30"))

_openai_gate = pytest.mark.skipif(
    not LLM_API_KEY, reason="LLM_API_KEY environment variable not set"
)
_anthropic_gate = pytest.mark.skipif(
    not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY environment variable not set"
)


# ---------------------------------------------------------------------------
# Sequence capture (offline, deterministic) and replay (live)
# ---------------------------------------------------------------------------


async def _captured_sequence(
    *, context_pool: bool, salt: str = ""
) -> list[CapturedCall]:
    """Produce the framework's outbound calls for the two-turn tool loop.

    ``salt`` namespaces the scenario's cache prefix (system prompt + user
    turns) so scenarios and reruns don't read each other's provider cache —
    identical FakeModel bytes would otherwise cross-hit between scenarios.
    """
    from tests.cache_audit.harness import LARGE_SYSTEM_PROMPT

    system_prompt = (f"Experiment {salt}.\n" if salt else "") + LARGE_SYSTEM_PROMPT
    world, entity_id, recorder, systems = build_audit_world(
        responses=scripted_two_turn_responses(),
        system_prompt=system_prompt,
    )
    pre_tick = []
    if context_pool:
        from ecs_agent.components import (
            PromptContextQueueComponent,
            UserPromptConfigComponent,
        )
        from ecs_agent.systems.prompt_context_collector import (
            PromptContextCollectorSystem,
        )

        world.add_component(
            entity_id, UserPromptConfigComponent(enable_context_pool=True)
        )
        world.add_component(entity_id, PromptContextQueueComponent())
        pre_tick = [PromptContextCollectorSystem(priority=-5)]

    await run_turn(
        world,
        entity_id,
        systems,
        "What is the weather in Paris and London?",
        pre_tick_systems=pre_tick,
    )
    await run_turn(
        world,
        entity_id,
        systems,
        "And how about tomorrow?",
        pre_tick_systems=pre_tick,
    )
    return recorder.calls


def _openai_model(api_format: ApiFormat) -> OpenAIModel:
    config = ProviderConfig(
        provider_id="live-cache-experiment",
        base_url=live_openai_base_url(api_format),
        api_key=LLM_API_KEY,
        api_format=api_format,
    )
    return OpenAIModel(config=config, model=live_openai_model())


def _claude_model() -> ClaudeModel:
    config = ProviderConfig(
        provider_id="live-cache-experiment",
        base_url=ANTHROPIC_BASE_URL,
        api_key=ANTHROPIC_API_KEY,
        api_format=ApiFormat.ANTHROPIC_MESSAGES,
        enable_prompt_caching=True,
    )
    return ClaudeModel(config=config, model=ANTHROPIC_MODEL, max_tokens=64)


async def _complete(model: object, call: CapturedCall) -> Usage:
    """One live completion; returns usage or skips on transient endpoint noise."""
    try:
        result = await model.complete(call.messages, tools=call.tools, stream=False)  # type: ignore[attr-defined]
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        pytest.skip(f"transient live endpoint error: {type(exc).__name__}: {exc}")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (429,) or exc.response.status_code >= 500:
            pytest.skip(f"transient live endpoint error: {exc.response.status_code}")
        raise
    assert isinstance(result, CompletionResult)
    assert result.usage is not None, "live endpoint returned no usage"
    return result.usage


def _cached_tokens(usage: Usage) -> int:
    """Provider-agnostic 'served from cache' token count."""
    if usage.cache_read_tokens is not None:
        return usage.cache_read_tokens
    return usage.cached_input_tokens or 0


def _row(scenario: str, index: int, usage: Usage) -> str:
    stats = compute_cache_stats(usage)
    hit = f"{stats.hit_rate:.0%}" if stats and stats.hit_rate is not None else "n/a"
    return (
        f"[cache-experiment] {scenario} call={index} "
        f"prompt={usage.prompt_tokens} cached_read={_cached_tokens(usage)} "
        f"cache_write={usage.cache_creation_tokens} hit_rate={hit}"
    )


async def _replay_openai(
    model: OpenAIModel, calls: list[CapturedCall], scenario: str
) -> list[Usage]:
    """Replay a captured sequence; poll warm calls through async cache writes.

    The first call primes the cache. Every subsequent call repeats until the
    gateway reports cached tokens or the lag budget is exhausted. Both the
    first attempt (what an uninterrupted agent loop experiences against an
    async-write gateway) and the final poll (steady-state prefix validity)
    are printed; the final observation is recorded.

    Caveat: each poll repetition also primes the gateway with its own bytes,
    so a step whose prompt diverged from the prior step can still converge to
    a hit against *itself* — read the ``first`` column for divergence impact.
    """
    usages: list[Usage] = []
    for call in calls:
        usage = await _complete(model, call)
        first_attempt = usage
        if call.call_index > 0 and _cached_tokens(usage) == 0:
            waited = 0.0
            while waited < CACHE_WRITE_LAG_BUDGET:
                await asyncio.sleep(CACHE_POLL_INTERVAL)
                waited += CACHE_POLL_INTERVAL
                usage = await _complete(model, call)
                if _cached_tokens(usage) > 0:
                    break
        usages.append(usage)
        print(
            _row(scenario, call.call_index, usage)
            + f" first_attempt_read={_cached_tokens(first_attempt)}"
        )
    return usages


_SYNTHETIC_THINKING = "I need to look up the requested data before answering."


def _with_replayed_thinking(calls: list[CapturedCall]) -> list[CapturedCall]:
    """Give every assistant history message a constant thinking block.

    Thinking-mode gateways require assistant turns to carry their thinking
    content back; a real conversation stores the model's own
    ``reasoning_content``, so the synthetic replay mirrors that. The constant
    string keeps the bytes identical across calls (prefix-stable).
    """
    for call in calls:
        for message in call.messages:
            if message.role == "assistant" and message.reasoning_content is None:
                message.reasoning_content = _SYNTHETIC_THINKING
    return calls


async def _replay_anthropic(
    model: ClaudeModel, calls: list[CapturedCall], scenario: str
) -> list[Usage]:
    """Replay a captured sequence back-to-back (Anthropic cache is readable
    as soon as the priming response starts)."""
    usages: list[Usage] = []
    for call in _with_replayed_thinking(calls):
        usage = await _complete(model, call)
        usages.append(usage)
        print(_row(scenario, call.call_index, usage))
    return usages


def _skip_if_no_cache_support(usages: list[Usage], family: str) -> None:
    if all(
        (usage.cache_read_tokens or 0) == 0
        and (usage.cache_creation_tokens or 0) == 0
        and (usage.cached_input_tokens or 0) == 0
        for usage in usages
    ):
        pytest.skip(f"{family} endpoint reports no prompt-cache usage fields")


def _aggregate_hit_rate(usages: list[Usage], *, warm_only: bool = True) -> float:
    """Cache hit rate over the scenario (warm calls only by default — call 0
    can never read a freshly salted prefix)."""
    scoped = usages[1:] if warm_only else usages
    read = sum(_cached_tokens(u) for u in scoped)
    total = 0
    for usage in scoped:
        stats = compute_cache_stats(usage)
        total += stats.total_prompt_tokens if stats else usage.prompt_tokens or 0
    return read / total if total else 0.0


def _run_salt(scenario: str) -> str:
    import uuid

    return f"{scenario}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Anthropic Messages experiments
# ---------------------------------------------------------------------------


@_anthropic_gate
@pytest.mark.asyncio
async def test_anthropic_clean_agentic_loop_cache_hits() -> None:
    """Default framework loop: each later call must read the previous prefix."""
    calls = await _captured_sequence(
        context_pool=False, salt=_run_salt("clean")
    )
    usages = await _replay_anthropic(_claude_model(), calls, "anthropic-clean-loop")
    _skip_if_no_cache_support(usages, "anthropic")

    print(
        f"[cache-experiment] anthropic-clean-loop warm hit_rate="
        f"{_aggregate_hit_rate(usages):.0%}"
    )
    warm_reads = [_cached_tokens(u) for u in usages[1:]]
    assert any(read > 0 for read in warm_reads), (
        f"no warm call read the cache: {warm_reads} — the framework's outbound "
        "prefix must be reusable call-over-call"
    )
    # Reads should not shrink as the conversation grows (incremental caching).
    assert warm_reads[-1] >= warm_reads[0], warm_reads


@_anthropic_gate
@pytest.mark.asyncio
async def test_anthropic_context_pool_injection_cache_penalty() -> None:
    """Context-pool injection rewrites the last user message per call; measure
    the cache penalty against the clean loop (see the printed tables)."""
    calls = await _captured_sequence(context_pool=True, salt=_run_salt("pool"))
    usages = await _replay_anthropic(_claude_model(), calls, "anthropic-context-pool")
    _skip_if_no_cache_support(usages, "anthropic")

    print(
        f"[cache-experiment] anthropic-context-pool warm hit_rate="
        f"{_aggregate_hit_rate(usages):.0%}"
    )
    # Structural expectation: history beyond the rewritten user turn cannot be
    # read back; reads stay pinned near the static prefix instead of growing
    # with the conversation. The precise numbers are the experiment output.
    assert len(usages) == 4


@_anthropic_gate
@pytest.mark.asyncio
async def test_anthropic_breakpoint_lookback_window_probe() -> None:
    """Measure whether a wide parallel tool batch (>20 new content blocks
    between consecutive trailing breakpoints) forfeits the message-prefix read.

    narrow: 3 tool calls  ->  8 new blocks  -> expect prefix read to include
            the primed message tokens.
    wide:  12 tool calls  -> ~25 new blocks -> the previous entry sits beyond
            the ~20-block lookback; expect the read to collapse to the static
            (tools+system) portion.
    """
    filler = "Compare humidity, wind and UV index too. " * 120  # ≈1.2k tokens

    def _probe_calls(width: int, tag: str) -> list[CapturedCall]:
        system = Message(
            role="system",
            content=f"Experiment {tag}.\n" + _claude_system_prompt(),
            cache_control=True,
        )
        user = Message(role="user", content=f"[{tag}] {filler}")
        base = [system, user]
        assistant = Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id=f"call_{tag}_{i}",
                    name="lookup_weather",
                    arguments={"city": f"City{i}"},
                )
                for i in range(width)
            ],
        )
        tool_results = [
            Message(
                role="tool",
                content=f"Weather in City{i}: 21C." + " detail" * 30,
                tool_call_id=f"call_{tag}_{i}",
            )
            for i in range(width)
        ]
        follow_up = [*base, assistant, *tool_results]
        tools = [echo_tool_schema()]
        return [
            CapturedCall(call_index=0, messages=base, tools=tools),
            CapturedCall(call_index=1, messages=follow_up, tools=tools),
        ]

    model = _claude_model()
    salt = _run_salt("probe")
    narrow = await _replay_anthropic(
        model, _probe_calls(3, f"{salt}-narrow"), "anthropic-narrow"
    )
    wide = await _replay_anthropic(
        model, _probe_calls(12, f"{salt}-wide"), "anthropic-wide"
    )
    _skip_if_no_cache_support([*narrow, *wide], "anthropic")

    narrow_read = _cached_tokens(narrow[1])
    wide_read = _cached_tokens(wide[1])
    print(
        f"[cache-experiment] lookback probe: narrow(3 calls) warm read={narrow_read}, "
        f"wide(12 calls) warm read={wide_read} "
        f"(both primed a ~equal prefix; a collapsed wide read demonstrates the "
        f"~20-block lookback window)"
    )
    assert narrow_read > 0, "narrow probe should read its primed prefix"


def _claude_system_prompt() -> str:
    from tests.cache_audit.harness import LARGE_SYSTEM_PROMPT

    return LARGE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# OpenAI Chat Completions / Responses experiments
# ---------------------------------------------------------------------------


@_openai_gate
@pytest.mark.asyncio
async def test_openai_chat_clean_agentic_loop_cache_hits() -> None:
    calls = (
        await _captured_sequence(context_pool=False, salt=_run_salt("chat-clean"))
    )[:3]
    model = _openai_model(ApiFormat.OPENAI_CHAT_COMPLETIONS)
    usages = await _replay_openai(model, calls, "openai-chat-clean-loop")
    _skip_if_no_cache_support(usages, "openai-chat")

    print(
        f"[cache-experiment] openai-chat-clean-loop warm hit_rate="
        f"{_aggregate_hit_rate(usages):.0%}"
    )
    warm_reads = [_cached_tokens(u) for u in usages[1:]]
    assert any(read > 0 for read in warm_reads), (
        f"no warm call read the cache within the lag budget: {warm_reads}"
    )


@_openai_gate
@pytest.mark.asyncio
async def test_openai_responses_clean_agentic_loop_cache_hits() -> None:
    calls = (
        await _captured_sequence(
            context_pool=False, salt=_run_salt("responses-clean")
        )
    )[:3]
    model = _openai_model(ApiFormat.OPENAI_RESPONSES)
    usages = await _replay_openai(model, calls, "openai-responses-clean-loop")
    _skip_if_no_cache_support(usages, "openai-responses")

    print(
        f"[cache-experiment] openai-responses-clean-loop warm hit_rate="
        f"{_aggregate_hit_rate(usages):.0%}"
    )
    warm_reads = [_cached_tokens(u) for u in usages[1:]]
    assert any(read > 0 for read in warm_reads), (
        f"no warm call read the cache within the lag budget: {warm_reads}"
    )


@_openai_gate
@pytest.mark.asyncio
async def test_openai_responses_identical_repeat_probe() -> None:
    """Minimal cache-support probe for the Responses endpoint: the same
    system+user request twice (no tools). Distinguishes 'endpoint does not
    cache / report today' from 'the framework's agentic request shape misses'.
    """
    from tests.cache_audit.harness import LARGE_SYSTEM_PROMPT

    salt = _run_salt("responses-probe")
    call = CapturedCall(
        call_index=1,  # index>0 so the poll loop applies on the repeat
        messages=[
            Message(role="system", content=f"Experiment {salt}.\n" + LARGE_SYSTEM_PROMPT),
            Message(role="user", content="Reply with the single word: ok"),
        ],
        tools=None,
    )
    model = _openai_model(ApiFormat.OPENAI_RESPONSES)
    prime = await _complete(model, CapturedCall(0, call.messages, None))
    print(_row("openai-responses-probe", 0, prime))
    usages = await _replay_openai(model, [call], "openai-responses-probe")
    print(
        "[cache-experiment] responses-probe verdict: "
        + (
            "endpoint caches identical repeats"
            if _cached_tokens(usages[0]) > 0
            else "endpoint reported no cached tokens within the lag budget"
        )
    )


@_openai_gate
@pytest.mark.asyncio
async def test_openai_responses_tool_request_repeat_probe() -> None:
    """Second bisection probe: identical repeat of a tool-loop request
    (tools + function_call/function_call_output input items). Together with
    the plain probe this separates 'endpoint caches nothing' / 'tool-bearing
    requests are excluded' / 'prefix extension fails'."""
    calls = await _captured_sequence(
        context_pool=False, salt=_run_salt("responses-tools-probe")
    )
    step = calls[1]  # tools + function items present
    model = _openai_model(ApiFormat.OPENAI_RESPONSES)
    prime = await _complete(model, CapturedCall(0, step.messages, step.tools))
    print(_row("openai-responses-tools-probe", 0, prime))
    usages = await _replay_openai(
        model,
        [CapturedCall(1, step.messages, step.tools)],
        "openai-responses-tools-probe",
    )
    print(
        "[cache-experiment] responses-tools-probe verdict: "
        + (
            "tool-bearing identical repeats cache"
            if _cached_tokens(usages[0]) > 0
            else "tool-bearing repeats reported no cached tokens in budget"
        )
    )


@_openai_gate
@pytest.mark.asyncio
async def test_openai_chat_context_pool_injection_cache_penalty() -> None:
    """Measure the injected-context penalty on the chat endpoint (report)."""
    calls = (
        await _captured_sequence(context_pool=True, salt=_run_salt("chat-pool"))
    )[:3]
    model = _openai_model(ApiFormat.OPENAI_CHAT_COMPLETIONS)
    usages = await _replay_openai(model, calls, "openai-chat-context-pool")
    _skip_if_no_cache_support(usages, "openai-chat")
    print(
        f"[cache-experiment] openai-chat-context-pool warm hit_rate="
        f"{_aggregate_hit_rate(usages):.0%}"
    )
    assert len(usages) == 3
