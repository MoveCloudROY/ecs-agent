"""Prompt-cache prefix audit harness.

Provider prompt caches (Anthropic ``cache_control`` prefix cache, OpenAI
automatic prefix cache on Chat Completions and Responses) hit only when the
rendered prompt of request N is a byte-prefix of request N+1. This module
captures the exact outbound message lists the framework produces across a
multi-turn agentic run, renders them through the *real* provider adapters,
and reports the first divergence between consecutive requests.

Components under test are the production ones: ``ReasoningSystem``,
``ToolExecutionSystem``, ``PromptContextCollectorSystem``,
``prepare_outbound_messages`` and the three wire adapters.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    PendingToolCallsComponent,
    TerminalComponent,
    ToolRegistryComponent,
)
from ecs_agent.core import World
from ecs_agent.providers.anthropic_messages_adapter import (
    AnthropicMessagesAdapter,
    AnthropicMessagesAdapterConfig,
    AnthropicMessagesRequest,
)
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.fake_model import FakeModel
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.observability.events import LLMObservationStartedEvent
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import (
    CompletionResult,
    Message,
    ToolCall,
    ToolSchema,
)

# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CapturedCall:
    """One outbound model call: deep-copied messages and tools."""

    call_index: int
    messages: list[Message]
    tools: list[ToolSchema] | None


class OutboundCallRecorder:
    """Subscribe to LLMObservationStartedEvent and snapshot each call."""

    def __init__(self) -> None:
        self.calls: list[CapturedCall] = []

    def attach(self, world: World) -> None:
        world.event_bus.subscribe(LLMObservationStartedEvent, self._on_started)

    async def _on_started(self, event: LLMObservationStartedEvent) -> None:
        self.calls.append(
            CapturedCall(
                call_index=len(self.calls),
                messages=copy.deepcopy(event.messages),
                tools=copy.deepcopy(event.tools),
            )
        )


# ---------------------------------------------------------------------------
# Wire rendering (real adapters)
# ---------------------------------------------------------------------------

_ANTHROPIC = "anthropic_messages"
_OPENAI_CHAT = "openai_chat_completions"
_OPENAI_RESPONSES = "openai_responses"

WIRE_FORMATS = (_ANTHROPIC, _OPENAI_CHAT, _OPENAI_RESPONSES)


def _provider_config(api_format: ApiFormat) -> ProviderConfig:
    return ProviderConfig(
        provider_id="audit",
        base_url="http://audit.invalid",
        api_key="audit-key",
        api_format=api_format,
        enable_prompt_caching=True,
    )


def render_wire_body(
    wire_format: str, messages: list[Message], tools: list[ToolSchema] | None
) -> dict[str, Any]:
    """Build the exact request body a provider adapter would send."""
    if wire_format == _ANTHROPIC:
        adapter = AnthropicMessagesAdapter(
            AnthropicMessagesAdapterConfig(
                provider=_provider_config(ApiFormat.ANTHROPIC_MESSAGES),
                model="audit-model",
                max_tokens=1024,
            )
        )
        return adapter.build_request_body(
            AnthropicMessagesRequest(messages=messages, tools=tools)
        )

    if wire_format == _OPENAI_CHAT:
        model = OpenAIModel(
            config=_provider_config(ApiFormat.OPENAI_CHAT_COMPLETIONS),
            model="audit-model",
        )
        return model._chat_adapter._build_request_body(messages, tools, None)

    if wire_format == _OPENAI_RESPONSES:
        model = OpenAIModel(
            config=_provider_config(ApiFormat.OPENAI_RESPONSES),
            model="audit-model",
        )
        return model._responses_adapter._build_request_body(
            messages, tools, None, None
        )

    raise ValueError(f"unknown wire format: {wire_format}")


# ---------------------------------------------------------------------------
# Prompt units in provider render order
# ---------------------------------------------------------------------------


def _strip_cache_control(value: Any) -> Any:
    """Remove cache_control markers: they place breakpoints, they are not
    prompt content, and the adapter intentionally moves them forward each
    request."""
    if isinstance(value, dict):
        return {
            key: _strip_cache_control(item)
            for key, item in value.items()
            if key != "cache_control"
        }
    if isinstance(value, list):
        return [_strip_cache_control(item) for item in value]
    return value


def prompt_units(wire_format: str, body: dict[str, Any]) -> list[tuple[str, str]]:
    """Split a request body into ``(kind, serialized)`` units in the order the
    provider renders them into the prompt.

    Anthropic renders tools -> system -> messages. OpenAI renders the tool
    schemas and message list as one prompt as well; instructions (Responses)
    sit ahead of the input list.
    """
    units: list[tuple[str, str]] = []

    def _dump(value: Any) -> str:
        return json.dumps(
            _strip_cache_control(value), ensure_ascii=False, sort_keys=True
        )

    if wire_format == _ANTHROPIC:
        for tool in body.get("tools") or []:
            units.append(("tool", _dump(tool)))
        system_value = body.get("system")
        if isinstance(system_value, str):
            units.append(("system", _dump(system_value)))
        elif isinstance(system_value, list):
            for block in system_value:
                units.append(("system", _dump(block)))
        for message in body.get("messages", []):
            units.append((f"message:{message.get('role')}", _dump(message)))
        return units

    if wire_format == _OPENAI_CHAT:
        for tool in body.get("tools") or []:
            units.append(("tool", _dump(tool)))
        for message in body.get("messages", []):
            units.append((f"message:{message.get('role')}", _dump(message)))
        return units

    if wire_format == _OPENAI_RESPONSES:
        units.append(("instructions", _dump(body.get("instructions"))))
        for tool in body.get("tools") or []:
            units.append(("tool", _dump(tool)))
        for item in body.get("input", []):
            kind = item.get("type", "item")
            role = item.get("role")
            label = f"input:{kind}" + (f":{role}" if role else "")
            units.append((label, _dump(item)))
        return units

    raise ValueError(f"unknown wire format: {wire_format}")


# ---------------------------------------------------------------------------
# Divergence audit
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Divergence:
    """First point where request N+1 stops extending request N."""

    call_index: int  # index of the *later* call
    unit_index: int
    prev_kind: str | None
    curr_kind: str | None
    prev_excerpt: str
    curr_excerpt: str
    reusable_bytes: int
    prev_total_bytes: int

    @property
    def retention(self) -> float:
        """Fraction of the previous prompt still reusable as a cache prefix."""
        if self.prev_total_bytes == 0:
            return 1.0
        return self.reusable_bytes / self.prev_total_bytes

    def describe(self) -> str:
        return (
            f"call {self.call_index - 1} -> {self.call_index}: prompt diverges at "
            f"unit {self.unit_index} ({self.prev_kind!r} vs {self.curr_kind!r}); "
            f"only {self.reusable_bytes}/{self.prev_total_bytes} bytes "
            f"({self.retention:.0%}) of the previous prompt remain cacheable.\n"
            f"  prev: {self.prev_excerpt}\n  curr: {self.curr_excerpt}"
        )


def _unit_bytes(units: list[tuple[str, str]]) -> int:
    return sum(len(serialized.encode("utf-8")) for _, serialized in units)


def _first_difference_excerpt(prev: str, curr: str, width: int = 160) -> tuple[str, str]:
    limit = min(len(prev), len(curr))
    pos = next(
        (i for i in range(limit) if prev[i] != curr[i]),
        limit,
    )
    start = max(0, pos - 40)
    return (
        prev[start : pos + width],
        curr[start : pos + width],
    )


def audit_pair(
    wire_format: str,
    prev_units: list[tuple[str, str]],
    curr_units: list[tuple[str, str]],
    call_index: int,
) -> Divergence | None:
    """Return the first divergence, or None when prev is a unit-prefix of curr."""
    prev_total = _unit_bytes(prev_units)
    reusable = 0
    for index, (prev_kind, prev_serialized) in enumerate(prev_units):
        if index >= len(curr_units):
            return Divergence(
                call_index=call_index,
                unit_index=index,
                prev_kind=prev_kind,
                curr_kind=None,
                prev_excerpt=prev_serialized[:200],
                curr_excerpt="<missing — prompt shrank>",
                reusable_bytes=reusable,
                prev_total_bytes=prev_total,
            )
        curr_kind, curr_serialized = curr_units[index]
        if prev_serialized != curr_serialized:
            prev_excerpt, curr_excerpt = _first_difference_excerpt(
                prev_serialized, curr_serialized
            )
            return Divergence(
                call_index=call_index,
                unit_index=index,
                prev_kind=prev_kind,
                curr_kind=curr_kind,
                prev_excerpt=prev_excerpt,
                curr_excerpt=curr_excerpt,
                reusable_bytes=reusable,
                prev_total_bytes=prev_total,
            )
        reusable += len(prev_serialized.encode("utf-8"))
    return None


@dataclass(slots=True)
class AuditReport:
    wire_format: str
    call_count: int
    divergences: list[Divergence] = field(default_factory=list)
    block_counts: list[int] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.divergences


def audit_captured_calls(
    calls: list[CapturedCall], wire_format: str
) -> AuditReport:
    """Audit consecutive captured calls for the byte-prefix property."""
    unit_lists = [
        prompt_units(wire_format, render_wire_body(wire_format, c.messages, c.tools))
        for c in calls
    ]
    report = AuditReport(wire_format=wire_format, call_count=len(calls))
    for index in range(1, len(unit_lists)):
        divergence = audit_pair(
            wire_format, unit_lists[index - 1], unit_lists[index], index
        )
        if divergence is not None:
            report.divergences.append(divergence)
    if wire_format == _ANTHROPIC:
        for call in calls:
            body = render_wire_body(wire_format, call.messages, call.tools)
            report.block_counts.append(
                sum(
                    len(m["content"]) if isinstance(m["content"], list) else 1
                    for m in body["messages"]
                )
            )
    return report


# ---------------------------------------------------------------------------
# Scenario world
# ---------------------------------------------------------------------------

LARGE_SYSTEM_PROMPT = (
    "You are a meticulous assistant for the cache audit experiment. Apply "
    "every one of the standing guidelines below to each reply.\n\n"
    + "\n".join(
        f"Guideline {i}: Consider correctness, cite assumptions explicitly, "
        f"prefer structured output, and double-check units before answering."
        for i in range(220)
    )
)


def echo_tool_schema(name: str = "lookup_weather") -> ToolSchema:
    return ToolSchema(
        name=name,
        description="Look up the current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        concurrency_safe=True,
    )


async def _weather_handler(city: str) -> str:
    return f"Weather in {city}: 21C, clear skies, light wind. " + "detail " * 40


def scripted_two_turn_responses(
    tool_name: str = "lookup_weather", calls_per_turn: int = 2
) -> list[CompletionResult]:
    """Model script: turn 1 issues tool calls then answers; turn 2 repeats."""

    def _tool_call_result(turn: int) -> CompletionResult:
        return CompletionResult(
            message=Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id=f"call_t{turn}_{i}",
                        name=tool_name,
                        arguments={"city": f"City{turn}{i}"},
                    )
                    for i in range(calls_per_turn)
                ],
            )
        )

    def _final(turn: int) -> CompletionResult:
        return CompletionResult(
            message=Message(
                role="assistant",
                content=f"Turn {turn} summary: all cities are clear.",
            )
        )

    return [_tool_call_result(1), _final(1), _tool_call_result(2), _final(2)]


def build_audit_world(
    *,
    responses: list[CompletionResult],
    system_prompt: str = LARGE_SYSTEM_PROMPT,
    volatile_suffix: str | None = None,
    tool_schemas: dict[str, ToolSchema] | None = None,
) -> tuple[World, int, OutboundCallRecorder, list[Any]]:
    """Build a minimal production-shaped world for the audit.

    Returns ``(world, entity_id, recorder, systems)`` where ``systems`` is the
    per-tick processing order.
    """
    from ecs_agent.components import RenderedSystemPromptComponent

    world = World()
    entity_id = world.create_entity()

    model = FakeModel(responses=responses)
    world.add_component(entity_id, LLMComponent(model=model, system_prompt=""))
    world.add_component(entity_id, ConversationComponent(messages=[]))
    world.add_component(
        entity_id,
        RenderedSystemPromptComponent(
            text=system_prompt + (volatile_suffix or ""),
            placeholder_snapshot={},
            stable_text=system_prompt,
            volatile_text=volatile_suffix or "",
        ),
    )

    schemas = tool_schemas or {"lookup_weather": echo_tool_schema()}
    handlers = {name: _weather_handler for name in schemas}
    world.add_component(
        entity_id, ToolRegistryComponent(tools=dict(schemas), handlers=handlers)
    )

    recorder = OutboundCallRecorder()
    recorder.attach(world)

    systems = [ReasoningSystem(priority=0), ToolExecutionSystem(priority=10)]
    return world, entity_id, recorder, systems


async def run_turn(
    world: World,
    entity_id: int,
    systems: list[Any],
    user_text: str,
    *,
    pre_tick_systems: list[Any] | None = None,
    max_ticks: int = 12,
) -> None:
    """Append a user message and tick systems until the turn produces a final
    assistant reply (no pending tool calls)."""
    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    conversation.messages.append(Message(role="user", content=user_text))
    world.remove_component(entity_id, TerminalComponent)

    for _ in range(max_ticks):
        for system in pre_tick_systems or []:
            await system.process(world)
        for system in systems:
            await system.process(world)
        if world.has_component(entity_id, PendingToolCallsComponent):
            continue
        last = conversation.messages[-1]
        if last.role == "assistant" and not last.tool_calls:
            return
    raise AssertionError("turn did not settle within max_ticks")
