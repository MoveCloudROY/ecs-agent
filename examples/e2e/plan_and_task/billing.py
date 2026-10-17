"""Billing, token-usage and cache hit-rate subscriber for the plan-and-task workflow.

Emits structured log events:
* ``plan_task_llm_usage`` — per-invocation token counts and raw cache fields.
* ``plan_task_llm_cache_stats`` — per-invocation cache hit-rate (only when model
  returns cache token data such as DashScope / OpenAI prompt-cache).
* ``plan_task_session_billing_summary`` — cumulative totals for the whole session.

``BillingSubscriber`` is also a raw-event observability plugin with
``propagate_to_children=True``: installed via ``ecs_agent.plugins.install_plugins``
(as ``main.py`` and the TUI entrypoint do), the delegation path propagates it onto
every subagent child world, so reviewer/writer subagent LLM calls are billed too
and their ``plan_task_llm_usage`` lines carry the child world's name. The bare
``subscribe(event_bus)`` path remains for main-world-only wiring; do not use both
on the same world or main-world invocations are counted twice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ecs_agent.accounting.models import LLMInvocationEvent
from ecs_agent.accounting.normalization import compute_cache_stats
from ecs_agent.logging import get_logger
from ecs_agent.observability.install import EventSubscription
from ecs_agent.observability.sinks import TelemetrySink

if TYPE_CHECKING:
    from ecs_agent.core.event_bus import EventBus

logger = get_logger(__name__)


class BillingSubscriber:
    name = "plan-task-billing"
    propagate_to_children = True

    def __init__(self) -> None:
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_tokens: int = 0
        self._total_cached_input_tokens: int = 0
        self._invocation_count: int = 0
        self._non_success_count: int = 0

    def subscribe(self, event_bus: EventBus) -> None:
        event_bus.subscribe(LLMInvocationEvent, self._handle_llm_invocation)

    def log_session_summary(self) -> None:
        logger.info(
            "plan_task_session_billing_summary",
            invocation_count=self._invocation_count,
            non_success_invocation_count=self._non_success_count,
            total_prompt_tokens=self._total_prompt_tokens,
            total_completion_tokens=self._total_completion_tokens,
            total_tokens=self._total_tokens,
            total_cached_input_tokens=self._total_cached_input_tokens,
        )

    # -- ObservabilityPlugin capabilities ---------------------------------

    def telemetry_sink(self) -> TelemetrySink | None:
        """No record-pipeline capability; billing taps raw events only."""
        return None

    def event_subscriptions(self, world: Any) -> tuple[EventSubscription, ...]:
        """Bind a usage handler labeled with the world's name.

        Child (subagent) worlds are named after their subagent, so propagated
        subscriptions attribute usage lines to ``advisor``/``qa``/etc.
        """
        world_name = getattr(world, "name", None)

        async def handle(event: LLMInvocationEvent) -> None:
            await self._handle_llm_invocation(event, world_name=world_name)

        return ((LLMInvocationEvent, handle),)

    async def start(self, world: Any) -> None:
        """No resources to allocate."""
        _ = world

    async def flush(self) -> None:
        """Nothing buffered."""

    async def shutdown(self) -> None:
        """No resources to release."""

    # -- event handling ----------------------------------------------------

    async def _handle_llm_invocation(
        self, event: LLMInvocationEvent, *, world_name: str | None = None
    ) -> None:
        usage = event.usage
        self._invocation_count += 1
        # An invocation that was cancelled (subagent timeout, shutdown) or that
        # errored reports no usage at all. Counting those as ordinary zero-token
        # calls is how a run full of timeouts reads as "the provider stopped
        # returning usage" — carry the framework's own labels through instead.
        if event.status != "success":
            self._non_success_count += 1

        prompt_tokens = usage.prompt_tokens or 0
        completion_tokens = usage.completion_tokens or 0
        total_tokens = usage.total_tokens or (prompt_tokens + completion_tokens)
        cached_input_tokens = usage.cached_input_tokens or 0

        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_tokens += total_tokens
        self._total_cached_input_tokens += cached_input_tokens

        logger.info(
            "plan_task_llm_usage",
            entity_id=event.entity_id,
            world=world_name,
            provider_id=event.provider_id,
            model=event.model,
            request_id=event.request_id,
            status=event.status,
            generation_status=event.generation_status,
            finish_reason=event.finish_reason,
            stream_completeness=usage.stream_completeness.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_creation_tokens=usage.cache_creation_tokens,
            cache_read_tokens=usage.cache_read_tokens,
        )

        cache_stats = compute_cache_stats(usage)
        if cache_stats is not None:
            logger.info(
                "plan_task_llm_cache_stats",
                entity_id=event.entity_id,
                world=world_name,
                provider_id=event.provider_id,
                model=event.model,
                request_id=event.request_id,
                cache_read_tokens=cache_stats.cache_read_tokens,
                total_prompt_tokens=cache_stats.total_prompt_tokens,
                cache_hit_rate=cache_stats.hit_rate,
            )
