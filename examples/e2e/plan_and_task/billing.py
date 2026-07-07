"""Billing, token-usage and cache hit-rate subscriber for the plan-and-task workflow.

Emits structured log events:
* ``plan_task_llm_usage`` — per-invocation token counts and raw cache fields.
* ``plan_task_llm_cache_stats`` — per-invocation cache hit-rate (only when model
  returns cache token data such as DashScope / OpenAI prompt-cache).
* ``plan_task_session_billing_summary`` — cumulative totals for the whole session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ecs_agent.accounting.models import LLMInvocationEvent
from ecs_agent.accounting.normalization import compute_cache_stats
from ecs_agent.logging import get_logger

if TYPE_CHECKING:
    from ecs_agent.core.event_bus import EventBus

logger = get_logger(__name__)


class BillingSubscriber:
    def __init__(self) -> None:
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_tokens: int = 0
        self._total_cached_input_tokens: int = 0
        self._invocation_count: int = 0

    def subscribe(self, event_bus: EventBus) -> None:
        event_bus.subscribe(LLMInvocationEvent, self._handle_llm_invocation)

    def log_session_summary(self) -> None:
        logger.info(
            "plan_task_session_billing_summary",
            invocation_count=self._invocation_count,
            total_prompt_tokens=self._total_prompt_tokens,
            total_completion_tokens=self._total_completion_tokens,
            total_tokens=self._total_tokens,
            total_cached_input_tokens=self._total_cached_input_tokens,
        )

    async def _handle_llm_invocation(self, event: LLMInvocationEvent) -> None:
        usage = event.usage
        self._invocation_count += 1

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
            provider_id=event.provider_id,
            model=event.model,
            request_id=event.request_id,
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
                provider_id=event.provider_id,
                model=event.model,
                request_id=event.request_id,
                cache_read_tokens=cache_stats.cache_read_tokens,
                total_prompt_tokens=cache_stats.total_prompt_tokens,
                cache_hit_rate=cache_stats.hit_rate,
            )
