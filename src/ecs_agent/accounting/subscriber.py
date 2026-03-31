"""Event-driven accounting subscriber for invocation cost and cache metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ecs_agent.accounting.catalog import PricingCatalog
from ecs_agent.accounting.models import CostRecord, LLMInvocationEvent, PromptCacheStats
from ecs_agent.accounting.normalization import compute_cache_stats
from ecs_agent.logging import get_logger

if TYPE_CHECKING:
    from ecs_agent.core.event_bus import EventBus

logger = get_logger(__name__)


class AccountingSubscriber:
    """Subscribes to LLM invocation events and computes accounting metrics."""

    def __init__(self, *, pricing_catalog: PricingCatalog | None = None) -> None:
        self._pricing_catalog = (
            PricingCatalog() if pricing_catalog is None else pricing_catalog
        )
        self._records: list[LLMInvocationEvent] = []
        self._agg_cache_read: dict[tuple[str, str], int] = {}
        self._agg_total_prompt: dict[tuple[str, str], int] = {}

    def subscribe(self, event_bus: EventBus) -> None:
        event_bus.subscribe(LLMInvocationEvent, self._handle_llm_invocation)

    def get_aggregate_stats(
        self, provider_id: str, model: str
    ) -> PromptCacheStats | None:
        key = (provider_id, model)
        cache_read_tokens = self._agg_cache_read.get(key)
        total_prompt_tokens = self._agg_total_prompt.get(key)
        if cache_read_tokens is None or total_prompt_tokens is None:
            return None

        hit_rate: float | None = None
        if total_prompt_tokens > 0:
            hit_rate = cache_read_tokens / total_prompt_tokens

        return PromptCacheStats(
            cache_read_tokens=cache_read_tokens,
            total_prompt_tokens=total_prompt_tokens,
            hit_rate=hit_rate,
        )

    async def _handle_llm_invocation(self, event: LLMInvocationEvent) -> None:
        self._records.append(event)

        cache_stats = compute_cache_stats(event.usage)
        if cache_stats is not None:
            self._update_aggregate_cache_stats(event=event, cache_stats=cache_stats)

        cost_record = self._compute_cost(event)
        if cost_record is None:
            logger.warning(
                "accounting_pricing_not_found",
                provider_id=event.provider_id,
                model=event.model,
                request_id=event.request_id,
            )
        else:
            logger.info(
                "accounting_invocation_recorded",
                entity_id=event.entity_id,
                provider_id=event.provider_id,
                model=event.model,
                request_id=event.request_id,
                total_cost=cost_record.total_cost,
                currency=cost_record.currency,
                catalog_version=self._pricing_catalog.version,
                cache_hit_rate=cache_stats.hit_rate
                if cache_stats is not None
                else None,
            )

    def _update_aggregate_cache_stats(
        self,
        *,
        event: LLMInvocationEvent,
        cache_stats: PromptCacheStats,
    ) -> None:
        key = (event.provider_id, event.model)
        self._agg_cache_read[key] = (
            self._agg_cache_read.get(key, 0) + cache_stats.cache_read_tokens
        )
        self._agg_total_prompt[key] = (
            self._agg_total_prompt.get(key, 0) + cache_stats.total_prompt_tokens
        )

    def _compute_cost(self, event: LLMInvocationEvent) -> CostRecord | None:
        pricing = self._pricing_catalog.get_pricing(event.provider_id, event.model)
        if pricing is None:
            return None

        prompt_tokens = event.usage.prompt_tokens or 0
        completion_tokens = event.usage.completion_tokens or 0
        cached_input_tokens = event.usage.cached_input_tokens or 0
        cache_read_tokens = event.usage.cache_read_tokens or cached_input_tokens
        cache_write_tokens = event.usage.cache_creation_tokens or 0
        non_cached_prompt_tokens = max(prompt_tokens - cached_input_tokens, 0)

        input_cost = (non_cached_prompt_tokens / 1_000_000) * pricing.input_per_million

        cache_write_cost = None
        if pricing.cache_write_per_million is not None:
            cache_write_cost = (
                cache_write_tokens / 1_000_000
            ) * pricing.cache_write_per_million
            input_cost += cache_write_cost

        cached_input_cost = None
        if pricing.cached_input_per_million is not None:
            cached_input_cost = (
                cache_read_tokens / 1_000_000
            ) * pricing.cached_input_per_million

        output_cost = (completion_tokens / 1_000_000) * pricing.output_per_million

        total_cost = input_cost + output_cost
        if cached_input_cost is not None:
            total_cost += cached_input_cost

        return CostRecord(
            input_cost=input_cost,
            cached_input_cost=cached_input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            currency="USD",
            is_estimated=False,
        )
