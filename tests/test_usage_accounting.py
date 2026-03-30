"""Tests for canonical usage accounting models and normalization."""

import pytest

from ecs_agent.accounting import AccountingSubscriber
from ecs_agent.accounting.catalog import ModelPricing, PricingCatalog
from ecs_agent.accounting.models import (
    CostRecord,
    LLMInvocationEvent,
    StreamCompleteness,
    UsageRecord,
)
from ecs_agent.accounting.normalization import (
    compute_cache_stats,
    normalize_anthropic_usage,
    normalize_openai_usage,
)
from ecs_agent.core import EventBus
from ecs_agent.types import LLMInvocationEvent as ExportedLLMInvocationEvent
from ecs_agent.types import Usage


def test_openai_cache_normalization_with_prompt_cache_fields() -> None:
    usage = normalize_openai_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 40},
        }
    )

    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 120
    assert usage.cached_input_tokens == 40
    assert usage.stream_completeness is StreamCompleteness.COMPLETE


def test_normalize_openai_usage_accepts_responses_api_keys() -> None:
    usage = normalize_openai_usage(
        {
            "input_tokens": 7,
            "output_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 2},
        }
    )

    assert usage.prompt_tokens == 7
    assert usage.completion_tokens == 3
    assert usage.total_tokens is None
    assert usage.cached_input_tokens == 2


def test_normalize_openai_usage_does_not_coerce_missing_fields_to_zero() -> None:
    usage = normalize_openai_usage({})

    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.total_tokens is None
    assert usage.cached_input_tokens is None


def test_anthropic_cache_normalization_with_cache_fields() -> None:
    usage = normalize_anthropic_usage(
        {
            "input_tokens": 80,
            "output_tokens": 10,
            "cache_creation_input_tokens": 30,
            "cache_read_input_tokens": 20,
        }
    )

    assert usage.prompt_tokens == 80
    assert usage.completion_tokens == 10
    assert usage.total_tokens == 90
    assert usage.cache_creation_tokens == 30
    assert usage.cache_read_tokens == 20


def test_compute_cache_stats_openai_compatibility_formula() -> None:
    usage = UsageRecord(prompt_tokens=100, cached_input_tokens=40)

    stats = compute_cache_stats(usage)

    assert stats is not None
    assert stats.cache_read_tokens == 40
    assert stats.total_prompt_tokens == 100
    assert stats.hit_rate == 0.4


def test_compute_cache_stats_anthropic_formula() -> None:
    usage = UsageRecord(
        prompt_tokens=50,
        cache_creation_tokens=30,
        cache_read_tokens=20,
    )

    stats = compute_cache_stats(usage)

    assert stats is not None
    assert stats.cache_read_tokens == 20
    assert stats.total_prompt_tokens == 100
    assert stats.hit_rate == 0.2


def test_compute_cache_stats_returns_none_when_cache_fields_missing() -> None:
    usage = UsageRecord(prompt_tokens=42)

    assert compute_cache_stats(usage) is None


def test_compute_cache_stats_zero_denominator_yields_none_hit_rate() -> None:
    usage = UsageRecord(
        prompt_tokens=0,
        cached_input_tokens=0,
        cache_creation_tokens=0,
        cache_read_tokens=0,
    )

    stats = compute_cache_stats(usage)

    assert stats is not None
    assert stats.total_prompt_tokens == 0
    assert stats.hit_rate is None


def test_interrupted_stream_policy_rejects_non_estimated_cost() -> None:
    usage = UsageRecord(stream_completeness=StreamCompleteness.PARTIAL)

    try:
        LLMInvocationEvent(
            entity_id=1,
            provider_id="openai",
            model="gpt-4o",
            usage=usage,
            cost=CostRecord(total_cost=1.2, is_estimated=False),
        )
    except ValueError as exc:
        assert "interrupted" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-estimated interrupted cost")


def test_interrupted_stream_policy_allows_none_or_estimated_cost() -> None:
    usage = UsageRecord(stream_completeness=StreamCompleteness.UNKNOWN)

    event_without_cost = LLMInvocationEvent(
        entity_id=1,
        provider_id="openai",
        model="gpt-4o",
        usage=usage,
        cost=None,
    )
    assert event_without_cost.cost is None

    event_with_estimated_cost = LLMInvocationEvent(
        entity_id=1,
        provider_id="openai",
        model="gpt-4o",
        usage=usage,
        cost=CostRecord(total_cost=0.4, is_estimated=True),
    )
    assert event_with_estimated_cost.cost is not None
    assert event_with_estimated_cost.cost.is_estimated is True


def test_pricing_catalog_defaults_include_required_models() -> None:
    catalog = PricingCatalog()

    assert catalog.version.startswith("v")
    assert catalog.get_pricing("openai", "gpt-4o") is not None
    assert catalog.get_pricing("openai", "gpt-4o-mini") is not None
    assert catalog.get_pricing("anthropic", "claude-3-5-sonnet") is not None
    assert catalog.get_pricing("anthropic", "claude-3-haiku") is not None


def test_pricing_catalog_override_takes_precedence() -> None:
    custom = {
        "openai": {
            "gpt-4o": ModelPricing(
                input_per_million=1.0,
                output_per_million=2.0,
                cached_input_per_million=0.5,
                cache_write_per_million=None,
            )
        }
    }
    catalog = PricingCatalog(catalog=custom, version="v-custom")

    pricing = catalog.get_pricing("openai", "gpt-4o")
    assert pricing is not None
    assert pricing.input_per_million == 1.0
    assert catalog.version == "v-custom"


def test_usage_backward_compatibility_alias_supports_existing_shape() -> None:
    usage = Usage(prompt_tokens=5, completion_tokens=6, total_tokens=11)

    assert isinstance(usage, UsageRecord)
    assert usage.prompt_tokens == 5
    assert usage.completion_tokens == 6
    assert usage.total_tokens == 11


def test_types_exports_canonical_llm_invocation_event() -> None:
    assert ExportedLLMInvocationEvent is LLMInvocationEvent


def test_embedding_usage_maps_input_tokens_to_prompt_tokens() -> None:
    usage = normalize_openai_usage({"prompt_tokens": 42, "total_tokens": 42})

    assert usage.prompt_tokens == 42
    assert usage.completion_tokens is None
    assert usage.total_tokens == 42


@pytest.mark.asyncio
async def test_accounting_subscriber_single_terminal_event_records_once() -> None:
    bus = EventBus()
    subscriber = AccountingSubscriber()
    subscriber.subscribe(bus)

    await bus.publish(
        LLMInvocationEvent(
            entity_id=1,
            provider_id="openai",
            model="gpt-4o",
            usage=UsageRecord(
                prompt_tokens=100, completion_tokens=50, cached_input_tokens=20
            ),
        )
    )

    assert len(subscriber._records) == 1


@pytest.mark.asyncio
async def test_accounting_subscriber_aggregate_cache_stats_are_token_weighted() -> None:
    bus = EventBus()
    subscriber = AccountingSubscriber()
    subscriber.subscribe(bus)

    await bus.publish(
        LLMInvocationEvent(
            entity_id=1,
            provider_id="openai",
            model="gpt-4o",
            usage=UsageRecord(prompt_tokens=100, cached_input_tokens=90),
        )
    )
    await bus.publish(
        LLMInvocationEvent(
            entity_id=2,
            provider_id="openai",
            model="gpt-4o",
            usage=UsageRecord(prompt_tokens=900, cached_input_tokens=90),
        )
    )

    aggregate = subscriber.get_aggregate_stats("openai", "gpt-4o")
    assert aggregate is not None
    assert aggregate.cache_read_tokens == 180
    assert aggregate.total_prompt_tokens == 1000
    assert aggregate.hit_rate == 0.18


@pytest.mark.asyncio
async def test_accounting_subscriber_failure_is_logged_and_publish_continues(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bus = EventBus()
    subscriber = AccountingSubscriber()
    subscriber.subscribe(bus)

    async def bad_handler(event: LLMInvocationEvent) -> None:
        _ = event
        raise RuntimeError("accounting explode")

    bus.subscribe(LLMInvocationEvent, bad_handler)

    await bus.publish(
        LLMInvocationEvent(
            entity_id=9,
            provider_id="openai",
            model="gpt-4o",
            usage=UsageRecord(prompt_tokens=10, completion_tokens=5),
        )
    )
    captured = capsys.readouterr()

    assert len(subscriber._records) == 1
    assert "event_bus_subscriber_error" in captured.out
    assert "accounting explode" in captured.out
