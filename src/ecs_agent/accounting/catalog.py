"""Versioned in-repo pricing catalog contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None
    cache_write_per_million: float | None


DEFAULT_CATALOG_VERSION = "v1"

DEFAULT_PRICING: dict[str, dict[str, ModelPricing]] = {
    "openai": {
        "gpt-4o": ModelPricing(
            input_per_million=5.0,
            output_per_million=15.0,
            cached_input_per_million=2.5,
            cache_write_per_million=None,
        ),
        "gpt-4o-mini": ModelPricing(
            input_per_million=0.15,
            output_per_million=0.6,
            cached_input_per_million=0.075,
            cache_write_per_million=None,
        ),
    },
    "anthropic": {
        "claude-3-5-sonnet": ModelPricing(
            input_per_million=3.0,
            output_per_million=15.0,
            cached_input_per_million=0.3,
            cache_write_per_million=3.75,
        ),
        "claude-3-haiku": ModelPricing(
            input_per_million=0.25,
            output_per_million=1.25,
            cached_input_per_million=0.03,
            cache_write_per_million=0.3,
        ),
    },
}


class PricingCatalog:
    """Pricing catalog lookup with versioned contract."""

    def __init__(
        self,
        *,
        catalog: dict[str, dict[str, ModelPricing]] | None = None,
        version: str = DEFAULT_CATALOG_VERSION,
    ) -> None:
        self._catalog = DEFAULT_PRICING if catalog is None else catalog
        self.version = version

    def get_pricing(self, provider_id: str, model: str) -> ModelPricing | None:
        model_models = self._catalog.get(provider_id)
        if model_models is None:
            return None
        return model_models.get(model)
