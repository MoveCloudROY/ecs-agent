"""Canonical provider/model identifier helpers."""

from dataclasses import dataclass


@dataclass(slots=True)
class ModelId:
    provider: str
    model: str


def parse_model_id(raw: str) -> ModelId:
    if not raw:
        raise ValueError("Model ID must not be empty")

    if ":" in raw:
        raise ValueError(
            "Model ID must use 'provider/model', not colon-delimited format"
        )

    if raw.count("/") != 1:
        raise ValueError("Model ID must be in 'provider/model' format")

    provider, model = raw.split("/", maxsplit=1)
    if not provider or not model:
        raise ValueError("Model ID provider and model must both be non-empty")

    return ModelId(provider=provider, model=model)


def format_model_id(model_id: ModelId) -> str:
    if not model_id.provider or not model_id.model:
        raise ValueError("Model ID provider and model must both be non-empty")

    if "/" in model_id.provider or "/" in model_id.model:
        raise ValueError("Model ID fields must not contain '/'")

    if ":" in model_id.provider or ":" in model_id.model:
        raise ValueError("Model ID fields must not contain ':'")

    return f"{model_id.provider}/{model_id.model}"
