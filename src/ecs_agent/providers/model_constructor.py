"""Unified Model(...) factory for constructing LLMModel instances.

Provides a single entry point for creating any supported LLMModel without
requiring callers to manually build a ProviderConfig or choose an
implementation class.

Decision rules
--------------
- api_format only       → infer model_type from format
- model_type only       → infer api_format from model_type default
- both given            → validate compatibility; api_format overrides the default
- neither given         → raise ValueError
- type ↔ format mismatch → raise ValueError with "conflict" in the message
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from ecs_agent.providers.claude_model import ClaudeModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.providers.protocol import LLMModel


class ModelType(StrEnum):
    """Canonical string keys for supported model implementations."""

    OPENAI = "openai"
    CLAUDE = "claude"
    LITELLM = "litellm"


# Formats supported by each model_type (excluding LiteLLM which accepts all)
_OPENAI_FORMATS: frozenset[ApiFormat] = frozenset(
    {ApiFormat.OPENAI_CHAT_COMPLETIONS, ApiFormat.OPENAI_RESPONSES}
)
_CLAUDE_FORMATS: frozenset[ApiFormat] = frozenset({ApiFormat.ANTHROPIC_MESSAGES})

_SUPPORTED_FORMATS: dict[str, frozenset[ApiFormat]] = {
    ModelType.OPENAI: _OPENAI_FORMATS,
    ModelType.CLAUDE: _CLAUDE_FORMATS,
}

_DEFAULT_FORMAT: dict[str, ApiFormat] = {
    ModelType.OPENAI: ApiFormat.OPENAI_CHAT_COMPLETIONS,
    ModelType.CLAUDE: ApiFormat.ANTHROPIC_MESSAGES,
}

# api_format → model_type (only for unambiguous mappings)
_FORMAT_TO_MODEL_TYPE: dict[ApiFormat, str] = {
    ApiFormat.OPENAI_CHAT_COMPLETIONS: ModelType.OPENAI,
    ApiFormat.OPENAI_RESPONSES: ModelType.OPENAI,
    ApiFormat.ANTHROPIC_MESSAGES: ModelType.CLAUDE,
}


def _resolve_model_type(model_type: str | type | None) -> str | None:
    """Normalise model_type to a canonical ModelType string, or None."""
    if model_type is None:
        return None
    if isinstance(model_type, str):
        lower = model_type.lower()
        if lower not in (ModelType.OPENAI, ModelType.CLAUDE, ModelType.LITELLM):
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Supported values: {', '.join(ModelType)}."
            )
        return lower
    # Class references
    if model_type is OpenAIModel:
        return ModelType.OPENAI
    if model_type is ClaudeModel:
        return ModelType.CLAUDE
    try:
        from ecs_agent.providers.litellm_model import LiteLLMModel

        if model_type is LiteLLMModel:
            return ModelType.LITELLM
    except ImportError:
        pass
    raise ValueError(
        f"Unknown model_type class '{model_type.__name__}'. "
        "Supported classes: OpenAIModel, ClaudeModel, LiteLLMModel."
    )


def _resolve_api_format(api_format: ApiFormat | str | None) -> ApiFormat | None:
    """Normalise api_format to ApiFormat enum, or None."""
    if api_format is None:
        return None
    if isinstance(api_format, ApiFormat):
        return api_format
    try:
        return ApiFormat(api_format)
    except ValueError:
        raise ValueError(
            f"Unknown api_format '{api_format}'. "
            f"Supported values: {', '.join(v.value for v in ApiFormat)}."
        )


def Model(
    model_id: str,
    *,
    base_url: str,
    api_key: str,
    api_format: ApiFormat | str | None = None,
    model_type: str | type | None = None,
    provider_id: str = "",
    extra_headers: dict[str, str] | None = None,
    timeout: float | None = None,
    enable_store: bool = False,
    enable_prompt_caching: bool = True,
    **kwargs: Any,
) -> LLMModel:
    """Unified factory for constructing LLMModel instances.

    Args:
        model_id: Model name (e.g. "gpt-4o", "claude-3-5-haiku-20241022").
        base_url: Provider base URL.
        api_key: Authentication key.
        api_format: Wire format to use. Accepts an ApiFormat enum value or its
            string name (e.g. "openai_chat_completions"). When both api_format
            and model_type are provided, api_format overrides the model_type
            default; a mismatch raises ValueError.
        model_type: Implementation class to use. Accepts a ModelType / string
            ("openai", "claude", "litellm") or the actual class (OpenAIModel,
            ClaudeModel, LiteLLMModel). When omitted, inferred from api_format.
        provider_id: Optional label stored in ProviderConfig (unused at runtime).
        extra_headers: Additional HTTP headers forwarded on every request.
        timeout: Global request timeout in seconds (overrides per-phase defaults).
        enable_store: Enable conversation storage (Responses API feature).
        enable_prompt_caching: Emit Anthropic prompt-cache breakpoints (Claude
            adapter only; ignored by other formats). Defaults to True.
        **kwargs: Extra keyword arguments forwarded to the underlying model
            constructor (e.g. connect_timeout, max_tokens).

    Returns:
        An LLMModel instance satisfying the LLMModel protocol.

    Raises:
        ValueError: If api_format and model_type conflict, if neither is
            provided, if api_format cannot be used for model auto-selection,
            or if an unknown type / format string is supplied.
    """
    resolved_type = _resolve_model_type(model_type)
    resolved_format = _resolve_api_format(api_format)

    if resolved_type is None and resolved_format is None:
        raise ValueError(
            "At least one of 'api_format' or 'model_type' must be provided."
        )

    # Infer model_type from api_format
    if resolved_type is None:
        # resolved_format is guaranteed non-None by the check above
        inferred = _FORMAT_TO_MODEL_TYPE.get(resolved_format)  # type: ignore[arg-type]
        if inferred is None:
            raise ValueError(
                f"Cannot infer model_type from api_format '{resolved_format}'. "
                f"Formats that support auto-selection: "
                f"{', '.join(f.value for f in _FORMAT_TO_MODEL_TYPE)}."
            )
        resolved_type = inferred

    # Infer api_format from model_type (LiteLLM does not need one)
    if resolved_format is None and resolved_type != ModelType.LITELLM:
        resolved_format = _DEFAULT_FORMAT.get(resolved_type)

    # Validate compatibility when both are known
    if resolved_format is not None and resolved_type != ModelType.LITELLM:
        supported = _SUPPORTED_FORMATS.get(resolved_type, frozenset())
        if resolved_format not in supported:
            raise ValueError(
                f"model_type '{resolved_type}' and api_format '{resolved_format}' conflict. "
                f"'{resolved_type}' supports: "
                f"{', '.join(f.value for f in supported)}."
            )

    # LiteLLM path — does not use ProviderConfig
    if resolved_type == ModelType.LITELLM:
        try:
            from ecs_agent.providers.litellm_model import LiteLLMModel
        except ImportError as exc:
            raise ImportError(
                "litellm must be installed to use model_type='litellm'. "
                "Install with: pip install litellm"
            ) from exc
        return LiteLLMModel(model=model_id, api_key=api_key, base_url=base_url, **kwargs)

    assert resolved_format is not None
    config = ProviderConfig(
        provider_id=provider_id,
        base_url=base_url,
        api_key=api_key,
        api_format=resolved_format,
        extra_headers=extra_headers or {},
        timeout=timeout,
        enable_store=enable_store,
        enable_prompt_caching=enable_prompt_caching,
    )

    if resolved_type == ModelType.OPENAI:
        return OpenAIModel(config=config, model=model_id, **kwargs)
    if resolved_type == ModelType.CLAUDE:
        return ClaudeModel(config=config, model=model_id, **kwargs)

    raise ValueError(f"Unhandled model_type '{resolved_type}'.")  # pragma: no cover
