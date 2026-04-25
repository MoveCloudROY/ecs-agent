"""Provider registry and factory for LLM models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ecs_agent.providers.config import ApiFormat, ProviderConfig, ProviderEntry
from ecs_agent.providers.model_factory import create_model
from ecs_agent.providers.model_id import ModelId, parse_model_id
from ecs_agent.providers.protocol import LLMModel


class ProviderRegistry:
    def __init__(self, entries: dict[str, ProviderEntry]) -> None:
        self._entries = dict(entries)

    @classmethod
    def from_toml(cls, path: str | Path) -> ProviderRegistry:
        import tomllib

        with Path(path).open("rb") as handle:
            payload = tomllib.load(handle)

        providers = payload.get("providers")
        if not isinstance(providers, dict):
            raise ValueError("TOML must contain a [providers] table")
        return cls.from_dict(providers)

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, Any]]) -> ProviderRegistry:
        entries: dict[str, ProviderEntry] = {}
        for provider_id, raw_entry in data.items():
            if not isinstance(raw_entry, dict):
                raise ValueError(f"Provider entry '{provider_id}' must be a dictionary")
            entries[provider_id] = ProviderEntry.from_dict(raw_entry)
        return cls(entries)

    def get_entry(self, provider_id: str) -> ProviderEntry:
        try:
            return self._entries[provider_id]
        except KeyError as exc:
            available = ", ".join(self.provider_ids()) or "none"
            raise KeyError(
                f"Provider '{provider_id}' not found in registry. Available providers: {available}"
            ) from exc

    def provider_ids(self) -> list[str]:
        return sorted(self._entries.keys())


def get_model(
    model_id: str | ModelId,
    *,
    registry: ProviderRegistry,
    api_key: str | None = None,
) -> LLMModel:
    parsed = parse_model_id(model_id) if isinstance(model_id, str) else model_id
    entry = registry.get_entry(parsed.provider)

    resolved_api_key = _resolve_api_key(entry=entry, explicit_api_key=api_key)
    if resolved_api_key is None:
        raise ValueError(
            f"No API key available for provider '{parsed.provider}'. "
            "Pass api_key explicitly or configure api_key/api_key_env in registry entry."
        )

    config = ProviderConfig(
        provider_id=parsed.provider,
        base_url=entry.base_url,
        api_key=resolved_api_key,
        api_format=entry.api_format,
        extra_headers=dict(entry.extra_headers),
        timeout=entry.timeout,
    )

    if entry.api_format in (ApiFormat.OPENAI_EMBEDDINGS, ApiFormat.OPENAI_FILES):
        raise ValueError(
            f"api_format '{entry.api_format.value}' is not supported by get_model; "
            "use get_embedding_provider/get_file_service"
        )

    kwargs: dict[str, Any] = {}
    if entry.api_format == ApiFormat.ANTHROPIC_MESSAGES and entry.default_max_tokens is not None:
        kwargs["max_tokens"] = entry.default_max_tokens

    return create_model(config, parsed.model, **kwargs)


# Backward-compat alias
def get_llm_provider(
    model_id: str | ModelId,
    *,
    registry: ProviderRegistry,
    api_key: str | None = None,
) -> LLMModel:
    return get_model(model_id, registry=registry, api_key=api_key)


def _resolve_api_key(
    *, entry: ProviderEntry, explicit_api_key: str | None
) -> str | None:
    if explicit_api_key is not None:
        return explicit_api_key
    if entry.api_key is not None:
        return entry.api_key
    if entry.api_key_env is not None:
        return os.environ.get(entry.api_key_env)
    return None
