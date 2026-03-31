"""Provider connection configuration primitives."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ApiFormat(StrEnum):
    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    OPENAI_RESPONSES = "openai_responses"
    OPENAI_EMBEDDINGS = "openai_embeddings"
    OPENAI_FILES = "openai_files"
    ANTHROPIC_MESSAGES = "anthropic_messages"


@dataclass(slots=True)
class ProviderConfig:
    provider_id: str
    base_url: str
    api_key: str
    api_format: ApiFormat
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout: float | None = None


@dataclass(slots=True)
class ProviderEntry:
    base_url: str
    api_format: ApiFormat
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    timeout: float | None = None
    default_max_tokens: int = 4096

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderEntry":
        base_url_raw = data.get("base_url")
        if not isinstance(base_url_raw, str) or not base_url_raw:
            raise ValueError("Provider entry requires non-empty 'base_url'")

        api_format_raw = data.get("api_format")
        if not isinstance(api_format_raw, str):
            raise ValueError(
                f"Invalid api_format '{api_format_raw}'. Expected one of: "
                f"{', '.join(value.value for value in ApiFormat)}"
            )
        try:
            api_format = ApiFormat(api_format_raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid api_format '{api_format_raw}'. Expected one of: "
                f"{', '.join(value.value for value in ApiFormat)}"
            ) from exc

        api_key = data.get("api_key")
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError("Provider entry 'api_key' must be a string when provided")

        api_key_env = data.get("api_key_env")
        if api_key_env is not None and not isinstance(api_key_env, str):
            raise ValueError(
                "Provider entry 'api_key_env' must be a string when provided"
            )

        extra_headers_raw = data.get("extra_headers", {})
        if not isinstance(extra_headers_raw, dict):
            raise ValueError("Provider entry 'extra_headers' must be a dict[str, str]")
        if not all(
            isinstance(header_name, str) and isinstance(header_value, str)
            for header_name, header_value in extra_headers_raw.items()
        ):
            raise ValueError("Provider entry 'extra_headers' must be a dict[str, str]")

        timeout_raw = data.get("timeout")
        timeout: float | None
        if timeout_raw is None:
            timeout = None
        elif isinstance(timeout_raw, int | float):
            timeout = float(timeout_raw)
        else:
            raise ValueError("Provider entry 'timeout' must be a number when provided")

        default_max_tokens_raw = data.get("default_max_tokens", 4096)
        if not isinstance(default_max_tokens_raw, int):
            raise ValueError("Provider entry 'default_max_tokens' must be an integer")

        return cls(
            base_url=base_url_raw.rstrip("/"),
            api_format=api_format,
            api_key=api_key,
            api_key_env=api_key_env,
            extra_headers=dict(extra_headers_raw),
            timeout=timeout,
            default_max_tokens=default_max_tokens_raw,
        )
