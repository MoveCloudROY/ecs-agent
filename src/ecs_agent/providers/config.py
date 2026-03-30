"""Provider connection configuration primitives."""

from dataclasses import dataclass, field
from enum import StrEnum


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
