"""OpenAI-compatible embedding provider using httpx."""

from __future__ import annotations

from typing import Any

import httpx

from ecs_agent.accounting.models import UsageRecord
from ecs_agent.accounting.normalization import normalize_openai_usage
from ecs_agent.logging import get_logger
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.model_id import ModelId, format_model_id

logger = get_logger(__name__)


class OpenAIEmbeddingProvider:
    """OpenAI-compatible embedding provider using httpx AsyncClient."""

    def __init__(
        self,
        config: ProviderConfig | None = None,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        provider_id: str = "openai",
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
    ) -> None:
        resolved_config = self._resolve_config(
            config=config,
            api_key=api_key,
            base_url=base_url,
            provider_id=provider_id,
        )

        self._config = resolved_config
        self._model = model
        self._canonical_model_id = format_model_id(
            ModelId(provider=self._config.provider_id, model=self._model)
        )
        timeout_value = self._config.timeout
        if timeout_value is not None:
            self._timeout = httpx.Timeout(timeout_value)
        else:
            self._timeout = httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=10.0,
                pool=10.0,
            )
        self._client = httpx.AsyncClient(trust_env=False, timeout=self._timeout)
        self._last_usage: UsageRecord | None = None

    @property
    def canonical_model_id(self) -> str:
        return self._canonical_model_id

    @property
    def last_usage(self) -> UsageRecord | None:
        return self._last_usage

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts into vectors.

        Args:
            texts: List of strings to embed

        Returns:
            List of vectors (each vector is a list of floats)
        """
        # Empty input optimization - return empty list without API call
        if not texts:
            return []

        url = f"{self._config.base_url}/embeddings"
        headers = self._build_headers()

        request_body = {
            "model": self._model,
            "input": texts,
        }

        try:
            response = await self._client.post(url, json=request_body, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "embedding_http_error",
                status_code=exc.response.status_code,
                response_body=exc.response.text,
                exception=str(exc),
                model_id=self._canonical_model_id,
            )
            raise
        except httpx.RequestError as exc:
            request_method: str | None = None
            request_url: str | None = None
            try:
                request_method = exc.request.method
                request_url = str(exc.request.url)
            except RuntimeError:
                pass
            logger.error(
                "embedding_network_error",
                exception_type=type(exc).__name__,
                exception=str(exc),
                request_method=request_method,
                request_url=request_url,
                model_id=self._canonical_model_id,
            )
            raise

        response_data = response.json()
        self._last_usage = self._extract_usage(response_data)
        return self._parse_response(response_data)

    def _resolve_config(
        self,
        *,
        config: ProviderConfig | None,
        api_key: str | None,
        base_url: str,
        provider_id: str,
    ) -> ProviderConfig:
        if config is not None:
            if config.api_format is not ApiFormat.OPENAI_EMBEDDINGS:
                raise ValueError("ProviderConfig.api_format must be OPENAI_EMBEDDINGS")
            return config

        if api_key is None:
            raise ValueError("api_key is required when config is not provided")

        return ProviderConfig(
            provider_id=provider_id,
            base_url=base_url,
            api_key=api_key,
            api_format=ApiFormat.OPENAI_EMBEDDINGS,
        )

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self._config.extra_headers)
        return headers

    def _extract_usage(self, response_data: dict[str, Any]) -> UsageRecord:
        raw_usage = response_data.get("usage")
        if not isinstance(raw_usage, dict):
            usage = UsageRecord()
        else:
            usage = normalize_openai_usage(raw_usage)

        if usage.total_tokens is None and usage.prompt_tokens is not None:
            usage.total_tokens = usage.prompt_tokens

        usage.provider_id = self._config.provider_id
        usage.model = self._canonical_model_id
        return usage

    def _parse_response(self, response_data: dict[str, Any]) -> list[list[float]]:
        """Parse OpenAI embeddings API response.

        Args:
            response_data: JSON response from OpenAI API

        Returns:
            List of embedding vectors
        """
        # Extract embeddings from response data
        # OpenAI format: {"data": [{"embedding": [0.1, 0.2, ...], "index": 0}, ...]}
        data_items = response_data["data"]
        embeddings = [item["embedding"] for item in data_items]
        return embeddings
