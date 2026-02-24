"""OpenAI-compatible embedding provider using httpx."""


from typing import Any
import httpx

from ecs_agent.logging import get_logger

logger = get_logger(__name__)


class OpenAIEmbeddingProvider:
    """OpenAI-compatible embedding provider using httpx AsyncClient."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=10.0,
            pool=10.0,
        )
        self._client = httpx.AsyncClient(trust_env=False, timeout=self._timeout)

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

        url = f"{self._base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

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
            )
            raise

        response_data = response.json()
        return self._parse_response(response_data)

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
