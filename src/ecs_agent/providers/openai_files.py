from __future__ import annotations

from pathlib import Path

import httpx

from ecs_agent.logging import get_logger
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.types import FileRefPart

logger = get_logger(__name__)


class OpenAIFilesService:
    _SUPPORTED_PURPOSES: frozenset[str] = frozenset(
        {"assistants", "fine-tune", "batch", "vision"}
    )

    def __init__(
        self,
        config: ProviderConfig,
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
    ) -> None:
        if config.api_format is not ApiFormat.OPENAI_FILES:
            raise ValueError("ProviderConfig.api_format must be OPENAI_FILES")

        self._config = config
        timeout_value = config.timeout
        if timeout_value is not None:
            timeout = httpx.Timeout(timeout_value)
        else:
            timeout = httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=10.0,
                pool=10.0,
            )

        self._client = httpx.AsyncClient(trust_env=False, timeout=timeout)
        self.last_upload_metadata: dict[str, str | int] | None = None

    async def upload_file(self, path: str | Path, purpose: str) -> FileRefPart:
        if purpose not in self._SUPPORTED_PURPOSES:
            raise ValueError(
                "Unsupported purpose. Allowed values: assistants, fine-tune, batch, vision"
            )

        file_path = Path(path)
        if not file_path.exists():
            raise ValueError(f"File path does not exist: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"File path must be a regular file: {file_path}")

        url = f"{self._config.base_url}/files"
        headers = self._build_headers(include_content_type=False)
        data = {"purpose": purpose}
        files = {"file": (file_path.name, file_path.read_bytes())}

        try:
            response = await self._client.post(
                url, data=data, files=files, headers=headers
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "openai_file_upload_http_error",
                status_code=exc.response.status_code,
                response_body=exc.response.text,
                exception=str(exc),
                purpose=purpose,
            )
            raise
        except httpx.RequestError as exc:
            logger.error(
                "openai_file_upload_network_error",
                exception_type=type(exc).__name__,
                exception=str(exc),
                purpose=purpose,
            )
            raise

        payload = response.json()
        file_id = payload.get("id")
        if not isinstance(file_id, str) or not file_id:
            raise ValueError("OpenAI files response missing non-empty 'id'")

        filename = payload.get("filename")
        if not isinstance(filename, str) or not filename:
            filename = file_path.name

        bytes_count = payload.get("bytes")
        parsed_bytes_count = (
            bytes_count if isinstance(bytes_count, int) else file_path.stat().st_size
        )
        self.last_upload_metadata = {
            "file_id": file_id,
            "filename": filename,
            "bytes": parsed_bytes_count,
            "purpose": purpose,
            "provider_id": self._config.provider_id,
        }
        return FileRefPart(file_id=file_id, filename=filename)

    async def delete_file(self, file_id: str) -> None:
        if not file_id:
            raise ValueError("file_id must not be empty")

        url = f"{self._config.base_url}/files/{file_id}"
        headers = self._build_headers(include_content_type=False)

        try:
            response = await self._client.delete(url, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "openai_file_delete_http_error",
                status_code=exc.response.status_code,
                response_body=exc.response.text,
                exception=str(exc),
                file_id=file_id,
            )
            raise
        except httpx.RequestError as exc:
            logger.error(
                "openai_file_delete_network_error",
                exception_type=type(exc).__name__,
                exception=str(exc),
                file_id=file_id,
            )
            raise

    def _build_headers(self, *, include_content_type: bool) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        if include_content_type:
            headers["Content-Type"] = "application/json"
        headers.update(self._config.extra_headers)
        return headers
