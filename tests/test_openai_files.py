from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from ecs_agent.providers import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_files import OpenAIFilesService
from ecs_agent.types import FileRefPart


def _build_files_config() -> ProviderConfig:
    return ProviderConfig(
        provider_id="openai",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_format=ApiFormat.OPENAI_FILES,
    )


@pytest.mark.asyncio
async def test_openai_files_upload_file_returns_file_ref_and_tracks_metadata(
    tmp_path: Path,
) -> None:
    service = OpenAIFilesService(config=_build_files_config())
    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello", encoding="utf-8")

    request = httpx.Request("POST", "https://api.openai.com/v1/files")
    mock_response = httpx.Response(
        status_code=200,
        request=request,
        json={
            "id": "file_123",
            "filename": "doc.txt",
            "bytes": 5,
            "purpose": "assistants",
        },
    )

    with patch.object(service._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        result = await service.upload_file(file_path, "assistants")

    assert result == FileRefPart(file_id="file_123", filename="doc.txt")
    assert service.last_upload_metadata == {
        "file_id": "file_123",
        "filename": "doc.txt",
        "bytes": 5,
        "purpose": "assistants",
        "provider_id": "openai",
    }

    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args.args[0] == "https://api.openai.com/v1/files"
    assert call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert call_args.kwargs["data"] == {"purpose": "assistants"}
    assert call_args.kwargs["files"]["file"][0] == "doc.txt"


@pytest.mark.asyncio
async def test_openai_files_upload_invalid_file_raises_value_error() -> None:
    service = OpenAIFilesService(config=_build_files_config())

    with pytest.raises(ValueError, match="File path does not exist"):
        await service.upload_file("/missing/does-not-exist.txt", "assistants")


@pytest.mark.asyncio
async def test_openai_files_upload_unsupported_purpose_raises_value_error(
    tmp_path: Path,
) -> None:
    service = OpenAIFilesService(config=_build_files_config())
    file_path = tmp_path / "data.json"
    file_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported purpose"):
        await service.upload_file(file_path, "search")


@pytest.mark.asyncio
async def test_openai_files_delete_file_calls_delete_endpoint() -> None:
    service = OpenAIFilesService(config=_build_files_config())

    request = httpx.Request("DELETE", "https://api.openai.com/v1/files/file_123")
    mock_response = httpx.Response(
        status_code=200,
        request=request,
        json={"id": "file_123", "deleted": True},
    )

    with patch.object(
        service._client,
        "delete",
        new_callable=AsyncMock,
    ) as mock_delete:
        mock_delete.return_value = mock_response
        await service.delete_file("file_123")

    mock_delete.assert_called_once_with(
        "https://api.openai.com/v1/files/file_123",
        headers={"Authorization": "Bearer test-key"},
    )
