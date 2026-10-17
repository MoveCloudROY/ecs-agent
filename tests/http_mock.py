"""HTTP fixtures that exercise provider SDKs through a real httpx client."""

from collections.abc import AsyncIterator

import httpx


class SSEStream(httpx.AsyncByteStream):
    """Encode fixture SSE lines as incremental HTTP chunks and track cleanup."""

    def __init__(self, lines: AsyncIterator[str]) -> None:
        self.lines = lines
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for line in self.lines:
            # Existing fixtures specify one complete JSON event per data line.
            yield (line + ("\n\n" if line.startswith("data:") else "\n")).encode()

    async def aclose(self) -> None:
        self.closed = True
        close = getattr(self.lines, "aclose", None)
        if close is not None:
            await close()


def sse_response(lines: AsyncIterator[str]) -> httpx.Response:
    """Build a streaming response without buffering the event iterator."""
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=SSEStream(lines)
    )


class MockHTTP(httpx.MockTransport):
    """Serve queued responses/errors and record actual serialized requests."""

    def __init__(self) -> None:
        self.responses: list[httpx.Response | httpx.RequestError | httpx.HTTPStatusError] = []
        self.requests: list[httpx.Request] = []
        self.returned: list[httpx.Response] = []
        super().__init__(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"Unexpected HTTP request: {request.method} {request.url}")
        response = self.responses.pop(0)
        if isinstance(response, httpx.HTTPStatusError):
            response = response.response
        if isinstance(response, httpx.RequestError):
            response.request = request
            raise response
        self.returned.append(response)
        return response

    def client(self) -> httpx.AsyncClient:
        """Create a real HTTP client backed only by this in-memory transport."""
        return httpx.AsyncClient(transport=self, trust_env=False)
