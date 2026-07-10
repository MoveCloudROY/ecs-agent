"""Prometheus metrics endpoint helper tests."""

from __future__ import annotations

import socket
import urllib.request
from collections.abc import Callable
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST


def test_render_metrics_returns_prometheus_text_for_bound_registry() -> None:
    """render_metrics emits HELP/TYPE/sample lines for the provided registry."""
    from ecs_agent.plugins.prometheus import PrometheusMetrics, render_metrics
    from ecs_agent.types import RunCompletedEvent

    metrics = PrometheusMetrics()

    import asyncio

    asyncio.run(
        metrics.handle_run_completed(
            RunCompletedEvent(
                status="success",
                reason="max_ticks",
                duration_seconds=0.1,
                ticks=1,
                active_entities=0,
            )
        )
    )

    output = render_metrics(metrics)

    assert b"# HELP ecs_agent_runs_total Agent run outcomes." in output
    assert b"# TYPE ecs_agent_runs_total counter" in output
    assert b'ecs_agent_runs_total{status="success"} 1.0' in output


async def test_make_metrics_asgi_app_serves_bound_registry_without_framework() -> None:
    """ASGI helper returns a raw app that serves the bound registry."""
    from ecs_agent.plugins.prometheus import PrometheusMetrics, make_metrics_asgi_app
    from ecs_agent.types import RunnerTickCompletedEvent

    metrics = PrometheusMetrics()
    await metrics.handle_runner_tick_completed(
        RunnerTickCompletedEvent(
            tick=1,
            status="success",
            duration_seconds=0.1,
            active_entities=1,
        )
    )
    app = make_metrics_asgi_app(metrics)
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "method": "GET",
            "path": "/metrics",
            "headers": [],
            "query_string": b"",
        },
        receive,
        send,
    )

    start = next(message for message in sent if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    headers = {key.lower(): value for key, value in start["headers"]}

    assert start["status"] == 200
    assert headers[b"content-type"].startswith(CONTENT_TYPE_LATEST.encode())
    assert b'ecs_agent_runner_ticks_total{status="success"} 1.0' in body


def test_make_metrics_wsgi_app_serves_bound_registry_without_framework() -> None:
    """WSGI helper returns a raw app that serves the bound registry."""
    from ecs_agent.plugins.prometheus import PrometheusMetrics, make_metrics_wsgi_app
    from ecs_agent.types import ToolApprovedEvent

    metrics = PrometheusMetrics()

    import asyncio

    asyncio.run(
        metrics.handle_tool_approved(
            ToolApprovedEvent(
                entity_id=1,
                tool_call_id="call-raw-id",
                tool_name="read_file",
                policy="always_approve",
            )
        )
    )
    app = make_metrics_wsgi_app(metrics)
    status_headers: list[tuple[str, list[tuple[str, str]]]] = []

    def start_response(
        status: str,
        headers: list[tuple[str, str]],
        exc_info: object | None = None,
    ) -> Callable[[bytes], object]:
        status_headers.append((status, headers))

        def write(data: bytes) -> object:
            return None

        return write

    body = b"".join(
        app(
            {
                "REQUEST_METHOD": "GET",
                "PATH_INFO": "/metrics",
                "QUERY_STRING": "",
                "SERVER_NAME": "127.0.0.1",
                "SERVER_PORT": "0",
                "wsgi.url_scheme": "http",
            },
            start_response,
        )
    )

    status, headers = status_headers[0]
    header_map = {key.lower(): value for key, value in headers}
    assert status.startswith("200")
    assert header_map["content-type"].startswith(CONTENT_TYPE_LATEST)
    assert b'ecs_agent_tool_approved_total{policy="always_approve",tool="read_file"} 1.0' in body


def test_start_metrics_server_returns_cleanup_handle_for_bound_registry() -> None:
    """Standalone server can be started, scraped, and deterministically cleaned up."""
    from ecs_agent.plugins.prometheus import PrometheusMetrics, start_metrics_server
    from ecs_agent.types import LLMRetryEvent

    metrics = PrometheusMetrics()

    import asyncio

    asyncio.run(
        metrics.handle_llm_retry(
            LLMRetryEvent(provider_id="openai", model="gpt-4o-mini", reason="timeout", attempt=2)
        )
    )

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    handle = start_metrics_server(port, addr="127.0.0.1", metrics=metrics)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
            body = response.read()
            content_type = response.headers["Content-Type"]

        assert response.status == 200
        assert content_type.startswith(CONTENT_TYPE_LATEST)
        assert b'ecs_agent_llm_retries_total{model="gpt-4o-mini",provider="openai",reason="timeout"} 1.0' in body
    finally:
        handle.close(timeout=5)

    assert not handle.thread.is_alive()


def test_start_metrics_server_handle_can_be_tuple_unpacked() -> None:
    """Cleanup handle preserves access to the underlying server and thread."""
    from ecs_agent.plugins.prometheus import PrometheusMetrics, start_metrics_server

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    handle = start_metrics_server(port, addr="127.0.0.1", metrics=PrometheusMetrics())
    try:
        server, thread = handle
        assert server is handle.server
        assert thread is handle.thread
    finally:
        handle.close(timeout=5)
