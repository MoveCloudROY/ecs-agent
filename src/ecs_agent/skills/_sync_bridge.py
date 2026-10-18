"""Bridge for running a coroutine from synchronous skill-installation code."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any


def run_coroutine_sync(operation: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine to completion from a synchronous call site.

    Outside a running event loop the coroutine runs via ``asyncio.run``;
    inside one it runs on a fresh loop in a worker thread (blocking the
    caller until done).

    CAVEAT: the temporary loop is destroyed afterwards, so this must not be
    used to create loop-bound resources that outlive the call (open
    connections, sessions, background tasks). Such resources must be created
    on the loop that will use them — see MCPClient's owner-task lifecycle.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(operation)

    def _run_in_thread() -> Any:
        return asyncio.run(operation)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future: Future[Any] = executor.submit(_run_in_thread)
        return future.result()
