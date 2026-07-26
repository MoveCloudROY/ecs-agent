"""JSONL-over-stdio command loop for driving a plan-and-task debug session.

One JSON command per stdin line, one JSON result per stdout line — so an agent
can drive the workflow turn-by-turn from a shell (or a persistent terminal)
without writing Python, with full introspection between turns.

Commands (``cmd`` field):

- ``{"cmd": "send", "text": "/plan:start …"}``   → run one turn, return TurnResult
- ``{"cmd": "answer", "answers": [1]}``          → resolve a surfaced ask_question
- ``{"cmd": "answer"}`` (no answers)             → dismiss the question
- ``{"cmd": "snapshot"}``                        → return a StateSnapshot
- ``{"cmd": "artifact", "path": "plan/draft.md"}``→ return file contents
- ``{"cmd": "events", "turn": 0}``               → return that turn's event log
- ``{"cmd": "quit"}``                            → terminate

Model: ``--fake <script.json>`` for a deterministic ``FakeModel`` replay, else
the ``LLM_*`` environment (same as ``main.py`` via ``build_model_from_env``).
``--record <file.jsonl>`` tees every ``{command, result}`` pair for evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from ecs_agent.providers.fake_model import FakeModel
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.types import CompletionResult, Message, ToolCall
from examples.e2e.plan_and_task.debug.session import PlanTaskDebugSession


def _load_fake_model(path: Path) -> FakeModel:
    """Build a ``FakeModel`` from a JSON script of responses.

    Accepts a top-level list, or ``{"responses": [...]}``. Each response is a
    string (assistant content) or ``{"content": str, "tool_calls": [{id, name,
    arguments}]}``.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw["responses"] if isinstance(raw, dict) else raw
    responses: list[CompletionResult] = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            content, tool_calls = item, []
        else:
            content = item.get("content", "")
            tool_calls = [
                ToolCall(
                    id=call.get("id", f"call_{index}_{i}"),
                    name=call["name"],
                    arguments=call.get("arguments", {}),
                )
                for i, call in enumerate(item.get("tool_calls", []))
            ]
        responses.append(
            CompletionResult(
                message=Message(
                    role="assistant", content=content, tool_calls=tool_calls
                )
            )
        )
    return FakeModel(responses=responses)


def _build_model(args: argparse.Namespace) -> LLMModel:
    if args.fake:
        return _load_fake_model(Path(args.fake))
    from examples.e2e.plan_and_task.main import build_model_from_env

    return build_model_from_env()


async def _dispatch(
    session: PlanTaskDebugSession, command: dict[str, Any]
) -> dict[str, Any]:
    cmd = command.get("cmd")
    if cmd == "send":
        text = command.get("text", "")
        if not isinstance(text, str) or not text:
            return {"ok": False, "error": "send requires a non-empty 'text'."}
        result = await session.send(text, timeout=command.get("timeout"))
        return result.to_dict()
    if cmd == "answer":
        result = await session.answer(
            command.get("answers"), timeout=command.get("timeout")
        )
        return result.to_dict()
    if cmd == "snapshot":
        return {"ok": True, "snapshot": session.snapshot().to_dict()}
    if cmd == "artifact":
        path = command.get("path")
        if not isinstance(path, str):
            return {"ok": False, "error": "artifact requires a 'path'."}
        try:
            return {"ok": True, "path": path, "content": session.read_artifact(path)}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
    if cmd == "events":
        return {
            "ok": True,
            "events": session.events(
                turn=command.get("turn"), kinds=command.get("kinds")
            ),
        }
    return {"ok": False, "error": f"unknown command: {cmd!r}"}


async def _run(args: argparse.Namespace) -> None:
    model = _build_model(args)
    record_handle = (
        Path(args.record).open("a", encoding="utf-8") if args.record else None
    )
    session = await PlanTaskDebugSession.build(
        model,
        base_dir=Path(args.base_dir) if args.base_dir else None,
        surface_questions=not args.auto_answer,
        enable_tool_sink=not args.no_tool_sink,
        max_turn_seconds=args.max_turn_seconds,
        close_model=True,
    )

    def emit(payload: dict[str, Any], command: dict[str, Any] | None = None) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        print(line, flush=True)
        if record_handle is not None:
            record_handle.write(
                json.dumps({"command": command, "result": payload}, ensure_ascii=False)
                + "\n"
            )
            record_handle.flush()

    emit({"ok": True, "event": "ready", "snapshot": session.snapshot().to_dict()})
    loop = asyncio.get_running_loop()
    try:
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                command = json.loads(line)
            except json.JSONDecodeError as exc:
                emit({"ok": False, "error": f"invalid JSON: {exc}"})
                continue
            if command.get("cmd") == "quit":
                break
            try:
                result = await _dispatch(session, command)
            except Exception as exc:  # surface, never crash the loop
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            emit(result, command)
    finally:
        await session.aclose()
        if record_handle is not None:
            record_handle.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m examples.e2e.plan_and_task.debug",
        description="Drive the plan-and-task world turn-by-turn over JSONL stdio.",
    )
    parser.add_argument(
        "--fake",
        metavar="SCRIPT.json",
        help="Use a deterministic FakeModel from a JSON response script.",
    )
    parser.add_argument(
        "--base-dir",
        help="Workflow scratchbook root (default: the example dir). Use a temp "
        "dir to avoid polluting the committed scratchbook/.",
    )
    parser.add_argument(
        "--auto-answer",
        action="store_true",
        help="Auto-answer ask_question via the answer policy instead of "
        "surfacing questions to the caller.",
    )
    parser.add_argument(
        "--no-tool-sink",
        action="store_true",
        help="Disable the tool-results scratchbook sink. It is ON by default to "
        "match main.py; disabling keeps large tool outputs inline in context.",
    )
    parser.add_argument(
        "--record", metavar="FILE.jsonl", help="Append every command/result pair."
    )
    parser.add_argument("--max-turn-seconds", type=float, default=180.0)
    args = parser.parse_args(argv)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
