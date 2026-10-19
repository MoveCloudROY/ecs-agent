"""Coder/reviewer evidence retrieval with offline and real LLM modes.

Run: uv run python -m examples.e2e.cross_session_records --output /tmp/session-demo
Add --live --trace and set LLM_API_KEY to let a real model drive both agents.
LLM_BASE_URL, LLM_MODEL, and LLM_API_FORMAT configure the provider. Defaults are
https://api.deepseek.com, deepseek-flash, and responses. Offline mode uses
FakeModel for the coder and a scripted reviewer. Both modes save interaction.json
(full projected messages), interaction.txt (abbreviated display), and result.json.
Normal output is concise. Add --stress-output only to exercise long-line paging
with explicitly labeled synthetic padding.
"""

import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any
from uuid import uuid4

from ecs_agent.components import ConversationComponent, PendingToolCallsComponent, SessionRefComponent, ToolRegistryComponent
from ecs_agent.core import World
from ecs_agent.history import append_conversation_message, replace_conversation_history
from ecs_agent.providers import FakeModel
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.providers.registry import ProviderRegistry, get_model
from ecs_agent.scratchbook import ArtifactRegistry
from ecs_agent.sessions import SessionService, SQLiteSessionStore, install_session_recording
from ecs_agent.sessions.service import project_value
from ecs_agent.sessions.types import SessionRecord
from ecs_agent.systems.subagent.child_world import ChildWorldBuilder
from ecs_agent.systems.subagent.delegation import DelegationExecutor
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.tools.builtins import BuiltinToolsSkill
from ecs_agent.types import CompletionResult, EntityId, InheritancePolicy, Message, SubagentConfig, ToolCall, ToolSchema


async def _call(world: World, entity: EntityId, name: str, arguments: dict[str, Any]) -> str:
    call = ToolCall(id=uuid4().hex, name=name, arguments=arguments)
    append_conversation_message(world, entity, Message(role="assistant", content="", tool_calls=[call]))
    world.add_component(entity, PendingToolCallsComponent(tool_calls=[call]))
    await ToolExecutionSystem().process(world)
    conversation = world.get_component(entity, ConversationComponent)
    assert conversation is not None
    return conversation.messages[-1].content


def _write_interaction(service: SessionService, sessions: dict[str, str], output_dir: Path) -> None:
    """Export visible messages and compaction events, with full tool bodies."""
    entries: list[dict[str, Any]] = []
    display: list[str] = []
    for agent, session_id in sessions.items():
        records = list(service.store.records(session_id))
        calls = {r.payload["tool_call_id"]: r.payload for r in records if r.kind == "tool_started"}
        for record in records:
            if record.kind not in {"message", "compaction", "history_replaced"}:
                continue
            payload = project_value(record.payload)
            entry: dict[str, Any] = {
                "agent": agent, "session_id": session_id,
                "record_id": record.record_id, "timestamp": record.timestamp,
                "kind": record.kind,
            }
            if record.kind == "message":
                entry.update({
                    "role": payload.get("role"),
                    "content": "".join(service.content_chunks(record)),
                    "tool_calls": payload.get("tool_calls"),
                    "tool_call_id": payload.get("tool_call_id"),
                })
                entry["content_chars"] = len(entry["content"])
                if entry["role"] == "tool":
                    entry["tool_name"] = calls.get(entry["tool_call_id"], {}).get("tool_name", "unknown")
            else:
                entry["details"] = {
                    key: payload[key] for key in (
                        "reason", "previous_head", "previous_message_count",
                        "replacement_message_count", "summary",
                    ) if key in payload
                }
            entries.append(entry)
            display.append(f"[{len(entries):02d}] {agent} / {entry.get('role', record.kind)}")
            for call in entry.get("tool_calls") or []:
                display.append(f"CALL {call['name']} id={call['id']} arguments={json.dumps(call['arguments'], ensure_ascii=False)}")
            if entry.get("role") == "tool":
                display.append(f"RESULT {entry['tool_name']} id={entry['tool_call_id']} chars={entry['content_chars']}")
            content = str(entry.get("content", entry.get("details", "")))
            if entry.get("role") == "tool" and len(content) > 1200:
                content = content[:800] + "\n[display abbreviated; full body in interaction.json]\n" + content[-300:]
            display.extend([content, ""])
    (output_dir / "interaction.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "interaction.txt").write_text("\n".join(display), encoding="utf-8")


def _verify_reads(service: SessionService, reviewer_session: str, source: SessionRecord) -> dict[str, Any]:
    """Verify contiguous pages actually returned to the reviewer, not its claims."""
    records = list(service.store.records(reviewer_session))
    calls = {r.payload["tool_call_id"]: r.payload for r in records if r.kind == "tool_started"}
    pages: dict[int, tuple[dict[str, Any], str]] = {}
    read_offsets: list[int] = []
    search_calls = [call for call in calls.values() if call["tool_name"] == "sessions_search"]
    search_scoped = bool(search_calls) and all(call["arguments"].get("session_id") == source.session_id for call in search_calls)
    search_self_hits = 0
    for record in records:
        if record.payload.get("role") != "tool":
            continue
        call = calls.get(record.payload.get("tool_call_id"), {})
        if call.get("tool_name") == "sessions_search":
            response = "".join(service.content_chunks(record))
            if response.startswith("{"):
                search_self_hits += sum(item.get("session_id") == reviewer_session for item in json.loads(response).get("items", []))
        if call.get("tool_name") != "read_file":
            continue
        if call["arguments"].get("file_path") != service.tool_result_uri(source):
            continue
        response = "".join(service.content_chunks(record))
        header, separator, body = response.partition("\n---\n")
        if not separator:
            continue
        metadata = json.loads(header)
        # Support the optional line-number display without counting it as text.
        lines = body.splitlines()
        if call["arguments"].get("line_numbers"):
            lines = [line.partition(": ")[2] for line in lines]
        text = "".join(json.loads(line.removeprefix("text: ")) for line in lines if line.startswith("text: "))
        pages[metadata["start_line"]] = metadata, text
        read_offsets.append(metadata["start_line"])
    offset: int | None = 1
    chunks: list[str] = []
    visited: set[int] = set()
    while offset is not None and offset in pages and offset not in visited:
        visited.add(offset)
        metadata, text = pages[offset]
        chunks.append(text)
        offset = metadata["next_offset"]
    return {
        "pages_read": len(read_offsets),
        "read_offsets": read_offsets,
        "duplicate_page_reads": len(read_offsets) - len(pages),
        "search_scoped": search_scoped,
        "search_self_hits": search_self_hits,
        "evidence_complete": offset is None and "".join(chunks) == "".join(service.content_chunks(source)),
        "tools_used": sorted({call["tool_name"] for call in calls.values()}),
        "record_read": any(
            call["tool_name"] == "read_file" and call["arguments"].get("file_path") == service.record_uri(source)
            for call in calls.values()
        ),
    }


async def run_demo(
    output_dir: Path, *, model: LLMModel | None = None, stress_output: bool = False,
) -> dict[str, Any]:
    """Run checks and verify cross-session evidence; the caller owns model cleanup.

    Supplying a model makes both agents use actual model-driven delegation.
    Omitting it preserves the deterministic offline demonstration. Stress mode
    deliberately adds labeled synthetic padding to test lossless long-line paging.
    """
    started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    service = SessionService(SQLiteSessionStore(output_dir / "sessions.sqlite3"), artifact_registry=ArtifactRegistry(output_dir))
    root = World()
    entity = root.create_entity()
    root.add_component(entity, ConversationComponent(messages=[]))
    schema, handler = BuiltinToolsSkill().bind_workspace(str(output_dir)).tools()["read_file"]
    marker = "CHECK_RUN=" + uuid4().hex
    validation_runs = 0
    sessions: dict[str, str] = {}

    async def validate() -> str:
        nonlocal validation_runs
        validation_runs += 1
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c",
            "assert 2 + 2 == 4; print('PASS arithmetic: 2 + 2 == 4'); "
            "assert '会话'.encode('utf-8').decode('utf-8') == '会话'; "
            "print('PASS utf8_roundtrip: 会话 -> UTF-8 -> 会话'); print('2 checks passed')",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(output.decode())
        padding = ""
        if stress_output:
            padding = (
                "PAGING_STRESS_FIXTURE: synthetic padding for long-line paging; not additional checks.\n"
                + "".join(f"{index:04d}:" + "x" * 24 + ";" for index in range(1100)) + "\n"
            )
        return "TEST_OUTPUT\nScope: two demonstration smoke checks, not the project test suite.\n" + padding + output.decode() + "exit_code=0\n" + marker

    root.add_component(entity, ToolRegistryComponent(
        tools={"read_file": schema, "validate": ToolSchema(name="validate", description="Run two demonstration smoke checks (arithmetic and UTF-8 round trip); return named results, process exit code, and a unique run marker. This is not the project test suite.", parameters={"type": "object", "properties": {}})},
        handlers={"read_file": handler, "validate": validate},
    ))
    install_session_recording(root, entity, service, workspace_id=str(output_dir.resolve()), agent_name="main")
    try:
        coder_config = SubagentConfig(name="coder", model=model if model is not None else FakeModel([
            CompletionResult(message=Message(role="assistant", content="Run validation", tool_calls=[ToolCall(id=uuid4().hex, name="validate", arguments={})])),
            CompletionResult(message=Message(role="assistant", content="Validation finished")),
        ]), max_ticks=8, inheritance_policy=InheritancePolicy(inherit_tools=["validate"]))
        builder = ChildWorldBuilder()
        coder, coder_id = builder.assemble_child_world(root, entity, coder_config)
        coder_ref = coder.get_component(coder_id, SessionRefComponent)
        assert coder_ref is not None
        sessions["coder"] = coder_ref.session_id
        outcome = await DelegationExecutor().run_delegation(
            coder, coder_id,
            "Call validate exactly once, then briefly report the named check results and exit code. "
            "These are two demonstration smoke checks, not the project test suite. "
            "Any PAGING_STRESS_FIXTURE is synthetic paging-test data, not extra executions. Do not invent evidence.",
            coder_config,
        )
        if not outcome.success:
            raise RuntimeError(outcome.error)
        source = next((r for r in service.store.records(coder_ref.session_id) if r.payload.get("role") == "tool"), None)
        if source is None:
            raise RuntimeError("Coder produced no validation tool result")
        # Deliberately replace working context; this is host-driven compaction,
        # not an LLM summarizer. Durable original evidence must remain readable.
        replace_conversation_history(coder, coder_id, [Message(role="user", content="Earlier context was compacted")], reason="compaction_summary")
        reviewer_config = SubagentConfig(
            name="reviewer", model=model if model is not None else FakeModel([]), max_ticks=20,
            inheritance_policy=InheritancePolicy(inherit_tools=["read_file", "sessions_list", "sessions_search"]),
        )
        reviewer, reviewer_id = builder.assemble_child_world(root, entity, reviewer_config)
        reviewer_ref = reviewer.get_component(reviewer_id, SessionRefComponent)
        assert reviewer_ref is not None
        sessions["reviewer"] = reviewer_ref.session_id
        if model is not None:
            review = await DelegationExecutor().run_delegation(
                reviewer, reviewer_id,
                "Review the coder's actual test evidence after its working context was compacted. "
                "First discover the coder session with sessions_list(agent_name='coder'). "
                "Only after receiving its session_id, search TEST_OUTPUT with sessions_search using that exact session_id. "
                "and read the matching record with read_file. Follow its tool_result_uri and use "
                "read_file with limit=500 plus next_offset until has_more=false. Read ALL pages. "
                "Report the named check results, exit code, exact check result phrase, exact CHECK_RUN marker at the end, "
                "and cite the source session_id and record_id. Do not run checks yourself. "
                "The output explicitly describes two demonstration smoke checks; do not claim the project test suite passed. "
                "Any PAGING_STRESS_FIXTURE is synthetic paging-test data, not extra executions. "
                "Write a concise final explanation in Chinese (at most 8 lines), preserving evidence strings exactly.",
                reviewer_config,
            )
            if not review.success:
                raise RuntimeError(review.error)
            report = review.as_tuple()[0]
        else:
            report = await _scripted_review(reviewer, reviewer_id, marker)
        verification = _verify_reads(service, reviewer_ref.session_id, source)
        report_cites_source = all(value in report for value in (coder_ref.session_id, source.record_id, marker, "2 checks passed"))
        result = {
            "mode": "llm" if model is not None else "offline",
            "model": model.model_id if model is not None else "fake/scripted",
            "report": report, "session_id": coder_ref.session_id,
            "record_id": source.record_id, "reviewer_session_id": reviewer_ref.session_id,
            "marker": marker, "validation_runs": validation_runs,
            "stress_output": stress_output,
            "validation_output_chars": len("".join(service.content_chunks(source))),
            "tool_call_counts": {
                agent: dict(Counter(r.payload["tool_name"] for r in service.store.records(session_id) if r.kind == "tool_started"))
                for agent, session_id in sessions.items()
            },
            **verification, "report_cites_source": report_cites_source,
            "accepted": verification["evidence_complete"] and verification["record_read"]
                and report_cites_source and validation_runs == 1
                and verification["search_scoped"] and verification["duplicate_page_reads"] == 0
                and {"sessions_list", "sessions_search", "read_file"}.issubset(verification["tools_used"]),
            "elapsed_seconds": round(perf_counter() - started, 3),
        }
        (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    finally:
        try:
            _write_interaction(service, sessions, output_dir)
        finally:
            service.close()


async def _scripted_review(reviewer: World, reviewer_id: EntityId, marker: str) -> str:
    """Use the normal tool execution path for deterministic offline retrieval."""
    catalog = json.loads(await _call(reviewer, reviewer_id, "sessions_list", {"agent_name": "coder"}))
    coder_session = catalog["items"][0]["session_id"]
    hits = json.loads(await _call(reviewer, reviewer_id, "sessions_search", {"query": "TEST_OUTPUT", "session_id": coder_session}))
    hit = hits["items"][0]
    record_page = await _call(reviewer, reviewer_id, "read_file", {"file_path": hit["read_uri"]})
    tool_uri = next(line.removeprefix("tool_result_uri: ") for line in record_page.splitlines() if line.startswith("tool_result_uri: "))
    offset = 1
    chunks: list[str] = []
    while True:
        page = await _call(reviewer, reviewer_id, "read_file", {"file_path": tool_uri, "offset": offset, "limit": 500})
        header, body = page.split("\n---\n", 1)
        metadata = json.loads(header)
        chunks.extend(json.loads(line.removeprefix("text: ")) for line in body.splitlines() if line.startswith("text: "))
        if metadata["next_offset"] is None:
            break
        offset = metadata["next_offset"]
    evidence = "".join(chunks)
    report = f"2 checks passed. Evidence: session={coder_session}, record={hit['record_id']}; {marker}"
    append_conversation_message(reviewer, reviewer_id, Message(role="assistant", content=report))
    if not evidence.endswith(marker) or "2 checks passed" not in evidence:
        raise RuntimeError("Offline reviewer did not retrieve the expected evidence")
    return report


def main() -> None:
    """Print evidence references and the number of resource pages read."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/ecs-session-demo"))
    parser.add_argument("--live", action="store_true", help="Use a real LLM for both agents; requires LLM_API_KEY")
    parser.add_argument("--trace", action="store_true", help="Print the abbreviated visible interaction")
    parser.add_argument("--stress-output", action="store_true", help="Add labeled synthetic output to test multi-page long-line reads")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "deepseek-flash"))
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-format", default=os.getenv("LLM_API_FORMAT", "responses"), choices=["responses", "chat", "openai_responses", "openai_chat_completions"])
    args = parser.parse_args()
    if args.live and not os.getenv("LLM_API_KEY"):
        parser.error("--live requires LLM_API_KEY; no offline fallback is performed")

    async def execute() -> dict[str, Any]:
        model = None
        try:
            if args.live:
                api_format = ApiFormat.OPENAI_RESPONSES if args.api_format in {"responses", "openai_responses"} else ApiFormat.OPENAI_CHAT_COMPLETIONS
                registry = ProviderRegistry.from_dict({"demo": {
                    "base_url": args.base_url, "api_format": api_format.value,
                    "api_key_env": "LLM_API_KEY", "timeout": 120,
                }})
                model = get_model(f"demo/{args.model}", registry=registry)
            result = await run_demo(args.output, model=model, stress_output=args.stress_output)
            if args.live:
                result.update({"base_url": args.base_url, "api_format": api_format.value})
                (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        finally:
            if model is not None:
                await model.aclose()

    result = asyncio.run(execute())
    if args.trace:
        print((args.output / "interaction.txt").read_text(encoding="utf-8"))
    # The trace already contains the final report; avoid printing it twice.
    printed_result = {key: value for key, value in result.items() if key != "report"} if args.trace else result
    print(json.dumps(printed_result, ensure_ascii=False, indent=2))
    if not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
