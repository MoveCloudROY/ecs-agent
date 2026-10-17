"""Long, branching Agent conversations audited at the SDK HTTP boundary.

Snapshot branches use the public WorldSerializer API. Native ConversationTree
contracts are tested separately: snapshot success must not imply tree support.
No request headers or credentials are persisted by this harness.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import time
from copy import deepcopy
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from ecs_agent.accounting.models import LLMInvocationEvent
from ecs_agent.components import (
    ConversationComponent, LLMComponent, PendingToolCallsComponent,
    StreamingComponent, SystemPromptComponent, TerminalComponent, ToolRegistryComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.serialization import WorldSerializer
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.tool_execution import ToolExecutionSystem
from ecs_agent.types import EntityId, Message, ToolSchema


def digest(value: Any) -> str:
    """Hash JSON values deterministically without estimating model tokens."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def without_nulls(value: Any) -> Any:
    """Match optional SDK output fields omitted during replay serialization."""
    if isinstance(value, dict):
        return {key: without_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [without_nulls(item) for item in value]
    return value


def corpus(namespace: str, count: int) -> str:
    """Produce distinct synthetic records, with no private project content."""
    return "\n".join(
        f"Record {namespace}/{index:04d}: archive={hashlib.sha256(f'{namespace}:{index}'.encode()).hexdigest()[:16]}; "
        f"sector={index % 17}; capacity={100 + index}; owner=synthetic; "
        "retain the original ordering and identifiers for later audit comparisons."
        for index in range(count)
    )


class CapturedStream(httpx.AsyncByteStream):
    """Observe HTTP bytes without consuming a live SSE stream ahead of the SDK."""

    def __init__(self, inner: httpx.AsyncByteStream, record: dict[str, Any], owner: CaptureTransport) -> None:
        self.inner, self.record, self.owner = inner, record, owner
        self.chunks: list[bytes] = []

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self.inner:
                self.chunks.append(chunk)
                yield chunk
        except BaseException as exc:
            self.record["stream_error"] = type(exc).__name__
            raise
        finally:
            self.capture()

    def capture(self) -> None:
        raw = b"".join(self.chunks)
        if self.record.get("encoding") == "gzip":
            try:
                raw = gzip.decompress(raw)
            except (OSError, EOFError):
                return
        text = raw.decode("utf-8", errors="replace")
        self.record["raw_response"] = text
        if self.record["body"].get("stream"):
            for line in text.splitlines():
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                try:
                    event = json.loads(line[5:])
                except ValueError:
                    continue
                if isinstance(event.get("response"), dict):
                    self.record["response"] = event["response"]
        else:
            try:
                self.record["response"] = json.loads(text)
            except ValueError:
                self.record["decode_error"] = True
        self.owner.save()

    async def aclose(self) -> None:
        await self.inner.aclose()
        self.capture()
        self.record["closed"] = True
        self.record["seconds"] = round(time.monotonic() - self.record["started"], 3)
        self.owner.save()


class CaptureTransport(httpx.AsyncBaseTransport):
    """Record SDK-serialized bodies and raw responses with a hard request cap."""

    def __init__(self, inner: httpx.AsyncBaseTransport, *, limit: int = 32, directory: Path | None = None) -> None:
        self.inner, self.limit, self.directory = inner, limit, directory
        self.records: list[dict[str, Any]] = []
        self.context: dict[str, Any] = {}

    def save(self) -> None:
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            target = self.directory / "requests.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.records, ensure_ascii=False, indent=2))
            temporary.replace(target)

    def predict_reuse(self, body: dict[str, Any]) -> dict[str, Any]:
        """Record a conditional reuse baseline BEFORE dispatching the request.

        Only prior input token counts are used; observed cache hits never feed
        the prediction. This assumes persisted, reachable cache and unchanged
        server rendering, not a guarantee of backend cache availability.
        """
        reference = None
        kind = "cold_control"
        for previous in reversed(self.records):
            if previous["body"] == body and previous.get("response", {}).get("usage"):
                reference, kind = previous, "exact_replay"
                break
        if reference is None:
            parent = self.context.get("parent")
            if self.records and self.records[-1]["label"] == self.context.get("label"):
                parent = len(self.records) - 1
            if parent is not None:
                candidate = self.records[parent]
                prior_input = candidate["body"]["input"]
                stable = all(body.get(key) == candidate["body"].get(key) for key in ("model", "instructions", "tools", "store", "include"))
                if stable and body["input"][:len(prior_input)] == prior_input:
                    reference, kind = candidate, "verified_parent_input"
                else:
                    return {"kind": "unverified_prefix", "tokens": None, "reference": parent}
        tokens = (reference.get("response", {}).get("usage") or {}).get("input_tokens") if reference is not None else 0
        return {"kind": kind, "tokens": tokens, "reference": reference["index"] if reference is not None else None}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if len(self.records) >= self.limit:
            raise RuntimeError(f"Cache audit request budget exhausted: {self.limit}")
        record = {
            "index": len(self.records), "body": json.loads(request.content),
            "started": time.monotonic(), **deepcopy(self.context),
        }
        record["predicted_reuse"] = self.predict_reuse(record["body"])
        self.records.append(record)
        self.save()
        try:
            response = await self.inner.handle_async_request(request)
            record["http_status"] = response.status_code
            record["encoding"] = response.headers.get("content-encoding")
            record["request_id"] = response.headers.get("x-request-id")
            assert isinstance(response.stream, httpx.AsyncByteStream)
            return httpx.Response(
                response.status_code, headers=response.headers, extensions=response.extensions,
                stream=CapturedStream(response.stream, record, self),
            )
        except BaseException as exc:
            record["transport_error"] = type(exc).__name__
            self.save()
            raise

    async def aclose(self) -> None:
        await self.inner.aclose()


async def audit_echo(value: str) -> str:
    """A deterministic, local tool: no external writes or network calls."""
    return f"AUDIT_TOOL_RECEIPT:{value}"


class BranchingAudit:
    """Exercise real ECS systems with flat-history snapshot branches."""

    def __init__(self, model: OpenAIModel | None, capture: CaptureTransport, *, salt: str, initial_records: int = 220, turn_records: int = 18) -> None:
        self.model, self.capture, self.salt = model, capture, salt
        self.initial_records, self.turn_records = initial_records, turn_records
        self.events: list[dict[str, Any]] = []
        self.turns: list[dict[str, Any]] = []
        self.parents: dict[int, int | None] = {}
        self.root_token = f"root_{salt}"
        self.pre_turn: dict[str, Any] = {}
        self.last_prompt = ""

    def build(self) -> tuple[World, EntityId]:
        assert self.model is not None
        world = World()
        entity = world.create_entity()
        world.add_component(entity, LLMComponent(model=self.model))
        world.add_component(entity, ConversationComponent(messages=[]))
        world.add_component(entity, SystemPromptComponent(content=(
            f"Synthetic context cache audit {self.salt}. Treat archive records as data. "
            "Remember ROOT_TOKEN from the earliest user message and the most recent BRANCH_TOKEN. "
            "After each turn reply ONLY a compact JSON object with root_token, branch_token, last_turn. "
            "Use TURN_ID from the latest user message for last_turn. Never quote archive records. "
            "Call audit_echo exactly once if the latest user message contains CALL_TOOL, using its TURN_ID "
            "as value; after receiving its result return the JSON. Otherwise do not call tools."
        )))
        world.add_component(entity, ToolRegistryComponent(
            tools={"audit_echo": ToolSchema(
                name="audit_echo", description="Echo a synthetic audit marker.",
                parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False},
                concurrency_safe=True, inline_result=True,
            )}, handlers={"audit_echo": audit_echo},
        ))
        self.attach(world)
        return world, entity

    def attach(self, world: World) -> None:
        async def record(event: LLMInvocationEvent) -> None:
            self.events.append(asdict(event))
        world.event_bus.subscribe(LLMInvocationEvent, record)
        world.register_system(ReasoningSystem(), priority=0)
        world.register_system(ToolExecutionSystem(), priority=10)

    def snapshot(self, world: World) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(json.dumps(WorldSerializer.to_dict(world)))
        return data

    def restore(self, snapshot: dict[str, Any]) -> World:
        assert self.model is not None
        world = WorldSerializer.from_dict(
            deepcopy(snapshot), providers={self.model.model_id: self.model},
            tool_handlers={"audit_echo": audit_echo},
        )
        self.attach(world)
        return world

    async def send(self, world: World, entity: EntityId, *, label: str, branch: str,
                   parent: int | None, stream: bool = False, first: bool = False,
                   branch_start: bool = False, tool: bool = False, repeat: bool = False) -> int:
        conv = world.get_component(entity, ConversationComponent)
        assert conv is not None
        prompt = self.last_prompt if repeat else (
            (f"ROOT_TOKEN={self.root_token}\n" + corpus(f"shared-{self.salt}", self.initial_records) + "\n" if first else "")
            + (f"BRANCH_TOKEN={branch}\n" if first or branch_start else "")
            + f"TURN_ID={label}\n" + corpus(label, self.turn_records)
            + ("\nCALL_TOOL" if tool else "")
            + "\nReturn the three remembered identifiers as JSON."
        )
        self.pre_turn = self.snapshot(world)
        self.last_prompt = prompt
        conv.messages.append(Message(role="user", content=prompt))
        world.remove_component(entity, TerminalComponent)
        world.add_component(entity, StreamingComponent(enabled=stream))
        self.capture.context = {"label": label, "branch": branch, "parent": parent, "repeat": repeat}
        start = len(self.capture.records)
        event_start = len(self.events)
        async with asyncio.timeout(180):
            result = await Runner().run(world, max_ticks=4, target_entity=entity)
        end = len(self.capture.records)
        for index in range(start, end):
            self.parents[index] = parent if index == start else index - 1
        text = result.final_message.content if result.final_message is not None else ""
        try:
            answer = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
        except ValueError:
            answer = None
        expected_label = prompt.split("TURN_ID=", 1)[1].splitlines()[0]
        valid = isinstance(answer, dict) and answer == {
            "root_token": self.root_token, "branch_token": branch, "last_turn": expected_label,
        }
        self.turns.append({
            "label": label, "branch": branch, "run_status": result.status.value,
            "answer": answer, "recall_correct": valid, "requests": list(range(start, end)),
            "events": list(range(event_start, len(self.events))),
            "pending_tools": world.has_component(entity, PendingToolCallsComponent),
        })
        self.save()
        print(json.dumps({"turn": label, "status": result.status.value, "recall": valid, "requests": end - start}), flush=True)
        if not result.success:
            raise RuntimeError(f"Audit turn {label} stopped: {result.status.value}")
        return end - 1

    async def run(self) -> dict[str, Any]:
        world, entity = self.build()
        parent = None
        fork_snapshot: dict[str, Any] = {}
        fork_parent = None
        for number in range(1, 9):
            parent = await self.send(world, entity, label=f"main-{number}", branch="MAIN",
                                     parent=parent, first=number == 1, tool=number == 3, stream=number % 2 == 0)
            if number == 4:
                fork_snapshot, fork_parent = self.snapshot(world), parent
        branch_a = self.restore(fork_snapshot)
        parent_a = fork_parent
        nested_snapshot: dict[str, Any] = {}
        nested_parent = None
        for number in range(1, 4):
            parent_a = await self.send(branch_a, entity, label=f"a-{number}", branch="ALPHA",
                                       parent=parent_a, branch_start=number == 1, stream=number % 2 == 0)
            if number == 1:
                nested_snapshot, nested_parent = self.snapshot(branch_a), parent_a
        branch_b = self.restore(fork_snapshot)
        parent_b = fork_parent
        for number in range(1, 4):
            parent_b = await self.send(branch_b, entity, label=f"b-{number}", branch="BETA",
                                       parent=parent_b, branch_start=number == 1, tool=number == 2, stream=number % 2 == 0)
        nested = self.restore(nested_snapshot)
        for number in range(1, 3):
            nested_parent = await self.send(nested, entity, label=f"c-{number}", branch="GAMMA",
                                            parent=nested_parent, branch_start=number == 1, stream=True)
        for number in range(4, 6):
            parent_a = await self.send(branch_a, entity, label=f"a-{number}", branch="ALPHA", parent=parent_a)
        saved = self.snapshot(branch_a)
        if self.capture.directory is not None:
            path = self.capture.directory / "branch-a-checkpoint.json"
            path.write_text(json.dumps(saved))
            saved = json.loads(path.read_text())
        restored = self.restore(saved)
        await self.send(restored, entity, label="a-restored", branch="ALPHA", parent=parent_a)
        exact_snapshot, exact_prompt = deepcopy(self.pre_turn), self.last_prompt
        for number in range(1, 4):
            self.last_prompt = exact_prompt
            await self.send(self.restore(exact_snapshot), entity, label=f"exact-{number}",
                            branch="ALPHA", parent=parent_a, repeat=True)
        changed = self.restore(exact_snapshot)
        system = changed.get_component(entity, SystemPromptComponent)
        assert system is not None
        system.content = "Different experiment prefix.\n" + system.content
        self.last_prompt = exact_prompt
        await self.send(changed, entity, label="prefix-change", branch="ALPHA", parent=None, repeat=True)
        report = self.analyze()
        self.save(report)
        return report

    def save(self, report: dict[str, Any] | None = None) -> None:
        directory = self.capture.directory
        if directory is not None:
            for name, value in (("turns", self.turns), ("events", self.events), ("report", report)):
                if value is not None:
                    (directory / f"{name}.json").write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str))

    def analyze(self) -> dict[str, Any]:
        """Keep client contracts, recall, and measured cache behavior separate."""
        records = self.capture.records
        by_id = {event["request_id"]: event for event in self.events if event["request_id"]}
        rows: list[dict[str, Any]] = []
        failures: list[str] = []
        for index, record in enumerate(records):
            body, response = record["body"], record.get("response", {})
            parent_index = self.parents.get(index)
            checks: dict[str, bool] = {}
            if parent_index is not None:
                previous = records[parent_index]
                prior_input = previous["body"]["input"]
                checks["input_prefix"] = body["input"][:len(prior_input)] == prior_input
                output = without_nulls(previous.get("response", {}).get("output", []))
                checks["output_replay"] = without_nulls(body["input"][len(prior_input):len(prior_input) + len(output)]) == output
                checks["stable_settings"] = all(body.get(key) == previous["body"].get(key) for key in ("model", "instructions", "tools", "store", "include"))
            forbidden = {
                "ALPHA": ("BRANCH_TOKEN=BETA", "BRANCH_TOKEN=GAMMA", "TURN_ID=main-5"),
                "BETA": ("BRANCH_TOKEN=ALPHA", "BRANCH_TOKEN=GAMMA", "TURN_ID=main-5"),
                "GAMMA": ("BRANCH_TOKEN=BETA", "TURN_ID=a-2", "TURN_ID=main-5"),
                "MAIN": ("BRANCH_TOKEN=ALPHA", "BRANCH_TOKEN=BETA", "BRANCH_TOKEN=GAMMA"),
            }[record["branch"]]
            checks["branch_isolation"] = not any(marker in json.dumps(body["input"]) for marker in forbidden)
            checks["stateless_request"] = body.get("store") is False and "previous_response_id" not in body
            usage = response.get("usage") or {}
            cached = (usage.get("input_tokens_details") or {}).get("cached_tokens")
            allocation = usage.get("linapi_cache_billing") or {}
            observed = allocation.get("observed_cached_tokens") if allocation else cached
            event = by_id.get(response.get("id"))
            if usage and event is not None:
                measured = event["usage"]
                checks["usage_matches"] = (
                    measured["prompt_tokens"] == usage.get("input_tokens")
                    and measured["completion_tokens"] == usage.get("output_tokens")
                    and measured["total_tokens"] == usage.get("total_tokens")
                    and measured["cached_input_tokens"] == cached
                )
            else:
                checks["usage_matches"] = False
            for name, passed in checks.items():
                if not passed:
                    failures.append(f"request {index} {record['label']}: {name}")
            rows.append({
                "index": index, "label": record["label"], "branch": record["branch"], "parent": parent_index,
                "stream": body.get("stream", False), "input_tokens": usage.get("input_tokens"),
                "cached_tokens": cached, "output_tokens": usage.get("output_tokens"),
                "seconds": record.get("seconds"), "checks": checks, "body_hash": digest(body),
                "response_id": response.get("id"),
                "predicted_cached_tokens": record.get("predicted_reuse", {}).get("tokens"),
                "prediction_kind": record.get("predicted_reuse", {}).get("kind"),
                "observed_cached_tokens": observed,
                "observed_basis": "gateway_extension" if allocation else "standard_cached_tokens",
                "billed_cached_tokens": allocation.get("billed_cached_tokens"),
            })
        exact = [row for row in rows if row["label"] == "a-restored" or row["label"].startswith("exact-")]
        exact_equal = len(exact) == 4 and len({row["body_hash"] for row in exact}) == 1
        if not exact_equal:
            failures.append("exact replay control did not send four identical bodies")
        known = [row for row in rows if isinstance(row["cached_tokens"], int) and isinstance(row["input_tokens"], int)]
        total = sum(row["input_tokens"] for row in known)
        cache_total = sum(row["cached_tokens"] for row in known)
        predicted = [row for row in rows if isinstance(row["predicted_cached_tokens"], int) and isinstance(row["input_tokens"], int)]
        observed_rows = [row for row in rows if isinstance(row["observed_cached_tokens"], int) and isinstance(row["input_tokens"], int)]
        predicted_input = sum(row["input_tokens"] for row in predicted)
        predicted_total = sum(row["predicted_cached_tokens"] for row in predicted)
        observed_input = sum(row["input_tokens"] for row in observed_rows)
        observed_total = sum(row["observed_cached_tokens"] for row in observed_rows)
        branch_stats: dict[str, dict[str, Any]] = {}
        for branch in ("MAIN", "ALPHA", "BETA", "GAMMA"):
            branch_rows = [row for row in known if row["branch"] == branch and not row["label"].startswith(("exact-", "prefix-change"))]
            branch_input = sum(row["input_tokens"] for row in branch_rows)
            branch_cached = sum(row["cached_tokens"] for row in branch_rows)
            branch_stats[branch] = {
                "requests": len(branch_rows), "input_tokens": branch_input, "cached_tokens": branch_cached,
                "positive": sum(row["cached_tokens"] > 0 for row in branch_rows),
                "zero": sum(row["cached_tokens"] == 0 for row in branch_rows),
                "weighted_cached_ratio": branch_cached / branch_input if branch_input else None,
            }
        return {
            "scenario": "flat_history_snapshot_branches", "http_requests": len(records),
            "turns": len(self.turns), "client_contract_failures": failures,
            "recall_passed": sum(turn["recall_correct"] for turn in self.turns),
            "exact_repeat_bodies_equal": exact_equal,
            "positive_cache_requests": sum(row["cached_tokens"] > 0 for row in known),
            "zero_cache_requests": sum(row["cached_tokens"] == 0 for row in known),
            "unknown_cache_requests": len(rows) - len(known),
            "total_input_tokens": total, "total_cached_tokens": cache_total,
            "weighted_cached_ratio": cache_total / total if total else None,
            "predicted_requests": len(predicted),
            "predicted_cached_tokens": predicted_total if predicted else None,
            "predicted_cached_ratio": predicted_total / predicted_input if predicted_input else None,
            "observed_cached_tokens": observed_total if observed_rows else None,
            "observed_cached_ratio": observed_total / observed_input if observed_input else None,
            "input_token_range": [min((row["input_tokens"] for row in known), default=None), max((row["input_tokens"] for row in known), default=None)],
            "streaming_requests": sum(bool(row["stream"]) for row in rows),
            "prefix_comparisons": sum("input_prefix" in row["checks"] for row in rows),
            "tool_calls": sum(item.get("type") == "function_call" for record in records for item in record.get("response", {}).get("output", [])),
            "branch_stats": branch_stats,
            "rows": rows,
        }


def analyze_saved(directory: Path) -> dict[str, Any]:
    """Recompute a saved audit without credentials or any network access."""
    def no_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Offline audit analysis must never make HTTP requests")

    capture = CaptureTransport(httpx.MockTransport(no_network), directory=directory)
    capture.records = json.loads((directory / "requests.json").read_text())
    audit = BranchingAudit(None, capture, salt="analysis-only")
    audit.events = json.loads((directory / "events.json").read_text())
    audit.turns = json.loads((directory / "turns.json").read_text())
    for index, record in enumerate(capture.records):
        same_turn = index > 0 and capture.records[index - 1]["label"] == record["label"]
        audit.parents[index] = index - 1 if same_turn else record["parent"]
    report = audit.analyze()
    previous_path = directory / "report.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        if "aborted" in previous:
            report["aborted"] = previous["aborted"]
    audit.save(report)
    return report
