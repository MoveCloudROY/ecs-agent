"""Exercise long snapshot branches through real ECS systems and the OpenAI SDK."""

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_model import OpenAIModel
from tests.cache_audit.branching import BranchingAudit, CaptureTransport, analyze_saved


class ChunkStream(httpx.AsyncByteStream):
    """Split fixture SSE frames across transport chunks like a real network."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for offset in range(0, len(self.data), 79):
            yield self.data[offset:offset + 79]


class ScriptedResponses(httpx.MockTransport):
    """Respond from the received history; usage is synthetic test data only."""

    def __init__(self, *, cache_mode: str = "mixed") -> None:
        self.index = 0
        self.cache_mode = cache_mode
        super().__init__(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.index += 1
        texts = [
            part["text"] for item in body["input"] if item.get("role") == "user"
            for part in item["content"] if part.get("type") == "input_text"
        ]
        combined = "\n".join(texts)
        label = re.findall(r"TURN_ID=([^\n]+)", combined)[-1]
        root = re.findall(r"ROOT_TOKEN=([^\n]+)", combined)[0]
        branch = re.findall(r"BRANCH_TOKEN=([^\n]+)", combined)[-1]
        output: list[dict[str, Any]] = [{
            "type": "reasoning", "id": f"rs_{self.index}", "summary": [],
            "encrypted_content": f"opaque_fixture_{self.index}",
        }]
        if "CALL_TOOL" in texts[-1] and body["input"][-1].get("type") != "function_call_output":
            output.append({
                "type": "function_call", "id": f"fc_{self.index}", "call_id": f"call_{self.index}",
                "name": "audit_echo", "arguments": json.dumps({"value": label}), "status": "completed",
            })
        else:
            output.append({
                "type": "message", "id": f"msg_{self.index}", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": json.dumps({
                    "root_token": root, "branch_token": branch, "last_turn": label,
                }), "annotations": []}],
            })
        input_tokens = len(request.content) // 4
        details = {} if self.cache_mode == "missing" else {
            "cached_tokens": 0 if self.cache_mode == "zero" or self.index % 4 == 1 else input_tokens - 256,
        }
        response = {
            "object": "response", "id": f"resp_{self.index}", "status": "completed", "store": False,
            "model": body["model"], "output": output,
            "usage": {"input_tokens": input_tokens, "output_tokens": 30, "total_tokens": input_tokens + 30,
                      "input_tokens_details": details, "output_tokens_details": {"reasoning_tokens": 5}},
        }
        if not body.get("stream"):
            return httpx.Response(200, json=response)
        events: list[dict[str, Any]] = []
        for index, item in enumerate(output):
            events.append({"type": "response.output_item.done", "output_index": index, "item": item})
        events.append({"type": "response.completed", "response": response})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=ChunkStream(
            "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode(),
        ))


@pytest.mark.parametrize("cache_mode", ["mixed", "zero", "missing"])
async def test_long_context_snapshot_branches_preserve_wire_history_and_usage(cache_mode: str, tmp_path: Path) -> None:
    capture = CaptureTransport(ScriptedResponses(cache_mode=cache_mode), directory=tmp_path)
    model = OpenAIModel(config=ProviderConfig(
        provider_id="branch-audit", base_url="https://audit.invalid/v1", api_key="fixture",
        api_format=ApiFormat.OPENAI_RESPONSES, enable_store=False,
    ), model="test-model")
    await model._client.aclose()
    model._client = httpx.AsyncClient(transport=capture)
    audit = BranchingAudit(model, capture, salt="offline")
    try:
        report = await audit.run()
        assert report["http_requests"] == 25
        assert report["turns"] == 23
        assert report["client_contract_failures"] == []
        assert report["recall_passed"] == 23
        assert all(record.get("closed") for record in capture.records)
        assert sum(record["body"].get("stream", False) for record in capture.records) >= 8
        assert len(capture.records[0]["body"]["input"][0]["content"][0]["text"]) > 30000
        assert report["exact_repeat_bodies_equal"]
        assert report["predicted_requests"] == 25
        assert capture.records[0]["predicted_reuse"]["tokens"] == 0
        assert capture.records[-1]["predicted_reuse"]["tokens"] == 0
        # Sibling branches predict from their actual shared ancestor; cache
        # fixture values (including zero/missing) cannot influence the baseline.
        assert capture.records[12]["predicted_reuse"]["reference"] == 4
        assert capture.records[21]["predicted_reuse"]["reference"] == 20
        assert capture.records[21]["predicted_reuse"]["tokens"] == capture.records[20]["response"]["usage"]["input_tokens"]
        assert (tmp_path / "branch-a-checkpoint.json").is_file()
        assert analyze_saved(tmp_path) == report
        if cache_mode == "zero":
            assert report["zero_cache_requests"] == 25
            assert report["weighted_cached_ratio"] == 0
        elif cache_mode == "missing":
            assert report["unknown_cache_requests"] == 25
            assert report["weighted_cached_ratio"] is None
        else:
            assert report["positive_cache_requests"] > 0
            assert report["zero_cache_requests"] > 0
        changed = capture.records[-1]["body"]
        original = capture.records[-2]["body"]
        assert changed["input"] == original["input"]
        assert changed["instructions"] != original["instructions"]
        # The audit must detect an actual sibling-history leak and replay damage.
        capture.records[12]["body"]["input"].append({
            "role": "user", "content": [{"type": "input_text", "text": "BRANCH_TOKEN=ALPHA"}],
        })
        capture.records[2]["body"]["input"][0]["content"][0]["text"] += "corrupted-prefix"
        damaged = audit.analyze()["client_contract_failures"]
        assert any("branch_isolation" in failure for failure in damaged)
        assert any("input_prefix" in failure for failure in damaged)
    finally:
        await model.aclose()
