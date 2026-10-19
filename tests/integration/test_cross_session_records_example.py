"""Example acceptance requires retrieved evidence, including in model-driven mode."""

import json

from ecs_agent.providers import FakeModel
from ecs_agent.types import CompletionResult, Message, ToolCall

async def test_coder_reviewer_example_reads_compacted_evidence(tmp_path):
    from examples.e2e.cross_session_records import run_demo

    result = await run_demo(tmp_path)
    assert result["pages_read"] == 1
    assert result["evidence_complete"]
    assert result["session_id"].startswith("sess_")
    assert result["record_id"].startswith("rec_")
    assert "2 checks passed" in result["report"]
    assert (tmp_path / "sessions.sqlite3").is_file()
    assert result["accepted"]
    assert result["search_scoped"]
    assert result["search_self_hits"] == 0
    assert result["duplicate_page_reads"] == 0
    assert result["tool_call_counts"]["coder"] == {"validate": 1}
    trace = json.loads((tmp_path / "interaction.json").read_text())
    assert any(item["agent"] == "coder" and item.get("role") == "tool" for item in trace)
    assert any(item["agent"] == "reviewer" and item.get("tool_calls") for item in trace)
    assert any(item["kind"] == "history_replaced" and item["details"]["reason"] == "compaction_summary" for item in trace)
    assert (tmp_path / "interaction.txt").is_file()
    validation = next(item for item in trace if item["agent"] == "coder" and item.get("role") == "tool")
    assert validation["tool_name"] == "validate"
    assert validation["content_chars"] < 1000
    assert "validation-log" not in validation["content"]
    assert "PASS arithmetic" in validation["content"]
    assert "PASS utf8_roundtrip" in validation["content"]
    assert "exit_code=0" in validation["content"]
    display = (tmp_path / "interaction.txt").read_text()
    assert "CALL validate" in display
    assert "RESULT validate" in display


async def test_explicit_stress_mode_preserves_lossless_multi_page_coverage(tmp_path):
    from examples.e2e.cross_session_records import run_demo

    result = await run_demo(tmp_path, stress_output=True)
    assert result["accepted"]
    assert result["stress_output"]
    assert result["pages_read"] >= 3
    assert result["read_offsets"][0] == 1
    assert result["duplicate_page_reads"] == 0
    trace = json.loads((tmp_path / "interaction.json").read_text())
    validation = next(item for item in trace if item["agent"] == "coder" and item.get("role") == "tool")
    assert "PAGING_STRESS_FIXTURE" in validation["content"]
    assert "validation-log" not in validation["content"]


async def test_unscoped_search_is_detected_even_when_final_evidence_is_correct(tmp_path, monkeypatch):
    from examples.e2e import cross_session_records as demo

    original_call = demo._call

    async def unscoped_call(world, entity, name, arguments):
        if name == "sessions_search":
            arguments = {key: value for key, value in arguments.items() if key != "session_id"}
        return await original_call(world, entity, name, arguments)

    monkeypatch.setattr(demo, "_call", unscoped_call)
    result = await demo.run_demo(tmp_path)
    assert result["evidence_complete"]
    assert result["report_cites_source"]
    assert not result["search_scoped"]
    assert result["search_self_hits"] > 0
    assert not result["accepted"]


async def test_duplicate_page_reads_are_counted_and_rejected(tmp_path, monkeypatch):
    from examples.e2e import cross_session_records as demo

    original_call = demo._call

    async def duplicate_read(world, entity, name, arguments):
        if name == "read_file" and "/tool-result?" in arguments.get("file_path", ""):
            await original_call(world, entity, name, arguments)
        return await original_call(world, entity, name, arguments)

    monkeypatch.setattr(demo, "_call", duplicate_read)
    result = await demo.run_demo(tmp_path)
    assert result["evidence_complete"]
    assert result["duplicate_page_reads"] == 1
    assert not result["accepted"]


async def test_model_claim_without_reading_evidence_is_rejected(tmp_path):
    from examples.e2e.cross_session_records import run_demo

    model = FakeModel([
        CompletionResult(message=Message(role="assistant", content="", tool_calls=[
            ToolCall(id="validate_once", name="validate", arguments={}),
        ])),
        CompletionResult(message=Message(role="assistant", content="2 checks passed")),
        CompletionResult(message=Message(role="assistant", content="2 checks passed; trust me")),
    ])
    result = await run_demo(tmp_path, model=model)

    assert result["mode"] == "llm"
    assert result["model"] == "fake"
    assert result["validation_runs"] == 1
    assert result["pages_read"] == 0
    assert not result["evidence_complete"]
    assert not result["accepted"]
    trace = json.loads((tmp_path / "interaction.json").read_text())
    assert trace[-1]["content"] == "2 checks passed; trust me"
