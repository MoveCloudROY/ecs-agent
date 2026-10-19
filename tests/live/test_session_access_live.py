"""Live coder/reviewer evidence retrieval using persistent session resources."""

import asyncio
from pathlib import Path
import sys
from uuid import uuid4

import pytest

from ecs_agent.components import ConversationComponent, SessionRefComponent, ToolRegistryComponent
from ecs_agent.core import World
from ecs_agent.history import replace_conversation_history
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.registry import ProviderRegistry, get_model
from ecs_agent.scratchbook.artifact_registry import ArtifactRegistry
from ecs_agent.sessions import SessionService, SQLiteSessionStore, install_session_recording
from ecs_agent.systems.subagent.child_world import ChildWorldBuilder
from ecs_agent.systems.subagent.delegation import DelegationExecutor
from ecs_agent.tools.builtins import BuiltinToolsSkill
from ecs_agent.types import InheritancePolicy, Message, SubagentConfig, ToolSchema

from tests.live.api_format import live_openai_base_url, live_openai_model, resolve_live_api_format


async def test_live_reviewer_reads_compacted_coder_tool_evidence(live_api_key, tmp_path):
    api_format = resolve_live_api_format()
    if api_format is not ApiFormat.OPENAI_RESPONSES:
        pytest.skip("This integration exercises the Responses API; set LLM_API_FORMAT=responses")
    providers = ProviderRegistry.from_dict({"session-live": {
        "base_url": live_openai_base_url(api_format), "api_format": api_format.value,
    }})
    model = get_model(f"session-live/{live_openai_model()}", registry=providers, api_key=live_api_key)
    service = SessionService(SQLiteSessionStore(tmp_path / "sessions.sqlite3"), artifact_registry=ArtifactRegistry(tmp_path))
    root = World()
    entity = root.create_entity()
    root.add_component(entity, ConversationComponent(messages=[]))
    read_schema, read_handler = BuiltinToolsSkill().bind_workspace(str(tmp_path)).tools()["read_file"]
    marker = "CHECK_RUN=" + uuid4().hex
    executions = []

    async def run_validation():
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pytest", "tests/test_session_store.py", "-q",
            cwd=Path(__file__).resolve().parents[2],
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await asyncio.wait_for(process.communicate(), timeout=30)
        assert process.returncode == 0
        executions.append(1)
        # A long single line forces multiple bounded resource reads after the
        # coder's working context is replaced. The evidence is only at the end.
        return "TEST_OUTPUT\n" + "validation-log " * 2200 + "\n" + output.decode() + "\n" + marker

    root.add_component(entity, ToolRegistryComponent(tools={
        "read_file": read_schema,
        "run_validation": ToolSchema(name="run_validation", description="Run the local session persistence tests and return the complete validation log.", parameters={"type": "object", "properties": {}, "required": []}),
    }, handlers={"read_file": read_handler, "run_validation": run_validation}))
    install_session_recording(root, entity, service, workspace_id="live-test", agent_name="main")
    builder = ChildWorldBuilder()
    executor = DelegationExecutor()
    try:
        coder_config = SubagentConfig(name="coder", model=model, max_ticks=8, inheritance_policy=InheritancePolicy(inherit_tools=["run_validation"]))
        coder, coder_id = builder.assemble_child_world(root, entity, coder_config)
        coder_outcome = await executor.run_delegation(coder, coder_id, "Call run_validation exactly once, then report whether the tests passed. Do not invent evidence.", coder_config)
        assert coder_outcome.success
        assert executions == [1]
        replace_conversation_history(coder, coder_id, [Message(role="user", content="The coder context has been compacted. Retrieve original evidence from session records.")], reason="compaction_summary")
        reviewer_config = SubagentConfig(name="reviewer", model=model, max_ticks=16, inheritance_policy=InheritancePolicy(inherit_tools=["read_file", "sessions_list", "sessions_search"]))
        reviewer, reviewer_id = builder.assemble_child_world(root, entity, reviewer_config)
        outcome = await executor.run_delegation(
            reviewer, reviewer_id,
            "Review the coder's actual test evidence. Discover the coder session, search TEST_OUTPUT, and read the matching record with read_file. Follow its tool_result_uri and use read_file with limit=500 plus next_offset until has_more=false. Read all pages of the full tool result. Report the test count, the exact CHECK_RUN marker at the end, and cite the source session_id and record_id. Do not run tests yourself.",
            reviewer_config,
        )
        assert outcome.success
        assert marker in outcome.as_tuple()[0]
        coder_ref = coder.get_component(coder_id, SessionRefComponent)
        assert coder_ref.session_id in outcome.as_tuple()[0]
        tool_record = next(record for record in service.store.records(coder_ref.session_id) if record.payload.get("role") == "tool")
        assert tool_record.record_id in outcome.as_tuple()[0]
        reviewer_ref = reviewer.get_component(reviewer_id, SessionRefComponent)
        calls = [record.payload for record in service.store.records(reviewer_ref.session_id) if record.kind == "tool_started"]
        assert any(call["tool_name"] == "sessions_list" for call in calls)
        assert any(call["tool_name"] == "sessions_search" for call in calls)
        full_reads = [call for call in calls if call["tool_name"] == "read_file" and "/tool-result?" in call["arguments"].get("file_path", "")]
        assert len(full_reads) >= 3
        assert any(int(call["arguments"].get("offset", 1)) > 1 for call in full_reads)
    finally:
        await model.aclose()
        service.close()
