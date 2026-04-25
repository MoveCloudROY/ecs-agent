"""Live acceptance tests for the plan-and-task E2E example.

These tests require LLM_API_KEY to be set and will skip otherwise.
They use the DashScope-compatible model configuration.

Anthropic tests additionally require LLM_API_FORMAT=anthropic_messages plus
compatible LLM_BASE_URL / LLM_MODEL (e.g. cc2.caaa.tech / kimi-for-coding).
"""

import os
import re
from pathlib import Path

import pytest

from ecs_agent.providers import OpenAIModel
from ecs_agent.providers.claude_model import ClaudeModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.protocol import LLMModel
from ecs_agent.types import CompletionResult, Message
from examples.e2e.plan_and_task.scratchbook_adapter import (
    PlanTaskScratchbookAdapter as ArtifactAdapter,
)
from examples.e2e.plan_and_task.controller import PlanController
from examples.e2e.plan_and_task.runtime import derive_workflow_id_from_llm
from examples.e2e.plan_and_task.task_exec import TaskExec


_WRITING_SOFTWARE_DESCRIPTION = """我希望开发一份辅助写作软件，目前实现的前端位于 @frontend/，请为其在 @backend/ 中补全后端，并与前端进行对接
需求如下：
1. 支持网文、长篇小说、剧本创作
2. 支持创意头脑风暴，通过多个LLM对话获取潜在的设定集，并记录进备选库
3. 以每部作品为单位管理
4. 每部作品具备一份设定集，包括
	- 世界观构建：完善设定，例如力量体系（等级划分）、地理环境、社会规则
	- 大纲、时间线设定：规划好每个小节的具体内容，大高潮（爆点）的位置，通常每 10-20 章需要一个小高潮。
	- 角色卡片生成与管理：主角的核心动机（他想要什么？）、性格缺陷、记忆点
   均支持通过LLM自动生成，审核
5. 支持大模型基于上下文和设定集自动按章节生成，其中：
	- 支持前 3 章的精细化生成
	- 支持前 30-50 章的快速试验，以确定是否合格
   为实现这一点：
   	- 需要具备伏笔记录表功能
6. 支持大模型校对、审稿、润色建议
7. 支持自动获取读者反馈，筛选分析，由作者选择是否接受

技术建议：
	通过知识图谱 (Knowledge Graph / 关系型数据库)：用于存储确定性的实体与关系。
	- 节点 (Nodes)：人物、法宝、功法、门派、地理位置。
	- 边 (Edges)：师徒关系、敌对关系、所属关系、地理毗邻。
	向量数据库 (Vector Database)：用于存储描述性文本和历史剧情。
	- 人物的性格侧写、功法的具体运行路线、某场经典战役的详细过程。
	- 技术选型：Milvus 或 Qdrant（支持高并发和复杂的元数据过滤）。"""


@pytest.fixture
def live_model(live_api_key: str) -> OpenAIModel:
    """Create a real OpenAIModel for live tests."""
    base_url = os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.getenv("LLM_MODEL", "qwen3.5-flash")
    config = ProviderConfig(
        provider_id="openai",
        base_url=base_url,
        api_key=live_api_key,
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )
    return OpenAIModel(config=config, model=model)


@pytest.mark.asyncio
async def test_live_plan_controller_starts_plan_interview(
    live_model: OpenAIModel, tmp_path: Path
) -> None:
    """Verify handle_plan_start creates artifacts and transitions to DRAFT_INTERVIEW."""
    workflow_id = "live-test-workflow-start"
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    controller = PlanController()

    state = controller.handle_plan_start(adapter, _WRITING_SOFTWARE_DESCRIPTION)

    assert state.phase == "DRAFT_INTERVIEW"
    assert state.workflow_id == workflow_id
    assert (adapter.plan_dir / "draft.md").exists()

    reloaded = adapter.read_state()
    assert reloaded.phase == "DRAFT_INTERVIEW"
    assert reloaded.workflow_id == workflow_id


@pytest.mark.asyncio
async def test_live_plan_controller_completes_review_cycle(
    live_model: OpenAIModel, tmp_path: Path
) -> None:
    """Verify review verdicts allow transition to finalized state."""
    workflow_id = "live-test-workflow-review"
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    controller = PlanController()

    state = controller.handle_plan_start(adapter, "Review cycle test")

    # Full review cycle: advisor → qa → write_plan_completed → plan_qa
    state = controller.handle_advisor_review(
        state, adapter, "approved", notes="Advisor approved"
    )
    state = controller.handle_qa_review(state, adapter, "approved", notes="QA approved")
    # handle_qa_review auto-transitions to WRITE_PLAN when approved
    state = controller.handle_write_plan_completed(state, adapter)
    state = controller.handle_plan_qa_review(state, adapter, "approved", notes="Plan QA approved")

    state = controller.handle_plan_finalize(state, adapter)

    assert state.phase == "TASK_READY"
    assert (adapter.plan_dir / "workflow_plan.md").exists()


@pytest.mark.asyncio
async def test_live_task_exec_loads_finalized_plan(
    live_model: OpenAIModel, tmp_path: Path
) -> None:
    """Verify TaskExec can load a finalized plan and populate TaskRecord list."""
    workflow_id = "live-test-workflow-exec"
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    controller = PlanController()

    state = controller.handle_plan_start(adapter, "Task exec test")
    state = controller.handle_advisor_review(state, adapter, "approved")
    state = controller.handle_qa_review(state, adapter, "approved")
    state = controller.handle_write_plan_completed(state, adapter)
    state = controller.handle_plan_qa_review(state, adapter, "approved")
    state = controller.handle_plan_finalize(state, adapter)

    task_exec = TaskExec(state=state)
    plan = task_exec.load_plan(adapter)
    tasks = task_exec.build_todo_queue(plan)

    assert len(tasks) > 0
    assert tasks[0].task_id == "task-001"
    assert tasks[0].status == "pending"

    # Verify /task:start path: initialize_task_queue transitions state and persists queue
    next_state = task_exec.initialize_task_queue(state, adapter)
    assert next_state.phase == "TASK_RUNNING"
    assert next_state.current_task_id == "task-001"
    assert (adapter.state_dir / "task_queue.json").exists()

    # Use live model to generate task output — proves LLM API is reachable
    result = await live_model.complete(
        [Message(role="user", content="Confirm: this plan is ready to execute.")]
    )
    assert isinstance(result, CompletionResult)
    assert result.message.content.strip()  # Non-empty string assertion

    # Prove full delegated task lifecycle: dispatch → completion
    dispatched_state = task_exec.record_subagent_dispatch(
        next_state, adapter, "task-001", "live-session-001"
    )
    assert dispatched_state.phase == "TASK_RUNNING"

    completed_state = task_exec.record_task_completion(
        dispatched_state,
        adapter,
        "task-001",
        evidence_refs=None,
        summary=result.message.content.strip(),
    )
    assert completed_state.phase == "TASK_COMPLETED"
    assert "task-001" in completed_state.completed_task_ids


@pytest.mark.asyncio
async def test_live_derive_workflow_id_from_llm_returns_valid_slug(
    live_model: OpenAIModel,
) -> None:
    slug = await derive_workflow_id_from_llm(
        _WRITING_SOFTWARE_DESCRIPTION, live_model
    )

    assert re.match(r"^[a-z][a-z0-9-]*$", slug), (
        f"Expected a valid slug (lowercase letters, digits, hyphens), got: {slug!r}"
    )
    assert len(slug) <= 50, f"Slug too long: {slug!r}"
    assert len(slug) >= 3, f"Slug too short: {slug!r}"


@pytest.mark.asyncio
async def test_live_controller_advisor_retry_loop_revise_then_approved(
    live_model: OpenAIModel, tmp_path: Path
) -> None:
    workflow_id = "live-test-advisor-retry"
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    controller = PlanController()

    state = controller.handle_plan_start(adapter, "Advisor retry loop test")

    state = controller.handle_advisor_review(
        state, adapter, "revise", notes="Draft needs more detail in scope section."
    )
    assert state.phase == "DRAFT_ADVISOR_REVIEW"
    assert state.phase != "DRAFT_QA_REVIEW"

    state = controller.handle_advisor_review(
        state, adapter, "approved", notes="Looks good now."
    )
    assert state.phase == "DRAFT_ADVISOR_REVIEW"

    missing = controller._missing_approved_reviews(state.review_verdicts)
    assert "DRAFT_ADVISOR_REVIEW" not in missing

    # upsert_verdict replaces (not accumulates) — only the final "approved" verdict remains
    advisor_verdicts = [
        v for v in state.review_verdicts if v.phase == "DRAFT_ADVISOR_REVIEW"
    ]
    assert len(advisor_verdicts) == 1
    assert advisor_verdicts[0].verdict == "approved"

    result = await live_model.complete(
        [
            Message(
                role="user",
                content="Confirm: advisor retry loop (revise → approved) is correctly handled.",
            )
        ]
    )
    assert isinstance(result, CompletionResult)
    assert result.message.content.strip()


_ANTHROPIC_SKIP = pytest.mark.skipif(
    os.getenv("LLM_API_FORMAT") != ApiFormat.ANTHROPIC_MESSAGES,
    reason="Set LLM_API_FORMAT=anthropic_messages to run Anthropic live tests",
)


@pytest.fixture
def anthropic_model(live_api_key: str) -> ClaudeModel:
    base_url = os.getenv("LLM_BASE_URL", "https://api.anthropic.com")
    model = os.getenv("LLM_MODEL", "claude-3-5-haiku-20241022")
    config = ProviderConfig(
        provider_id="anthropic",
        base_url=base_url,
        api_key=live_api_key,
        api_format=ApiFormat.ANTHROPIC_MESSAGES,
    )
    return ClaudeModel(config=config, model=model)


@_ANTHROPIC_SKIP
@pytest.mark.asyncio
async def test_anthropic_model_completes_simple_message(
    anthropic_model: ClaudeModel,
) -> None:
    result = await anthropic_model.complete(
        [Message(role="user", content="Reply with exactly: pong")]
    )
    assert isinstance(result, CompletionResult)
    assert result.message.content.strip()


@_ANTHROPIC_SKIP
@pytest.mark.asyncio
async def test_anthropic_derive_workflow_id_returns_valid_slug(
    anthropic_model: ClaudeModel,
) -> None:
    slug = await derive_workflow_id_from_llm("Build a simple todo list app", anthropic_model)
    assert re.match(r"^[a-z][a-z0-9-]*$", slug), (
        f"Expected a valid slug, got: {slug!r}"
    )
    assert 3 <= len(slug) <= 50


@_ANTHROPIC_SKIP
@pytest.mark.asyncio
async def test_anthropic_plan_controller_starts_plan_interview(
    anthropic_model: ClaudeModel, tmp_path: Path
) -> None:
    workflow_id = "live-anthropic-test-start"
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    controller = PlanController()

    state = controller.handle_plan_start(adapter, "Build a simple todo list app")

    assert state.phase == "DRAFT_INTERVIEW"
    assert state.workflow_id == workflow_id
    assert (adapter.plan_dir / "draft.md").exists()

    result = await anthropic_model.complete(
        [Message(role="user", content="Confirm: plan interview started correctly.")]
    )
    assert isinstance(result, CompletionResult)
    assert result.message.content.strip()
