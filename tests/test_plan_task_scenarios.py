"""Offline scenario coverage extracted from formerly credential-gated files."""

from pathlib import Path
import pytest
from tests.support.plan_task import write_plan
from ecs_agent.core import World
from ecs_agent.phases import bind_phase_graph
from ecs_agent.types import EntityId
from examples.e2e.plan_and_task.scratchbook_adapter import (
    PlanTaskScratchbookAdapter as ArtifactAdapter,
)
from examples.e2e.plan_and_task.controller import PlanController
from examples.e2e.plan_and_task.phase_graph import PLAN_TASK_PHASE_GRAPH

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
   \t- 需要具备伏笔记录表功能
6. 支持大模型校对、审稿、润色建议
7. 支持自动获取读者反馈，筛选分析，由作者选择是否接受

技术建议：
	通过知识图谱 (Knowledge Graph / 关系型数据库)：用于存储确定性的实体与关系。
	- 节点 (Nodes)：人物、法宝、功法、门派、地理位置。
	- 边 (Edges)：师徒关系、敌对关系、所属关系、地理毗邻。
	向量数据库 (Vector Database)：用于存储描述性文本和历史剧情。
	- 人物的性格侧写、功法的具体运行路线、某场经典战役的详细过程。
	- 技术选型：Milvus 或 Qdrant（支持高并发和复杂的元数据过滤）。"""


async def _bound_world() -> tuple[World, EntityId]:
    """Minimal world with the plan-task phase graph bound at IDLE.

    PlanController transitions run through ecs_agent.phases, so tests driving
    the controller directly need a world with a bound PhaseComponent.
    """
    world = World()
    entity_id = world.create_entity()
    await bind_phase_graph(world, entity_id, PLAN_TASK_PHASE_GRAPH, agent_key="main")
    return world, entity_id


@pytest.mark.asyncio
async def test_live_plan_controller_starts_plan_interview(tmp_path: Path) -> None:
    """Verify handle_plan_start creates artifacts and transitions to DRAFT_INTERVIEW."""
    workflow_id = "live-test-workflow-start"
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    world, entity_id = await _bound_world()
    controller = PlanController(world, entity_id)

    state = await controller.handle_plan_start(adapter, _WRITING_SOFTWARE_DESCRIPTION)

    assert state.phase == "DRAFT_INTERVIEW"
    assert state.workflow_id == workflow_id
    assert (adapter.plan_dir / "draft.md").exists()

    reloaded = adapter.read_state()
    assert reloaded.phase == "DRAFT_INTERVIEW"
    assert reloaded.workflow_id == workflow_id


@pytest.mark.asyncio
async def test_live_plan_controller_completes_review_cycle(tmp_path: Path) -> None:
    """Verify review verdicts allow transition to finalized state."""
    workflow_id = "live-test-workflow-review"
    adapter = ArtifactAdapter(base_dir=tmp_path, workflow_id=workflow_id)
    world, entity_id = await _bound_world()
    controller = PlanController(world, entity_id)

    state = await controller.handle_plan_start(adapter, "Review cycle test")

    # Full review cycle: advisor → qa → write_plan_completed → plan_qa
    state = await controller.handle_advisor_review(
        state, adapter, "approved", notes="Advisor approved"
    )
    state = await controller.handle_qa_review(
        state, adapter, "approved", notes="QA approved"
    )
    # handle_qa_review auto-transitions to WRITE_PLAN when approved
    write_plan(adapter.plan_dir, workflow_id)
    state = await controller.handle_write_plan_completed(state, adapter)
    state = await controller.handle_plan_qa_review(
        state, adapter, "approved", notes="Plan QA approved"
    )

    state = await controller.handle_plan_finalize(state, adapter)

    assert state.phase == "TASK_READY"
    assert (adapter.plan_dir / "workflow_plan.md").exists()
