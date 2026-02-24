from __future__ import annotations

import math

import pytest

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    PlanComponent,
    PlanSearchComponent,
)
from ecs_agent.core import World
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.tree_search import TreeNode, TreeSearchSystem
from ecs_agent.types import CompletionResult, MCTSNodeScoredEvent, Message


class RecordingFakeProvider(FakeProvider):
    def __init__(self, responses: list[CompletionResult]) -> None:
        super().__init__(responses=responses)
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        tools: list[object] | None = None,
        stream: bool = False,
        response_format: dict[str, object] | None = None,
    ) -> CompletionResult:
        _ = tools
        _ = stream
        _ = response_format
        self.calls.append(list(messages))
        result = await super().complete(messages)
        assert isinstance(result, CompletionResult)
        return result


def _make_world(
    provider: RecordingFakeProvider,
    plan_search: PlanSearchComponent | None,
    include_plan: bool = False,
) -> tuple[World, int]:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, LLMComponent(provider=provider, model="fake"))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Solve this")]),
    )
    if plan_search is not None:
        world.add_component(entity_id, plan_search)
    if include_plan:
        world.add_component(entity_id, PlanComponent(steps=["existing plan"]))
    return world, int(entity_id)


def test_ucb1_calculation_matches_expected_value() -> None:
    system = TreeSearchSystem()
    node = TreeNode(id=1, parent_id=0, action="a", score=5.0, visits=10)
    value = system._ucb1(node=node, parent_visits=100, exploration_weight=1.414)
    expected = 0.5 + 1.414 * math.sqrt(math.log(100) / 10)
    assert value == pytest.approx(expected, rel=1e-9)


def test_select_phase_chooses_highest_ucb1_child() -> None:
    system = TreeSearchSystem()
    nodes = {
        0: TreeNode(id=0, parent_id=None, action="root", visits=30, children=[1, 2]),
        1: TreeNode(id=1, parent_id=0, action="safe", score=9.0, visits=10),
        2: TreeNode(id=2, parent_id=0, action="explore", score=4.0, visits=2),
    }

    selected = system._select_leaf(nodes=nodes, root_id=0, exploration_weight=1.414)
    assert selected == 2


@pytest.mark.asyncio
async def test_expand_phase_limits_children_to_max_branching() -> None:
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(
                message=Message(
                    role="assistant",
                    content="action one\naction two\naction three\naction four",
                )
            ),
            CompletionResult(message=Message(role="assistant", content="0.5")),
        ]
    )
    world, entity_id = _make_world(
        provider,
        PlanSearchComponent(max_depth=3, max_branching=2, exploration_weight=1.414),
    )
    system = TreeSearchSystem()

    await system.process(world)

    tree = system._nodes_by_entity[entity_id]
    root = tree[0]
    assert len(root.children) == 2


@pytest.mark.asyncio
async def test_simulate_phase_clamps_score_to_zero_one_range() -> None:
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="1.7")),
        ]
    )
    system = TreeSearchSystem()
    world, entity_id = _make_world(provider, PlanSearchComponent())
    nodes = {
        0: TreeNode(id=0, parent_id=None, action="root", visits=1, children=[1]),
        1: TreeNode(id=1, parent_id=0, action="candidate", visits=0),
    }

    score = await system._simulate(
        world=world,
        entity_id=entity_id,
        nodes=nodes,
        node_id=1,
        conversation=world.get_component(entity_id, ConversationComponent),
        llm_component=world.get_component(entity_id, LLMComponent),
    )
    assert score == 1.0


def test_backpropagate_updates_scores_and_visits_to_root() -> None:
    system = TreeSearchSystem()
    nodes = {
        0: TreeNode(id=0, parent_id=None, action="root"),
        1: TreeNode(id=1, parent_id=0, action="a"),
        2: TreeNode(id=2, parent_id=1, action="b"),
    }

    system._backpropagate(nodes=nodes, node_id=2, score=0.8)

    assert nodes[2].visits == 1
    assert nodes[1].visits == 1
    assert nodes[0].visits == 1
    assert nodes[2].score == pytest.approx(0.8)
    assert nodes[1].score == pytest.approx(0.8)
    assert nodes[0].score == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_depth_limit_stops_search_and_sets_best_plan() -> None:
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="first\nsecond")
            ),
            CompletionResult(message=Message(role="assistant", content="0.9")),
        ]
    )
    world, entity_id = _make_world(
        provider,
        PlanSearchComponent(max_depth=1, max_branching=3, exploration_weight=1.414),
    )
    system = TreeSearchSystem()

    await system.process(world)

    plan_search = world.get_component(entity_id, PlanSearchComponent)
    assert plan_search is not None
    assert plan_search.search_active is False
    assert len(plan_search.best_plan) == 1


@pytest.mark.asyncio
async def test_branching_limit_enforced_for_node_children() -> None:
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="a\nb\nc\nd\ne")
            ),
            CompletionResult(message=Message(role="assistant", content="0.5")),
        ]
    )
    world, entity_id = _make_world(
        provider,
        PlanSearchComponent(max_depth=2, max_branching=3, exploration_weight=1.414),
    )
    system = TreeSearchSystem()

    await system.process(world)

    root = system._nodes_by_entity[entity_id][0]
    assert len(root.children) <= 3


@pytest.mark.asyncio
async def test_mutual_exclusion_skips_entity_with_existing_plan_component() -> None:
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="should not be used")
            ),
        ]
    )
    world, entity_id = _make_world(
        provider,
        PlanSearchComponent(max_depth=2, max_branching=2, exploration_weight=1.414),
        include_plan=True,
    )
    system = TreeSearchSystem()

    await system.process(world)

    plan_search = world.get_component(entity_id, PlanSearchComponent)
    assert plan_search is not None
    assert plan_search.best_plan == []
    assert provider.calls == []


@pytest.mark.asyncio
async def test_entity_without_plan_search_component_is_skipped() -> None:
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(message=Message(role="assistant", content="unused")),
        ]
    )
    world, _ = _make_world(provider, None)
    system = TreeSearchSystem()

    await system.process(world)

    assert provider.calls == []


def test_extract_best_path_returns_highest_scoring_leaf_path() -> None:
    system = TreeSearchSystem()
    nodes = {
        0: TreeNode(id=0, parent_id=None, action="root", children=[1, 2]),
        1: TreeNode(
            id=1, parent_id=0, action="left", score=1.0, visits=2, children=[3]
        ),
        2: TreeNode(
            id=2, parent_id=0, action="right", score=1.8, visits=2, children=[4]
        ),
        3: TreeNode(id=3, parent_id=1, action="left-leaf", score=0.1, visits=1),
        4: TreeNode(id=4, parent_id=2, action="right-leaf", score=0.9, visits=1),
    }

    path = system._extract_best_path(nodes=nodes, root_id=0)
    assert path == ["right", "right-leaf"]


@pytest.mark.asyncio
async def test_integration_populates_best_plan_and_publishes_scored_event() -> None:
    provider = RecordingFakeProvider(
        responses=[
            CompletionResult(
                message=Message(role="assistant", content="collect data\nanswer")
            ),
            CompletionResult(message=Message(role="assistant", content="0.75")),
        ]
    )
    world, entity_id = _make_world(
        provider,
        PlanSearchComponent(max_depth=1, max_branching=2, exploration_weight=1.414),
    )
    seen: list[MCTSNodeScoredEvent] = []

    async def on_scored(event: MCTSNodeScoredEvent) -> None:
        seen.append(event)

    world.event_bus.subscribe(MCTSNodeScoredEvent, on_scored)
    system = TreeSearchSystem()

    await system.process(world)

    plan_search = world.get_component(entity_id, PlanSearchComponent)
    assert plan_search is not None
    assert plan_search.best_plan != []
    assert len(seen) == 1
    assert seen[0].entity_id == entity_id
