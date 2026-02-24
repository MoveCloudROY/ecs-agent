"""Tree search system using Monte Carlo Tree Search (MCTS)."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

from ecs_agent.components import (
    ConversationComponent,
    ErrorComponent,
    LLMComponent,
    PlanComponent,
    PlanSearchComponent,
    TerminalComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.types import CompletionResult, EntityId, MCTSNodeScoredEvent, Message

logger = get_logger(__name__)


@dataclass
class TreeNode:
    id: int
    parent_id: int | None
    action: str
    score: float = 0.0
    visits: int = 0
    children: list[int] = field(default_factory=list)


class TreeSearchSystem:
    def __init__(self, priority: int = 0) -> None:
        self.priority = priority
        self._nodes_by_entity: dict[int, dict[int, TreeNode]] = {}
        self._next_node_id_by_entity: dict[int, int] = {}
        self._root_id_by_entity: dict[int, int] = {}

    async def process(self, world: World) -> None:
        for entity_id, components in world.query(
            PlanSearchComponent, LLMComponent, ConversationComponent
        ):
            plan_search, llm_component, conversation = components
            assert isinstance(plan_search, PlanSearchComponent)
            assert isinstance(llm_component, LLMComponent)
            assert isinstance(conversation, ConversationComponent)

            if world.has_component(entity_id, PlanComponent):
                continue

            int_entity_id = int(entity_id)

            if not plan_search.search_active:
                self._initialize_tree(int_entity_id)
                plan_search.best_plan = []
                plan_search.search_active = True

            try:
                nodes = self._nodes_by_entity[int_entity_id]
                root_id = self._root_id_by_entity[int_entity_id]

                if plan_search.max_depth <= 0:
                    plan_search.search_active = False
                    plan_search.best_plan = []
                    continue

                selected_id = self._select_leaf(
                    nodes=nodes,
                    root_id=root_id,
                    exploration_weight=plan_search.exploration_weight,
                )
                selected_depth = self._depth(nodes, selected_id)

                if (
                    selected_depth < plan_search.max_depth
                    and not nodes[selected_id].children
                ):
                    new_child_ids = await self._expand(
                        world=world,
                        entity_id=entity_id,
                        nodes=nodes,
                        node_id=selected_id,
                        plan_search=plan_search,
                        conversation=conversation,
                        llm_component=llm_component,
                    )
                    if new_child_ids:
                        selected_id = new_child_ids[0]
                        selected_depth = self._depth(nodes, selected_id)

                score = await self._simulate(
                    world=world,
                    entity_id=entity_id,
                    nodes=nodes,
                    node_id=selected_id,
                    conversation=conversation,
                    llm_component=llm_component,
                )
                self._backpropagate(nodes=nodes, node_id=selected_id, score=score)

                await world.event_bus.publish(
                    MCTSNodeScoredEvent(
                        entity_id=entity_id,
                        node_id=selected_id,
                        score=score,
                    )
                )

                if (
                    selected_depth >= plan_search.max_depth
                    or not self._has_expandable_node(
                        nodes=nodes,
                        max_depth=plan_search.max_depth,
                        max_branching=plan_search.max_branching,
                    )
                ):
                    plan_search.best_plan = self._extract_best_path(
                        nodes=nodes, root_id=root_id
                    )
                    plan_search.search_active = False
            except (IndexError, StopIteration):
                world.add_component(
                    entity_id,
                    TerminalComponent(reason="provider_exhausted"),
                )
            except Exception as exc:
                logger.error(
                    "tree_search_error",
                    entity_id=entity_id,
                    exception=str(exc),
                )
                world.add_component(
                    entity_id,
                    ErrorComponent(
                        error=str(exc),
                        system_name="TreeSearchSystem",
                        timestamp=time.time(),
                    ),
                )
                world.add_component(
                    entity_id,
                    TerminalComponent(reason="tree_search_error"),
                )

    def _initialize_tree(self, entity_id: int) -> None:
        root = TreeNode(id=0, parent_id=None, action="root")
        self._nodes_by_entity[entity_id] = {0: root}
        self._next_node_id_by_entity[entity_id] = 1
        self._root_id_by_entity[entity_id] = 0

    def _ucb1(
        self,
        node: TreeNode,
        parent_visits: int,
        exploration_weight: float,
    ) -> float:
        if node.visits == 0:
            return float("inf")

        safe_parent_visits = max(parent_visits, 1)
        exploitation = node.score / node.visits
        exploration = exploration_weight * math.sqrt(
            math.log(safe_parent_visits) / node.visits
        )
        return exploitation + exploration

    def _select_leaf(
        self,
        nodes: dict[int, TreeNode],
        root_id: int,
        exploration_weight: float,
    ) -> int:
        current_id = root_id

        while nodes[current_id].children:
            parent = nodes[current_id]
            current_id = max(
                parent.children,
                key=lambda child_id: self._ucb1(
                    node=nodes[child_id],
                    parent_visits=max(parent.visits, 1),
                    exploration_weight=exploration_weight,
                ),
            )

        return current_id

    async def _expand(
        self,
        world: World,
        entity_id: EntityId,
        nodes: dict[int, TreeNode],
        node_id: int,
        plan_search: PlanSearchComponent,
        conversation: ConversationComponent,
        llm_component: LLMComponent,
    ) -> list[int]:
        _ = world
        prompt = (
            "Generate candidate next actions for planning. "
            f"Return one action per line, at most {plan_search.max_branching} lines."
        )
        messages = [
            Message(role="system", content=prompt),
            *conversation.messages,
            Message(
                role="user", content=f"Current path: {self._path_text(nodes, node_id)}"
            ),
        ]
        result = await llm_component.provider.complete(messages)
        if not isinstance(result, CompletionResult):
            raise TypeError("Streaming response not supported in TreeSearchSystem")

        actions = self._parse_actions(result.message.content, plan_search.max_branching)
        new_child_ids: list[int] = []
        for action in actions:
            if len(nodes[node_id].children) >= plan_search.max_branching:
                break
            next_id = self._next_node_id_by_entity[int(entity_id)]
            self._next_node_id_by_entity[int(entity_id)] = next_id + 1
            nodes[next_id] = TreeNode(id=next_id, parent_id=node_id, action=action)
            nodes[node_id].children.append(next_id)
            new_child_ids.append(next_id)

        return new_child_ids

    async def _simulate(
        self,
        world: World,
        entity_id: int,
        nodes: dict[int, TreeNode],
        node_id: int,
        conversation: ConversationComponent | None,
        llm_component: LLMComponent | None,
    ) -> float:
        _ = world
        if conversation is None or llm_component is None:
            return 0.0

        action_path = self._extract_actions_to_node(nodes, node_id)
        prompt = (
            "Score this candidate plan path from 0 to 1. "
            "Return only a number. "
            f"Path: {' -> '.join(action_path)}"
        )
        messages = [
            *conversation.messages,
            Message(role="user", content=prompt),
        ]

        result = await llm_component.provider.complete(messages)
        if not isinstance(result, CompletionResult):
            raise TypeError("Streaming response not supported in TreeSearchSystem")

        score = self._parse_score(result.message.content)
        return min(max(score, 0.0), 1.0)

    def _backpropagate(
        self, nodes: dict[int, TreeNode], node_id: int, score: float
    ) -> None:
        current_id: int | None = node_id
        while current_id is not None:
            node = nodes[current_id]
            node.visits += 1
            node.score += score
            current_id = node.parent_id

    def _extract_best_path(self, nodes: dict[int, TreeNode], root_id: int) -> list[str]:
        path: list[str] = []
        current_id = root_id
        while nodes[current_id].children:
            best_child_id = max(
                nodes[current_id].children,
                key=lambda child_id: self._average_score(nodes[child_id]),
            )
            current_id = best_child_id
            if nodes[current_id].action != "root":
                path.append(nodes[current_id].action)
        return path

    def _average_score(self, node: TreeNode) -> float:
        if node.visits == 0:
            return 0.0
        return node.score / node.visits

    def _extract_actions_to_node(
        self, nodes: dict[int, TreeNode], node_id: int
    ) -> list[str]:
        actions: list[str] = []
        current_id: int | None = node_id
        while current_id is not None:
            node = nodes[current_id]
            if node.action != "root":
                actions.append(node.action)
            current_id = node.parent_id
        actions.reverse()
        return actions

    def _depth(self, nodes: dict[int, TreeNode], node_id: int) -> int:
        depth = 0
        current_id = node_id
        while nodes[current_id].parent_id is not None:
            depth += 1
            parent_id = nodes[current_id].parent_id
            assert parent_id is not None
            current_id = parent_id
        return depth

    def _path_text(self, nodes: dict[int, TreeNode], node_id: int) -> str:
        actions = self._extract_actions_to_node(nodes, node_id)
        if not actions:
            return "(root)"
        return " -> ".join(actions)

    def _parse_actions(self, content: str, max_branching: int) -> list[str]:
        if max_branching <= 0:
            return []
        unique_actions: list[str] = []
        seen: set[str] = set()
        for raw_line in content.splitlines():
            action = raw_line.strip()
            if not action or action in seen:
                continue
            seen.add(action)
            unique_actions.append(action)
            if len(unique_actions) >= max_branching:
                break
        return unique_actions

    def _parse_score(self, content: str) -> float:
        stripped = content.strip()
        if not stripped:
            return 0.0

        try:
            return float(stripped)
        except ValueError:
            match = re.search(r"[-+]?\d*\.?\d+", stripped)
            if match is None:
                return 0.0
            return float(match.group(0))

    def _has_expandable_node(
        self,
        nodes: dict[int, TreeNode],
        max_depth: int,
        max_branching: int,
    ) -> bool:
        if max_branching <= 0:
            return False

        for node_id, node in nodes.items():
            if (
                self._depth(nodes, node_id) < max_depth
                and len(node.children) < max_branching
            ):
                return True
        return False


__all__ = ["TreeSearchSystem"]
