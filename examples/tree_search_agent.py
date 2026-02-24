"""Tree Search Agent — MCTS for plan exploration.

This example demonstrates Monte Carlo Tree Search (MCTS) for exploring and optimizing
plan branches. The agent uses MCTS to:

1. Select promising branches using UCB1 exploration strategy
2. Expand nodes by asking the LLM to generate candidate actions (one per line)
3. Simulate branches by asking the LLM to score potential paths (returns float 0.0-1.0)
4. Backpropagate scores to improve future branch selection

This creates an adaptive tree search where the agent discovers and scores different
plan paths, eventually converging on the best_plan (highest cumulative scores).

Usage:
  uv run python examples/tree_search_agent.py

The example uses FakeProvider with deterministic responses:
- Expansion responses: one action per line, max max_branching lines
- Simulation responses: a float value 0.0-1.0 (or text containing a number)
"""

from __future__ import annotations

import asyncio

from ecs_agent.components import (
    ConversationComponent,
    LLMComponent,
    PlanSearchComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.tree_search import TreeSearchSystem
from ecs_agent.types import CompletionResult, Message


async def main() -> None:
    """Run a Tree Search Agent exploring problem-solving strategies."""

    # =========================================================================
    # MCTS Setup: Fake provider with alternating expand/simulate responses
    # =========================================================================
    # TreeSearchSystem follows this pattern each iteration:
    # 1. Select a leaf node (using UCB1)
    # 2. If leaf is at depth < max_depth and has no children: Expand
    #    - Calls provider.complete(), expects response with actions (one per line)
    #    - Creates up to max_branching children from the actions
    # 3. Simulate the selected/newly-expanded node
    #    - Calls provider.complete(), expects response with a score (float 0.0-1.0)
    #    - _parse_score extracts the number from the response
    # 4. Backpropagate the score up the tree
    # 5. Repeat until max_depth reached or no more expandable nodes

    # Note: Each MCTS iteration may call expand once and simulate once,
    # so we need pairs of responses: (expansion actions, simulation score)

    # Response pair 0: Expand root node → 2 candidate strategies
    response_expand_0 = CompletionResult(
        message=Message(
            role="assistant",
            content=("Systematic step-by-step approach\nDivide-and-conquer strategy"),
        ),
    )
    # Response 1: Simulate first strategy
    response_score_0 = CompletionResult(
        message=Message(role="assistant", content="0.75"),
    )

    # Response pair 1: Now select will pick the first strategy (UCB1)
    # or the second (unexplored). Let's say it explores the second.
    # No expansion needed yet (both children of root already exist after response_expand_0)
    # Just simulate the second strategy
    response_score_1 = CompletionResult(
        message=Message(role="assistant", content="0.85"),
    )

    # Response pair 2: Select expanded first strategy for refinement
    # Expand it to get sub-strategies
    response_expand_1 = CompletionResult(
        message=Message(
            role="assistant",
            content=("Break into manageable sub-problems\nVerify each step thoroughly"),
        ),
    )
    # Response 3: Simulate first sub-strategy
    response_score_2 = CompletionResult(
        message=Message(role="assistant", content="0.80"),
    )

    # Response pair 3: Simulate second sub-strategy of first main strategy
    response_score_3 = CompletionResult(
        message=Message(role="assistant", content="0.82"),
    )

    # Response pair 4: Expand the divide-and-conquer main strategy
    response_expand_2 = CompletionResult(
        message=Message(
            role="assistant",
            content=("Identify independent subproblems\nParallelize when possible"),
        ),
    )
    # Response 5: Simulate first sub-strategy of divide-and-conquer
    response_score_4 = CompletionResult(
        message=Message(role="assistant", content="0.88"),
    )

    # Response 6: Simulate second sub-strategy of divide-and-conquer
    response_score_5 = CompletionResult(
        message=Message(role="assistant", content="0.86"),
    )

    # =========================================================================
    # Create the ECS World
    # =========================================================================
    world = World()

    # Create provider with pre-defined responses in order
    provider = FakeProvider(
        responses=[
            response_expand_0,
            response_score_0,
            response_score_1,
            response_expand_1,
            response_score_2,
            response_score_3,
            response_expand_2,
            response_score_4,
            response_score_5,
        ]
    )

    # Create the agent entity
    agent = world.create_entity()

    # Attach required components for TreeSearchSystem
    world.add_component(
        agent,
        LLMComponent(
            provider=provider,
            model="fake-model",
            system_prompt=(
                "You are a planning expert using MCTS to explore solution strategies. "
                "When asked to generate actions, return one per line. "
                "When asked to score a path, return a number from 0 to 1."
            ),
        ),
    )

    world.add_component(
        agent,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content="Find the best strategy to solve a complex algorithmic problem",
                ),
            ],
            max_messages=100,
        ),
    )

    # MCTS configuration: max_depth=1, max_branching=2
    # This will explore 1 decision level with at most 2 options per level
    # With max_depth=1, one call to process() completes the MCTS search
    world.add_component(
        agent,
        PlanSearchComponent(
            max_depth=1,
            max_branching=2,
            exploration_weight=1.414,  # UCB1 parameter: balance explore vs exploit
        ),
    )

    # =========================================================================
    # Register the TreeSearchSystem
    # =========================================================================
    world.register_system(TreeSearchSystem(priority=0), priority=0)

    # =========================================================================
    # Run the MCTS loop
    # =========================================================================
    print("=" * 70)
    print("TREE SEARCH AGENT — MCTS for Plan Exploration")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  max_depth: 1 (explore up to 1 decision level)")
    print("  max_branching: 2 (at most 2 options per level)")
    print("  exploration_weight: 1.414 (UCB1 balance: explore vs. exploit)")
    print()
    print("Tree Search Process:")
    print("  1. Select: Use UCB1 to pick promising leaf nodes")
    print("  2. Expand: Ask LLM for candidate actions (one per line)")
    print("  3. Simulate: Ask LLM to score each candidate (float 0.0-1.0)")
    print("  4. Backpropagate: Update node statistics with scores")
    print("  5. Repeat until max_depth reached or no more expandable nodes")
    print()

    # Run for one tick - TreeSearchSystem completes one MCTS iteration
    # (select → expand → simulate → backpropagate) then sets search_active=False
    runner = Runner()
    await runner.run(world, max_ticks=1)

    # =========================================================================
    # Display Results
    # =========================================================================
    print()
    print("=" * 70)
    print("MCTS RESULTS")
    print("=" * 70)
    print()

    search = world.get_component(agent, PlanSearchComponent)
    if search:
        print(f"Search completed: search_active={search.search_active}")
        if search.best_plan:
            best_path = " → ".join(search.best_plan)
            print(f"Best plan (highest-scoring path): {best_path}")
        else:
            print("Best plan: (none found - may have exhausted responses)")
        print()
        print("Summary:")
        print(f"  - Exploration depth: {search.max_depth} levels")
        print(f"  - Branching factor: {search.max_branching} children/node")
        print()

    conv = world.get_component(agent, ConversationComponent)
    if conv:
        print(f"Conversation history ({len(conv.messages)} messages):")
        for i, msg in enumerate(conv.messages):
            role_label = {
                "user": "User",
                "assistant": "LLM",
                "system": "System",
            }.get(msg.role, msg.role.title())
            preview = msg.content[:70] if msg.content else "(empty)"
            if len(msg.content or "") > 70:
                preview += "..."
            print(f"  {i + 1}. [{role_label}] {preview}")

    print()
    print("=" * 70)
    print("Tree Search Agent completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
