# Tree-Structured Conversations

The `ConversationTreeComponent` enables branching conversation history, allowing agents to explore multiple reasoning paths and compare outcomes.

## Overview

Tree-structured conversations support:
- **Branching**: Create alternative conversation paths from any message
- **Navigation**: Switch between branches to explore different reasoning
- **Linearization**: Convert tree to flat message list for LLM compatibility
- **Message Tracking**: Parent-child relationships via `parent_message_id`

## Core Types

### ConversationMessage

Messages in a tree have parent-child relationships:

```python
from ecs_agent.types import ConversationMessage

root_msg = ConversationMessage(
    id="msg_root",
    parent_message_id=None,  # Root message has no parent
    role="user",
    content="What's the capital of France?",
)

child_msg = ConversationMessage(
    id="msg_1",
    parent_message_id="msg_root",  # Links to parent
    role="assistant",
    content="Paris is the capital of France.",
)
```

### ConversationBranch

Branches track conversation paths from root to leaf:

```python
from ecs_agent.types import ConversationBranch

branch = ConversationBranch(
    branch_id="branch_main",
    leaf_message_id="msg_1",  # Current end of this branch
)
```

### ConversationTreeComponent

The component stores all messages and branches:

```python
from ecs_agent.components import ConversationTreeComponent

world.add_component(
    entity,
    ConversationTreeComponent(
        messages={"msg_root": root_msg, "msg_1": child_msg},
        current_branch_id="branch_main",
        branches={"branch_main": branch},
    ),
)
```

## Usage

### Creating a Tree Conversation

```python
from ecs_agent import World
from ecs_agent.components import ConversationTreeComponent
from ecs_agent.types import ConversationMessage, ConversationBranch

world = World()
entity = world.create_entity()

# Create root message
root = ConversationMessage(
    id="msg_0",
    parent_message_id=None,
    role="user",
    content="Explain quantum entanglement.",
)

# Create initial branch
branch = ConversationBranch(
    branch_id="main",
    leaf_message_id="msg_0",
)

# Add tree component
world.add_component(
    entity,
    ConversationTreeComponent(
        messages={"msg_0": root},
        current_branch_id="main",
        branches={"main": branch},
    ),
)
```

### Adding Messages to a Branch

Use the `conversation_tree` module to add messages:

```python
from ecs_agent.conversation_tree import add_message_to_tree

tree = world.get_component(entity, ConversationTreeComponent)
if tree:
    new_msg = ConversationMessage(
        id="msg_1",
        parent_message_id="msg_0",
        role="assistant",
        content="Quantum entanglement is...",
    )
    
    add_message_to_tree(tree, new_msg, branch_id="main")
```

### Creating Alternative Branches

Branch from any message to explore alternatives:

```python
from ecs_agent.conversation_tree import create_branch

# Create alternative response branch
alt_msg = ConversationMessage(
    id="msg_alt",
    parent_message_id="msg_0",  # Same parent as msg_1
    role="assistant",
    content="Let me explain it differently...",
)

create_branch(tree, alt_msg, new_branch_id="alternative", parent_message_id="msg_0")
```

### Switching Branches

Change the active branch to explore different paths:

```python
from ecs_agent.conversation_tree import switch_branch

switch_branch(tree, "alternative")
# Now tree.current_branch_id == "alternative"
```

### Linearizing for Systems

Most systems expect flat message lists. Convert tree to linear:

```python
from ecs_agent.conversation_tree import linearize_branch

messages = linearize_branch(tree, branch_id="main")
# Returns list[Message] following parent_message_id chain
```

## Integration with ReasoningSystem

The `ReasoningSystem` can work with both flat and tree conversations:

```python
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.components import LLMComponent, ConversationComponent

# Option 1: Use ConversationComponent (flat)
world.add_component(
    entity,
    ConversationComponent(messages=[...]),
)

# Option 2: Use ConversationTreeComponent (branching)
# System will automatically linearize current branch
world.add_component(
    entity,
    ConversationTreeComponent(...),
)

# Both work with ReasoningSystem
world.register_system(ReasoningSystem(priority=0), priority=0)
```

## Use Cases

### Multi-Path Reasoning

Explore multiple solution approaches:

```python
# Create main branch
add_message_to_tree(tree, user_msg, branch_id="main")

# Try approach A
create_branch(tree, approach_a_msg, new_branch_id="approach_a", parent_message_id=user_msg.id)

# Try approach B
create_branch(tree, approach_b_msg, new_branch_id="approach_b", parent_message_id=user_msg.id)

# Compare outcomes
results_a = linearize_branch(tree, "approach_a")
results_b = linearize_branch(tree, "approach_b")
```

### Rollback and Retry

Backtrack to earlier messages and try alternatives:

```python
# Find message to backtrack to
target_msg = tree.messages["msg_3"]

# Create new branch from that point
retry_msg = ConversationMessage(
    id="msg_retry",
    parent_message_id=target_msg.id,
    role="user",
    content="Let's try a different approach...",
)

create_branch(tree, retry_msg, new_branch_id="retry", parent_message_id=target_msg.id)
switch_branch(tree, "retry")
```

### Conversation Visualization

Tree structure enables visual representation:

```
msg_0 (user)
├── msg_1 (assistant) [branch: main]
│   └── msg_2 (user)
│       └── msg_3 (assistant)
└── msg_alt (assistant) [branch: alternative]
    └── msg_alt_2 (user)
```

## API Reference

### Functions

- `add_message_to_tree(tree, message, branch_id)` — Append message to branch
- `create_branch(tree, message, new_branch_id, parent_message_id)` — Create new branch
- `switch_branch(tree, branch_id)` — Change active branch
- `linearize_branch(tree, branch_id)` — Convert branch to flat message list
- `get_branch_messages(tree, branch_id)` — Get all messages in branch
- `find_common_ancestor(tree, msg_id_1, msg_id_2)` — Find branch point

### Component Fields

- `messages: dict[str, ConversationMessage]` — All messages by ID
- `current_branch_id: str | None` — Active branch
- `branches: dict[str, ConversationBranch]` — All branches by ID

## See Also

- [Conversation Component](../components.md#conversationcomponent) — Flat conversation
- [Memory System](../systems.md#memorysystem) — Message management
- [Serialization](./serialization.md) — Persisting trees
