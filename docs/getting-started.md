# Getting Started

This guide walks you through installing the ecs-agent framework, setting up your environment, and building your first agentic system.

## Installation

The ecs-agent package requires Python 3.11 or higher.

### Standard Installation

Install the package directly from PyPI:

```bash
pip install ecs-agent
```

### From Source

If you want to contribute or use the latest development version, clone the repository and install in editable mode:

```bash
git clone https://github.com/your-repo/ecs-agent.git
cd ecs
pip install -e ".[dev]"
```

### Dependencies

The framework relies on several core libraries:

*   **Runtime:** `httpx>=0.24.0`, `tenacity>=8.2.0`, `structlog>=23.1.0`, `pydantic>=2.0.0`
*   **Development:** `pytest>=7.4.0`, `pytest-asyncio>=0.21.0`, `mypy>=1.5.0`

## Environment Setup

To use real Large Language Models (LLMs), you need to configure your environment variables. You can find a template in the `.env.example` file.

*   `LLM_API_KEY`: Your API key (required for OpenAI or DashScope examples).
*   `LLM_BASE_URL`: The endpoint URL (default: `https://dashscope.aliyuncs.com/compatible-mode/v1`).
*   `LLM_MODEL`: The model name to use (default: `qwen3.5-plus`).

## Your First Agent

Let's build a simple chat agent using a mock provider. This allows you to test the ECS architecture without an active API key.

### Complete Chat Agent Code

```python
import asyncio
from ecs_agent.core import World, Runner
from ecs_agent.components import LLMComponent, ConversationComponent
from ecs_agent.providers import FakeProvider
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.types import CompletionResult, Message

async def main():
    # 1. Create a World
    # The World acts as the container for all entities and systems.
    world = World()

    # 2. Create a FakeProvider with pre-configured responses
    # This simulates an LLM response for testing purposes.
    provider = FakeProvider([
        CompletionResult(
            message=Message(role="assistant", content="Hello! I'm doing great, how can I help you today?")
        )
    ])

    # 3. Create an entity and add LLMComponent + ConversationComponent
    # We define our agent as an entity with specific behavioral components.
    agent = world.create_entity()
    world.add_component(agent, LLMComponent(provider=provider, model="fake", system_prompt="You are a helpful assistant."))
    world.add_component(agent, ConversationComponent(messages=[Message(role="user", content="Hello, how are you?")]))

    # 4. Register systems
    # Systems contain the logic. Priority determines execution order.
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # 5. Run with Runner
    # The Runner advances the world through discrete ticks.
    runner = Runner()
    await runner.run(world, max_ticks=3)
    # The Runner also supports infinite execution with max_ticks=None,
    # running until a TerminalComponent signals completion.

    # 6. Read the conversation result
    conv = world.get_component(agent, ConversationComponent)
    for msg in conv.messages:
        print(f"{msg.role}: {msg.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Breakdown

1.  **World Creation**: Everything lives inside a `World` instance.
2.  **Mock Provider**: `FakeProvider` bypasses real network calls, making it perfect for CI/CD or initial development.
3.  **Entity Configuration**: We add an `LLMComponent` to handle model settings and a `ConversationComponent` to store the chat history.
4.  **System Registration**: 
    *   `ReasoningSystem` triggers the LLM call.
    *   `MemorySystem` manages how context is stored or pruned.
    *   `ErrorHandlingSystem` catches and logs failures during the tick.
5.  **Execution**: The `Runner` executes the loop. Each tick allows systems to process the current state of entities.

## Your First Tool Agent

To create an agent that can interact with external functions, you use the `ToolRegistryComponent` and `ToolExecutionSystem`.

### Implementation Pattern

1.  **Define a Tool**: Create an async function for your tool logic.
    ```python
    async def add(a: str, b: str) -> str:
        return str(int(a) + int(b))
    ```

2.  **Add Tool Registry**: Attach `ToolRegistryComponent` to your entity with the tool schema.
    ```python
    from ecs_agent.types import ToolSchema
    from ecs_agent.components import ToolRegistryComponent

    schema = ToolSchema(
        name="add",
        description="Adds two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            }
        }
    )
    
    world.add_component(agent, ToolRegistryComponent(
        tools={"add": schema},
        handlers={"add": add}
    ))
    ```

3.  **Register Tool Execution**: Add `ToolExecutionSystem` to your world.
    ```python
    from ecs_agent.systems.tool_execution import ToolExecutionSystem
    world.register_system(ToolExecutionSystem(), priority=5)
    ```

The `ReasoningSystem` will now see these tools in the LLM context and generate `ToolCall` objects, which the `ToolExecutionSystem` then processes.

## Next Steps

Now that you have your first agent running, explore the deeper architectural details:

*   [Architecture Overview](architecture.md): Understand the Entity-Component-System pattern.
*   [Core Concepts](core-concepts.md): Learn about Components, Systems, and the World.
*   [Examples](examples.md): See more complex patterns like multi-agent orchestration.
