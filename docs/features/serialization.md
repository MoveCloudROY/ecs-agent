# Serialization

The `WorldSerializer` provides a robust mechanism to save the entire state of a `World` — including all entities and their components — into a structured format like JSON, and reload it later.

## API Reference

The `WorldSerializer` class offers several methods for serializing and deserializing `World` states:

- `WorldSerializer.to_dict(world: World) -> dict`: Converts a `World` instance into a serializable dictionary.
- `WorldSerializer.from_dict(data: dict, providers: dict[str, LLMModel], tool_handlers: dict[str, Callable]) -> World`: Reconstructs a `World` from a dictionary.
- `WorldSerializer.save(world: World, path: Path | str)`: Directly saves the `World` to a JSON file.
- `WorldSerializer.load(path: Path | str, providers: dict[str, LLMModel], tool_handlers: dict[str, Callable]) -> World`: Directly loads a `World` from a JSON file.

## How it Works

The serialization process relies on the `COMPONENT_REGISTRY`, which maps component class names to their corresponding types for correct deserialization.

### Non-Serializable Fields
Some fields within components are naturally non-serializable, such as live LLM model instances or tool handler callables.
- `LLMComponent.model`: Replaced with `"<non-serializable>"` during serialization.
- `ToolRegistryComponent.handlers`: Replaced with `"<non-serializable>"` during serialization.

### Re-Injection on Load
When loading a `World`, you must provide a dictionary of `providers` (mapping model names to `LLMModel` instances) and `tool_handlers` (mapping tool names to their corresponding callable functions). The `WorldSerializer` uses these to re-inject the necessary live objects back into the components.

## Usage Example

The following example demonstrates a complete save-and-load cycle:

```python
import asyncio
from pathlib import Path
from ecs_agent.core import World
from ecs_agent.components import LLMComponent, ConversationComponent
from ecs_agent.providers import Model
from ecs_agent.providers.config import ApiFormat
from ecs_agent.serialization import WorldSerializer

async def main():
    # 1. Create and setup the original world
    world = World()
    agent_id = world.create_entity()
    model = Model(
        "qwen3.5-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="...",
        api_format=ApiFormat.OPENAI_CHAT_COMPLETIONS,
    )
    
    world.add_component(agent_id, LLMComponent(model=model))
    world.add_component(agent_id, ConversationComponent(messages=[]))
    
    # 2. Save to JSON
    state_path = Path("agent_state.json")
    WorldSerializer.save(world, state_path)
    print(f"World state saved to {state_path}")
    
    # 3. Load back from JSON
    # Note: We must re-provide the live LLM model and tool handlers (if any)
    loaded_world = WorldSerializer.load(
        state_path,
        providers={"qwen3.5-plus": model},
        tool_handlers={}
    )
    print("World state loaded back successfully.")

if __name__ == "__main__":
    asyncio.run(main())
```

## Caveats

- **Private Internals**: The `WorldSerializer` directly accesses private `World` internals (`world._components._components`) to efficiently iterate and extract state.
- **Dependency Management**: Deserialization requires all necessary models and tool handlers to be available at load time. If a component references a model that is not included in the `providers` map, the field will remain as `"<non-serializable>"`, which may lead to errors when the agent attempts to use it.
