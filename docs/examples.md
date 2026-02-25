# Examples Gallery

Overview of the included examples for the ECS-based LLM Agent framework.

## Overview Table

| Example Name | Description | API Key Required? | Key Features |
| :--- | :--- | :--- | :--- |
| [Simple Chat Agent](#simple-chat-agent) | Minimal single-agent chat with FakeProvider | No | World + LLMComponent + Runner |
| [Tool-Using Agent](#tool-using-agent) | LLM calls a tool, system executes it | No | ToolRegistryComponent, ToolExecutionSystem |
| [Multi-Agent Collaboration](#multi-agent-collaboration) | Two agents exchange messages | No | CollaborationComponent, CollaborationSystem |
| [ReAct Pattern Agent](#react-pattern-agent) | PlanningSystem drives multi-step plan | Yes (OpenAI) | PlanningSystem, ToolExecutionSystem |
| [Plan-and-Execute Agent](#plan-and-execute-with-replanning) | Dynamic replanning based on results | Yes (OpenAI) | ReplanningSystem, RetryProvider |
| [Streaming Responses](#streaming-responses) | Real-time response streaming | Optional | stream=True, Delta iteration |
| [Retry with Backoff](#retry-with-exponential-backoff) | Automatic retries for transient errors | Optional | RetryProvider, RetryConfig |
| [Structured Output](#structured-output-with-pydantic) | JSON mode with Pydantic validation | Optional | pydantic_to_response_format |
| [World Serialization](#world-serialization) | Save/load world state to/from JSON | No | WorldSerializer.save/load |
| [Tool Approval Agent](#tool-approval-agent) | Policy-based tool call approval flow | No | ToolApprovalComponent, ToolApprovalSystem |
| [Tree Search Agent](#tree-search-agent) | MCTS planning for complex goals | No | PlanSearchComponent, TreeSearchSystem |
| [RAG Agent](#rag-agent) | Vector search retrieval-augmented generation | No | RAGTriggerComponent, RAGSystem |
| [Sub-Agent Delegation](#sub-agent-delegation) | Parent agent delegates to child agents | No | OwnerComponent, CollaborationComponent |
| [Claude Agent](#claude-agent) | Native Anthropic Claude provider | Yes (Anthropic) | ClaudeProvider |
| [LiteLLM Agent](#litellm-agent) | Unified access to 100+ LLM providers | Yes (varies) | LiteLLMProvider |
| [System Streaming](#system-streaming) | System-level streaming with events | Optional | StreamingComponent, StreamStartEvent |
| [Context Management](#context-management) | Checkpoint, undo, and compaction demo | Optional | CheckpointSystem, CompactionSystem |

---

## No API Key Required

### Simple Chat Agent
- **File:** `examples/chat_agent.py`
- **What it demonstrates:** Basic agent setup using the ECS pattern with a `FakeProvider`.
- **Run:** `python examples/chat_agent.py`
- **Pattern:** `World` + entity + `LLMComponent` + `ConversationComponent` + `ReasoningSystem` + `MemorySystem` + `ErrorHandlingSystem` + `Runner`.

#### Key Code
```python
# Create Agent Entity
agent_id = world.create_entity()
world.add_component(agent_id, LLMComponent(provider=provider, model="fake"))
world.add_component(agent_id, ConversationComponent(messages=[Message(role="user", content="Hello!")]))

# Register Systems
world.register_system(ReasoningSystem(priority=0), priority=0)
world.register_system(MemorySystem(), priority=10)
world.register_system(ErrorHandlingSystem(priority=99), priority=99)
```

#### Expected Output
A simple conversation printout:
```
Conversation:
  user: Hello, how are you?
  assistant: Hello! I'm doing great, thank you for asking! How can I help you today?
```

---

### Tool-Using Agent
- **File:** `examples/tool_agent.py`
- **What it demonstrates:** An agent that can call tools (e.g., an `add` function).
- **Run:** `python examples/tool_agent.py`
- **Pattern:** `ToolRegistryComponent` with `ToolSchema` and async handlers + `ToolExecutionSystem(priority=5)`.

#### Key Code
```python
# Register tools
world.add_component(
    agent_id,
    ToolRegistryComponent(
        tools={"add": ToolSchema(name="add", description="Add two numbers", ...)},
        handlers={"add": add},
    ),
)

# Register Systems (ToolExecutionSystem is key)
world.register_system(ReasoningSystem(priority=0), priority=0)
world.register_system(ToolExecutionSystem(priority=5), priority=5)
```

#### Expected Output
The agent calls the tool and then provides the final answer:
```
Conversation:
  user: What is 2 + 3?
  assistant: [tool_calls: add]
  tool (tool result): 5
  assistant: The answer is 5
```

---

### Multi-Agent Collaboration
- **File:** `examples/multi_agent.py`
- **What it demonstrates:** Two agents (researcher and summarizer) exchanging messages.
- **Run:** `python examples/multi_agent.py`
- **Pattern:** `CollaborationComponent(peers, inbox)` + `CollaborationSystem`.

#### Key Code
```python
# Set up collaboration: Agent A sends message to Agent B
world.add_component(agent_a_id, CollaborationComponent(peers=[agent_b_id], inbox=[]))
world.add_component(agent_b_id, CollaborationComponent(
    peers=[agent_a_id],
    inbox=[(agent_a_id, Message(role="assistant", content="I found interesting data."))],
))

# Register CollaborationSystem
world.register_system(CollaborationSystem(priority=5), priority=5)
```

#### Expected Output
Conversations for both agents:
```
Agent A (researcher) conversation:
  user: Start researching the topic.
  assistant: I've analyzed the data and found interesting patterns.

Agent B (summarizer) conversation:
  assistant (from researcher): I found interesting data.
  assistant: Thank you! I'll summarize the key findings for you.
```

---

### World Serialization
- **File:** `examples/serialization_demo.py`
- **What it demonstrates:** Saving and loading the entire world state to/from JSON.
- **Run:** `python examples/serialization_demo.py`
- **Pattern:** `WorldSerializer.save(world, path)` → `WorldSerializer.load(path, providers={...}, tool_handlers={...})`.

#### Key Code
```python
# Save World to JSON
WorldSerializer.save(world, "world_state.json")

# Load World from JSON
loaded_world = WorldSerializer.load(
    "world_state.json",
    providers={"fake-model": provider},
    tool_handlers=tool_handlers,
)
```

#### Expected Output
Verification that loaded state matches original:
```
STEP 2: SAVE WORLD TO JSON
✓ World saved to world_state.json
✓ JSON contains 3 entities

STEP 3: LOAD WORLD FROM JSON
✓ World loaded from world_state.json

STEP 4: VERIFY LOADED STATE
✓ All checks passed!
```

---

## API Key Required

### ReAct Pattern Agent
- **File:** `examples/react_agent.py`
- **What it demonstrates:** The full Reasoning + Acting (ReAct) pattern with a multi-step plan.
- **Run:** `uv run python examples/react_agent.py`
- **Required Env:** `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`.
- **Pattern:** `PlanComponent` + `PlanningSystem` + `ToolExecutionSystem` + `PlanStepCompletedEvent` subscription.

#### Key Code
```python
# Attach plan steps
plan_steps = ["Look up weather in Beijing", "Look up population in Beijing", ...]
world.add_component(main_agent, PlanComponent(steps=plan_steps))

# Register systems (order: planning -> tool execution)
world.register_system(PlanningSystem(priority=0), priority=0)
world.register_system(ToolExecutionSystem(priority=5), priority=5)

# Subscribe to progress
world.event_bus.subscribe(PlanStepCompletedEvent, on_step_completed)
```

#### Expected Output
Step-by-step progress and final comparison:
```
Running ReAct agent with 5-step plan...
  ✓ Step 1 completed: Look up the weather in Beijing...
  ✓ Step 2 completed: Look up the weather in Shanghai...
...
[Thought] Based on the data, Beijing is cooler but larger...
```

---

### Plan-and-Execute with Replanning
- **File:** `examples/plan_and_execute_agent.py`
- **What it demonstrates:** Dynamic replanning where the agent revises its plan after each step.
- **Run:** `uv run python examples/plan_and_execute_agent.py`
- **Required Env:** `LLM_API_KEY`, timeouts, and retries.
- **Pattern:** `PlanningSystem` + `ToolExecutionSystem` + `ReplanningSystem` + `RetryProvider` + `PlanRevisedEvent`.

#### Key Code
```python
# Wrap provider with retry logic
provider = RetryProvider(base_provider, retry_config=RetryConfig(max_attempts=3))

# Register systems including ReplanningSystem
world.register_system(PlanningSystem(priority=0), priority=0)
world.register_system(ToolExecutionSystem(priority=5), priority=5)
world.register_system(ReplanningSystem(priority=7), priority=7)

# Subscribe to replanning events
world.event_bus.subscribe(PlanRevisedEvent, on_plan_revised)
```

#### Expected Output
Initial plan steps followed by potential revisions:
```
Running Plan-and-Execute agent...
  ✓ Step 1 completed: Check weather for Beijing...
  📋 Plan revised!
    Old plan: [...]
    New plan: [...]
```

---

## API Key Optional

### Streaming Responses
- **File:** `examples/streaming_agent.py`
- **What it demonstrates:** Real-time character-by-character or word-by-word output from an LLM.
- **Run:** `uv run python examples/streaming_agent.py`
- **Pattern:** `async for delta in delta_iterator: sys.stdout.write(delta.content)`.

#### Key Code
```python
# Call provider with streaming enabled
delta_iterator = await provider.complete(messages, stream=True)

# Print chunks in real-time
async for delta in delta_iterator:
    if delta.content:
        sys.stdout.write(delta.content)
        sys.stdout.flush()

# Access usage stats from final delta
if delta.finish_reason:
    print(f"Total tokens: {delta.usage.total_tokens}")
```

#### Expected Output
The response appearing gradually:
```
Streaming response:
------------------------------------------------------------
Streaming in LLMs refers to the technique where...

------------------------------------------------------------
Tokens used:
  Prompt tokens:     15
  Completion tokens: 35
  Total tokens:      50
```

---

### Retry with Exponential Backoff
- **File:** `examples/retry_agent.py`
- **What it demonstrates:** Automatic retries for transient HTTP errors like 429 or 500.
- **Run:** `uv run python examples/retry_agent.py`
- **Pattern:** `RetryProvider` wrapping a base provider with a custom `RetryConfig`.

#### Key Code
```python
# Custom retry config
retry_config = RetryConfig(
    max_attempts=5,
    multiplier=2.0,
    min_wait=2.0,
    max_wait=30.0,
    retry_status_codes=(429, 500, 502, 503, 504),
)

# Wrap provider
provider = RetryProvider(base_provider, retry_config=retry_config)
```

#### Expected Output
Logs showing retry attempts (if errors occur) and the final successful completion:
```
Retry Configuration:
  max_attempts: 5
  multiplier: 2.0
...
Completion Result:
  Role: assistant
  Content: This is a demonstration response...
✓ Completion succeeded
```

---

### Structured Output with Pydantic
- **File:** `examples/structured_output_agent.py`
- **What it demonstrates:** Extracting JSON data that strictly follows a Pydantic model.
- **Run:** `uv run python examples/structured_output_agent.py`
- **Pattern:** `pydantic_to_response_format(CityInfo)` → `provider.complete(messages, response_format=...)`.

#### Key Code
```python
# Convert Pydantic model to response format
response_format = pydantic_to_response_format(CityInfo)

# Call LLM with JSON mode enabled
result = await provider.complete(messages, response_format=response_format)

# Parse and validate JSON back into Pydantic model
city_info = CityInfo.model_validate_json(result.message.content)
print(f"City: {city_info.name}, Population: {city_info.population}M")
```

#### Expected Output
Formatted data extracted from the JSON response:
```
STRUCTURED OUTPUT RESULT
============================================================

City Name:    Tokyo
Country:      Japan
Population:   14.0 million
Climate:      Temperate
Landmarks:
  - Senso-ji Temple
  - Tokyo Tower
...
```
---

### Tool Approval Agent
- **File:** `examples/tool_approval_agent.py`
- **What it demonstrates:** Policy-based manual approval flow for sensitive tool calls.
- **Run:** `uv run python examples/tool_approval_agent.py`
- **Pattern:** `ToolApprovalComponent` + `ToolApprovalSystem`.

---

### Tree Search Agent
- **File:** `examples/tree_search_agent.py`
- **What it demonstrates:** MCTS planning for complex goals using Monte Carlo Tree Search.
- **Run:** `uv run python examples/tree_search_agent.py`
- **Pattern:** `PlanSearchComponent` + `TreeSearchSystem`.

---

### RAG Agent
- **File:** `examples/rag_agent.py`
- **What it demonstrates:** Retrieval-augmented generation using vector search.
- **Run:** `uv run python examples/rag_agent.py`
- **Pattern:** `RAGTriggerComponent` + `RAGSystem` + `EmbeddingComponent` + `VectorStoreComponent`.

---

### Sub-Agent Delegation
- **File:** `examples/subagent_delegation.py`
- **What it demonstrates:** Parent agent delegating tasks to child agents.
- **Run:** `uv run python examples/subagent_delegation.py`
- **Pattern:** `OwnerComponent` + `CollaborationComponent`.

---

### Claude Agent
- **File:** `examples/claude_agent.py`
- **What it demonstrates:** Native usage of the Anthropic Claude provider.
- **Run:** `uv run python examples/claude_agent.py`
- **Pattern:** `ClaudeProvider`.

---

### LiteLLM Agent
- **File:** `examples/litellm_agent.py`
- **What it demonstrates:** Unified access to 100+ LLM providers.
- **Run:** `uv run python examples/litellm_agent.py`
- **Pattern:** `LiteLLMProvider`.

---

### System Streaming
- **File:** `examples/streaming_system_agent.py`
- **What it demonstrates:** System-level streaming using the event bus.
- **Run:** `uv run python examples/streaming_system_agent.py`
- **Pattern:** `StreamingComponent` + `StreamStartEvent` + `StreamDeltaEvent` + `StreamEndEvent`.

---

### Context Management
- **File:** `examples/context_management_agent.py`
- **What it demonstrates:** Checkpoint, undo, and conversation compaction lifecycle.
- **Run:** `uv run python examples/context_management_agent.py`
- **Pattern:** `CheckpointSystem` + `CompactionSystem` + `CheckpointComponent` + `CompactionConfigComponent`.
