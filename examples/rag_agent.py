"""Retrieval-augmented generation (RAG) agent example.

This example demonstrates:
- Setting up a vector store with sample documents
- Dual-mode provider support (fake/real) for both LLM and embedding
- Registering RAGSystem to retrieve relevant context before reasoning
- Showing how RAG context is injected into the conversation

Environment variables:
- LLM_API_KEY: API key (if set, uses OpenAI-compatible providers; if unset, uses FakeProvider)
- LLM_BASE_URL: Base URL (default: https://dashscope.aliyuncs.com/compatible-mode/v1)
- LLM_MODEL: LLM model name (default: qwen3.5-flash)
- EMBEDDING_MODEL: Embedding model name (default: text-embedding-v3)
- EMBEDDING_DIMENSION: Embedding dimension (default: 1024)
"""

import asyncio
import os

from ecs_agent.components import (
    ConversationComponent,
    EmbeddingComponent,
    LLMComponent,
    RAGTriggerComponent,
    VectorStoreComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider, OpenAIProvider
from ecs_agent.providers.embedding_provider import OpenAIEmbeddingProvider
from ecs_agent.providers.fake_embedding_provider import FakeEmbeddingProvider
from ecs_agent.providers.vector_store import InMemoryVectorStore
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.rag import RAGSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message


async def main() -> None:
    """Run a RAG agent that retrieves context before reasoning."""
    # --- Read environment variables ---
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    model = os.environ.get("LLM_MODEL", "qwen3.5-flash")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-v3")
    embedding_dim = int(os.environ.get("EMBEDDING_DIMENSION", "1024"))

    # Create World
    world = World()

    # --- Create LLM provider (dual-mode) ---
    if api_key:
        print(f"Using OpenAIProvider with model: {model}")
        provider = OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
    else:
        print("No LLM_API_KEY set. Using FakeProvider for demonstration.")
        provider = FakeProvider(
            responses=[
                CompletionResult(
                    message=Message(
                        role="assistant",
                        content="Based on the retrieved context, RAG is a powerful technique that combines retrieval and generation for better answers.",
                    )
                )
            ]
        )

    # --- Create embedding provider (dual-mode) ---
    if api_key:
        print(f"Using OpenAIEmbeddingProvider with model: {embedding_model}")
        embedding_provider = OpenAIEmbeddingProvider(
            api_key=api_key, base_url=base_url, model=embedding_model
        )
        dimension = embedding_dim
    else:
        print("Using FakeEmbeddingProvider for demonstration.")
        embedding_provider = FakeEmbeddingProvider(dimension=8)
        dimension = 8

    # Create and populate vector store with sample documents
    vector_store = InMemoryVectorStore(dimension=dimension)

    # Create Agent Entity
    agent_id = world.create_entity()
    world.add_component(
        agent_id,
        LLMComponent(
            provider=provider,
            model=model if api_key else "fake",
            system_prompt="You are a helpful assistant that answers questions about AI and machine learning.",
        ),
    )

    # Add initial conversation with RAG query
    world.add_component(
        agent_id,
        ConversationComponent(
            messages=[
                Message(
                    role="user",
                    content="Tell me about retrieval-augmented generation.",
                )
            ]
        ),
    )

    # Add RAG components to trigger retrieval before reasoning
    world.add_component(
        agent_id,
        RAGTriggerComponent(query="retrieval-augmented generation", top_k=3),
    )
    world.add_component(
        agent_id,
        EmbeddingComponent(provider=embedding_provider, dimension=dimension),
    )
    world.add_component(
        agent_id,
        VectorStoreComponent(store=vector_store),
    )

    # Register Systems
    # RAG runs BEFORE reasoning (priority -10 < 0)
    world.register_system(RAGSystem(priority=-10), priority=-10)
    world.register_system(ReasoningSystem(priority=0), priority=0)
    world.register_system(MemorySystem(), priority=10)
    world.register_system(ErrorHandlingSystem(priority=99), priority=99)

    # Run
    runner = Runner()
    await runner.run(world, max_ticks=3)

    # Print results
    conv = world.get_component(agent_id, ConversationComponent)
    if conv is not None:
        print("Conversation:")
        for msg in conv.messages:
            content_preview = msg.content[:100]
            if len(msg.content) > 100:
                content_preview += "..."
            print(f"  {msg.role}: {content_preview}")
    else:
        print("No conversation found")


if __name__ == "__main__":
    asyncio.run(main())
