"""Retrieval-augmented generation (RAG) agent example.

This example demonstrates:
- Setting up a vector store with sample documents
- Using FakeEmbeddingProvider for deterministic embeddings (no API key needed)
- Registering RAGSystem to retrieve relevant context before reasoning
- Showing how RAG context is injected into the conversation
"""

import asyncio

from ecs_agent.components import (
    ConversationComponent,
    EmbeddingComponent,
    LLMComponent,
    RAGTriggerComponent,
    VectorStoreComponent,
)
from ecs_agent.core import Runner, World
from ecs_agent.providers import FakeProvider
from ecs_agent.providers.fake_embedding_provider import FakeEmbeddingProvider
from ecs_agent.providers.vector_store import InMemoryVectorStore
from ecs_agent.systems.error_handling import ErrorHandlingSystem
from ecs_agent.systems.memory import MemorySystem
from ecs_agent.systems.rag import RAGSystem
from ecs_agent.systems.reasoning import ReasoningSystem
from ecs_agent.types import CompletionResult, Message


async def main() -> None:
    """Run a RAG agent that retrieves context before reasoning."""
    # Create World
    world = World()

    # Set up embedding provider (deterministic, no API key needed)
    embedding_provider = FakeEmbeddingProvider(dimension=8)

    # Create and populate vector store with sample documents
    vector_store = InMemoryVectorStore(dimension=8)
    sample_docs = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning involves training algorithms on data to make predictions.",
        "Retrieval-augmented generation combines neural networks with information retrieval.",
    ]

    # Embed and add documents to store
    doc_vectors = await embedding_provider.embed(sample_docs)
    for i, (doc_text, vector) in enumerate(zip(sample_docs, doc_vectors)):
        await vector_store.add(f"doc_{i}", vector, metadata={"text": doc_text})

    # Create FakeProvider with pre-configured response
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

    # Create Agent Entity
    agent_id = world.create_entity()
    world.add_component(
        agent_id,
        LLMComponent(
            provider=provider,
            model="fake",
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
        EmbeddingComponent(provider=embedding_provider, dimension=8),
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
