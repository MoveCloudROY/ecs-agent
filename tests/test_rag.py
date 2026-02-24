from __future__ import annotations

import pytest

from ecs_agent.components import (
    ConversationComponent,
    EmbeddingComponent,
    KVStoreComponent,
    RAGTriggerComponent,
    VectorStoreComponent,
)
from ecs_agent.core import World
from ecs_agent.providers.fake_embedding_provider import FakeEmbeddingProvider
from ecs_agent.providers.vector_store import InMemoryVectorStore
from ecs_agent.systems.rag import RAGSystem
from ecs_agent.types import Message, RAGRetrievalCompletedEvent


class RecordingEmbeddingProvider(FakeEmbeddingProvider):
    def __init__(self, dimension: int = 8) -> None:
        super().__init__(dimension=dimension)
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return await super().embed(texts)


@pytest.mark.asyncio
async def test_rag_trigger_with_query_retrieves_and_injects_context() -> None:
    world = World()
    entity_id = world.create_entity()
    provider = FakeEmbeddingProvider(dimension=8)
    store = InMemoryVectorStore(dimension=8)
    vector = (await provider.embed(["Python is great"]))[0]
    await store.add("doc-1", vector, metadata={"text": "Python is great"})

    world.add_component(entity_id, RAGTriggerComponent(query="Python", top_k=1))
    world.add_component(entity_id, EmbeddingComponent(provider=provider, dimension=8))
    world.add_component(entity_id, VectorStoreComponent(store=store))
    world.add_component(
        entity_id,
        ConversationComponent(
            messages=[Message(role="user", content="Tell me about Python")]
        ),
    )

    await RAGSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    trigger = world.get_component(entity_id, RAGTriggerComponent)
    assert conversation is not None
    assert trigger is not None
    assert conversation.messages[0].role == "system"
    assert conversation.messages[0].content.startswith("[RAG Context] ")
    assert "Python is great" in conversation.messages[0].content
    assert conversation.messages[-1] == Message(
        role="user",
        content="Tell me about Python",
    )
    assert trigger.retrieved_docs == ["Python is great"]
    assert trigger.query == ""


@pytest.mark.asyncio
async def test_empty_query_skips_retrieval() -> None:
    world = World()
    entity_id = world.create_entity()
    provider = RecordingEmbeddingProvider(dimension=8)

    world.add_component(entity_id, RAGTriggerComponent(query=""))
    world.add_component(entity_id, EmbeddingComponent(provider=provider, dimension=8))
    world.add_component(
        entity_id,
        VectorStoreComponent(store=InMemoryVectorStore(dimension=8)),
    )
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Hi")]),
    )

    await RAGSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    trigger = world.get_component(entity_id, RAGTriggerComponent)
    assert conversation is not None
    assert trigger is not None
    assert provider.calls == []
    assert conversation.messages == [Message(role="user", content="Hi")]
    assert trigger.retrieved_docs == []


@pytest.mark.asyncio
async def test_entity_without_rag_trigger_component_is_skipped() -> None:
    world = World()
    entity_id = world.create_entity()
    world.add_component(entity_id, KVStoreComponent(store={"k": "v"}))

    await RAGSystem().process(world)

    kv_component = world.get_component(entity_id, KVStoreComponent)
    assert kv_component is not None
    assert kv_component.store == {"k": "v"}


@pytest.mark.asyncio
async def test_retrieved_docs_are_formatted_as_system_messages_with_prefix() -> None:
    world = World()
    entity_id = world.create_entity()
    provider = FakeEmbeddingProvider(dimension=8)
    store = InMemoryVectorStore(dimension=8)

    vector_1 = (await provider.embed(["Doc one"]))[0]
    vector_2 = (await provider.embed(["Doc two"]))[0]
    await store.add("doc-1", vector_1, metadata={"text": "Doc one"})
    await store.add("doc-2", vector_2, metadata={"text": "Doc two"})

    world.add_component(entity_id, RAGTriggerComponent(query="Doc", top_k=2))
    world.add_component(entity_id, EmbeddingComponent(provider=provider, dimension=8))
    world.add_component(entity_id, VectorStoreComponent(store=store))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Need docs")]),
    )

    await RAGSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    rag_messages = [m for m in conversation.messages if m.role == "system"]
    assert len(rag_messages) == 2
    assert rag_messages[0].content.startswith("[RAG Context] ")
    assert rag_messages[1].content.startswith("[RAG Context] ")


@pytest.mark.asyncio
async def test_rag_messages_inserted_before_last_user_message() -> None:
    world = World()
    entity_id = world.create_entity()
    provider = FakeEmbeddingProvider(dimension=8)
    store = InMemoryVectorStore(dimension=8)
    vector = (await provider.embed(["Background fact"]))[0]
    await store.add("doc-1", vector, metadata={"text": "Background fact"})

    messages = [
        Message(role="user", content="first"),
        Message(role="assistant", content="reply"),
        Message(role="user", content="latest"),
    ]
    world.add_component(entity_id, RAGTriggerComponent(query="latest", top_k=1))
    world.add_component(entity_id, EmbeddingComponent(provider=provider, dimension=8))
    world.add_component(entity_id, VectorStoreComponent(store=store))
    world.add_component(entity_id, ConversationComponent(messages=list(messages)))

    await RAGSystem().process(world)

    conversation = world.get_component(entity_id, ConversationComponent)
    assert conversation is not None
    assert conversation.messages[-1] == Message(role="user", content="latest")
    assert conversation.messages[-2].role == "system"
    assert conversation.messages[-2].content.startswith("[RAG Context] ")


@pytest.mark.asyncio
async def test_retrieval_publishes_rag_completed_event() -> None:
    world = World()
    entity_id = world.create_entity()
    provider = FakeEmbeddingProvider(dimension=8)
    store = InMemoryVectorStore(dimension=8)
    vector = (await provider.embed(["RAG event doc"]))[0]
    await store.add("doc-1", vector, metadata={"text": "RAG event doc"})

    seen: list[RAGRetrievalCompletedEvent] = []

    async def handler(event: RAGRetrievalCompletedEvent) -> None:
        seen.append(event)

    world.event_bus.subscribe(RAGRetrievalCompletedEvent, handler)
    world.add_component(entity_id, RAGTriggerComponent(query="RAG", top_k=1))
    world.add_component(entity_id, EmbeddingComponent(provider=provider, dimension=8))
    world.add_component(entity_id, VectorStoreComponent(store=store))
    world.add_component(
        entity_id,
        ConversationComponent(messages=[Message(role="user", content="Tell me")]),
    )

    await RAGSystem().process(world)

    assert len(seen) == 1
    assert seen[0].entity_id == entity_id
    assert seen[0].query == "RAG"
    assert seen[0].num_results == 1


def test_default_priority_is_lower_than_reasoning_system() -> None:
    assert RAGSystem().priority == -10
