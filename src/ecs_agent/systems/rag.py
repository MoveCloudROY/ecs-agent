from __future__ import annotations

from typing import Any

from ecs_agent.components import (
    ConversationComponent,
    EmbeddingComponent,
    RAGTriggerComponent,
    VectorStoreComponent,
)
from ecs_agent.core import World
from ecs_agent.types import Message, RAGRetrievalCompletedEvent


class RAGSystem:
    def __init__(self, priority: int = -10) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        for entity_id, components in world.query(
            RAGTriggerComponent,
            EmbeddingComponent,
            VectorStoreComponent,
            ConversationComponent,
        ):
            rag_trigger, embedding, vector_store, conversation = components
            assert isinstance(rag_trigger, RAGTriggerComponent)
            assert isinstance(embedding, EmbeddingComponent)
            assert isinstance(vector_store, VectorStoreComponent)
            assert isinstance(conversation, ConversationComponent)

            query = rag_trigger.query.strip()
            if query == "":
                continue

            vectors = await embedding.provider.embed([query])
            if not vectors:
                continue

            results = await vector_store.store.search(
                vectors[0], top_k=rag_trigger.top_k
            )

            retrieved_docs: list[str] = []
            rag_messages: list[Message] = []
            for doc_id, _score in results:
                text = _extract_text(vector_store.store, doc_id)
                if text is None:
                    continue
                retrieved_docs.append(text)
                rag_messages.append(
                    Message(role="system", content=f"[RAG Context] {text}")
                )

            if rag_messages:
                insert_at = _find_last_user_message_index(conversation.messages)
                conversation.messages[insert_at:insert_at] = rag_messages

            rag_trigger.retrieved_docs = retrieved_docs
            rag_trigger.query = ""

            await world.event_bus.publish(
                RAGRetrievalCompletedEvent(
                    entity_id=entity_id,
                    query=query,
                    num_results=len(retrieved_docs),
                )
            )


def _extract_text(store: Any, doc_id: str) -> str | None:
    metadata = getattr(store, "_metadata", None)
    if not isinstance(metadata, dict):
        return None

    raw = metadata.get(doc_id)
    if not isinstance(raw, dict):
        return None

    text = raw.get("text")
    if not isinstance(text, str):
        return None

    return text


def _find_last_user_message_index(messages: list[Message]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            return index
    return len(messages)


__all__ = ["RAGSystem"]
