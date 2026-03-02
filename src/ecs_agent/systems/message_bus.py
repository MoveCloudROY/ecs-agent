"""Message bus system with bounded pub/sub buffering."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any

from ecs_agent.components import (
    MessageBusConfigComponent,
    MessageBusConversationComponent,
    MessageBusSubscriptionComponent,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger, log_bus_deliver, log_bus_publish
from ecs_agent.observability import extract_trace_id, generate_traceparent
from ecs_agent.types import (
    EntityId,
    Message,
    MessageBusDeliveredEvent,
    MessageBusEnvelope,
    MessageBusPublishedEvent,
)

logger = get_logger(__name__)


class MessageBusSystem:
    """Core publish/subscribe buffering behavior for the message bus."""

    def __init__(
        self,
        *,
        buffer_size: int = 1000,
        publish_timeout: float = 2.0,
        request_timeout: float = 30.0,
        dead_subscriber_failures: int = 3,
    ) -> None:
        self._config = MessageBusConfigComponent(
            max_queue_size=buffer_size,
            publish_timeout=publish_timeout,
            request_timeout=request_timeout,
        )
        self._dead_subscriber_failures = dead_subscriber_failures
        self._subscriptions: dict[str, set[str]] = {}
        self._queues: dict[str, dict[str, asyncio.Queue[Any]]] = {}
        self._subscriber_failures: dict[tuple[str, str], int] = {}
        self._conversations: dict[str, MessageBusConversationComponent] = {}
        self._pending_requests: dict[str, asyncio.Future[MessageBusEnvelope]] = {}
        self._world: World | None = None
        self._bus_entity_id: EntityId | None = None

    async def process(self, world: World) -> None:
        self._world = world

        config_rows = world.query(MessageBusConfigComponent)
        if config_rows:
            entity_id, (config,) = config_rows[0]
            self._bus_entity_id = entity_id
            self._config = config

        subscription_rows = world.query(MessageBusSubscriptionComponent)
        if subscription_rows:
            _, (subscription_component,) = subscription_rows[0]
            self._subscriptions = {
                topic: set(subscribers)
                for topic, subscribers in subscription_component.subscriptions.items()
            }

        conversation_rows = world.query(MessageBusConversationComponent)
        self._conversations = {
            str(entity_id): conversation
            for entity_id, (conversation,) in conversation_rows
        }

    def subscribe(self, topic: str, subscriber_id: str) -> asyncio.Queue[Any]:
        subscribers = self._subscriptions.setdefault(topic, set())
        subscribers.add(subscriber_id)

        topic_queues = self._queues.setdefault(topic, {})
        queue = topic_queues.get(subscriber_id)
        if queue is None:
            queue = asyncio.Queue(maxsize=self._config.max_queue_size)
            topic_queues[subscriber_id] = queue

        return queue

    async def publish(
        self,
        topic: str,
        envelope: MessageBusEnvelope | None = None,
        message: dict[str, Any] | None = None,
    ) -> None:
        if envelope is None:
            if message is None:
                raise ValueError("publish requires envelope or message")
            envelope = self._build_envelope(message)

        trace_id = self._safe_trace_id(envelope.traceparent)
        correlation_id = envelope.correlationid
        payload = envelope.data if envelope.data is not None else envelope

        log_bus_publish(
            logger=logger,
            topic=topic,
            trace_id=trace_id,
            correlation_id=correlation_id,
            payload_type=type(payload).__name__,
        )
        await self._emit_published_event(topic=topic, envelope=envelope)

        for subscriber_id in sorted(self._subscriptions.get(topic, set())):
            queue = self._queues.setdefault(topic, {}).get(subscriber_id)
            if queue is None:
                queue = self.subscribe(topic=topic, subscriber_id=subscriber_id)

            await self._deliver_to_subscriber(
                topic=topic,
                subscriber_id=subscriber_id,
                queue=queue,
                payload=payload,
                envelope=envelope,
                trace_id=trace_id,
                correlation_id=correlation_id,
            )

    async def _deliver_to_subscriber(
        self,
        *,
        topic: str,
        subscriber_id: str,
        queue: asyncio.Queue[Any],
        payload: Any,
        envelope: MessageBusEnvelope,
        trace_id: str,
        correlation_id: str,
    ) -> None:
        try:
            await asyncio.wait_for(
                queue.put(payload),
                timeout=self._config.publish_timeout,
            )
        except TimeoutError as exc:
            failures_key = (topic, subscriber_id)
            failures = self._subscriber_failures.get(failures_key, 0) + 1
            self._subscriber_failures[failures_key] = failures

            logger.error(
                "bus_delivery_timeout",
                topic=topic,
                subscriber_id=subscriber_id,
                trace_id=trace_id,
                correlation_id=correlation_id,
                publish_timeout_seconds=self._config.publish_timeout,
                consecutive_failures=failures,
                exception=str(exc),
            )

            if failures >= self._dead_subscriber_failures:
                self._remove_subscriber(topic=topic, subscriber_id=subscriber_id)

            raise

        self._subscriber_failures[(topic, subscriber_id)] = 0
        log_bus_deliver(
            logger=logger,
            topic=topic,
            subscriber_id=subscriber_id,
            trace_id=trace_id,
            correlation_id=correlation_id,
        )
        self._append_conversation_message(
            subscriber_id=subscriber_id, envelope=envelope
        )
        await self._emit_delivered_event(envelope=envelope, subscriber_id=subscriber_id)

    def _append_conversation_message(
        self,
        *,
        subscriber_id: str,
        envelope: MessageBusEnvelope,
    ) -> None:
        conversation = self._conversations.get(subscriber_id)
        if conversation is None:
            return

        conversation.messages.append(self._as_message(envelope))
        if len(conversation.messages) > conversation.max_messages:
            overflow = len(conversation.messages) - conversation.max_messages
            if overflow > 0:
                del conversation.messages[:overflow]

    def _as_message(self, envelope: MessageBusEnvelope) -> Message:
        payload = envelope.data
        if isinstance(payload, str):
            content = payload
        else:
            content = json.dumps(payload, default=str)
        return Message(role="system", content=content)

    def _remove_subscriber(self, *, topic: str, subscriber_id: str) -> None:
        topic_subscribers = self._subscriptions.get(topic)
        if topic_subscribers is not None:
            topic_subscribers.discard(subscriber_id)
            if not topic_subscribers:
                del self._subscriptions[topic]

        topic_queues = self._queues.get(topic)
        if topic_queues is not None:
            topic_queues.pop(subscriber_id, None)
            if not topic_queues:
                del self._queues[topic]

        self._subscriber_failures.pop((topic, subscriber_id), None)
        logger.warning(
            "bus_subscriber_removed", topic=topic, subscriber_id=subscriber_id
        )

    async def _emit_published_event(
        self,
        *,
        topic: str,
        envelope: MessageBusEnvelope,
    ) -> None:
        if self._world is None:
            return

        entity_id = (
            self._bus_entity_id if self._bus_entity_id is not None else EntityId(0)
        )
        await self._world.event_bus.publish(
            MessageBusPublishedEvent(
                entity_id=entity_id, envelope=envelope, topic=topic
            )
        )

    async def _emit_delivered_event(
        self,
        *,
        envelope: MessageBusEnvelope,
        subscriber_id: str,
    ) -> None:
        if self._world is None:
            return

        entity_id = (
            self._bus_entity_id if self._bus_entity_id is not None else EntityId(0)
        )
        await self._world.event_bus.publish(
            MessageBusDeliveredEvent(
                entity_id=entity_id,
                subscriber_id=self._subscriber_entity_id(subscriber_id),
                envelope=envelope,
            )
        )

    def _subscriber_entity_id(self, subscriber_id: str) -> EntityId:
        if subscriber_id.isdigit():
            return EntityId(int(subscriber_id))
        return self._bus_entity_id if self._bus_entity_id is not None else EntityId(0)

    def _build_envelope(self, payload: dict[str, Any]) -> MessageBusEnvelope:
        return MessageBusEnvelope(
            id=str(uuid.uuid4()),
            source="ecs://message-bus",
            type="ecs.bus.publish",
            specversion="1.0",
            correlationid=str(uuid.uuid4()),
            traceparent=generate_traceparent(),
            data=payload,
            time=datetime.now(),
        )

    def _safe_trace_id(self, traceparent: str) -> str:
        try:
            return extract_trace_id(traceparent)
        except ValueError:
            return traceparent

    async def request(self, topic: str, message: dict[str, Any]) -> dict[str, Any]:
        del topic
        del message
        raise TimeoutError("request/response not implemented in this task")

    async def respond(self, correlation_id: str, message: dict[str, Any]) -> bool:
        del correlation_id
        del message
        return False
