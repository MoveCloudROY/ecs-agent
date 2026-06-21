"""Message bus system with bounded pub/sub buffering."""

from __future__ import annotations

import asyncio
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
    MessageBusResponseEvent,
    MessageBusTimeoutEvent,
)

logger = get_logger(__name__)


class MessageBusSystem:
    """Core publish/subscribe buffering behavior for the message bus."""

    def __init__(
        self,
        *,
        priority: int = 5,
        buffer_size: int = 1000,
        publish_timeout: float = 2.0,
        request_timeout: float = 30.0,
        dead_subscriber_failures: int = 3,
    ) -> None:
        self.priority = priority
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
        self._pending_request_topics: dict[str, str] = {}
        self._pending_request_event_ids: dict[str, str] = {}
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
    ) -> bool:
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

        delivered = False
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
            delivered = True

        if delivered:
            self._append_published_message(envelope=envelope)

        return delivered

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
        await self._emit_delivered_event(envelope=envelope, subscriber_id=subscriber_id)

    def _append_published_message(self, *, envelope: MessageBusEnvelope) -> None:
        conversation = self._conversations.get(envelope.source)
        if conversation is None:
            return

        conversation.messages.append(
            Message(role="user", content=f"From: {envelope.source}: {envelope.data}")
        )
        if len(conversation.messages) > conversation.max_messages:
            overflow = len(conversation.messages) - conversation.max_messages
            if overflow > 0:
                del conversation.messages[:overflow]

    def get_conversation(self, entity_id: EntityId) -> list[Message]:
        conversation = self._conversations.get(str(entity_id))
        if conversation is None:
            return []
        return list(conversation.messages)

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

    @property
    def _default_entity_id(self) -> EntityId:
        return self._bus_entity_id if self._bus_entity_id is not None else EntityId(0)

    async def _emit_published_event(
        self,
        *,
        topic: str,
        envelope: MessageBusEnvelope,
    ) -> None:
        if self._world is None:
            return

        await self._world.event_bus.publish(
            MessageBusPublishedEvent(
                entity_id=self._default_entity_id, envelope=envelope, topic=topic
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

        await self._world.event_bus.publish(
            MessageBusDeliveredEvent(
                entity_id=self._default_entity_id,
                subscriber_id=self._subscriber_entity_id(subscriber_id),
                envelope=envelope,
            )
        )

    def _subscriber_entity_id(self, subscriber_id: str) -> EntityId:
        if subscriber_id.isdigit():
            return EntityId(int(subscriber_id))
        return self._default_entity_id

    def _build_envelope(self, payload: dict[str, Any]) -> MessageBusEnvelope:
        return MessageBusEnvelope(
            id=str(uuid.uuid4()),
            source=f"ecs://entity/{self._default_entity_id}",
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

    async def request(
        self,
        topic: str,
        message: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        correlation_id = str(uuid.uuid4())
        inbox_topic = f"ecs.bus.inbox.{correlation_id}"
        requester_id = f"requester-{correlation_id}"
        request_timeout = self._config.request_timeout if timeout is None else timeout

        self.subscribe(topic=inbox_topic, subscriber_id=requester_id)
        response_future: asyncio.Future[MessageBusEnvelope] = (
            asyncio.get_running_loop().create_future()
        )
        # Enforce max_pending_requests limit
        if len(self._pending_requests) >= self._config.max_pending_requests:
            raise ValueError(
                f"Max pending requests ({self._config.max_pending_requests}) exceeded. "
                f"Current: {len(self._pending_requests)}"
            )
        
        self._pending_requests[correlation_id] = response_future

        request_payload = dict(message)
        request_payload["reply_to"] = inbox_topic

        request_envelope = MessageBusEnvelope(
            id=str(uuid.uuid4()),
            source=f"ecs://entity/{self._bus_entity_id if self._bus_entity_id is not None else EntityId(0)}",
            type="ecs.bus.request",
            specversion="1.0",
            correlationid=correlation_id,
            traceparent=generate_traceparent(),
            data=request_payload,
            subject=topic,
            time=datetime.now(),
        )

        self._pending_request_topics[correlation_id] = inbox_topic
        self._pending_request_event_ids[correlation_id] = request_envelope.id

        await self.publish(topic=topic, envelope=request_envelope)

        try:
            response = await asyncio.wait_for(
                asyncio.shield(response_future),
                timeout=request_timeout,
            )
        except asyncio.TimeoutError as exc:
            from ecs_agent.logging import log_bus_timeout
            log_bus_timeout(
                logger=logger,
                request_id=request_envelope.id,
                trace_id=self._safe_trace_id(request_envelope.traceparent),
                correlation_id=correlation_id,
                timeout_seconds=request_timeout,
            )
            await self._emit_timeout_event(correlation_id=correlation_id)
            raise TimeoutError(
                f"message bus request timed out after {request_timeout}s"
            ) from exc
        finally:
            self._pending_requests.pop(correlation_id, None)
            self._pending_request_topics.pop(correlation_id, None)
            self._pending_request_event_ids.pop(correlation_id, None)
            self._remove_subscriber(topic=inbox_topic, subscriber_id=requester_id)

        if isinstance(response.data, dict):
            return response.data

        return {"data": response.data}

    async def respond(self, correlation_id: str, message: dict[str, Any]) -> bool:
        response_future = self._pending_requests.get(correlation_id)
        if response_future is None:
            logger.info(
                "bus_response_ignored",
                correlation_id=correlation_id,
                reason="unknown_or_expired_request",
            )
            return False

        if response_future.done():
            logger.info(
                "bus_response_ignored",
                correlation_id=correlation_id,
                reason="duplicate_or_late_response",
            )
            return False

        response_topic = self._pending_request_topics.get(correlation_id)
        if response_topic is None:
            logger.info(
                "bus_response_ignored",
                correlation_id=correlation_id,
                reason="missing_response_topic",
            )
            return False

        response_envelope = MessageBusEnvelope(
            id=str(uuid.uuid4()),
            source=f"ecs://entity/{self._bus_entity_id if self._bus_entity_id is not None else EntityId(0)}",
            type="ecs.bus.response",
            specversion="1.0",
            correlationid=correlation_id,
            causationid=self._pending_request_event_ids.get(correlation_id),
            traceparent=generate_traceparent(),
            data=dict(message),
            subject=response_topic,
            time=datetime.now(),
        )

        await self.publish(topic=response_topic, envelope=response_envelope)

        if response_future.done():
            logger.info(
                "bus_response_ignored",
                correlation_id=correlation_id,
                reason="duplicate_or_late_response",
            )
            return False

        response_future.set_result(response_envelope)
        logger.info(
            "bus_response_delivered",
            topic=response_topic,
            correlation_id=correlation_id,
            trace_id=self._safe_trace_id(response_envelope.traceparent),
        )
        await self._emit_response_event(
            correlation_id=correlation_id,
            envelope=response_envelope,
        )
        return True

    async def _emit_timeout_event(self, *, correlation_id: str) -> None:
        if self._world is None:
            return

        await self._world.event_bus.publish(
            MessageBusTimeoutEvent(
                entity_id=self._default_entity_id,
                correlation_id=correlation_id,
            )
        )

    async def _emit_response_event(
        self,
        *,
        correlation_id: str,
        envelope: MessageBusEnvelope,
    ) -> None:
        if self._world is None:
            return

        await self._world.event_bus.publish(
            MessageBusResponseEvent(
                entity_id=self._default_entity_id,
                correlation_id=correlation_id,
                envelope=envelope,
            )
        )
