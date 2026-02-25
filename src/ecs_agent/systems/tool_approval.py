"""Tool approval system for pending tool calls."""

from __future__ import annotations

import asyncio
import time

from ecs_agent.components import (
    ConversationComponent,
    ErrorComponent,
    PendingToolCallsComponent,
    TerminalComponent,
    ToolApprovalComponent,
)
from ecs_agent.core.world import World
from ecs_agent.types import (
    ApprovalPolicy,
    EntityId,
    Message,
    ToolApprovedEvent,
    ToolApprovalRequestedEvent,
    ToolCall,
    ToolDeniedEvent,
)


class ToolApprovalSystem:
    """System for tool call approval flow."""

    def __init__(self, priority: int = -5) -> None:
        self.priority = priority

    async def process(self, world: World) -> None:
        for entity_id, components in world.query(
            PendingToolCallsComponent,
            ToolApprovalComponent,
            ConversationComponent,
        ):
            pending, approval, conversation = components
            assert isinstance(pending, PendingToolCallsComponent)
            assert isinstance(approval, ToolApprovalComponent)
            assert isinstance(conversation, ConversationComponent)

            try:
                approved_calls: list[ToolCall] = []
                for tool_call in pending.tool_calls:
                    if approval.policy is ApprovalPolicy.ALWAYS_APPROVE:
                        await self._approve(entity_id, tool_call, approval, world)
                        approved_calls.append(tool_call)
                        continue

                    if approval.policy is ApprovalPolicy.ALWAYS_DENY:
                        await self._deny(
                            entity_id,
                            tool_call,
                            approval,
                            conversation,
                            world,
                            "Denied by ALWAYS_DENY policy",
                        )
                        continue

                    if approval.policy is ApprovalPolicy.REQUIRE_APPROVAL:
                        decision = await self._request_approval(
                            entity_id,
                            tool_call,
                            approval.timeout,
                            world,
                        )
                        if decision:
                            await self._approve(entity_id, tool_call, approval, world)
                            approved_calls.append(tool_call)
                        else:
                            await self._deny(
                                entity_id,
                                tool_call,
                                approval,
                                conversation,
                                world,
                                "Denied by approval handler",
                            )

                if approved_calls:
                    pending.tool_calls = approved_calls
                else:
                    world.remove_component(entity_id, PendingToolCallsComponent)
            except Exception as exc:
                world.add_component(
                    entity_id,
                    ErrorComponent(
                        error=str(exc),
                        system_name="ToolApprovalSystem",
                        timestamp=time.time(),
                    ),
                )
                world.add_component(
                    entity_id,
                    TerminalComponent(reason="tool_approval_error"),
                )

    async def _request_approval(
        self,
        entity_id: EntityId,
        tool_call: ToolCall,
        timeout: float | None,
        world: World,
    ) -> bool:
        approval_future: asyncio.Future[bool] = (
            asyncio.get_running_loop().create_future()
        )
        event = ToolApprovalRequestedEvent(
            entity_id=entity_id,
            tool_call=tool_call,
            approval_future=approval_future,
        )
        await world.event_bus.publish(event)

        # EventBus.publish uses gather(return_exceptions=True), so handler failures
        # can be swallowed and the future may never be set; timeout defaults to deny.
        try:
            return await asyncio.wait_for(approval_future, timeout=timeout)
        except asyncio.TimeoutError:
            await self._publish_denied(
                world,
                entity_id,
                tool_call.id,
                f"Approval timeout after {timeout}s",
            )
            return False

    async def _approve(
        self,
        entity_id: EntityId,
        tool_call: ToolCall,
        approval: ToolApprovalComponent,
        world: World,
    ) -> None:
        approval.approved_calls.append(tool_call.id)
        await world.event_bus.publish(
            ToolApprovedEvent(entity_id=entity_id, tool_call_id=tool_call.id)
        )

    async def _deny(
        self,
        entity_id: EntityId,
        tool_call: ToolCall,
        approval: ToolApprovalComponent,
        conversation: ConversationComponent,
        world: World,
        reason: str,
    ) -> None:
        approval.denied_calls.append(tool_call.id)
        conversation.messages.append(
            Message(role="system", content=f"Tool call {tool_call.id} denied by policy")
        )
        await self._publish_denied(world, entity_id, tool_call.id, reason)

    async def _publish_denied(
        self,
        world: World,
        entity_id: EntityId,
        tool_call_id: str,
        reason: str,
    ) -> None:
        await world.event_bus.publish(
            ToolDeniedEvent(
                entity_id=entity_id, tool_call_id=tool_call_id, reason=reason
            )
        )


__all__ = ["ToolApprovalSystem"]
