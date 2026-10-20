"""Batch-scoped tool authorization shared by preflight systems and execution.

Decisions are ephemeral: checkpoints retain policy and audit history, never an
executable approval. Only ToolExecutionSystem consumes calls and lands results.
"""

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass, field

from ecs_agent.components import (
    PendingToolCallsComponent,
    PermissionComponent,
    ToolApprovalComponent,
    ToolRuntimeStateComponent,
    ToolStateNamespace,
)
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger
from ecs_agent.tools.permissions import tool_is_allowed
from ecs_agent.types import (
    ApprovalPolicy,
    EntityId,
    ToolApprovalRequestedEvent,
    ToolApprovedEvent,
    ToolCall,
    ToolDeniedEvent,
)

logger = get_logger(__name__)
_NAMESPACE = "ecs.tool_authorization"


@dataclass(slots=True, frozen=True)
class AuthorizationDecision:
    """Decision bound to the call payload and policies observed at approval."""

    fingerprint: str
    reason: str | None = None
    approval_checked: bool = True


@dataclass(slots=True)
class CallAuthorization:
    """One call's shared decision and one-time execution claim."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    decision: AuthorizationDecision | None = None
    consumed: bool = False
    denial_published: bool = False


@dataclass(slots=True)
class AuthorizationBatch:
    """An entity's current queue, with serialized ownership of result writes."""

    pending: PendingToolCallsComponent
    calls: dict[str, CallAuthorization] = field(default_factory=dict)
    execution_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    completed: bool = False

    def call_state(self, call: ToolCall) -> CallAuthorization:
        """Return shared state for a call in this batch."""
        return self.calls.setdefault(call.id, CallAuthorization())


def authorization_batch(
    world: World, entity: EntityId, pending: PendingToolCallsComponent,
) -> AuthorizationBatch:
    """Get entity-local, non-serializable authorization for exactly this queue."""
    runtime = world.get_component(entity, ToolRuntimeStateComponent)
    if runtime is None:
        runtime = ToolRuntimeStateComponent()
        world.add_component(entity, runtime)
    namespace = runtime.namespaces.setdefault(_NAMESPACE, ToolStateNamespace())
    batch = namespace.values.get("batch")
    if not isinstance(batch, AuthorizationBatch) or batch.pending is not pending:
        batch = AuthorizationBatch(pending=pending)
        namespace.values["batch"] = batch
    return batch


def authorization_fingerprint(world: World, entity: EntityId, call: ToolCall) -> str:
    """Capture payload and effective policy, including policy replacement."""
    permissions = world.get_component(entity, PermissionComponent)
    approval = world.get_component(entity, ToolApprovalComponent)
    return json.dumps(
        [
            call.id, call.name, call.arguments,
            None if permissions is None else [
                id(permissions), permissions.allowed_tools, permissions.denied_tools,
            ],
            None if approval is None else [id(approval), approval.policy.value, approval.timeout],
        ],
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


async def _request_approval(
    world: World, entity: EntityId, call: ToolCall, timeout: float | None,
) -> str | None:
    if not world.event_bus.has_subscribers(ToolApprovalRequestedEvent):
        return "No tool approval handler is registered"
    future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
    requested_call = deepcopy(call)
    event = ToolApprovalRequestedEvent(
        entity_id=entity, tool_call=requested_call, approval_future=future,
    )
    try:
        # Cover subscribers as well as the eventual answer with the timeout.
        async with asyncio.timeout(timeout):
            await world.event_bus.publish(event, raise_subscriber_errors=True)
            decision = await future
        if requested_call != call:
            return "Tool call changed during approval"
        return None if decision is True else "Denied by approval handler"
    except TimeoutError:
        return f"Approval timeout after {timeout}s"
    except Exception as exc:
        logger.error("tool_approval_failed", entity_id=int(entity), exception=str(exc))
        return "Tool approval handler failed"
    finally:
        if not future.done():
            future.cancel()


async def authorize_tool_call(
    world: World,
    entity: EntityId,
    pending: PendingToolCallsComponent,
    call: ToolCall,
    *,
    request_approval: bool = True,
) -> AuthorizationDecision:
    """Check current policy and share one approval request for an unchanged call.

    Cancellation propagates. Changed payload/policy invalidates cached decisions;
    continuously changing inputs are denied rather than retried forever.
    This function never consumes the queue or appends conversation messages.
    """
    batch = authorization_batch(world, entity, pending)
    state = batch.call_state(call)
    async with state.lock:
        for _ in range(3):
            if world.get_component(entity, PendingToolCallsComponent) is not pending:
                return AuthorizationDecision("", "Tool call queue was replaced")
            fingerprint = authorization_fingerprint(world, entity, call)
            cached = state.decision
            if cached is not None and (state.consumed or (
                cached.fingerprint == fingerprint
                and (cached.approval_checked or not request_approval)
            )):
                return cached

            permissions = world.get_component(entity, PermissionComponent)
            approval = world.get_component(entity, ToolApprovalComponent)
            reason: str | None = None
            checked = True
            if not tool_is_allowed(permissions, call.name):
                reason = f"tool '{call.name}' denied by permission policy"
            elif not request_approval:
                checked = False
            elif approval is not None:
                if approval.policy is ApprovalPolicy.ALWAYS_DENY:
                    reason = "Denied by ALWAYS_DENY policy"
                elif approval.policy is ApprovalPolicy.REQUIRE_APPROVAL:
                    reason = await _request_approval(world, entity, call, approval.timeout)
                elif approval.policy is not ApprovalPolicy.ALWAYS_APPROVE:
                    reason = "Unknown approval policy"

            if fingerprint != authorization_fingerprint(world, entity, call):
                continue
            decision = AuthorizationDecision(fingerprint, reason, checked)
            state.decision = decision
            if reason is None and checked and approval is not None:
                if call.id not in approval.approved_calls:
                    approval.approved_calls.append(call.id)
                await world.event_bus.publish(ToolApprovedEvent(
                    entity_id=entity, tool_call_id=call.id, tool_name=call.name,
                    policy=approval.policy.value,
                ))
                if fingerprint != authorization_fingerprint(world, entity, call):
                    continue
            return decision
        decision = AuthorizationDecision(
            authorization_fingerprint(world, entity, call),
            "Tool authorization changed repeatedly",
        )
        state.decision = decision
        return decision


async def publish_tool_denial(
    world: World, entity: EntityId, state: CallAuthorization, call: ToolCall, reason: str,
) -> None:
    """Publish a committed denial once; execution owns the result message."""
    if state.denial_published:
        return
    state.denial_published = True
    approval = world.get_component(entity, ToolApprovalComponent)
    if approval is not None and call.id not in approval.denied_calls:
        approval.denied_calls.append(call.id)
    await world.event_bus.publish(ToolDeniedEvent(
        entity_id=entity, tool_call_id=call.id, tool_name=call.name, reason=reason,
    ))
