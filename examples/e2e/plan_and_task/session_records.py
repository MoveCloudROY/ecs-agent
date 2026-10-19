"""Record-service ownership and explicit evidence sharing for workflow resumes."""

from pathlib import Path
from weakref import WeakKeyDictionary

from ecs_agent.components import SessionPolicyComponent, SessionRefComponent
from ecs_agent.core import World
from ecs_agent.logging import get_logger
from ecs_agent.scratchbook import ArtifactRegistry
from ecs_agent.sessions import SessionService, SQLiteSessionStore, install_session_recording
from ecs_agent.systems.subagent import SubagentSystem
from ecs_agent.types import EntityId
from examples.e2e.plan_and_task.state_models import RuntimeState

_RUNTIMES: WeakKeyDictionary[World, tuple[SubagentSystem, EntityId]] = WeakKeyDictionary()
logger = get_logger(__name__)


def install_session_records(world: World, entity: EntityId, base_dir: Path, runtime: SubagentSystem) -> None:
    """Let the root application own one durable store shared by every child World."""
    registry = ArtifactRegistry(base_dir)
    service = SessionService(SQLiteSessionStore(base_dir / "sessions.sqlite3"), artifact_registry=registry)
    try:
        install_session_recording(world, entity, service, workspace_id=str(base_dir.resolve()), agent_name="main")
    except Exception as exc:
        service.close()
        logger.error("plan_task_session_install_failed", exception=str(exc))
        raise
    _RUNTIMES[world] = runtime, entity


def remember_workflow_sessions(world: World, entity: EntityId, state: RuntimeState) -> None:
    """Persist workflow root IDs and grant access only to those task trees."""
    service = world.session_service
    ref = world.get_component(entity, SessionRefComponent)
    if service is None or ref is None:
        return
    if ref.session_id not in state.record_session_ids:
        state.record_session_ids.append(ref.session_id)
    current = service.store.get_session(ref.session_id)
    roots: set[str] = set()
    for session_id in state.record_session_ids:
        try:
            info = service.store.get_session(session_id)
        except FileNotFoundError:
            continue
        if info.workspace_id == current.workspace_id:
            roots.add(info.root_session_id)
    policy = world.get_component(entity, SessionPolicyComponent)
    if policy is None:
        policy = SessionPolicyComponent()
        world.add_component(entity, policy)
    policy.shared_session_ids = sorted(
        info.session_id for info in service.store.sessions()
        if info.workspace_id == current.workspace_id and info.root_session_id in roots
        and info.root_session_id != current.root_session_id
    )


async def close_session_records(world: World) -> None:
    """Settle background tasks before closing the root-owned record service."""
    runtime = _RUNTIMES.pop(world, None)
    try:
        if runtime is not None:
            system, entity = runtime
            sessions = await system._runtime_manager.get_all_sessions()
            for session_id, metadata in sessions.items():
                if metadata.status in {"queued", "running"}:
                    await system._runtime_manager.cancel_session(session_id)
            await system._runtime_manager.sync_to_component(world, entity)
    finally:
        if world.session_service is not None:
            world.session_service.close()
