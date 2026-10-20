"""Dependency validation and repeat-safe system registration for authoring."""

from ecs_agent.authoring.definitions import InstalledStep, SystemStep
from ecs_agent.core.world import World
from ecs_agent.logging import get_logger

logger = get_logger(__name__)


def resolve_steps(steps: tuple[SystemStep, ...]) -> tuple[tuple[SystemStep, int], ...]:
    """Resolve a dependency graph without invoking system factories."""
    by_name: dict[str, SystemStep] = {}
    for step in steps:
        if not step.name or not callable(step.factory):
            raise ValueError("System steps require a name and a callable factory")
        if step.priority is not None and (isinstance(step.priority, bool) or not isinstance(step.priority, int)):
            raise ValueError(f"System step '{step.name}': priority must be an integer")
        if step.name in by_name and by_name[step.name] != step:
            raise ValueError(f"Conflicting system role: {step.name}")
        by_name[step.name] = step
    parents: dict[str, set[str]] = {name: set() for name in by_name}
    for name, step in by_name.items():
        for dependency in (*step.after, *step.before):
            if dependency not in by_name:
                raise ValueError(f"System step '{name}' references missing role '{dependency}'")
        parents[name].update(step.after)
        for following in step.before:
            parents[following].add(name)
    ordered: list[str] = []
    while len(ordered) < len(by_name):
        ready = sorted(name for name in by_name if name not in ordered and parents[name].issubset(ordered))
        if not ready:
            raise ValueError(f"System dependency cycle: {sorted(set(by_name) - set(ordered))}")
        ordered.extend(ready)
    ceilings: dict[str, int | None] = {}
    for name in reversed(ordered):
        candidates: list[int] = []
        for child in ordered:
            ceiling = ceilings.get(child)
            if name in parents[child] and ceiling is not None:
                candidates.append(ceiling - 1)
        fixed = by_name[name].priority
        if fixed is not None:
            candidates.append(fixed)
        ceilings[name] = min(candidates, default=None)
    priorities: dict[str, int] = {}
    for name in ordered:
        lower = max((priorities[parent] + 1 for parent in parents[name]), default=None)
        upper = ceilings[name]
        priority = by_name[name].priority
        if priority is None:
            priority = lower if lower is not None else min(0, upper) if upper is not None else 0
        if (lower is not None and priority < lower) or (upper is not None and priority > upper):
            raise ValueError(f"System step '{name}' has a priority incompatible with its dependencies")
        priorities[name] = priority
    return tuple((by_name[name], priorities[name]) for name in sorted(ordered, key=lambda name: (priorities[name], name)))


def install_steps(world: World, steps: tuple[SystemStep, ...], *, expected_types: dict[str, str] | None = None) -> tuple[InstalledStep, ...]:
    """Install named roles once, rejecting conflicts before registering anything."""
    if world._runner_task is not None or world._agent_run is not None or world._processing_task is not None:
        raise RuntimeError("Cannot install systems while a World is being driven")
    resolved = resolve_steps(steps)
    prepared = []
    report = []
    try:
        for step, priority in resolved:
            previous = world._authoring_systems.get(step.name)
            if previous is not None:
                old_step, old_priority, system_name = previous
                if old_step != step or old_priority != priority:
                    raise ValueError(f"Conflicting installed system role: {step.name}")
            else:
                system = step.factory()
                if not callable(getattr(system, "process", None)):
                    raise ValueError(f"System role '{step.name}' factory did not return a System")
                system_name = f"{type(system).__module__}.{type(system).__name__}"
                prepared.append((step, priority, system, system_name))
            if expected_types is not None and expected_types.get(step.name) != system_name:
                raise ValueError(f"System type for '{step.name}' differs from the checkpoint manifest")
            report.append(InstalledStep(step.name, system_name, priority, step.after, step.before))
    except Exception as exc:
        logger.error("runtime_install_failed", exception=str(exc))
        raise
    for step, priority, system, system_name in prepared:
        world.register_system(system, priority=priority)
        world._authoring_systems[step.name] = (step, priority, system_name)
        world._authoring_instances[step.name] = system
    return tuple(report)
