"""Shared tool policy semantics for execution, installation and compilation."""

from ecs_agent.components import PermissionComponent


def tool_is_allowed(policy: PermissionComponent | None, name: str) -> bool:
    """Check deny-first policy, preserving unrestricted versus empty allowlists."""
    if policy is None:
        return True
    return name not in policy.denied_tools and (
        policy.allowed_tools is None or name in policy.allowed_tools
    )


def compile_tool_permissions(tools: dict[str, bool] | None) -> PermissionComponent | None:
    """Compile an optional closed allowlist without losing explicit denials."""
    if tools is None:
        return None
    return PermissionComponent(
        allowed_tools=[name for name, enabled in tools.items() if enabled],
        denied_tools=[name for name, enabled in tools.items() if not enabled],
    )


def intersect_permissions(
    inherited: PermissionComponent | None, local: PermissionComponent | None,
) -> PermissionComponent | None:
    """Combine inherited and local policies without widening either policy."""
    if inherited is None and local is None:
        return None
    allowed: list[str] | None = None
    denied: list[str] = []
    for policy in (inherited, local):
        if policy is None:
            continue
        if policy.allowed_tools is not None:
            allowed = list(policy.allowed_tools) if allowed is None else [
                name for name in allowed if name in policy.allowed_tools
            ]
        denied.extend(name for name in policy.denied_tools if name not in denied)
    return PermissionComponent(allowed_tools=allowed, denied_tools=denied)
