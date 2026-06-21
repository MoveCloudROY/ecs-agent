"""Skills module public API."""

from ecs_agent.skills.discovery import (
    DiscoveryManager,
    DiscoveryReport,
    SkillDiscovery,
    discover_skills,
)
from ecs_agent.skills.manager import SkillManager
from ecs_agent.skills.skill import Skill
from ecs_agent.skills.script_skill import ScriptSkill
from ecs_agent.skills.web_search import WebSearchSkill

__all__ = [
    "DiscoveryManager",
    "DiscoveryReport",
    "Skill",
    "ScriptSkill",
    "SkillDiscovery",
    "discover_skills",
    "SkillManager",
    "WebSearchSkill",
]
