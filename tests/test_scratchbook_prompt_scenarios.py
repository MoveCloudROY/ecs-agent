"""Offline scenario coverage extracted from formerly credential-gated files."""

from __future__ import annotations
import pytest
from ecs_agent.components import LLMComponent
from ecs_agent.components.definitions import RenderedSystemPromptComponent
from ecs_agent.core import World
from ecs_agent.prompts.contracts import PromptTemplateSource, SystemPromptConfigSpec
from ecs_agent.providers.config import ApiFormat
from ecs_agent.scratchbook.prompt_definition import (
    ScratchbookArtifactPromptDef,
    ScratchbookPromptConfig,
)
from ecs_agent.systems.system_prompt_render_system import SystemPromptRenderSystem
from ecs_agent.providers.fake_model import FakeModel

_ENDPOINT_PARAMS = [
    pytest.param(ApiFormat.OPENAI_CHAT_COMPLETIONS, id="chat-completions"),
    pytest.param(ApiFormat.OPENAI_RESPONSES, id="responses"),
]


def _build_scratchbook_prompt_config() -> ScratchbookPromptConfig:
    return ScratchbookPromptConfig(
        overview_default_template=(
            "Scratchbook path: ${scratchbook_path}\n"
            "Artifact types:\n${artifact_types}\n"
            "Use builtin tools to inspect or update artifacts when allowed."
        ),
        scratchbook_root_path=".sisyphus/notepads/scratchbook-prompt-model",
        artifacts=[
            ScratchbookArtifactPromptDef(
                artifact_type_id="plan notes",
                path=".sisyphus/notepads/scratchbook-prompt-model/plan.md",
                purpose="Tracks active task intent and rationale.",
                readonly=False,
                read_when="Before changing task behavior.",
            ),
            ScratchbookArtifactPromptDef(
                artifact_type_id="run log",
                path=".sisyphus/notepads/scratchbook-prompt-model/live-run.log",
                purpose="Stores immutable live execution traces.",
                readonly=True,
                read_when="When validating endpoint-mode behavior.",
            ),
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("api_format", _ENDPOINT_PARAMS)
async def test_live_scratchbook_placeholders_render_into_system_prompt_snapshot(
    api_format: ApiFormat,
) -> None:
    model = FakeModel([])
    world = World()
    entity_id = world.create_entity()

    world.add_component(
        entity_id,
        LLMComponent(
            model=model,
        ),
    )
    world.add_component(
        entity_id,
        SystemPromptConfigSpec(
            template_source=PromptTemplateSource(
                inline=(
                    "SB-LIVE-CHECK\n"
                    "${_scratchbook_overview}\n"
                    "plan_path=${_scratchbook_artifact_path_plan_notes}\n"
                    "run_log_path=${_scratchbook_artifact_path_run_log}"
                )
            )
        ),
    )
    world.add_component(entity_id, _build_scratchbook_prompt_config())

    await SystemPromptRenderSystem().process(world)

    rendered = world.get_component(entity_id, RenderedSystemPromptComponent)
    assert rendered is not None
    assert "${_scratchbook_overview}" not in rendered.text
    assert "SB-LIVE-CHECK" in rendered.text
    assert (
        "Scratchbook path: .sisyphus/notepads/scratchbook-prompt-model" in rendered.text
    )
    assert (
        "plan_path=.sisyphus/notepads/scratchbook-prompt-model/plan.md" in rendered.text
    )
    assert (
        "run_log_path=.sisyphus/notepads/scratchbook-prompt-model/live-run.log"
        in rendered.text
    )
    assert {
        "_scratchbook_path",
        "_scratchbook_artifact_types",
        "_scratchbook_artifacts",
        "_scratchbook_overview",
        "_scratchbook_artifact_plan_notes",
        "_scratchbook_artifact_path_plan_notes",
        "_scratchbook_artifact_run_log",
        "_scratchbook_artifact_path_run_log",
    }.issubset(set(rendered.placeholder_snapshot))
