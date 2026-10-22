"""Per-test model ownership using the production provider constructors."""

from contextlib import AsyncExitStack
from typing import Any, TypeVar

from ecs_agent.providers import Model
from ecs_agent.providers.claude_model import ClaudeModel
from ecs_agent.providers.config import ApiFormat
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.providers.protocol import LLMModel
from tests.support.config import LiveSettings, Profile

ModelT = TypeVar("ModelT", bound=LLMModel)


class LiveModelFactory:
    """Own clients until fixture teardown, including partially failed scenarios."""

    def __init__(self, settings: LiveSettings, role: str = "primary") -> None:
        self.settings = settings
        self.profile = settings.profile(role)
        self._stack = AsyncExitStack()
        self._owned: set[int] = set()

    def track(self, model: ModelT) -> ModelT:
        """Register a directly constructed model without changing its API path."""
        if id(model) not in self._owned:
            self._owned.add(id(model))
            self._stack.push_async_callback(model.aclose)
        return model

    async def aclose(self) -> None:
        """Close all registered models even if an earlier close fails."""
        await self._stack.aclose()

    def key(self, role: str | None = None) -> str:
        """Resolve the current or explicitly requested role's credential."""
        return self._profile(role).key(self.settings.environment)

    def _profile(self, role: str | None) -> Profile:
        return self.profile if role is None else self.settings.profile(role)

    def connection(self, role: str | None = None) -> dict[str, Any]:
        """Model keyword arguments for direct constructor contract tests."""
        p = self._profile(role)
        return dict(
            base_url=p.base_url,
            api_key=self.key(role),
            timeout=p.timeouts.request_seconds,
            stream_read_timeout=p.timeouts.stream_read_seconds,
            enable_store=False,
        )

    def registry_entry(self, role: str | None = None) -> dict[str, Any]:
        """Non-secret configuration for direct registry contract tests."""
        p = self._profile(role)
        return dict(
            base_url=p.base_url,
            api_format=p.api_format.value,
            timeout=p.timeouts.request_seconds,
            enable_store=False,
        )

    def create(self, *, role: str | None = None, **kwargs: Any) -> LLMModel:
        """Create a model without allowing connection fields to be mixed."""
        p = self._profile(role)
        forbidden = {
            "model",
            "model_id",
            "api_key",
            "base_url",
            "api_format",
            "timeout",
            "stream_read_timeout",
        } & kwargs.keys()
        if forbidden:
            raise ValueError(
                f"Use a separate profile to change connection fields: {sorted(forbidden)}"
            )
        options = {**self.connection(role), **kwargs}
        return self.track(Model(p.model, api_format=p.api_format, **options))

    def openai(self, *, role: str | None = None, **kwargs: Any) -> OpenAIModel:
        """Create an OpenAI-family client with the selected profile's protocol."""
        if self._profile(role).api_format is ApiFormat.ANTHROPIC_MESSAGES:
            raise ValueError("This scenario requires a Chat or Responses profile")
        model = self.create(role=role, **kwargs)
        assert isinstance(model, OpenAIModel)
        return model

    def claude(self, **kwargs: Any) -> ClaudeModel:
        """Create an Anthropic client only for an Anthropic profile."""
        if self.profile.api_format is not ApiFormat.ANTHROPIC_MESSAGES:
            raise ValueError("This scenario requires an Anthropic profile")
        model = self.create(**kwargs)
        assert isinstance(model, ClaudeModel)
        return model
