"""Factory for creating LLMModel instances from ProviderConfig."""

from ecs_agent.providers.claude_model import ClaudeModel
from ecs_agent.providers.config import ApiFormat, ProviderConfig
from ecs_agent.providers.openai_model import OpenAIModel
from ecs_agent.providers.protocol import LLMModel


def create_model(config: ProviderConfig, model: str, **kwargs: object) -> LLMModel:
    """Create an LLMModel from a ProviderConfig and model name.

    Args:
        config: Provider configuration including base_url, api_key, api_format.
        model: Model name (e.g. 'gpt-4o', 'claude-3-5-haiku').
        **kwargs: Extra keyword arguments forwarded to the model constructor.

    Returns:
        An LLMModel instance that satisfies the LLMModel protocol.

    Raises:
        ValueError: If api_format is not supported for completion.
    """
    match config.api_format:
        case ApiFormat.OPENAI_CHAT_COMPLETIONS | ApiFormat.OPENAI_RESPONSES:
            return OpenAIModel(config=config, model=model, **kwargs)
        case ApiFormat.ANTHROPIC_MESSAGES:
            return ClaudeModel(config=config, model=model, **kwargs)
        case _:
            raise ValueError(
                f"api_format '{config.api_format.value}' is not supported by create_model"
            )
