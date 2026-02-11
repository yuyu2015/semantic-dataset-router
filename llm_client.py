"""
OpenRouter LLM client: send prompts and parse responses in a structured format.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from openai import OpenAI

from config import AppSettings, get_settings


class LLMResponse(BaseModel):
    """Parsed response from the LLM."""

    content: str
    model: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_message: dict = Field(description="Full message object for debugging")


class OpenRouterClient:
    """
    Client for OpenRouter chat completions (OpenAI-compatible API).
    Uses AppSettings (OPENROUTER_API_KEY, etc.) when not overridden.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        settings: AppSettings | None = None,
    ):
        s = settings or get_settings()
        self._api_key = api_key if api_key is not None else s.openrouter_api_key
        if not self._api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY in .env or pass api_key=..."
            )
        self._client = OpenAI(
            base_url=base_url or s.openrouter_base_url,
            api_key=self._api_key,
        )
        self.default_model = default_model if default_model is not None else s.default_llm_model

    def chat(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_message: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Send the prompt to OpenRouter and return a parsed LLMResponse.

        :param prompt: User content (or full prompt including instruction).
        :param model: OpenRouter model id (e.g. openai/gpt-4o-mini). Uses default if None.
        :param system_message: Optional system message (some models use this for behavior).
        :param temperature: Sampling temperature.
        :param max_tokens: Maximum completion tokens.
        :return: LLMResponse with content, usage, model, finish_reason.
        """
        model = model or self.default_model
        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        usage = response.usage

        raw = getattr(message, "model_dump", None)
        raw_message = raw() if callable(raw) else {"role": getattr(message, "role", ""), "content": content}

        return LLMResponse(
            content=content,
            model=response.model,
            finish_reason=choice.finish_reason or "unknown",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            raw_message=raw_message,
        )


def create_client(
    api_key: str | None = None,
    default_model: str | None = None,
    settings: AppSettings | None = None,
) -> OpenRouterClient:
    """Create an OpenRouter client with optional overrides."""
    return OpenRouterClient(
        api_key=api_key,
        default_model=default_model,
        settings=settings,
    )
