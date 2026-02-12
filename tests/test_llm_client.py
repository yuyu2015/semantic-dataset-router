import pytest

import llm_client
from config import AppSettings
from llm_client import LLMResponse, OpenRouterClient


class FakeMessage:
    def __init__(self, content: str):
        self.role = "assistant"
        self.content = content

    def model_dump(self):
        return {"role": self.role, "content": self.content}


class FakeChoice:
    def __init__(self, content: str):
        self.finish_reason = "stop"
        self.message = FakeMessage(content)


class FakeUsage:
    def __init__(self):
        self.prompt_tokens = 5
        self.completion_tokens = 7
        self.total_tokens = 12


class FakeResponse:
    def __init__(self, content: str, model: str):
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage()
        self.model = model


class FakeCompletions:
    def __init__(self, model: str):
        self._model = model

    def create(self, model: str, messages, temperature: float, max_tokens: int):
        # Echo back a simple deterministic response that depends on model name
        text = f"model={model}, messages={len(messages)}"
        return FakeResponse(text, model=self._model)


class FakeChat:
    def __init__(self, model: str):
        self.completions = FakeCompletions(model)


class FakeOpenAI:
    def __init__(self, base_url: str, api_key: str):
        # Validate we received values from settings
        assert base_url
        assert api_key
        self._model = "fake/model"
        self.chat = FakeChat(self._model)


def test_openrouterclient_raises_when_no_api_key():
    """Constructing OpenRouterClient without an API key should raise."""
    settings = AppSettings(openrouter_api_key=None)
    with pytest.raises(ValueError):
        OpenRouterClient(settings=settings)


def test_openrouterclient_chat_uses_fake_openai(monkeypatch):
    """chat() should return a parsed LLMResponse without calling external services."""
    # Patch OpenAI class used inside llm_client
    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAI)

    settings = AppSettings(
        openrouter_api_key="test-key",
        openrouter_base_url="https://example.test/api",
        default_llm_model="provider/model",
    )

    client = OpenRouterClient(settings=settings)
    # Also exercise the system_message branch
    resp = client.chat("hello", system_message="sys", temperature=0.5, max_tokens=16)

    assert isinstance(resp, LLMResponse)
    assert "model=" in resp.content
    assert resp.model == "fake/model"
    assert resp.finish_reason == "stop"
    assert resp.prompt_tokens == 5
    assert resp.completion_tokens == 7
    assert resp.total_tokens == 12
    assert resp.raw_message["role"] == "assistant"


def test_create_client_helper_uses_openrouterclient(monkeypatch):
    """create_client should delegate to OpenRouterClient without hitting network."""
    monkeypatch.setattr(llm_client, "OpenAI", FakeOpenAI)

    settings = AppSettings(
        openrouter_api_key="key",
        openrouter_base_url="https://example.test/api",
        default_llm_model="provider/model",
    )

    client = llm_client.create_client(settings=settings)
    resp = client.chat("hi")
    assert isinstance(resp, LLMResponse)

