from typer.testing import CliRunner

import main
from llm_client import LLMResponse
from router import RouterResult


runner = CliRunner()


def test_hello_command_runs():
    result = runner.invoke(main.app, ["hello"])
    assert result.exit_code == 0
    assert "Semantic Dataset Router" in result.stdout


class FakeRouter:
    def __init__(self, model_name: str = "x"):
        self.model_name = model_name

    def route(self, query: str, threshold: float = 0.25):
        # Simple deterministic result
        best_id = "shopping_habits" if threshold <= 0.5 else None
        best_score = 0.9 if best_id else 0.4
        return RouterResult(
            query=query,
            best_dataset_id=best_id,
            best_score=best_score,
            threshold=threshold,
            all_scores={"shopping_habits": best_score, "weekly_searches": 0.1},
            context_df=None,
            context_preview='[{"foo": 1}]' if best_id else "",
        )


class FakeOpenRouterClient:
    def __init__(self, default_model: str | None = None):
        self.default_model = default_model
        self.called = False

    def chat(self, prompt: str, model: str | None = None, temperature: float = 0.7, max_tokens: int = 1024):
        self.called = True
        return LLMResponse(
            content="answer from fake client",
            model=model or (self.default_model or "fake/model"),
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            raw_message={"role": "assistant", "content": "answer from fake client"},
        )


def test_route_command_uses_router(monkeypatch):
    monkeypatch.setattr(main, "DatasetRouter", FakeRouter)

    result = runner.invoke(main.app, ["route", "some question", "--threshold", "0.1"])
    assert result.exit_code == 0
    # Should show scores table and selected dataset
    assert "Similarity scores" in result.stdout
    assert "shopping_habits" in result.stdout


def test_route_command_handles_no_selected_dataset(monkeypatch):
    """Route command should print the 'none selected' branch when no dataset passes threshold."""
    monkeypatch.setattr(main, "DatasetRouter", FakeRouter)

    result = runner.invoke(main.app, ["route", "q", "--threshold", "0.9"])
    assert result.exit_code == 0
    assert "none" in result.stdout


def test_query_command_prints_prompt_without_llm(monkeypatch):
    monkeypatch.setattr(main, "DatasetRouter", FakeRouter)

    result = runner.invoke(
        main.app,
        ["query", "some question", "--instruction-version", "1", "--scores"],
    )
    assert result.exit_code == 0
    # Prompt panel title, routing info, and question should be present
    assert "Routing" in result.stdout
    assert "Prompt" in result.stdout
    assert "User question: some question" in result.stdout


def test_query_command_calls_llm_when_flag_set(monkeypatch):
    monkeypatch.setattr(main, "DatasetRouter", FakeRouter)
    fake_client = FakeOpenRouterClient(default_model="fake/model")
    monkeypatch.setattr(main, "OpenRouterClient", lambda default_model=None: fake_client)

    result = runner.invoke(
        main.app,
        ["query", "some question", "--call", "--instruction-version", "1"],
    )
    assert result.exit_code == 0
    assert "Prompt (sent to LLM)" in result.stdout
    assert "answer from fake client" in result.stdout


def test_query_command_exits_on_missing_api_key(monkeypatch):
    """Simulate OpenRouterClient raising ValueError and ensure CLI exits with code 1."""

    class RaisingClient:
        def __init__(self, default_model=None):
            raise ValueError("missing key")

    monkeypatch.setattr(main, "DatasetRouter", FakeRouter)
    monkeypatch.setattr(main, "OpenRouterClient", RaisingClient)

    result = runner.invoke(
        main.app,
        ["query", "q", "--call", "--instruction-version", "1"],
    )
    assert result.exit_code != 0
    assert "Error:" in result.stdout

