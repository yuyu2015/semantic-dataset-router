import os

import pytest

from config import AppSettings, get_settings
from general_instructions import get_instruction, DEFAULT_VERSION, INSTRUCTIONS


def test_appsettings_defaults_from_env(monkeypatch):
    """AppSettings should read values from environment and fall back to defaults."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.test/api")
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "provider/model-x")
    monkeypatch.setenv("EMBEDDING_MODEL", "embed-model-x")
    monkeypatch.setenv("ROUTING_THRESHOLD", "0.42")

    settings = AppSettings()

    assert settings.openrouter_api_key == "test-key"
    assert settings.openrouter_base_url == "https://example.test/api"
    assert settings.default_llm_model == "provider/model-x"
    assert settings.embedding_model == "embed-model-x"
    assert pytest.approx(settings.routing_threshold, rel=1e-6) == 0.42


def test_get_settings_uses_env(tmp_path, monkeypatch):
    """get_settings should load from .env file when present."""
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=from-dot-env\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    settings = get_settings()
    assert settings.openrouter_api_key == "from-dot-env"


def test_get_instruction_default_and_specific():
    """get_instruction should return default when version is None and specific when provided."""
    default_text = get_instruction(None)
    assert default_text == INSTRUCTIONS[DEFAULT_VERSION].strip()

    for version, text in INSTRUCTIONS.items():
        assert get_instruction(version) == text.strip()


def test_get_instruction_unknown_version_raises():
    with pytest.raises(KeyError):
        get_instruction("non-existent-version")

