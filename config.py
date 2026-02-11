"""
Application settings loaded from environment variables via Pydantic Settings.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application configuration from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter LLM
    openrouter_api_key: str | None = Field(
        default=None,
        description="OpenRouter API key (required when using --call).",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL.",
    )
    default_llm_model: str = Field(
        default="openai/gpt-4o-mini",
        description="Default OpenRouter model id.",
    )

    # Embedding / router (optional overrides)
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for dataset embeddings.",
    )
    routing_threshold: float = Field(
        default=0.25,
        description="Min cosine similarity to attach a dataset to the prompt.",
    )


def get_settings() -> AppSettings:
    """Return application settings (loads from env / .env once)."""
    return AppSettings()
