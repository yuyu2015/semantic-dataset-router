"""
Versioned general instructions for the LLM prompt.
Add or edit entries in INSTRUCTIONS; use DEFAULT_VERSION for the default.
"""

from __future__ import annotations

# Version key -> instruction text (included in prompts when no --instruction override)
INSTRUCTIONS: dict[str, str] = {
    "1": (
        "Answer the user's question concisely. "
        "When dataset context is provided, use it to support your answer; "
        "otherwise answer from general knowledge."
    ),
    # Add new versions below, e.g.:
    # "2": "You are a helpful analyst. Answer based on the data when provided, otherwise briefly.",
}

DEFAULT_VERSION = "1"


def get_instruction(version: str | None = None) -> str:
    """Return the instruction text for the given version. Uses DEFAULT_VERSION if version is None or missing."""
    v = version or DEFAULT_VERSION
    if v not in INSTRUCTIONS:
        raise KeyError(
            f"Unknown instruction version {v!r}. Available: {list(INSTRUCTIONS.keys())}"
        )
    return INSTRUCTIONS[v].strip()
