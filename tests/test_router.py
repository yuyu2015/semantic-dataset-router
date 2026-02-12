import json

import numpy as np
import pandas as pd
import pytest

import router
from router import DatasetRouter, RouterResult, cosine_similarity, build_prompt


class FakeSentenceTransformer:
    """Minimal fake SentenceTransformer to avoid external model downloads."""

    def __init__(self, *args, **kwargs):
        self._call = 0

    def encode(self, inputs, normalize_embeddings=True):
        self._call += 1
        # First call: dataset descriptions -> two orthogonal unit vectors
        if self._call == 1:
            n = len(inputs)
            if n == 0:
                # Explicitly return an empty 2D array to cover this edge case
                return np.zeros((0, 3))
            if n == 1:
                return np.array([[1.0, 0.0, 0.0]])
            # Two-dataset case used in this project
            return np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ][:n]
            )
        # Second call: user query -> aligned with first dataset
        return np.array([[1.0, 0.0, 0.0]])


def test_fake_sentence_transformer_handle_zero_descriptions():
    """Explicitly cover the n == 0 branch of FakeSentenceTransformer.encode."""
    ft = FakeSentenceTransformer()
    out = ft.encode([])
    assert out.shape == (0, 3)


def test_cosine_similarity_vector_and_matrix_branch():
    """Cover both 1D and 2D branches in cosine_similarity."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert pytest.approx(cosine_similarity(a, b), rel=1e-6) == 1.0

    # 2D case: matrix of two vectors, we just ensure it runs
    b_mat = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sim = cosine_similarity(a, b_mat)
    assert isinstance(sim, float)


def test_dataset_router_routes_to_best_dataset_with_json_context(monkeypatch):
    """Router should select the best dataset and produce JSON context."""
    # Patch the SentenceTransformer used inside router
    monkeypatch.setattr(router, "SentenceTransformer", FakeSentenceTransformer)

    r = DatasetRouter()
    result = r.route("How much do people spend?", threshold=0.0, context_max_rows=5)

    # With FakeSentenceTransformer, first dataset is always best
    assert result.best_dataset_id == "shopping_habits"
    assert result.best_score >= 0.0
    assert "shopping_habits" in result.all_scores
    assert isinstance(result.context_preview, str)

    # context_preview should be JSON list of records
    parsed = json.loads(result.context_preview)
    assert isinstance(parsed, list)
    assert len(parsed) <= 5
    # Expect keys from the CSV
    assert {"Customer ID", "Age", "Gender", "Annual Income", "Spending Score"} <= set(
        parsed[0].keys()
    )


def test_dataset_router_no_context_when_below_threshold(monkeypatch):
    """When best_score < threshold, no context should be attached."""
    monkeypatch.setattr(router, "SentenceTransformer", FakeSentenceTransformer)

    r = DatasetRouter()
    # Use a very high threshold to force "no context"
    result = r.route("How much do people spend?", threshold=1.0)

    assert result.best_dataset_id is None
    assert result.context_df is None
    assert result.context_preview == ""


def test_dataset_router_missing_file_sets_no_context(monkeypatch, tmp_path):
    """If the CSV path does not exist, router should treat as no context."""
    monkeypatch.setattr(router, "SentenceTransformer", FakeSentenceTransformer)

    fake_path = tmp_path / "missing.csv"
    registry = {
        "fake": ("Some description", fake_path),
    }
    r = DatasetRouter(dataset_registry=registry)

    result = r.route("Question about fake data", threshold=0.0)
    # best_score >= threshold, but path does not exist -> best_dataset_id becomes None
    assert result.best_dataset_id is None
    assert result.context_df is None
    assert result.context_preview == ""


def test_build_prompt_uses_versioned_instruction_when_none(monkeypatch):
    """build_prompt should pull instruction from general_instructions when not provided."""
    rr = RouterResult(
        query="q",
        best_dataset_id=None,
        best_score=0.0,
        threshold=0.25,
        all_scores={},
        context_df=None,
        context_preview="",
    )

    from general_instructions import get_instruction, DEFAULT_VERSION

    prompt = build_prompt("What is this?", rr, general_instruction=None, instruction_version=None)
    assert get_instruction(DEFAULT_VERSION).split()[0] in prompt
    assert "User question: What is this?" in prompt


def test_build_prompt_includes_internal_knowledge_base_when_context_present():
    """When context is present, prompt should include the 'Internal Knowledge Base' section."""
    rr = RouterResult(
        query="q",
        best_dataset_id="shopping_habits",
        best_score=0.9,
        threshold=0.25,
        all_scores={"shopping_habits": 0.9},
        context_df=pd.DataFrame([{"x": 1}]),
        context_preview='[{"x": 1}]',
    )

    prompt = build_prompt("What is this?", rr, general_instruction="Instr")
    assert "Internal Knowledge Base:" in prompt
    assert '[{"x": 1}]' in prompt

