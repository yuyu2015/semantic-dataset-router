"""
Semantic dataset router: embed dataset descriptions, score user queries by
cosine similarity, and select dataset context when above threshold.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Default paths relative to project root
DATA_DIR = Path(__file__).resolve().parent / "data"

# Dataset registry: id -> (description for embedding, path to CSV)
DATASET_REGISTRY: dict[str, tuple[str, Path]] = {
    "shopping_habits": (
        "Consumer demographics and shopping behavior: age, gender, annual income, spending score. "
        "Data about how much people shop and spend.",
        DATA_DIR / "shopping_habits.csv",
    ),
    "weekly_searches": (
        "Weekly search engine search volume for popular programming languages: Python, Java, C++. "
        "Time series of search popularity for coding languages.",
        DATA_DIR / "weekly_searches_for_programming_languages.csv",
    ),
}

# Default model for embeddings (small and fast)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class RouterResult:
    """Result of routing a query to datasets."""

    query: str
    best_dataset_id: str | None
    best_score: float
    threshold: float
    all_scores: dict[str, float]
    context_df: pd.DataFrame | None
    context_preview: str  # First N rows as string for prompt


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors (or a vector and a matrix)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64)
    if b.ndim == 1:
        b = b.ravel()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    # b is (n, dim)
    norms_b = np.linalg.norm(b, axis=1, keepdims=True)
    norms_b = np.where(norms_b == 0, 1e-9, norms_b)
    return float(np.dot(a, (b / norms_b).T).item(0))


class DatasetRouter:
    """
    Embeds dataset descriptions with a sentence transformer, then routes user
    queries by cosine similarity against those embeddings.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        dataset_registry: dict[str, tuple[str, Path]] | None = None,
    ):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self.registry = dataset_registry or DATASET_REGISTRY
        self._descriptions: list[str] = []
        self._ids: list[str] = []
        self._embeddings: np.ndarray | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _build_embeddings(self) -> None:
        """Embed all dataset descriptions once."""
        self._ids = list(self.registry.keys())
        self._descriptions = [self.registry[did][0] for did in self._ids]
        self._embeddings = self.model.encode(
            self._descriptions, normalize_embeddings=True
        )

    def route(
        self,
        query: str,
        threshold: float = 0.25,
        context_max_rows: int = 50,
    ) -> RouterResult:
        """
        Route a user query to the best matching dataset by cosine similarity.
        If the best score is >= threshold, return that dataset as context;
        otherwise return no dataset context.
        """
        if self._embeddings is None:
            self._build_embeddings()

        query_emb = self.model.encode(
            [query], normalize_embeddings=True
        )[0]

        # scores: id -> similarity
        scores = {}
        for i, did in enumerate(self._ids):
            sim = cosine_similarity(query_emb, self._embeddings[i])
            scores[did] = float(sim)

        best_id = max(scores, key=scores.get)
        best_score = scores[best_id]

        context_df = None
        context_preview = ""

        if best_score >= threshold:
            _, path = self.registry[best_id]
            if path.exists():
                context_df = pd.read_csv(path).head(context_max_rows)
                context_preview = context_df.to_string(index=False)
            else:
                best_id = None  # path missing; treat as no context

        return RouterResult(
            query=query,
            best_dataset_id=best_id if best_score >= threshold else None,
            best_score=best_score,
            threshold=threshold,
            all_scores=scores,
            context_df=context_df,
            context_preview=context_preview,
        )


def build_prompt(
    user_query: str,
    router_result: RouterResult,
    general_instruction: str = "Answer the user's question concisely. When dataset context is provided, use it to support your answer; otherwise answer from general knowledge.",
) -> str:
    """
    Build a prompt that combines general instruction with optional dataset context.
    """
    parts = [general_instruction.strip(), "", f"User question: {user_query}"]

    if router_result.best_dataset_id and router_result.context_preview:
        parts.extend([
            "",
            "Relevant dataset context (use this if it helps answer the question):",
            f"Dataset: {router_result.best_dataset_id}",
            "",
            router_result.context_preview,
        ])

    parts.append("")
    parts.append("Answer:")
    return "\n".join(parts)
