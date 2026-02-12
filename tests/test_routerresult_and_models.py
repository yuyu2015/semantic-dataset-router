import pandas as pd

from router import RouterResult
from llm_client import LLMResponse


def test_router_result_pydantic_model_serialization():
    """RouterResult should be a valid Pydantic model and serializable (excluding DataFrame safely)."""
    df = pd.DataFrame([{"a": 1, "b": 2}])
    router_result = RouterResult(
        query="q",
        best_dataset_id="id",
        best_score=0.5,
        threshold=0.25,
        all_scores={"id": 0.5},
        context_df=df,
        context_preview='[{"a": 1, "b": 2}]',
    )

    dumped = router_result.model_dump()
    # DataFrame should be present as object, and preview as string
    assert dumped["query"] == "q"
    assert dumped["best_dataset_id"] == "id"
    assert dumped["context_preview"] == '[{"a": 1, "b": 2}]'
    assert isinstance(dumped["context_df"], pd.DataFrame)


def test_llmresponse_pydantic_model():
    """LLMResponse should behave as a simple Pydantic model."""
    resp = LLMResponse(
        content="hello",
        model="m",
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
        raw_message={"role": "assistant", "content": "hello"},
    )
    dumped = resp.model_dump()
    assert dumped["content"] == "hello"
    assert dumped["raw_message"]["role"] == "assistant"

