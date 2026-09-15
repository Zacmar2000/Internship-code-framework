from typing import Optional
from pathlib import Path

from RAG_system.pipeline.rag_pipeline_new import rag_query_with_memory
from RAG_system.config import (
    MODEL,
    K_RAG_NEW,
    LOGS,
    AUDIT,
    REASONING,
    REASONING_EFFORT,
    MODEL_INTERNAL,
    REASONING_INTERNAL,
    REASONING_EFFORT_INTERNAL
)


def chatbot_st(
    user_message,
    session,
    previous_context: Optional[str],
    audit_path: Path,
    prompt: str,
    retrieval_engine,
    reranker,
    rag_tools
):

    return rag_query_with_memory(
        session=session,
        user_query=user_message,
        model=MODEL,
        model_internal=MODEL_INTERNAL,
        past_context=previous_context,
        retrieval_engine=retrieval_engine,
        reranker = reranker,
        k=K_RAG_NEW,
        reasoning=REASONING,
        reasoning_internal=REASONING_INTERNAL,
        reasoning_effort=REASONING_EFFORT,
        reasoning_effort_internal=REASONING_EFFORT_INTERNAL,
        logs=LOGS,
        audit=AUDIT,
        audit_path=audit_path,
        prompt=prompt
    )