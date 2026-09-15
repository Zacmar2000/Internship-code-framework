from typing import Optional
from pathlib import Path

from RAG_system.pipeline.rag_pipeline import rag_query_with_memory
from RAG_system.config import (
    MODEL,
    K_RAG,
    LOGS,
    AUDIT,
    REASONING,
    REASONING_EFFORT,
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
        past_context=previous_context,
        retrieval_engine=retrieval_engine,
        rag_tools=rag_tools,
        k=K_RAG,
        logs=LOGS,
        audit=AUDIT,
        audit_path=audit_path,
        reasoning=REASONING,
        reasoning_effort=REASONING_EFFORT,
        prompt=prompt
    )