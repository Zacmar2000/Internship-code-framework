from pathlib import Path
from RAG_system.agent.create_answer_new import create_answer_new
from RAG_system.utils.logging_utils import setup_logging



def rag_query_with_memory(
    session,
    user_query,
    model,
    model_internal,
    prompt,
    retrieval_engine,
    reranker,
    k=3,
    reasoning=False,
    reasoning_internal = False,
    reasoning_effort="medium",
    reasoning_effort_internal="medium",
    past_context=None,
    audit=False,
    audit_path=Path("Log_chats/random_chats.jsonl"),
    logs=False
):
    
    if logs:
        logger, log_tool_result = setup_logging(max_tool_result_length=300)
    else:
        logger = None
        log_tool_result = None
    

    final_answer, request_id, previous_id = create_answer_new(
        session=session,
        prompt = prompt,
        user_query=user_query,
        model=model,
        model_internal=model_internal,
        k=k,
        retrieval_engine=retrieval_engine,
        reranker=reranker,
        reasoning=reasoning,
        reasoning_internal=reasoning_internal,
        reasoning_effort=reasoning_effort,
        reasoning_effort_internal=reasoning_effort_internal,
        past_context=past_context,
        audit=audit,
        audit_path=audit_path,
        logger=logger,
        log_tool_result=log_tool_result
    )

    return final_answer, request_id, previous_id