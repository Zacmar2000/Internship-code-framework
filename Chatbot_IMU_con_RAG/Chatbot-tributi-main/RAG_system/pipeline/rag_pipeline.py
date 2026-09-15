from pathlib import Path
import asyncio
from RAG_system.agent.agent_loop import run_agent_loop
from RAG_system.utils.logging_utils import setup_logging


def rag_query_with_memory(
    session,
    user_query,
    model,
    prompt,
    retrieval_engine,
    rag_tools,
    k=3,
    reasoning=False,
    reasoning_effort="medium",
    past_context=None,
    audit=False,
    audit_path=Path("Log_chats/random_chats.jsonl"),
    logs=False
):
    

    input_messages = [
        {"role": "system", "content": prompt}
    ]

    input_messages.extend(session.get_history()[-6:])
    input_messages.append({"role": "user", "content": user_query})

    if logs:
        logger, log_tool_result = setup_logging(max_tool_result_length=300)
    else:
        logger = None
        log_tool_result = None

    final_answer, request_id, new_context = asyncio.run(run_agent_loop(
        session=session,
        input_messages=input_messages,
        user_query=user_query,
        model=model,
        k=k,
        retrieval_engine=retrieval_engine,
        rag_tools=rag_tools,
        reasoning=reasoning,
        reasoning_effort=reasoning_effort,
        past_context=past_context,
        audit=audit,
        audit_path=audit_path,
        logger=logger,
        log_tool_result=log_tool_result
    )) # type: ignore

    return final_answer, request_id, new_context