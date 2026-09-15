import uuid
from pathlib import Path

from RAG_system.retrieval.data_retrieval_new import data_retrieval_new

def create_answer_new(
    session,
    prompt,
    user_query,
    model,
    model_internal,
    retrieval_engine,
    reranker,
    k=3,
    reasoning=False,
    reasoning_internal = False,
    reasoning_effort="medium",
    reasoning_effort_internal="medium",
    past_context=None,
    audit=False,
    audit_path: Path = Path("Log_chats/random_chats.jsonl"),
    logger=None,
    log_tool_result=None,
):
    
    client = retrieval_engine.client

    request_id = str(uuid.uuid4())

    audit_trace = None
    if audit:
        from RAG_system.utils.audit_trace import AuditTrace
        audit_trace = AuditTrace(request_id, model, user_query)

    if logger:
        logger.info("=== Nuova richiesta ===")
        logger.debug(f"User query: {user_query}")


    fonti = data_retrieval_new(user_query= user_query,session= session, retrieval_engine= retrieval_engine, reranker= reranker,
                                reasoning= reasoning_internal, reasoning_effort= reasoning_effort_internal, model= model_internal, k= k,
                                audit_trace=audit_trace,
                                logger=logger, log_tool_result=log_tool_result)

    if fonti:
        full_prompt = prompt + "\n\n ##FONTI:" + fonti
    
    else:
        full_prompt = prompt + "\n\n ##FONTI: Non sono state trovate fonti rilevanti per la risposta"

    input_messages = [
        {"role": "system", "content": full_prompt}
    ]

    input_messages.extend(session.get_history()[-6:])
    input_messages.append({"role": "user", "content": user_query})

    params = {
        "model": model,
        "input": input_messages
    }

    if reasoning:
        params["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**params)

    final_message = None

    previous_id = response.id

    for item in response.output:

        if item.type == "message":

            for c in getattr(item, "content", []):
                text = getattr(c, "text", None)

                if text:
                    final_message = text
                    break

    if not final_message:
        final_message = "Errore nella generazione della risposta."

    session.add_message("user", user_query)
    session.add_message("assistant", final_message)

    # Logging
    if logger:
        logger.debug(f"Risposta: {final_message}")

    # Audit
    if audit_trace:
        audit_trace.finalize(final_message, 0)
        audit_trace.save(audit_path)

    return final_message, request_id, previous_id