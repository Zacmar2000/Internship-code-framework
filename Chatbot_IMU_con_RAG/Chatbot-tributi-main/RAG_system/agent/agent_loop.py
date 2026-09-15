import json
import uuid
from pathlib import Path
import asyncio

from RAG_system.tools.tool_executor import ToolExecutor
from RAG_system.tools.tool_scheme import tools


MAX_TOOL_CALLS = 20


async def run_agent_loop(
    session,
    input_messages,
    user_query,
    model,
    retrieval_engine,
    rag_tools,
    k=3,
    reasoning=False,
    reasoning_effort="medium",
    past_context=None,
    audit=False,
    audit_path: Path = Path("Log_chats/random_chats.jsonl"),
    logger=None,
    log_tool_result=None
):

    client = retrieval_engine.client
    tool_executor = ToolExecutor(rag_tools)

    
    tool_calls_count = 0
    previous_id = None
    request_id = str(uuid.uuid4())

    audit_trace = None
    if audit:
        from RAG_system.utils.audit_trace import AuditTrace
        audit_trace = AuditTrace(request_id, model, user_query)

    if logger:
        logger.info("=== Nuova richiesta ===")
        logger.debug(f"User query: {user_query}")
        #logger.debug(f"History (last 6): {session.get_history()[-6:]}")

    if past_context:
        # Inserisci il contesto prima dell'ultimo messaggio
        input_messages = (
            input_messages[:-1] +
            [{
                "role": "assistant",
                "content": f"""Contesto recuperato in precedenza (dall'ultimo step):
            {past_context}

            Questo contesto, estratto con retrieve_context, è stato utilizzato per rispondere alla precedente domanda dell'utente.
            Riutilizzalo solo se è ancora rilevante per la richiesta attuale."""
            }] +
            [input_messages[-1]]
        )

    params = {
        "model": model,
        "input": input_messages,
        "tools": tools
    }

    if reasoning:
        params["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**params)

    new_context = ""  # Inizializza new_context

    args_used = set()

    vocabulary_used = []
    
    while True:

        if audit_trace:
            audit_trace.add_step(
                iteration=tool_calls_count,
                response_id=response.id,
                previous_id=previous_id
            )

            audit_trace.add_past_context(past_context)

        input_message = []
        tool_called = False
        final_message = None

        previous_id = response.id

        tool_calls = []

        # RACCOLTA
        for item in response.output:
            if item.type == "function_call":
                tool_called = True
                tool_calls_count += 1

                tool_calls.append({
                    "item": item,
                    "tool_number": tool_calls_count,
                    "tool_name": item.name,
                    "args": json.loads(item.arguments)
                })

                # Log
                if logger:
                    logger.debug(f"Tool call #{tool_calls_count}: {item.name}")
                    logger.debug(f"Args: {json.loads(item.arguments)}")

        # ESECUZIONE PARALLELA
        semaphore = asyncio.Semaphore(5)

        async def run_tool(call):
            async with semaphore:
                return await asyncio.to_thread(
                    tool_executor.execute,
                    tool_name=call["tool_name"],
                    k=k,
                    args=call["args"],
                    args_used=args_used,
                    vocabulary_used=vocabulary_used
                )

        results = await asyncio.gather(*[run_tool(call) for call in tool_calls])

        # POST-PROCESSING (ORDINATO)
        for call, (tool_result, agent_tool_results) in zip(tool_calls, results):

            tool_name = call["tool_name"]
            tool_number = call["tool_number"]
            args = call["args"]
            item = call["item"]

            args_used.add(json.dumps(args, sort_keys=True))

            if tool_name == "get_vocabulary":
                args = agent_tool_results.get("used", {}) # type: ignore
                for arg in args:
                    if arg not in vocabulary_used:
                        vocabulary_used.append(arg)
                agent_tool_results = agent_tool_results.get("agent_sources", []) # type: ignore
                if logger:
                    logger.debug(f"Args vocabulary: {args}")

            # Audit
            if audit_trace:
                audit_trace.add_tool_call(
                    tool_number=tool_number,
                    tool_name=tool_name,
                    args=args,
                    result=agent_tool_results
                )

            # Caso speciale retrieve_context
            if tool_name == "retrieve_context":
                final_results = ""

                for cat, text in tool_result.get("sources", {}).items(): # pyright: ignore[reportAttributeAccessIssue]
                    final_results += f"\n\n### {cat}\n{text}"

                    if audit_trace:
                        audit_trace.add_clean_extraction(
                            category=cat,
                            result=text,
                            tool_number=tool_number,
                            tool_name=tool_name
                        )

                tool_result = {"sources": final_results}
                new_context += "\n\n" + f"### Query: {args.get('query', 'N/A')} Year: {args.get('year', 'N/A')}" + "\n\n" + final_results # type: ignore

            # Log
            if log_tool_result:
                log_tool_result(tool_name, tool_result)

            # Output per LLM
            input_message.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(tool_result, ensure_ascii=False)
            })
        
        for item in response.output:
            if item.type == "message":

                for c in getattr(item, "content", []):
                    text = getattr(c, "text", None)

                    if text:
                        final_message = text

                        if audit_trace:
                            audit_trace.add_internal_message(
                                iteration=tool_calls_count,
                                message=text
                            )
                        break

        if final_message and not tool_called:

            session.add_message("user", user_query)
            session.add_message("assistant", final_message)

            # Logging
            if logger:
                logger.info(f"Risposta finale generata dopo {tool_calls_count} tool calls")
                logger.debug(f"Risposta: {final_message}")

            # Audit
            if audit_trace:
                audit_trace.finalize(final_message, tool_calls_count)
                audit_trace.save(audit_path)

            return final_message, request_id, new_context

        if tool_calls_count >= MAX_TOOL_CALLS:

            response = client.responses.create(
                model=model,
                input=input_message,
                previous_response_id=previous_id
            )
        elif final_message and logger:
            logger.debug(f"Messaggio interno: {final_message}")

        else:

            params = {
                "model": model,
                "input": input_message,
                "tools": tools,
                "previous_response_id": previous_id
            }

            if reasoning:
                params["reasoning"] = {"effort": reasoning_effort}

            response = client.responses.create(**params)