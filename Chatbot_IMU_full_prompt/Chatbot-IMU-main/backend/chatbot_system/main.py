import asyncio

from chatbot_system.utils.logging_utils import setup_logging

from chatbot_system.utils.history_utils import get_history, save_history

async def ask_chatbot(
    session_id,
    user_query,
    model,
    reasoning,
    reasoning_effort,
    logs,
    prompt,
    client,
    history_ttl,
    http_request=None  # Request object per verificare disconnessioni
):
    if logs:
        logger, _ = setup_logging(max_tool_result_length=300)
    else:
        logger = None

    # Recupera la storia
    history = await get_history(session_id=session_id)

    input_messages = [
            {"role": "system", "content": prompt}
        ]
    input_messages.extend(history)
    input_messages.append({"role": "user", "content": user_query})

    if logger:
        logger.info("=== Nuova richiesta ===")
        logger.debug(f"User query: {user_query}")
        logger.debug(f"Messaggi precedenti (user+assistant): {int(len(history)/2)}")

    params = {
        "model": model,
        "input": input_messages,
    }

    if reasoning:
        params["reasoning"] = {"effort": reasoning_effort}


    for attempt in range(3):
        try:
            response = client.responses.create(**params)
            output = response.output_text.strip()

            # aggiorna la history in memoria
            history.append({"role": "user", "content": user_query})
            history.append({"role": "assistant", "content": output})

            # Verifica se il client è ancora connesso prima di salvare
            if http_request and await http_request.is_disconnected():
                if logger:
                    logger.info("Client disconnected, saving with stopped status")
                # Sostituisci la risposta dell'assistente con il messaggio di stop
                history[-1]["content"] = "[stopped by user]"
                await save_history(session_id, history, history_ttl)
                return output

            await save_history(session_id, history, history_ttl)  # salva su Redis

            if logger:
                logger.info(f"Risposta finale generata dopo {attempt + 1} tentativi")
                logger.debug(f"Tokens usati in input: {response.usage.input_tokens} di cui {response.usage.input_tokens_details.cached_tokens} in cache")
                logger.debug(f"Tokens usati in output: {response.usage.output_tokens} di cui {response.usage.output_tokens_details.reasoning_tokens} per il reasoning")

                # logger.debug(f"Risposta: {output}")
                print()

            return output

        except Exception as e:
            if logger:
                logger.debug(f"Attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                # se falliscono tutti e 3 i tentativi, ritorna comunque la history aggiornata con l'errore
                history.append({"role": "error", "content": "Errore nella generazione della risposta"})
                await save_history(session_id, history, history_ttl)
                return ""
            await asyncio.sleep(2 ** attempt)
    else:
        return ""

