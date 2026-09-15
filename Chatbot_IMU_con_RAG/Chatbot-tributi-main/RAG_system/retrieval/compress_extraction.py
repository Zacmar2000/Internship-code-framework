import asyncio
import json

async def compress_one_category(
    sources,
    user_query,
    category,
    client,
    model,
    reasoning,
    reasoning_effort,
    session = None
):
    base_prompt = """
    # Ruolo e obiettivo
    Sei un assistente esperto in documentazione fiscale/comunale.

    # Istruzioni
    - Estrai SOLO le informazioni rilevanti.
    - NON inventare.
    - Mantieni articoli, eccezioni e condizioni.
    - Se una parte di testo si applica solo a un argomento, categoria, casistica o fattispecie specifica, specificalo esplicitamente e chiaramente.
    - Se è presente "eccezione", specifica che la parte estratta vale ESCLUSIVAMENTE per quella categoria, argomento o caso specifico.
    - Se il contenuto non è rilevante, restituisci una stringa vuota.

    ## PRIORITÀ FONTI

    In caso di conflitto:
    1. Comunale > Regionale > Nazionale
    2. Se più fonti dello stesso livello, dai priorità a quella più recente (anno più alto).

    # Formato di output
    - Restituisci solo le informazioni rilevanti oppure una stringa vuota se nulla è rilevante.
    """

    category_instructions = {
        "Normativa": "Estrai articoli, obblighi, eccezioni.",
        "Regolamento": "Estrai regole e vincoli.",
        "Aliquote": "Estrai numeri e condizioni.",
        "Istruzioni": "Estrai passaggi operativi.",
        "Privacy": "Estrai obblighi privacy."
    }

    prompt = base_prompt + "\n" + category_instructions.get(category, "")


    input_payload = {
            "query": user_query,
            "sources": sources
        }
    

    input_messages = [
        {"role": "system", "content": prompt}
    ]

    if session:
        input_messages.extend(session.get_history()[-6:])

    input_messages.append({"role": "user", "content": json.dumps(input_payload)})


    params = {
        "model": model,
        "input": input_messages
    }

    if reasoning:
        params["reasoning"] = {"effort": reasoning_effort}

    for attempt in range(3):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(client.responses.create, **params),
                timeout=30
            )

            output = response.output_text.strip()

            return category, output

        except Exception as e:
            if attempt == 2:
                return category, ""
            await asyncio.sleep(1.5 * (attempt + 1))