import datetime
import time
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from chatbot_system.main import ask_chatbot
from chatbot_system.schemas.chat import ChatRequest
from chatbot_system.schemas.prompt import PromptConfig
from chatbot_system.utils.logging_utils import save_message_log
from chatbot_system.utils.prompt_saver import save_modified_prompts
from chatbot_system.config import (
    model,
    reasoning,
    reasoning_effort,
    logs,
    main_prompt,
    prompt_aliquote,
    prompt_nazionale,
    prompt_comunale,
    prompt_file_rimanenti,
    prompt_vocab,
    client,
    history_ttl,
    redis_client,
)

prompt_state = {
    "main_prompt": main_prompt,
    "prompt_nazionale": prompt_nazionale,
    "prompt_comunale": prompt_comunale,
    "prompt_aliquote": prompt_aliquote,
    "prompt_file_rimanenti": prompt_file_rimanenti,
    "prompt_vocab": prompt_vocab,
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/prompts")
async def get_prompts():
    return prompt_state

@app.post("/prompts")
async def update_prompts(config: PromptConfig):
    global prompt_state

    if not config.main_prompt.strip():
        raise HTTPException(400, "main_prompt non può essere vuoto")

    prompt_state = config.model_dump()

    return {"status": "ok"}

@app.post("/prompts/save")
async def save_prompts(config: PromptConfig):
    global prompt_state, main_prompt, prompt_nazionale, prompt_comunale, prompt_aliquote, prompt_file_rimanenti, prompt_vocab

    if not config.main_prompt.strip():
        raise HTTPException(400, "main_prompt non può essere vuoto")

    prompt_state = config.model_dump()

    await save_modified_prompts(config)

    main_prompt = config.main_prompt
    prompt_nazionale = config.prompt_nazionale
    prompt_comunale = config.prompt_comunale
    prompt_aliquote = config.prompt_aliquote
    prompt_file_rimanenti = config.prompt_file_rimanenti
    prompt_vocab = config.prompt_vocab

    return {"status": "ok"}

@app.post("/prompts/reset")
async def reset_prompts():
    global prompt_state

    prompt_state = {
        "main_prompt": main_prompt,
        "prompt_nazionale": prompt_nazionale,
        "prompt_comunale": prompt_comunale,
        "prompt_aliquote": prompt_aliquote,
        "prompt_file_rimanenti": prompt_file_rimanenti,
        "prompt_vocab": prompt_vocab,
    }

    return {"status": "reset"}

@app.post("/prompts/reset/{key}")
async def reset_single_prompt(key: str):
    global prompt_state

    default_map = {
        "main_prompt": main_prompt,
        "prompt_nazionale": prompt_nazionale,
        "prompt_comunale": prompt_comunale,
        "prompt_aliquote": prompt_aliquote,
        "prompt_file_rimanenti": prompt_file_rimanenti,
        "prompt_vocab": prompt_vocab,
    }

    if key not in prompt_state:
        raise HTTPException(404, "Prompt non trovato")

    prompt_state[key] = default_map[key]

    return {"status": "reset_single", "key": key}


@app.post("/chat")
async def chatbot(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,  # Aggiungi Request per rilevare disconnessioni
):
    message = request.message
    session = request.session
    cfg = prompt_state

    final_prompt = cfg["main_prompt"] + "\n\n" + f"""
    # Fonti
    
    ## Data di oggi:
    ```{datetime.datetime.now().strftime('%d %B %Y')}```

    ## Regolamento Nazionale
    ```{cfg["prompt_nazionale"]}```

    ## Regolamento Comunale
    ```{cfg["prompt_comunale"]}```

    ## Aliquote
    ```{cfg["prompt_aliquote"]}```

    ## Resto dei documenti
    ```{cfg["prompt_file_rimanenti"]}```

    ## Vocabolario
    ```{cfg["prompt_vocab"]}```
    """

    try:
        # Verifica se il client è ancora connesso
        if await http_request.is_disconnected():
            print(f"DEBUG: Client disconnected for session {session}")
            return

        background_tasks.add_task(save_message_log, session, "user", message)

        start = time.perf_counter()

        response = await ask_chatbot(
            session_id=session,
            user_query=message,
            model=model,
            reasoning=reasoning,
            reasoning_effort=reasoning_effort,
            logs=logs,
            prompt=final_prompt,
            client=client,
            history_ttl=history_ttl,
            http_request=http_request,  # Passa il Request object
        )

        # Verifica di nuovo se il client è ancora connesso prima di inviare
        if await http_request.is_disconnected():
            print(f"DEBUG: Client disconnected before response for session {session}")
            background_tasks.add_task(save_message_log, session, "assistant", "[response interrupted by user]")
            end = time.perf_counter()
            return

        else:
            end = time.perf_counter()
            background_tasks.add_task(save_message_log, session, "assistant", response, end - start)

        return response

    except Exception as e:
        # Se è una disconnessione, non eseguire il logging dell'errore
        if "ClientDisconnect" not in str(type(e).__name__):
            print("DEBUG ERROR:", e)
            background_tasks.add_task(save_message_log, session, "error", str(e))
            raise HTTPException(status_code=500, detail="Errore interno")
        else:
            print(f"DEBUG: Client disconnected for session {session}")
    

@app.get("/redis-test")
async def test():
    await redis_client.set("foo", "bar")
    val = await redis_client.get("foo")
    return {"value": val}