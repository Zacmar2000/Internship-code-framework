import streamlit as st
import traceback
import os
import json
import uuid

from pathlib import Path
from datetime import datetime
from typing import Optional

from RAG_system.config import USE_NEW, get_reranker

if USE_NEW:
    from RAG_system.main_new import chatbot_st
    PROMPT_PATH = Path("Prompts/new_prompt.txt")

else:
    from RAG_system.main import chatbot_st
    PROMPT_PATH = Path("Prompts/main_prompt.txt")

from RAG_system.agent.chat_session import ChatSession

from RAG_system.resources import get_resources


with st.sidebar: 

    st.sidebar.page_link('chatbot_IMU.py', label='Home')
    st.sidebar.page_link('pages/prompt_editor.py', label='Prompt editor')

    st.markdown("---")

    if st.button("Nuova chat"):
        # Reset dello stato della chat
        st.session_state.chat_history = []
        st.session_state.previous_context = []
        st.session_state.chat_session = ChatSession(max_history=50)
        st.session_state.audit_path = Path(f"Log_chats\\chat_{datetime.now().strftime('%d-%m-%Y_%H-%M')}_session_{uuid.uuid4()}.jsonl")
        st.session_state.feedback = {}
        st.session_state.map_requests = {}
        st.session_state.generating = {}
        st.session_state.is_processing = False
        st.session_state.intro_displayed = False
        st.rerun()

retrieval_engine, rag_tools, client, map_documents = get_resources()

if USE_NEW:
    reranker = get_reranker()


# -------------------------------
# Inizializza session state
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_session" not in st.session_state:
    st.session_state.chat_session = ChatSession(max_history=50)

if "previous_context" not in st.session_state:
    st.session_state.previous_context = []

if "prompt" not in st.session_state:
    if PROMPT_PATH.exists():
        st.session_state.prompt = PROMPT_PATH.read_text(encoding="utf-8")
    else:
        st.session_state.prompt = ""


if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "generating" not in st.session_state:
    st.session_state.generating = {}

if "audit_path" not in st.session_state:
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    os.makedirs("Log_chats", exist_ok=True)
    st.session_state.audit_path = Path(f"Log_chats\\chat_{timestamp}_session_{uuid.uuid4()}.jsonl")


if "map_requests" not in st.session_state:
    st.session_state.map_requests = {}



def feedback_widget(request_index, disabled):

    map_req = st.session_state.map_requests
    request_id = map_req[request_index]

    # Inizializza stato feedback
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}

    current_value = st.session_state.feedback.get(request_id)

    st.write("**Valuta questa risposta:**")

    # Creiamo 5 colonne per evitare che vadano a capo
    cols = st.columns(5)

    for i in range(5):
        score = i + 1

        # Evidenzia la selezione corrente
        if current_value is not None and score <= current_value:
            label = "⭐"
        else:
            label = "☆"

        with cols[i]:
            if st.button(
                label,
                key=f"star_{request_id}_{score}",
                disabled = disabled
            ):
                # Salva nuovo voto in session_state
                st.session_state.feedback[request_id] = score

                # 🔹 Aggiorna audit file SOLO quando cambia il voto
                if st.session_state.audit_path.exists():
                    with open(st.session_state.audit_path, "r", encoding="utf-8") as f:
                        traces = [json.loads(line) for line in f]

                    for trace in traces:
                        if trace.get("request_id") == request_id:
                            trace["user_feedback"] = score
                            break

                    with open(st.session_state.audit_path, "w", encoding="utf-8") as f:
                        for trace in traces:
                            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

                st.rerun()



def should_reset_thread(client, model, last_user, last_answer, new_user):

    prompt = f"""
    Determina se la nuova domanda è STRETTAMENTE collegata alla conversazione precedente.

    Considera "strettamente collegata" SOLO se:
    - La nuova domanda si riferisce direttamente allo stesso argomento specifico
    - Oppure dipende chiaramente dalle informazioni della domanda/risposta precedente
    - Oppure è una continuazione naturale (approfondimento, chiarimento, seguito diretto)

    NON considerare collegata se:
    - Cambia anche leggermente argomento
    - Introduce un nuovo tema, anche se vagamente simile
    - È una domanda generale non dipendente dal contesto precedente

    Valuta internamente senza spiegare.

    Rispondi SOLO con uno di questi due token:
    - CONTINUA
    - RESET

    Domanda precedente:
    {last_user}

    Risposta precedente:
    {last_answer}

    Nuova domanda:
    {new_user}
    """

    response = client.responses.create(
        model=model,
        input=prompt,
    )
    print(response.output_text.strip())

    return response.output_text.strip() == "RESET"



def chat_gradio(user_message, session: ChatSession, previous_context: Optional[list[str]]):
    if previous_context and len(session.get_history())>=2:
        last_messages = session.get_history()[-2:]
        if last_messages[0]['role'] != "user" or last_messages[1]['role'] != "assistant":
            full_previous_context = None
        else:
            last_user = last_messages[0]['content']
            last_answer = last_messages[1]['content']

            reset = should_reset_thread(client, rag_tools.model, last_user, last_answer, user_message)

            if reset:
                full_previous_context = None
            else:
                full_previous_context = "\n\n".join(previous_context[-3:]) if previous_context else None
    else:
        full_previous_context = None

    answer, request_id, previous_context = chatbot_st(
        session=session,
        user_message=user_message,
        previous_context=full_previous_context,
        audit_path=st.session_state.audit_path,
        prompt=st.session_state.prompt,
        retrieval_engine=retrieval_engine,
        reranker=reranker if USE_NEW else None,
        rag_tools=rag_tools if not USE_NEW else None,
    ) # type: ignore
    return answer, request_id, previous_context, session

# -------------------------------
# Layout pagina
# -------------------------------
st.set_page_config(page_title="Chatbot IMU Comune di Venezia")
    
st.title("Chatbot IMU")



# -------------------------------
# Messaggio introduttivo
# -------------------------------

if "intro_displayed" not in st.session_state:
    st.session_state.intro_displayed = False

if not st.session_state.intro_displayed:
    with st.chat_message("assistant"):
        st.markdown(
            """
            Benvenuto nel servizio di assistenza informativa sull’IMU del Comune di Venezia.

            Il chatbot fornisce informazioni di carattere generale su aliquote, esenzioni e disciplina dell’Imposta Municipale Propria per gli ultimi 2 anni.

            Le risposte hanno finalità informative e si basano esclusivamente su contenuti ufficiali disponibili.
            In caso di informazioni non sufficienti, verrà indicato il canale istituzionale per ulteriori approfondimenti.
            """
        )


# -------------------------------
# Input dell’utente
# -------------------------------
user_input = st.chat_input("Scrivi qui la tua domanda...",
    disabled=st.session_state.is_processing)


if user_input:
    # Aggiungi subito messaggio dell'utente con placeholder bot
    st.session_state.chat_history.append((user_input, None))
    st.session_state.is_processing = True
    st.session_state.intro_displayed = True


# -------------------------------
# Mostra tutta la chat
# -------------------------------

any_generating = any(st.session_state.generating.values())

for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
    # Messaggio utente
    with st.chat_message("user"):
        st.markdown(user_msg)

    # Messaggio bot
    with st.chat_message("assistant"):
        if bot_msg is None and st.session_state.is_processing:


            if bot_msg in [None, "Il bot sta pensando..."] and not st.session_state.generating.get(i, False):

                # Metti il lock
                st.session_state.generating[i] = True
                st.session_state.chat_history[i] = (user_msg, "Il bot sta pensando...")

                with st.spinner("Il bot sta pensando..."):
                    try:
                        answer, request_id, previous_context, st.session_state.chat_session = chat_gradio(
                            user_msg,
                            st.session_state.chat_session,
                            st.session_state.previous_context,
                        )
                        if previous_context:
                            st.session_state.previous_context.append(previous_context)
                        else:
                            st.session_state.previous_context.append("")
                        st.session_state.map_requests[i] = request_id
                        st.session_state.chat_history[i] = (user_msg, answer)
                        bot_msg = answer
                    except Exception as e:
                        st.session_state.chat_history[i] = (user_msg, "Errore API. Riprova.")
                        bot_msg = "Errore API. Riprova."
                        st.error(f"Errore dettagliato: {e}")
                        traceback.print_exc()
                    finally:
                        st.session_state.generating[i] = False
                        st.session_state.is_processing = False
                    


        is_generating = st.session_state.generating.get(i, False)

        if is_generating:
            st.markdown("⏳ Sto scrivendo...")
        else:
            st.markdown(bot_msg)
            #feedback_widget(i, disabled=any_generating)

            if st.button(f"Vedi fonti utilizzate", key=f"log_{i}"):
                st.session_state.selected_log = i
                st.switch_page("pages/log_viewer.py")
            # with st.expander("Annota questa risposta", expanded=False):
            #     annotate_response(bot_msg, st.session_state.map_requests, i, disabled=any_generating)