import streamlit as st
import html
from RAG_system.resources import get_resources
from funzioni_varie.function_log import *

import uuid
from datetime import datetime
import os


if "audit_path" not in st.session_state:
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
    os.makedirs("Log_chats", exist_ok=True)
    st.session_state.audit_path = Path(f"Log_chats\\chat_{timestamp}_session_{uuid.uuid4()}.jsonl")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Visualizza fonti usate")

st.title("Visualizzazione fonti utilizzate")

st.sidebar.page_link('chatbot_IMU.py', label='Home')
st.sidebar.page_link('pages/prompt_editor.py', label='Prompt editor')


retrieval_engine, rag_tools, client, map_documents = get_resources()


def show_retrieve_widget_st(all_retrieve_data):
    if not all_retrieve_data or not isinstance(all_retrieve_data, dict):
        st.markdown(
            """
            <div style="
                background-color:#2b2b2b;
                color:#ff6b6b;
                padding:15px;
                border-radius:6px;
                font-family:monospace;">
                Nessun tool usato.
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # ===== Dropdown principale (tipo retrieve) =====
    retrieve_type = st.selectbox("Retrieve:", options=list(all_retrieve_data.keys()))

    data = all_retrieve_data[retrieve_type]
    if not data:
        st.markdown(
            """
            <div style="
                background-color:#2b2b2b;
                color:#ffc107;
                padding:15px;
                border-radius:6px;
                font-family:monospace;">
                Nessun dato disponibile per questo retrieve.
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # =======================
    # first_extraction
    # =======================
    if retrieve_type == "first_extraction":
        query_year_options = ["ALL"] + sorted({(k[1], k[2]) for k in data.keys()}, key=lambda t: (str(t[0]), str(t[1])))
        selected_qy = st.selectbox("Query + Year:", query_year_options, format_func=lambda x: "ALL" if x == "ALL" else f"{x[0]} | {x[1]}")

        if selected_qy == "ALL":
            name_options = sorted({k[0] for k in data.keys()})
        else:
            name_options = sorted({k[0] for k in data.keys() if k[1] == selected_qy[0] and k[2] == selected_qy[1]})

        selected_name = st.selectbox("Name:", ["ALL"] + name_options)

        selected_keys = []
        if selected_qy == "ALL":
            if selected_name == "ALL":
                selected_keys = list(data.keys())
            else:
                selected_keys = [k for k in data.keys() if k[0] == selected_name]
        else:
            if selected_name == "ALL":
                selected_keys = [k for k in data.keys() if k[1] == selected_qy[0] and k[2] == selected_qy[1]]
            else:
                selected_keys = [k for k in data.keys() if k[0] == selected_name and k[1] == selected_qy[0] and k[2] == selected_qy[1]]

        documents = list({doc for key in selected_keys for doc in data[key].keys()})
        document = st.selectbox("Documento:", documents)

        if selected_name == "ALL" or selected_qy == "ALL":
            chunks = []
            seen = set()
            for key in selected_keys:
                q_chunks = data.get(key, {})
                if document in q_chunks:
                    for chunk in q_chunks[document]:
                        if chunk[1] not in seen:
                            chunks.append(chunk)
                            seen.add(chunk[1])
        else:
            chunks = data[selected_keys[0]][document] if selected_keys else []

        titles = [f"{i+1}. {' | '.join(t[0])}" for i, t in enumerate(chunks)]
        selected_idx = st.selectbox("Titolo:", range(len(titles)), format_func=lambda i: titles[i])

        # Mostra testo
        text = chunks[selected_idx][1]
        st.markdown(
            f"""
            <div style="
                background-color: #1e1e1e;
                color: #e0e0e0;
                padding: 12px;
                border-radius: 5px;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: monospace;
                font-size: 14px;
                line-height: 1.5;">
                {html.escape(text)}
            </div>
            """,
            unsafe_allow_html=True
        )

    # =======================
    # first_rag
    # =======================
    elif retrieve_type == "first_rag":
        cat_options = list(data.keys())
        cat = st.selectbox("Categoria:", cat_options)

        # Mostra testo
        text = data[cat]
        st.markdown(
            f"""
            <div style="
                background-color: #1e1e1e;
                color: #e0e0e0;
                padding: 12px;
                border-radius: 5px;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: monospace;
                font-size: 14px;
                line-height: 1.5;">
                {html.escape(text)}
            </div>
            """,
            unsafe_allow_html=True
        )

    # =======================
    # retrieve_context
    # =======================
    elif retrieve_type == "retrieve_context":
        query_year_options = ["ALL"] + sorted({(k[1], k[2]) for k in data.keys()}, key=lambda t: (str(t[0]), str(t[1])))
        selected_qy = st.selectbox("Query + Year:", query_year_options, format_func=lambda x: "ALL" if x == "ALL" else f"{x[0]} | {x[1]}")

        if selected_qy == "ALL":
            name_options = sorted({k[0] for k in data.keys()})
        else:
            name_options = sorted({k[0] for k in data.keys() if k[1] == selected_qy[0] and k[2] == selected_qy[1]})

        selected_name = st.selectbox("Name:", name_options)

        selected_keys = []
        if selected_qy == "ALL":
            selected_keys = [k for k in data.keys() if k[0] == selected_name]
        else:
            selected_keys = [k for k in data.keys() if k[0] == selected_name and k[1] == selected_qy[0] and k[2] == selected_qy[1]]

        if selected_name == "raw_chunks":

            documents = list({doc for key in selected_keys for doc in data[key].keys()})
            document = st.selectbox("Documento:", documents)

            if selected_qy == "ALL":
                chunks = []
                seen = set()
                for key in selected_keys:
                    q_chunks = data.get(key, {})
                    if document in q_chunks:
                        for chunk in q_chunks[document]:
                            if chunk[1] not in seen:
                                chunks.append(chunk)
                                seen.add(chunk[1])
            else:
                chunks = data[selected_keys[0]][document] if selected_keys else []

            titles = [f"{i+1}. {' | '.join(t[0])}" for i, t in enumerate(chunks)]
            selected_idx = st.selectbox("Titolo:", range(len(titles)), format_func=lambda i: titles[i])

            text = chunks[selected_idx][1]

        else:
            cat_options = sorted({cat for key in selected_keys for cat in data.get(key, {}).keys()})
            if not cat_options:
                st.warning("Nessuna categoria disponibile per i filtri selezionati.")
                return

            cat = st.selectbox("Categoria:", cat_options)

            # Mostra testo: concatena tutte le occorrenze di cat tra i key selezionati
            texts = []
            for key in selected_keys:
                val = data.get(key, {})
                if isinstance(val, dict) and cat in val:
                    part = val[cat]
                    if part:
                        texts.append(part)

            text = "\n\n---\n\n".join(texts) if texts else ""

        # Mostra testo
        st.markdown(
            f"""
            <div style="
                background-color: #1e1e1e;
                color: #e0e0e0;
                padding: 12px;
                border-radius: 5px;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: monospace;
                font-size: 14px;
                line-height: 1.5;">
                {html.escape(text)}
            </div>
            """,
            unsafe_allow_html=True
        )

    # =======================
    # retrieve_normativa
    # =======================
    elif retrieve_type == "retrieve_normativa":
        document = st.selectbox("Normativa:", list(data.keys()))
        chunks = data[document]
        titles = [f"{i+1}. {' | '.join(ref)}" for i, (ref, _) in enumerate(chunks)]
        selected_idx = st.selectbox("Riferimento:", range(len(titles)), format_func=lambda i: titles[i])
        text = chunks[selected_idx][1]
        st.markdown(
            f"""
            <div style="
                background-color: #1e1e1e;
                color: #e0e0e0;
                padding: 12px;
                border-radius: 5px;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: monospace;
                font-size: 14px;
                line-height: 1.5;">
                {html.escape(text)}
            </div>
            """,
            unsafe_allow_html=True
        )

    # =======================
    # retrieve_vocabulary
    # =======================
    elif retrieve_type == "first_vocabulary":
        term = st.selectbox("Voce:", list(data.keys()))
        text = data[term]
        st.markdown(
            f"""
            <div style="
                background-color: #1e1e1e;
                color: #e0e0e0;
                padding: 12px;
                border-radius: 5px;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: monospace;
                font-size: 14px;
                line-height: 1.5;">
                {html.escape(text)}
            </div>
            """,
            unsafe_allow_html=True
        )

    # =======================
    # past_context
    # =======================

    elif retrieve_type == "past_context":
        st.markdown(
            f"""
            <div style="
                background-color: #1e1e1e;
                color: #e0e0e0;
                padding: 12px;
                border-radius: 5px;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: monospace;
                font-size: 14px;
                line-height: 1.5;">
                {html.escape(data)}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(f"<b>Widget non ancora implementato per {retrieve_type}</b>", unsafe_allow_html=True)


log = get_log(st.session_state.audit_path)

if "selected_log" not in st.session_state:
    st.write("Chat non ancora avviata")

else:
    # Dizionario con opzioni possibili in base all'indice (indice corrisponde al numero della risposta -1)
    pair_voc = show_titles_for_widget(map_documents= map_documents, all_metadata= retrieval_engine.metadata, vocabulary= retrieval_engine.vocabolario, logs= log, index= st.session_state.selected_log)

    show_retrieve_widget_st(pair_voc)


