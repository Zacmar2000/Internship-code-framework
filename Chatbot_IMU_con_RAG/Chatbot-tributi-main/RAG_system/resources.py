import json
from faiss import read_index
import re
from rank_bm25 import BM25Okapi
from openai import OpenAI
import streamlit as st

from RAG_system.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    VOCABOLARIO_PATH,
    MODEL_INTERNAL,
    REASONING_INTERNAL,
    REASONING_EFFORT_INTERNAL
)

from RAG_system.retrieval.retrieval_engine import RetrievalEngine
from RAG_system.tools.rag_tools import RAGTools


@st.cache_resource
def get_resources():

    return load_rag_resources()


def load_rag_resources():

    # OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    # FAISS
    index = read_index(FAISS_INDEX_PATH)

    # metadata
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # BM25
    tokenized_corpus = [doc["text"].split() for doc in metadata]
    bm25 = BM25Okapi(tokenized_corpus)

    # vocabolario
    with open(VOCABOLARIO_PATH, "r", encoding="utf-8") as f:
        vocabolario = json.load(f)

    # Retrieval engine
    retrieval_engine = RetrievalEngine(
        index=index,
        metadata=metadata,
        bm25=bm25,
        client=client,
        embedding_model=EMBEDDING_MODEL,
        vocabolario=vocabolario
    )

    rag_tools = RAGTools(retrieval_engine=retrieval_engine, model= MODEL_INTERNAL, reasoning= REASONING_INTERNAL, reasoning_effort= REASONING_EFFORT_INTERNAL)

    # Part for view of logs

    all_titles = list(set([el['metadata']['document'] for el in retrieval_engine.metadata]))

    # Regex per estrarre tipo + data + numero
    pattern = re.compile(
        r"(legge|decreto(?:-legge)?|decreto legislativo|decreto ministeriale) "  # Tipo atto
        r"(?:del\s*)?"  # opzionale "del"
        r"(\d{1,2} [a-z]+ \d{4}|\d{2}/\d{2}/\d{4})"  # Data in formato "30 dicembre 1992" o "04/05/2023"
        r"(?:, n\. \d+)?"  # Numero legge/decreto opzionale
        , flags=re.IGNORECASE
    )

    map_documents = {}

    for testo in all_titles:
        match = pattern.search(testo)
        if match:
            # Combiniamo tipo + data + numero se presente
            estratto = match.group(0)
            map_documents[testo] = estratto

    return retrieval_engine, rag_tools, client, map_documents