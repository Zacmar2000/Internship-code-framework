import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# -------------------
# API
# -------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------
# Modelli
# -------------------

REASONING = True
REASONING_EFFORT = "low"

if REASONING:
    MODEL = "gpt-5.4-mini"
else:
    MODEL = "gpt-4.1-mini"

EMBEDDING_MODEL = "text-embedding-3-large"

REASONING_INTERNAL = True
REASONING_EFFORT_INTERNAL = "low"

if REASONING_INTERNAL:
    MODEL_INTERNAL = "gpt-5.4-mini"
else:
    MODEL_INTERNAL = "gpt-4.1-mini"

# ------------------------------------
# Reranker (usato solo nel caso "NEW")
# ------------------------------------

@st.cache_resource
def get_reranker():
    from sentence_transformers import CrossEncoder
    import warnings
    warnings.filterwarnings("ignore", message="Accessing `__path__` from")
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

# -------------------
# RAG parameters
# -------------------

K_RAG_NEW = 10
K_RAG = 7

# -------------------
# logging / audit
# -------------------

AUDIT = True
LOGS = True

# -------------------
# paths
# -------------------

EMBEDDING_PATH = f"IMU Embeddings/{EMBEDDING_MODEL}"

FAISS_INDEX_PATH = f"{EMBEDDING_PATH}/embeddings.faiss"
METADATA_PATH = f"{EMBEDDING_PATH}/metadata.json"

VOCABOLARIO_PATH = "IMU Embeddings/vocabolario.json"


# -------------------
# New or old code
# -------------------


USE_NEW = False # Meglio tenerlo False. Il nuovo codice fa una ricerca RAG solamente all'inizio, usando la query dell'utente ma i risultati non sono soddisfacenti.
