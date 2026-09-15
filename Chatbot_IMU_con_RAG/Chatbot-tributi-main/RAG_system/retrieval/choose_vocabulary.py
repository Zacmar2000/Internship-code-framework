import json
import re
from pathlib import Path

from RAG_system.tools.rag_tools import RAGTools


def extract_selected(response, valid_options):
    try:
        raw_text = response.output_text
    except Exception:
        return []

    # 1. prova parsing diretto
    try:
        data = json.loads(raw_text)
        selected = data.get("selected", [])
    except Exception:
        # 2. prova a estrarre JSON con regex
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                selected = data.get("selected", [])
            except Exception:
                selected = []
        else:
            selected = []

    # 3. fallback: se ha restituito una lista pura
    if isinstance(selected, str):
        selected = [selected]

    # 4. FAILSAFE CRITICO 🚨
    # filtra SOLO valori presenti nella vocabulary
    selected = [s for s in selected if s in valid_options]

    return selected


def choose_vocab(query:str,
                rag_tools: RAGTools,
                vocabulary_used: list):
    
    retrieval_engine = rag_tools.retrieval_engine

    prompt_dir = Path(__file__).resolve().parents[2] / "Prompts"
    prompt = (prompt_dir / "prompt_decision_vocabulary.txt").read_text()

    prompt += "\n\nOpzioni disponibili (usa SOLO queste stringhe)\n" + json.dumps(retrieval_engine.titles_vocabulary, indent=4)

    client = retrieval_engine.client

    input_messages = [
        {"role": "system", "content": prompt}
    ]
    input_messages.append({"role": "user", "content": query})

    client = retrieval_engine.client

    params = {
        "model": rag_tools.model,
        "input": input_messages,
    }

    if rag_tools.reasoning:
        params["reasoning"] = {"effort": rag_tools.reasoning_effort}

    response = client.responses.create(**params)

    selected_all = extract_selected(response, retrieval_engine.titles_vocabulary)
    selected = [s for s in selected_all if s not in vocabulary_used]

    sources, agent_sources = rag_tools.retrieve_vocabulary(selected)

    return sources, agent_sources, selected




