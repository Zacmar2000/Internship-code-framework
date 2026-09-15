from typing import Optional, Union, Mapping, Any
from pathlib import Path
import json
import warnings
import re
from datetime import datetime

import tiktoken


def get_log(name: Optional[Union[str, Path]] = None) -> list[dict]:

    base_dir = Path("Log_chats")

    if not base_dir.exists():
        warnings.warn(f"La directory {base_dir} non esiste.")
        return []

    # Se name è None → prendi il file .jsonl più recente
    if name is None:
        jsonl_files = list(base_dir.glob("*.jsonl"))

        if not jsonl_files:
            warnings.warn("Nessun file .jsonl trovato nella directory.")
            return []

        full_path = max(jsonl_files, key=lambda f: f.stat().st_mtime)
    
    else:

        name = Path(name)

        # aggiunge suffisso .jsonl se mancante
        if name.suffix != ".jsonl":
            name = name.with_suffix(".jsonl")

        # Se name contiene già una directory o è assoluto → usalo così
        if name.is_absolute() or name.parent != Path("."):
            full_path = name
        else:
            full_path = base_dir / name

    # controllo esistenza file
    if not full_path.exists():
        warnings.warn(f"Il file {full_path} non esiste.")
        return []
    
    with open(full_path, "r", encoding="utf-8") as f:
        logs = [json.loads(line) for line in f if line.strip()]
    return logs


def get_history(logs):
    history = []
    for i, log in enumerate(logs[:-1]):
        history.extend( [
            {
                                "role": "user",
                                        "content": "Quali sono le prossime scadenze?"
                                        },
                                {
                                "role": "assistant",
                                        "content":  log.get("final_answer")}
        ])
        logs[i+1]['history'] = history[-6:]



def extract_hierarchy(metadata: Mapping[str, Any]) -> list[str]:
    if not metadata:
        return []
    
    titles = []
    for key, value in metadata.items():
        if not value:  # ignora valori vuoti o None
            continue
        if key == "title" or key.startswith("title_"):
            # determina il numero della gerarchia
            if key == "title":
                num = 0
            else:
                try:
                    num = int(key.split("_")[1])
                except ValueError:
                    continue  # ignora chiavi malformate
            titles.append((num, value))
    
    # ordina per numero decrescente
    titles.sort(key=lambda x: -x[0])
    
    # restituisci solo i testi non vuoti
    return [t[1] for t in titles]


def reconstruct_chunk(sub_chunks, overlap, encoding_name="cl100k_base"):
    """
    sub_chunks: list of dicts with keys:
        - "text"
        - "metadata" -> must include "sub_chunk"
    """
    enc = tiktoken.get_encoding(encoding_name)

    # Ensure correct order
    sub_chunks = sorted(sub_chunks, key=lambda x: x["metadata"]["sub_chunk"])

    reconstructed_tokens = []

    for i, chunk in enumerate(sub_chunks):
        tokens = enc.encode(chunk["text"])

        if i == 0:
            # keep everything from the first chunk
            reconstructed_tokens.extend(tokens)
        else:
            # drop overlapping prefix
            reconstructed_tokens.extend(tokens[overlap:])

    return enc.decode(reconstructed_tokens)

from itertools import groupby

def group_by_metadata(chunks, keys):
    # keys può essere una lista tipo ["title", "periodo"]

    def build_key(x):
        metadata = x["metadata"]
        values = []
        for k in keys:
            v = metadata.get(k)

            # se è lista (come periodo), la rendiamo tupla per renderla hashable
            if isinstance(v, list):
                v = tuple(v)

            values.append(v)

        return tuple(values)

    chunks = sorted(chunks, key=build_key)

    return {
        k: list(g)
        for k, g in groupby(chunks, key=build_key)
    }

def dict_title_to_full_text(final_chunks):
    grouped = group_by_metadata(final_chunks, keys=["title", "periodo"])

    full_chunks = {}

    for section, sub_chunks in grouped.items():
        # ricostruisci il testo completo
        text = reconstruct_chunk(sub_chunks, overlap=60)

        # estrai tutti gli embedding_id del gruppo
        embedding_ids = [chunk["embedding_id"] for chunk in sub_chunks]

        # salva come dizionario con testo + embedding_ids
        full_chunks[section] = {
            "text": text,
            "embedding_ids": embedding_ids
        }

    return full_chunks

def normalize_reference(parts):
    articolo = None
    comma = None

    for part in parts:
        # ---- ARTICOLO ----
        art_match = re.search(
            r'\b(?:art\.?|articolo)\s*(\d+(?:-[a-z]+)?)',
            part,
            re.IGNORECASE
        )
        if art_match:
            numero = art_match.group(1)
            articolo = f"Art. {numero}"

        # ---- COMMA ----
        comma_match = re.search(
            r'\bcomma\s*(\d+(?:-[a-z]+)?)',
            part,
            re.IGNORECASE
        )
        if comma_match:
            numero = comma_match.group(1)
            comma = f"Comma {numero}"

    result = []
    if articolo:
        result.append(articolo)
    if comma:
        result.append(comma)

    return result

def text_normativa( all_metadata, emb_ids: list = [], only_titles = True):
            
            filtered_norm = [chunk for chunk in all_metadata if chunk['embedding_id'] in emb_ids]

            if filtered_norm:

                metadata = filtered_norm[0].get("metadata")

                document_name = metadata["document"]

                hierarchy = extract_hierarchy(metadata)

                if only_titles:
                    hierarchy = normalize_reference(hierarchy)

                if len(filtered_norm)>1:
                    filtered_norm.sort(
                            key=lambda s: (s['metadata']['sub_chunk'])
                        )
                    filtered_norm_dict = dict_title_to_full_text(filtered_norm)
                    if len(filtered_norm_dict) > 1:

                        def parse_end_period(key):
                            # key = (title, periodo)
                            periodo = key[1]

                            if isinstance(periodo, list):
                                end_date = periodo[1]
                            else:
                                end_date = periodo[1]

                            return datetime.strptime(end_date, "%d/%m/%Y").date()

                        # prende la chiave con periodo più avanti
                        latest_key = max(filtered_norm_dict.keys(), key=parse_end_period)

                        text_final = filtered_norm_dict[latest_key].get("text")
                    else:
                        
                        final_values = next(iter(filtered_norm_dict.values()))
                        text_final = final_values.get("text")

                else:
                    text_final = filtered_norm[0].get('text')
                



                return {
                        "documento": document_name,
                        "paragrafi": hierarchy,
                        "testo": text_final,
                    }
            else:
                return {}
            


def get_fonti(map_documents, all_metadata, vocabulary, logs, index: int):

    all_chunks = {}

    all_tools = logs[index].get("tool_calls")
    if all_tools:
        for index_order,tool_call in enumerate(all_tools):

            # Retrieve_context
            if tool_call.get("tool_name") == "retrieve_context":
                if "retrieve_context" not in all_chunks:
                    all_chunks["retrieve_context"] = {}
                current_chunks = []
                query = tool_call.get("args").get("query")
                year = tool_call.get("args").get("year")
                sources = tool_call.get("result").get("agent_sources")
                if sources:
                    for emb_id in sources:
                        current_chunks.append(next(iter([chunk for chunk in all_metadata if chunk.get("embedding_id") == emb_id.get('emb_id')])))
                
                if current_chunks:
                    all_chunks["retrieve_context"][("raw_chunks",query,year)] = current_chunks

                clean_results = tool_call.get("clean_results")
                if clean_results:
                    for clean_res in clean_results:
                        cat = clean_res.get("category")
                        text_cat = clean_res.get("result")
                        if ("clean_extraction",query,year) not in all_chunks["retrieve_context"]:
                            all_chunks["retrieve_context"][("clean_extraction",query,year)] = {}
                        all_chunks["retrieve_context"][("clean_extraction",query,year)][cat] = text_cat


            
            # Retrieve_normativa
            elif tool_call.get("tool_name") == "retrieve_normativa":
                if "retrieve_normativa" not in all_chunks:
                    all_chunks["retrieve_normativa"] = {}
                emb_ids = tool_call.get("result")

                dict_text = text_normativa(all_metadata= all_metadata, emb_ids=emb_ids, only_titles= True)
                if dict_text:
                    text_norm = dict_text.get("testo")
                    document = dict_text.get("documento")
                    if document in map_documents:
                        document = map_documents[document]
                    paragrafi = dict_text.get("paragrafi")

                    if not all_chunks["retrieve_normativa"].get(document):
                        all_chunks["retrieve_normativa"][document] = []
                    
                    all_chunks["retrieve_normativa"][document].append((paragrafi,text_norm))
            
            elif tool_call.get("tool_name") == "get_day_of_week":
                if "get_day_of_week" not in all_chunks:
                    all_chunks["get_day_of_week"] = {}
                date = tool_call.get("args").get("date")
                result_date = tool_call.get("result")
                all_chunks['get_day_of_week'][date] = result_date

            elif tool_call.get("tool_name") == "get_vocabulary":
                if "first_vocabulary" not in all_chunks:
                    all_chunks["first_vocabulary"] = {}
                vocab_cat = tool_call.get("args")
                text_cat = tool_call.get("result")
                for arg,val in zip(vocab_cat,text_cat):
                    if val:
                        choosen = [vocab for vocab in vocabulary['IMU'] if vocab['title'] == arg][0]
                        all_chunks["first_vocabulary"][arg] = choosen['text']
            
    start_tools = logs[index].get("starting_calls")
    if start_tools:
        for index_order,tool_call in enumerate(start_tools):
            if "first_extraction" not in all_chunks:
                all_chunks["first_extraction"] = {}
            current_chunks = []
            cat = tool_call.get("category")
            query = tool_call.get("args").get("query")
            year = tool_call.get("args").get("year")
            sources = tool_call.get("result")
            if sources:
                for emb_id in sources:
                    current_chunks.append(next(iter([chunk for chunk in all_metadata if chunk.get("embedding_id") == emb_id.get('emb_id')])))
            
            if current_chunks:
                all_chunks["first_extraction"][(cat,query,year)] = current_chunks

    start_text = logs[index].get("starting_text")
    if start_text:
        for index_order,tool_call in enumerate(start_text):
            if "first_rag" not in all_chunks:
                all_chunks["first_rag"] = {}
            current_chunks = []
            cat = tool_call.get("category")
            text_cat = tool_call.get("result")
            if text_cat:
                all_chunks["first_rag"][cat] = text_cat

        
    if logs[index].get("past_context"):
        all_chunks["past_context"] = logs[index].get("past_context")
    
                        
    
    return all_chunks



def show_titles_for_widget(map_documents, all_metadata, vocabulary, logs, index, only_titles=False):
    all_chunks_tool = get_fonti(map_documents= map_documents, all_metadata= all_metadata, vocabulary=vocabulary, logs=logs, index=index)

    for el in ["first_extraction","retrieve_context"]:
        all_chunks = all_chunks_tool.get(el)
        
        title_text_pairs = {}

        if all_chunks is None:
            continue
        
        for query,chunks_call in all_chunks.items():
            current_titles = {}
            if query[0] != "clean_extraction":
                for chunk in chunks_call:
                    metadata = chunk.get("metadata")
                    if metadata:
                        document = metadata.get("document")
                        if document:
                            if document in map_documents:
                                document = map_documents[document]
                            if not current_titles.get(document):
                                current_titles[document] = []
                            if only_titles:
                                title = metadata.get("title")
                                if title:
                                    current_titles[document].append(([title], chunk.get("text", "")))
                            else:
                                hierarchy = extract_hierarchy(metadata=metadata)
                                if metadata.get("categoria") == 'Normativa':
                                    hierarchy = normalize_reference(hierarchy)
                                    if not hierarchy:
                                        hierarchy = extract_hierarchy(metadata= metadata)
                                current_titles[document].append((hierarchy,chunk.get("text", "")))
                            subchunk = metadata.get("sub_chunk")
                            if subchunk:
                                if subchunk>0:
                                    current_titles[document][-1][0][-1] = current_titles[document][-1][0][-1]+ f" subchunk_{subchunk}"

                title_text_pairs[query] = current_titles

            else:
                title_text_pairs[query] = chunks_call
        
        all_chunks_tool[el] = title_text_pairs
    
    return all_chunks_tool