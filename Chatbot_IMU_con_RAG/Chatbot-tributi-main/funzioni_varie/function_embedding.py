import os
from pathlib import Path
import pickle
import time
import json
import faiss
import numpy as np

from openai import OpenAI

def path_for_extraction(pdf_path, category = "chunk"):
    """
    Definizione Path per salvataggio files
    """

    output_root = Path(f"Extracted/{category}")

    pdf_path_extracted = output_root / Path(pdf_path).with_suffix(".pkl")

    # crea le cartelle se non esistono
    pdf_path_extracted.parent.mkdir(parents=True, exist_ok=True)

    return pdf_path_extracted

def path_for_embedding(model, IMU= False):
    
    # Cartella principale per gli embedding
    if IMU:
        folder_path = Path(f"IMU embeddings/{model}")
    else:
        folder_path = Path(f"Embeddings/{model}")
    
    # Creazione della cartella (incluse eventuali cartelle mancanti)
    folder_path.mkdir(parents=True, exist_ok=True)
    
    return folder_path

CHECKPOINT_FILE = "embeddings_checkpoint.json"

def save_checkpoint(data, last_index):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "last_index": last_index,
            "data": data
        }, f, ensure_ascii=False)

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return 0, []
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    return checkpoint["last_index"], checkpoint["data"]


def create_embeddings(documents, client, model="text-embedding-3-large", batch_size=60, time_between_tries=1.5, first_batch_size=60):
    all_embeddings = []

    start_index, saved_data = load_checkpoint()
    all_embeddings.extend(saved_data)

    print(f"Riparto dal documento {start_index}")

    i = start_index
    first_batch = True
    len_batch = 0

    if first_batch_size <= 0:
        first_batch = False
        time.sleep(time_between_tries)

    while i < len(documents):
        # Determina la dimensione del batch
        current_batch_size = first_batch_size if first_batch else batch_size
        batch = documents[i:i + current_batch_size]
        len_batch = len(batch)
        texts = [doc["text"] for doc in batch]

        try:
            response = client.embeddings.create(
                model=model,
                input=texts
            )

            for j, emb in enumerate(response.data):
                all_embeddings.append({
                    "embedding": emb.embedding,
                    "text": batch[j]["text"],
                    "metadata": batch[j]["metadata"]
                })

            i += len(batch)  # aggiorna l'indice

            save_checkpoint(all_embeddings, i)
            print(f"Batch fino al documento {i} completato")

            if i < len(documents):
                time.sleep(time_between_tries)
                first_batch = False      # i batch successivi useranno batch_size normale

        except Exception as e:
            print(f"Errore al batch {i}: {e}")
            print("Checkpoint salvato. Puoi riprendere più tardi.")
            save_checkpoint(all_embeddings, i)
            raise e  # fermati

    print("Embedding completato")
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint cancellato")
    
    final_batch_size = len_batch + (batch_size - first_batch_size) if first_batch else len_batch


    return all_embeddings, final_batch_size


def insert_pdf_in_faiss(
    pdf_path: Path,
    api_key: str,
    model: str = "text-embedding-3-large",
    Gemini: bool = False,
    batch_size: int = 60,
    time_between_tries: float = 1.5,
    last_batch_size: int = 0,
    on_duplicate_document: str = "ask",   # "ask" → input umano, "skip" → fermati, "force" → continua senza chiedere
    IMU: bool = False
):
    # ---- Client ----
    if Gemini:
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)

    #pdf_path = pdf_path.resolve()

    path_embedding = path_for_embedding(model=model,IMU=IMU)

    index_path = path_embedding / "embeddings.faiss"
    metadata_path = path_embedding / "metadata.json"

    # ---- Carica o crea index ----
    if index_path.exists() and metadata_path.exists():
        index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
    else:
        index = None
        all_metadata = []

    # ---- Controllo PDF già indicizzato ----
    already_indexed_pdfs = {m["source_pdf"] for m in all_metadata}

    name_indexed_pdfs = {m['metadata']['document'] for m in all_metadata}

    print()

    if str(Path(pdf_path)) in already_indexed_pdfs:
        print(f"{str(Path(pdf_path))} già indicizzato. Skip.")
        return last_batch_size

    print(f"Indicizzazione di {str(pdf_path)}")

    # ---- Carica chunk del PDF ----
    pdf_path_chunk = path_for_extraction(pdf_path)

    if not pdf_path_chunk.exists():
        print(f"{str(Path(pdf_path))} non esiste diviso in chunk. Skip.")
        return last_batch_size

    with open(pdf_path_chunk, "rb") as f:
        chunks = pickle.load(f)

    document_name = chunks[0]['metadata']['document']

    if document_name in name_indexed_pdfs:

        if on_duplicate_document == "skip":
            return last_batch_size
        
        elif on_duplicate_document == "ask":

            print(f"{document_name} già indicizzato, ma nome del file diverso. Continuare?")

            while True:
                answer = input("Vuoi continuare comunque? [s/n]: ").strip().lower()
                if answer in ("s", "si", "y", "yes"):
                    break
                elif answer in ("n", "no"):
                    print("⛔ Operazione annullata.")
                    return last_batch_size
                else:
                    print("Risposta non valida. Scrivi 's' o 'n'.")


    new_embeddings, len_batch = create_embeddings(
        chunks,
        client=client,
        model=model,
        batch_size=batch_size,
        time_between_tries=time_between_tries,
        first_batch_size= batch_size - last_batch_size
    )

    if not new_embeddings:
        print("Nessun embedding creato")
        return len_batch

    # ---- Crea index se non esiste ----
    if index is None:
        dimension = len(new_embeddings[0]["embedding"])
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
    

    vectors = np.array([e["embedding"] for e in new_embeddings], dtype="float32")
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    
    faiss.normalize_L2(vectors)

    next_id = (
        max(m["embedding_id"] for m in all_metadata) + 1
        if all_metadata
        else 0
    )

    ids = np.arange(next_id, next_id + len(vectors), dtype="int64")
    index.add_with_ids(vectors, ids)  # type: ignore

    # ---- Metadata ----

    metadata_only = []

    for i, e in enumerate(new_embeddings):
        meta = e["metadata"]

        if meta.get("title") == f"introduzione_{os.path.basename(pdf_path)}":
            meta["title"] = "Introduzione"

        metadata_only.append(
            {
                "embedding_id": int(ids[i]),
                "source_pdf": str(Path(pdf_path)),
                "text": e["text"],
                "metadata": meta,
            }
        )

    all_metadata.extend(metadata_only)

    # ---- Persistenza ----
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"{str(Path(pdf_path))} indicizzato con successo")

    print("---------")

    return len_batch


def remove_pdf_from_index(
    pdf_path: Path,
    model: str = "text-embedding-3-large",
    verbose: bool = True,
    ask_confirmation: bool = True,
    IMU: bool = False
):
    """
    Rimuove tutti gli embeddings e i metadata associati a un PDF dal FAISS index.

    Parametri:
    - pdf_path: Path del PDF da rimuovere
    - model: modello di embedding
    - verbose: se True stampa info
    - ask_confirmation: se True chiede conferma prima di cancellare
    """
    path_embedding = path_for_embedding(model=model, IMU = IMU)
    index_path = path_embedding / "embeddings.faiss"
    metadata_path = path_embedding / "metadata.json"

    if not index_path.exists() or not metadata_path.exists():
        if verbose:
            print("Index o metadata non trovati.")
        return

    # --- Carica index e metadata ---
    index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        all_metadata = json.load(f)

    # --- Trova gli ID dei chunk del pdf_path ---
    ids_to_remove = [
        m["embedding_id"] for m in all_metadata
        if m["source_pdf"] == str(Path(pdf_path))
    ]

    if not ids_to_remove:
        if verbose:
            print(f"Nessun embedding trovato per {str(Path(pdf_path))}")
        return

    # --- Chiedi conferma se richiesto ---
    if ask_confirmation:
        while True:
            answer = input(
                f"Vuoi davvero eliminare {len(ids_to_remove)} embeddings per '{str(Path(pdf_path))}'? [s/n]: "
            ).strip().lower()
            if answer in ("s", "si", "y", "yes"):
                break
            elif answer in ("n", "no"):
                if verbose:
                    print("Operazione annullata.")
                return
            else:
                print("Risposta non valida. Scrivi 's' o 'n'.")

    # --- Rimuovi dal index ---
    id_array = np.array(ids_to_remove, dtype='int64')
    index.remove_ids(id_array)

    # --- Rimuovi dal metadata ---
    all_metadata = [m for m in all_metadata if m["source_pdf"] != str(Path(pdf_path))]

    # --- Salva tutto ---
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"{len(ids_to_remove)} embeddings rimossi per {str(Path(pdf_path))}")

def remove_from_index_by_id(
    ids: list,
    model: str = "text-embedding-3-large",
    verbose: bool = True,
    ask_confirmation: bool = True,
    IMU: bool = False
):
    """
    Rimuove tutti gli embeddings e i metadata associati a un PDF dal FAISS index.

    Parametri:
    - ids: id da rimuovere
    - model: modello di embedding
    - verbose: se True stampa info
    - ask_confirmation: se True chiede conferma prima di cancellare
    """
    path_embedding = path_for_embedding(model=model, IMU=IMU)
    index_path = path_embedding / "embeddings.faiss"
    metadata_path = path_embedding / "metadata.json"

    if not index_path.exists() or not metadata_path.exists():
        if verbose:
            print("Index o metadata non trovati.")
        return

    # --- Carica index e metadata ---
    index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        all_metadata = json.load(f)


    ids_to_remove = [
        m["embedding_id"] for m in all_metadata
        if m["embedding_id"] in ids
    ]


    if not ids_to_remove:
        if verbose:
            print(f"Nessun embedding_id valido dato")
        return

    # --- Chiedi conferma se richiesto ---
    if ask_confirmation:
        while True:
            answer = input(
                f"Vuoi davvero eliminare {len(ids_to_remove)} embeddings ? [s/n]: "
            ).strip().lower()
            if answer in ("s", "si", "y", "yes"):
                break
            elif answer in ("n", "no"):
                if verbose:
                    print("Operazione annullata.")
                return
            else:
                print("Risposta non valida. Scrivi 's' o 'n'.")

    # --- Rimuovi dal index ---
    id_array = np.array(ids_to_remove, dtype='int64')
    index.remove_ids(id_array)

    # --- Rimuovi dal metadata ---
    all_metadata = [m for m in all_metadata if not (m["embedding_id"] in ids_to_remove)]

    # --- Salva tutto ---
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"{len(ids_to_remove)} embeddings rimossi")


def fix_metadata_titles(metadata_path: Path, pdf_path: Path):
    if not metadata_path.exists():
        print("Index o metadata non trovati.")
        return
    with open(metadata_path, "r", encoding="utf-8") as f:
        all_metadata = json.load(f)

    target = f"introduzione_{os.path.basename(pdf_path)}"

    changed = 0
    for batch in all_metadata:
        meta = batch.get("metadata", {})
        if meta.get("title") == target:
            meta["title"] = "Introduzione"
            changed += 1

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Aggiornati {changed} metadata")


def check_embeddings_done(
    model: str = "text-embedding-3-large",
    pdf_names = True,
    IMU: bool = False
):

    #pdf_path = pdf_path.resolve()

    path_embedding = path_for_embedding(model=model, IMU=IMU)

    metadata_path = path_embedding / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
    else:
        print("Nessun embedding per questo modello")
        return

    # ---- Controllo PDF già indicizzato ----
    already_indexed_pdfs = {m["source_pdf"] for m in all_metadata}

    name_files = {m['metadata']['document'] for m in all_metadata}

    if pdf_names:
        return already_indexed_pdfs
    else:
        return name_files