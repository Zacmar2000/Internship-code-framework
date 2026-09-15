from typing import List, Optional, Iterable

import numpy as np
from faiss import Index

from RAG_system.retrieval.faiss_search import embed_query, faiss_search
from RAG_system.retrieval.bm25_search import bm25_score


def is_in_year(year: int, metadata: dict) -> bool:
    anni = metadata.get("anno")

    if anni is None:
        return True

    if isinstance(anni, int):
        return year == anni

    if isinstance(anni, Iterable):
        return year in anni

    return False


def hybrid_search(
    query: str,
    model_emb: str,
    client,
    index: Index,
    all_metadata: List[dict],
    bm25,
    k: int = 5,
    year: Optional[int] = None,
    rrf_k: int = 60
):
    """
    Hybrid retrieval FAISS + BM25.

    Pipeline:
    1. embedding query
    2. FAISS search
    3. BM25 scoring
    4. score ibrido
    5. ranking finale
    """

    # =====================
    # embedding query
    # =====================

    q_emb = embed_query(
        text=query,
        model_emb=model_emb,
        client=client
    ).reshape(1, -1)

    q_emb = np.ascontiguousarray(q_emb, dtype="float32")

    # =====================
    # mapping utili
    # =====================

    id_to_position = {
        m["embedding_id"]: i
        for i, m in enumerate(all_metadata)
    }

    # =====================
    # BM25
    # =====================

    bm25_scores_full = bm25_score(query, bm25)

    # =====================
    # filtro anno
    # =====================

    if year is None:
        metadata_year = all_metadata
    else:
        metadata_year = [
            m for m in all_metadata
            if is_in_year(year, m["metadata"])
        ]

    # =====================
    # priorità regolamento IMU
    # =====================

    metadata_main = [
        m for m in metadata_year
        if (
            m["source_pdf"] == "IMU\\Normativa\\160_2019"
            and m["metadata"]["title"].startswith("Comma ")
        )
    ]

    ids_main = {m["embedding_id"] for m in metadata_main}

    metadata_other = [
        m for m in metadata_year
        if m["embedding_id"] not in ids_main
    ]

    bm25_scores_full = bm25_score(query, bm25)

    id_to_position = {
        m["embedding_id"]: i
        for i, m in enumerate(all_metadata)
    }

    results_full = []

    # =====================
    # search sui gruppi
    # =====================

    for metadata_subset in [metadata_main, metadata_other]:

        if not metadata_subset:
            results_full.append([])
            continue

        id_to_chunk = {
            chunk["embedding_id"]: chunk
            for chunk in metadata_subset
        }

        valid_ids = set(id_to_chunk.keys())

        # =====================
        # FAISS SEARCH
        # =====================

        k_tot = min(k * 10, 40)

        faiss_scores, faiss_indices = faiss_search(
            query_emb=q_emb,
            index=index,
            valid_ids=valid_ids,
            k=k_tot
        )

        faiss_results = []

        for score, idx in zip(faiss_scores, faiss_indices):

            if idx not in valid_ids:
                continue

            chunk = id_to_chunk.get(idx)

            if chunk is None:
                continue

            faiss_results.append(chunk)

        # =====================
        # BM25 SEARCH
        # =====================

        bm25_results = []

        for emb_id in valid_ids:

            pos = id_to_position.get(emb_id)

            if pos is None:
                continue

            score = bm25_scores_full[pos]

            if score <= 0:
                continue

            chunk = id_to_chunk.get(emb_id)

            if chunk is None:
                continue

            bm25_results.append((score, chunk))

        bm25_results.sort(
            key=lambda x: x[0],
            reverse=True
        )

        bm25_results = [c for _, c in bm25_results[:k_tot]]

        # =====================
        # RRF FUSION
        # =====================

        fused_scores = {}

        for rank, chunk in enumerate(faiss_results):

            emb_id = chunk["embedding_id"]

            score = 1 / (rrf_k + rank)

            fused_scores[emb_id] = fused_scores.get(emb_id, 0) + score

        for rank, chunk in enumerate(bm25_results):

            emb_id = chunk["embedding_id"]

            score = 1 / (rrf_k + rank)

            fused_scores[emb_id] = fused_scores.get(emb_id, 0) + score

        # =====================
        # costruzione risultati
        # =====================

        results = []

        for emb_id, score in fused_scores.items():

            chunk = id_to_chunk.get(emb_id)

            if chunk is None:
                continue

            chunk["metadata"]["score"] = score

            results.append(chunk)

        results.sort(
            key=lambda x: x["metadata"]["score"],
            reverse=True
        )

        results_full.append(results[:k])

    return results_full