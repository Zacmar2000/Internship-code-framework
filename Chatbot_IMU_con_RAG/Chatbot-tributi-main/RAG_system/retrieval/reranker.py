from collections import defaultdict

def build_query_from_session(user_query, session_history, max_tokens=500):
    """
    Arricchisce la query con i messaggi recenti della sessione.
    session_history: lista di dict {"role": ..., "content": ...}
    """
    # Prendi gli ultimi 6 messaggi (o meno se non ci sono)
    recent_history = session_history[-6:]
    
    # Concateno solo il contenuto
    history_text = " ".join(msg["content"] for msg in recent_history if "content" in msg)
    
    # Se la query è troppo breve, la concateno alla history
    if len(user_query.split()) < 3 and history_text:
        enriched_query = f"{user_query} {history_text}"
    else:
        enriched_query = user_query
    
    # Limita lunghezza
    return enriched_query[:max_tokens]

def global_rerank(
    user_query,
    category_results,   # dict: {cat: {sources: [chunks], year, query}}
    reranker,
    session=None,       # oggetto session con metodo get_history()
    top_k=10
):
    # Preparo la history della sessione
    session_history = session.get_history() if session else []
    
    enriched_query = build_query_from_session(user_query, session_history)

    all_chunks = []

    for cat, sources in category_results.items():
        chunks = sources.get("sources")
        if chunks:
            for c in chunks:
                c_copy = c.copy()
                c_copy["categoria"] = cat  # traccia origine
                c_copy["year"] = sources.get("year")  # traccia anno
                c_copy["query"] = sources.get("query")  # traccia query della categoria
                all_chunks.append(c_copy)

    if not all_chunks:
        return []

    all_chunks = deduplicate_chunks(all_chunks)

    # Passo al reranker
    pairs = [(enriched_query, c["text"]) for c in all_chunks]
    scores = reranker.predict(pairs)

    chunks_with_scores = []
    for score, chunk in zip(scores, all_chunks):
        chunk_copy = chunk.copy()
        chunk_copy["similarity"] = score
        chunks_with_scores.append((score, chunk_copy))

    # Ordino e prendo i top K
    ranked = sorted(chunks_with_scores, key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, c in ranked[:top_k]]

    return top_chunks

def regroup_by_category(chunks):
    grouped = defaultdict(list)

    for c in chunks:
        grouped[c["categoria"]].append(c)

    return grouped

def deduplicate_chunks(chunks):
    seen = set()
    unique = []

    for c in chunks:
        key = c["text"][:200]  # semplice fingerprint

        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique