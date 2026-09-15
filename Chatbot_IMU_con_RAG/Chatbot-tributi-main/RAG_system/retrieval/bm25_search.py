from rank_bm25 import BM25Okapi


def bm25_score(query: str, bm25: BM25Okapi):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    max_score = max(scores) + 1e-4

    norm_scores = [s / max_score for s in scores]

    return norm_scores