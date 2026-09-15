import numpy as np
from faiss import normalize_L2, SearchParametersIVF, PyCallbackIDSelector


def embed_query(text, model_emb, client):

    response = client.embeddings.create(
        model=model_emb,
        input=text
    )

    return np.array(response.data[0].embedding, dtype="float32")


def faiss_search(query_emb, index, valid_ids=None, k=20):

    normalize_L2(query_emb)

    if valid_ids is not None:

        def cond(i):
            return i in valid_ids

        sel = PyCallbackIDSelector(cond)

        params = SearchParametersIVF()
        params.sel = sel

        distances, indices = index.search(
            query_emb,
            k,
            params=params
        )

    else:

        distances, indices = index.search(query_emb, k)

    return distances[0], indices[0]