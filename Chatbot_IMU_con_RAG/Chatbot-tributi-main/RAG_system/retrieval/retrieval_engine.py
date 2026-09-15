from typing import List, Optional

from faiss import Index
from rank_bm25 import BM25Okapi

from RAG_system.retrieval.hybrid_search import hybrid_search
from RAG_system.retrieval.hybrid_search_new import hybrid_search_new


class RetrievalEngine:

    def __init__(
        self,
        index: Index,
        metadata: List[dict],
        bm25: BM25Okapi,
        client,
        embedding_model: str,
        vocabolario: dict
    ):
        """
        Retrieval engine per RAG.

        Parameters
        ----------
        index : FAISS index
        metadata : lista chunk metadata
        bm25 : BM25 index
        client : OpenAI client
        embedding_model : nome modello embedding
        """

        self.index = index
        self.metadata = metadata
        self.bm25 = bm25
        self.client = client
        self.embedding_model = embedding_model
        self.vocabolario = vocabolario
        self.titles_vocabulary = [el['title'] for el in self.vocabolario['IMU']]
    def search(
        self,
        query: str,
        year: Optional[int] = None,
        k: int = 5,
    ):
        """
        Esegue hybrid retrieval.
        """

        return hybrid_search(
            query=query,
            model_emb=self.embedding_model,
            client=self.client,
            index=self.index,
            all_metadata=self.metadata,
            bm25=self.bm25,
            k=k,
            year=year,
        )
    
    def search_new(
        self,
        query: str,
        year: Optional[int] = None,
        category: Optional[str] = None,
        k: int = 5,
    ):
        """
        Esegue hybrid retrieval.
        """

        return hybrid_search_new(
            query=query,
            model_emb=self.embedding_model,
            client=self.client,
            index=self.index,
            all_metadata=self.metadata,
            bm25=self.bm25,
            k=k,
            year=year,
            category=category
        )
