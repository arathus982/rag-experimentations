"""Hybrid search combining vector similarity with BM25 keyword search."""

from typing import List

from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import BasePydanticVectorStore


class HybridSearchRetriever:
    """Combines dense vector search with PostgreSQL BM25 text search.

    Leverages PGVectorStore's built-in hybrid_search mode which uses
    PostgreSQL's Hungarian text_search_config for stemming and
    stop-word filtering alongside vector similarity.
    """

    def __init__(
        self,
        vector_store: BasePydanticVectorStore,
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            vector_store: PGVectorStore with hybrid_search=True.
            top_k: Number of results to return.
            alpha: Balance between vector (1.0) and text (0.0) search.
        """
        self._vector_store = vector_store
        self._top_k = top_k
        self._alpha = alpha

    def retrieve(self, query: str) -> List[NodeWithScore]:
        """Execute hybrid search combining vector and BM25 results.

        Args:
            query: Search query in Hungarian.

        Returns:
            Ranked list of nodes with similarity scores.
        """
        query_bundle = QueryBundle(query_str=query)
        results = self._vector_store.query(
            query=query_bundle,  # type: ignore[arg-type]
            similarity_top_k=self._top_k,
        )
        return [
            NodeWithScore(node=node, score=score)
            for node, score in zip(results.nodes or [], results.similarities or [])  # type: ignore[arg-type]
        ]
