"""Ensemble retriever that merges multiple indices and reranks the combined pool."""

from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from rich.console import Console

from src.retrieval.reranker import BaseReranker

console = Console()


class EnsembleRetriever:
    """Retrieves from multiple VectorStoreIndex instances, deduplicates, and reranks.

    Each index is queried independently for `initial_top_k` results.
    The merged pool is deduplicated by content hash, then reranked to `final_top_k`.
    """

    def __init__(
        self,
        indices: List[VectorStoreIndex],
        reranker: BaseReranker,
        initial_top_k: int = 20,
        final_top_k: int = 10,
    ) -> None:
        self._retrievers = [idx.as_retriever(similarity_top_k=initial_top_k) for idx in indices]
        self._reranker = reranker
        self._final_top_k = final_top_k

    def retrieve(self, query: str) -> List[NodeWithScore]:
        """Retrieve, deduplicate, and rerank nodes for a query."""
        seen: set[int] = set()
        merged: List[NodeWithScore] = []

        for retriever in self._retrievers:
            for node in retriever.retrieve(query):
                content_hash = hash(node.node.get_content())
                if content_hash not in seen:
                    seen.add(content_hash)
                    merged.append(node)

        console.print(f"  [dim]Merged pool: {len(merged)} unique chunks → reranking to {self._final_top_k}[/dim]")
        return self._reranker.rerank(query, merged, self._final_top_k)
