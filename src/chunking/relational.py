"""Relational chunking via LlamaIndex PropertyGraphIndex.

Uses SimplePropertyGraphStore from llama-index-core with
PostgreSQL-backed persistence for entity-relation extraction.
"""

from typing import List, Optional

from llama_index.core import PropertyGraphIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.llms import LLM
from llama_index.core.schema import Document


class RelationalChunker:
    """Extracts entities and relationships from documents using a Property Graph.

    Uses an LLM (Qwen2.5-7B-Instruct) to identify Hungarian entities
    and relationships, storing them in a graph structure.
    """

    def __init__(self, llm: Optional[LLM] = None) -> None:
        self._graph_store = SimplePropertyGraphStore()
        self._llm = llm

    def build_index(
        self,
        documents: List[Document],
        embed_model: BaseEmbedding,
    ) -> PropertyGraphIndex:
        """Build a PropertyGraphIndex from documents.

        Extracts entities and relationships using the configured LLM,
        then stores them alongside vector embeddings.

        Args:
            documents: Documents to process.
            embed_model: Embedding model for vector representations.

        Returns:
            A PropertyGraphIndex for graph-based retrieval.
        """
        return PropertyGraphIndex.from_documents(
            documents,
            property_graph_store=self._graph_store,
            embed_model=embed_model,
            llm=self._llm,
            show_progress=True,
        )

    @property
    def graph_store(self) -> SimplePropertyGraphStore:
        """Access the underlying graph store."""
        return self._graph_store
