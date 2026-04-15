"""Relational chunking via LlamaIndex PropertyGraphIndex.

Uses SimplePropertyGraphStore from llama-index-core with
PostgreSQL-backed persistence for entity-relation extraction.
"""

from typing import List, Optional

from llama_index.core import PropertyGraphIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import ImplicitPathExtractor, SimpleLLMPathExtractor
from llama_index.core.schema import TransformComponent
from llama_index.core.llms import LLM
from llama_index.core.schema import Document


class RelationalChunker:
    """Extracts entities and relationships from documents using a Property Graph.

    When an LLM is provided, uses SimpleLLMPathExtractor for deep entity
    and relation extraction. Without an LLM, falls back to ImplicitPathExtractor
    which uses pattern matching — lighter but sufficient for structural extraction.
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

        Args:
            documents: Documents to process.
            embed_model: Embedding model for vector representations.

        Returns:
            A PropertyGraphIndex for graph-based retrieval.
        """
        extractors: List[TransformComponent]
        if self._llm:
            extractors = [SimpleLLMPathExtractor(llm=self._llm)]
        else:
            extractors = [ImplicitPathExtractor()]

        return PropertyGraphIndex.from_documents(
            documents,
            property_graph_store=self._graph_store,
            kg_extractors=extractors,
            embed_model=embed_model,
            show_progress=True,
        )

    @property
    def graph_store(self) -> SimplePropertyGraphStore:
        """Access the underlying graph store."""
        return self._graph_store
