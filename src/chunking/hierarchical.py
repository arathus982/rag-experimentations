"""Hierarchical chunking with parent-child node relationships."""

from typing import List

from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import BaseNode, Document
from llama_index.core.storage import StorageContext
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from src.config.constants import DEFAULT_TOP_K, HIERARCHICAL_CHUNK_SIZES


class HierarchicalChunker:
    """Wraps HierarchicalNodeParser with AutoMergingRetriever support.

    Creates a 3-level hierarchy (2048 -> 512 -> 128 tokens).
    Retrieves precise child chunks for similarity matching,
    then merges into parent context for LLM consumption.
    """

    def __init__(self) -> None:
        self._parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=HIERARCHICAL_CHUNK_SIZES,
        )

    def chunk(self, documents: List[Document]) -> List[BaseNode]:
        """Create hierarchical nodes (parent -> child -> leaf)."""
        return self._parser.get_nodes_from_documents(documents, show_progress=True)

    @staticmethod
    def get_leaf_nodes(nodes: List[BaseNode]) -> List[BaseNode]:
        """Extract only the leaf-level nodes for indexing."""
        return get_leaf_nodes(nodes)

    @staticmethod
    def build_retriever(
        vector_store: BasePydanticVectorStore,
        storage_context: StorageContext,
        top_k: int = DEFAULT_TOP_K,
    ) -> AutoMergingRetriever:
        """Build an AutoMergingRetriever that merges child nodes into parents.

        Args:
            vector_store: The vector store backing the index.
            storage_context: Storage context with the document store.
            top_k: Number of top results to retrieve.

        Returns:
            Configured AutoMergingRetriever.
        """
        from llama_index.core import VectorStoreIndex

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )
        base_retriever = index.as_retriever(similarity_top_k=top_k)
        return AutoMergingRetriever(
            vector_retriever=base_retriever,  # type: ignore[arg-type]
            storage_context=storage_context,
        )
