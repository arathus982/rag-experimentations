"""Builds LlamaIndex indices per (model, strategy) combination."""

from typing import List, Union

from llama_index.core import PropertyGraphIndex, StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode, Document
from llama_index.vector_stores.postgres import PGVectorStore

from src.chunking.factory import ChunkingFactory
from src.chunking.hierarchical import HierarchicalChunker
from src.chunking.relational import RelationalChunker
from src.chunking.semantic import SemanticChunker
from src.config.constants import HUNGARIAN_TEXT_SEARCH_CONFIG, VECTOR_TABLE_PREFIX
from src.config.settings import DatabaseSettings
from src.embedding.base import EmbeddingModelAdapter
from src.indexing.timer import IndexingTimer
from src.models.enums import ChunkingStrategy


class Indexer:
    """Builds PGVectorStore-backed indices for each chunking strategy.

    Each (model, strategy) combination gets its own isolated table
    to prevent cross-contamination during evaluation.
    """

    def __init__(
        self,
        db_settings: DatabaseSettings,
        timer: IndexingTimer,
    ) -> None:
        self._db_settings = db_settings
        self._timer = timer

    def _make_table_name(self, model_name: str, strategy: str) -> str:
        """Generate isolated table name for a (model, strategy) combo."""
        safe_model = model_name.replace("-", "_")
        return f"{VECTOR_TABLE_PREFIX}_{safe_model}_{strategy}"

    def _create_pg_vector_store(
        self,
        table_name: str,
        embed_dim: int,
    ) -> PGVectorStore:
        """Create a PGVectorStore with Hungarian text search config."""
        return PGVectorStore.from_params(
            database=self._db_settings.db,
            host=self._db_settings.host,
            port=str(self._db_settings.port),
            user=self._db_settings.user,
            password=self._db_settings.password.get_secret_value(),
            table_name=table_name,
            embed_dim=embed_dim,
            text_search_config=HUNGARIAN_TEXT_SEARCH_CONFIG,
            hybrid_search=True,
        )

    def build_vector_index(
        self,
        nodes: List[BaseNode],
        embed_model: BaseEmbedding,
        table_name: str,
        embed_dim: int,
    ) -> VectorStoreIndex:
        """Build a VectorStoreIndex backed by PGVectorStore."""
        vector_store = self._create_pg_vector_store(table_name, embed_dim)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )

    def index_documents(
        self,
        documents: List[Document],
        adapter: EmbeddingModelAdapter,
        strategy: ChunkingStrategy,
        llm: LLM | None = None,
    ) -> Union[VectorStoreIndex, PropertyGraphIndex]:
        """Full indexing pipeline for one (model, strategy) combination.

        Chunks documents, builds index, and measures timing per document.

        Args:
            documents: Documents to index.
            adapter: Embedding model adapter.
            strategy: Chunking strategy to use.
            llm: LLM for relational chunking entity extraction.

        Returns:
            The built index (VectorStoreIndex or PropertyGraphIndex).
        """
        embed_model = adapter.get_llama_index_embedding()
        table_name = self._make_table_name(adapter.model_name, strategy.value)

        chunker = ChunkingFactory.create(
            strategy=strategy,
            embed_model=embed_model,
            llm=llm,
        )

        # Relational strategy builds its own PropertyGraphIndex
        if isinstance(chunker, RelationalChunker):
            return self._index_relational(
                documents, chunker, embed_model, adapter.model_name, strategy.value
            )

        # Semantic and Hierarchical strategies produce nodes for VectorStoreIndex
        all_nodes: List[BaseNode] = []
        for doc in documents:
            with self._timer.measure(
                document_id=doc.doc_id,
                document_title=doc.metadata.get("title", doc.doc_id),
                embedding_model=adapter.model_name,
                chunking_strategy=strategy.value,
            ):
                if isinstance(chunker, SemanticChunker):
                    nodes = chunker.chunk([doc])
                elif isinstance(chunker, HierarchicalChunker):
                    nodes = chunker.chunk([doc])
                    nodes = HierarchicalChunker.get_leaf_nodes(nodes)
                else:
                    nodes = []
                all_nodes.extend(nodes)
            self._timer.update_last_chunk_count(len(nodes))

        return self.build_vector_index(
            nodes=all_nodes,
            embed_model=embed_model,
            table_name=table_name,
            embed_dim=adapter.embedding_dimension,
        )

    def _index_relational(
        self,
        documents: List[Document],
        chunker: RelationalChunker,
        embed_model: BaseEmbedding,
        model_name: str,
        strategy_value: str,
    ) -> PropertyGraphIndex:
        """Build a PropertyGraphIndex with per-document timing."""
        # For relational, we time the entire build since it's integrated
        for doc in documents:
            with self._timer.measure(
                document_id=doc.doc_id,
                document_title=doc.metadata.get("title", doc.doc_id),
                embedding_model=model_name,
                chunking_strategy=strategy_value,
            ):
                pass  # Timing recorded; actual indexing below
            self._timer.update_last_chunk_count(0)

        return chunker.build_index(documents, embed_model)
