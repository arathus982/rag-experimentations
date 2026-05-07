"""Enumerations for the RAG evaluation framework."""

from enum import Enum


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    RELATIONAL = "relational"
    DOCUMENT_AWARE = "document_aware"


class EmbeddingModelName(str, Enum):
    """Embedding models under evaluation."""

    GTE_MULTILINGUAL_BASE = "gte-multilingual-base"
    MULTILINGUAL_E5_SMALL = "multilingual-e5-small"
    BGE_M3 = "bge-m3"
    QWEN3_EMBEDDING_8B = "qwen3-embedding-8b"


class RerankerModel(str, Enum):
    """Available reranker models for ensemble retrieval."""

    BGE = "bge"
    QWEN3 = "qwen3"


class RagasMetric(str, Enum):
    """RAGAS evaluation metrics for retrieval quality."""

    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    CONTEXT_ENTITIES_RECALL = "context_entities_recall"
