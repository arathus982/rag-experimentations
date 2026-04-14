"""Enumerations for the RAG evaluation framework."""

from enum import Enum


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    RELATIONAL = "relational"


class EmbeddingModelName(str, Enum):
    """Embedding models under evaluation."""

    HARRIER_OSS_V1 = "harrier-oss-v1"
    QWEN3_EMBEDDING_8B = "qwen3-embedding-8b"
    BGE_M3 = "bge-m3"


class RagasMetric(str, Enum):
    """RAGAS evaluation metrics for retrieval quality."""

    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    CONTEXT_ENTITIES_RECALL = "context_entities_recall"
