"""Named constants for the RAG evaluation framework."""

from typing import List

# Chunk sizes for hierarchical splitting (tokens)
HIERARCHICAL_CHUNK_SIZES: List[int] = [2048, 512, 256]

# Semantic splitter thresholds
SEMANTIC_BUFFER_SIZE: int = 1
SEMANTIC_BREAKPOINT_PERCENTILE: int = 95

# Retrieval defaults
DEFAULT_TOP_K: int = 10
SIMILARITY_THRESHOLD: float = 0.7

# Hungarian text search config for PostgreSQL
HUNGARIAN_TEXT_SEARCH_CONFIG: str = "pg_catalog.hungarian"

# Supported image extensions from Confluence
SUPPORTED_IMAGE_EXTENSIONS: List[str] = [".png", ".jpg", ".jpeg", ".gif", ".svg"]

# PGVectorStore table name template: vectors_{model}_{strategy}
VECTOR_TABLE_PREFIX: str = "vectors"

# Default data directory
DEFAULT_DATA_DIR: str = "data"

# Document-aware chunker: documents below this line count are indexed whole
DOCUMENT_AWARE_SMALL_DOC_LINE_THRESHOLD: int = 50
