"""Semantic chunking via LlamaIndex SemanticSplitterNodeParser."""

from typing import List

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode, Document

from src.config.constants import SEMANTIC_BREAKPOINT_PERCENTILE, SEMANTIC_BUFFER_SIZE


class SemanticChunker:
    """Wraps SemanticSplitterNodeParser for Hungarian documents.

    Splits chunks where semantic meaning changes rather than at fixed
    token boundaries. Critical for Hungarian where agglutinative suffixes
    can change meaning mid-sentence.
    """

    def __init__(self, embed_model: BaseEmbedding) -> None:
        self._parser = SemanticSplitterNodeParser(
            buffer_size=SEMANTIC_BUFFER_SIZE,
            breakpoint_percentile_threshold=SEMANTIC_BREAKPOINT_PERCENTILE,
            embed_model=embed_model,
        )

    def chunk(self, documents: List[Document]) -> List[BaseNode]:
        """Split documents into semantic chunks."""
        return self._parser.get_nodes_from_documents(documents, show_progress=True)
