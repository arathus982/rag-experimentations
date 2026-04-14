"""Factory for creating chunking strategy instances."""

from typing import Optional, Union

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM

from src.chunking.hierarchical import HierarchicalChunker
from src.chunking.relational import RelationalChunker
from src.chunking.semantic import SemanticChunker
from src.models.enums import ChunkingStrategy


class ChunkingFactory:
    """Creates chunking strategy instances by enum name."""

    @staticmethod
    def create(
        strategy: ChunkingStrategy,
        embed_model: Optional[BaseEmbedding] = None,
        llm: Optional[LLM] = None,
    ) -> Union[SemanticChunker, HierarchicalChunker, RelationalChunker]:
        """Create a chunker for the specified strategy.

        Args:
            strategy: Which chunking strategy to use.
            embed_model: Required for semantic chunking.
            llm: Required for relational chunking (entity extraction).

        Returns:
            An initialized chunker instance.
        """
        match strategy:
            case ChunkingStrategy.SEMANTIC:
                if embed_model is None:
                    raise ValueError("Semantic chunking requires an embedding model")
                return SemanticChunker(embed_model)
            case ChunkingStrategy.HIERARCHICAL:
                return HierarchicalChunker()
            case ChunkingStrategy.RELATIONAL:
                return RelationalChunker(llm=llm)
