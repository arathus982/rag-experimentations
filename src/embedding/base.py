"""Protocol definition for embedding model adapters."""

from typing import List, Protocol, runtime_checkable

from llama_index.core.base.embeddings.base import BaseEmbedding


@runtime_checkable
class EmbeddingModelAdapter(Protocol):
    """Interface all embedding model adapters must satisfy."""

    @property
    def model_name(self) -> str:
        """Unique identifier for this model."""
        ...

    @property
    def embedding_dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...

    def get_llama_index_embedding(self) -> BaseEmbedding:
        """Return a LlamaIndex-compatible embedding model instance."""
        ...

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts into embedding vectors."""
        ...
