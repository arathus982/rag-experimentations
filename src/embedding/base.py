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

    def prefetch(self) -> None:
        """Download model weights to local cache without loading to GPU memory."""
        ...

    def unload(self) -> None:
        """Release GPU/MPS memory held by this model.

        Must be called after indexing is complete for this model and before
        loading the next model. Deletes internal references and flushes the
        device cache so the allocator reclaims the memory.
        """
        ...

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts into embedding vectors."""
        ...
