"""Embedding adapter for Qwen3-Embedding-8B."""

from typing import List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.models.enums import EmbeddingModelName


class QwenEmbeddingAdapter:
    """Adapter for Alibaba Qwen3-Embedding-8B.

    Notable for its 32k token context window, ideal for long documents.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._embed_model: Optional[HuggingFaceEmbedding] = None

    @property
    def model_name(self) -> str:
        return EmbeddingModelName.QWEN3_EMBEDDING_8B.value

    @property
    def embedding_dimension(self) -> int:
        return 4096

    def get_llama_index_embedding(self) -> BaseEmbedding:
        """Lazy-load and return the HuggingFace embedding model."""
        if self._embed_model is None:
            self._embed_model = HuggingFaceEmbedding(
                model_name=self._model_path,
                device=self._device,
            )
        return self._embed_model

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embedding vectors."""
        model = self.get_llama_index_embedding()
        return [model.get_text_embedding(t) for t in texts]
