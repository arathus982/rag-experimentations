"""Embedding adapter for BAAI/BGE-M3 with hybrid search support."""

from typing import Dict, List, Optional, Tuple

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.models.enums import EmbeddingModelName


class BgeM3EmbeddingAdapter:
    """Adapter for BAAI/BGE-M3.

    Supports dense, sparse, and multi-vector retrieval.
    The hybrid mode (dense + sparse) is recommended for Hungarian
    to capture both semantic meaning and specific word forms.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 4,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._batch_size = batch_size
        self._embed_model: Optional[HuggingFaceEmbedding] = None
        self._flag_model: Optional[object] = None

    @property
    def model_name(self) -> str:
        return EmbeddingModelName.BGE_M3.value

    @property
    def embedding_dimension(self) -> int:
        return 1024

    def get_llama_index_embedding(self) -> BaseEmbedding:
        """Lazy-load and return the HuggingFace embedding model."""
        if self._embed_model is None:
            model_kwargs = {} if self._device == "cpu" else {"dtype": "float16"}
            self._embed_model = HuggingFaceEmbedding(
                model_name=self._model_path,
                device=self._device,
                embed_batch_size=self._batch_size,
                model_kwargs=model_kwargs,
            )
        return self._embed_model

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into dense embedding vectors."""
        model = self.get_llama_index_embedding()
        return [model.get_text_embedding(t) for t in texts]

    def encode_hybrid(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        """Encode texts returning both dense and sparse vectors.

        Uses FlagEmbedding's BGEM3FlagModel for native hybrid support.

        Returns:
            Tuple of (dense_vectors, sparse_vectors).
            Sparse vectors are dicts mapping token_id -> weight.
        """
        if self._flag_model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._flag_model = BGEM3FlagModel(
                self._model_path,
                use_fp16=True,
            )

        output = self._flag_model.encode(  # type: ignore[union-attr]
            texts,
            return_dense=True,
            return_sparse=True,
        )
        dense_vectors = output["dense_vecs"].tolist()
        sparse_vectors = output["lexical_weights"]
        return dense_vectors, sparse_vectors
