"""Embedding adapter for Microsoft Harrier-OSS-v1."""

from typing import Dict, List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.models.enums import EmbeddingModelName


class HarrierEmbeddingAdapter:
    """Adapter for Microsoft Harrier-OSS-v1 (27B variant).

    Supports optional 4-bit quantization for 24GB GPUs.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        quantize: bool = False,
        batch_size: int = 4,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._quantize = quantize
        self._batch_size = batch_size
        self._embed_model: Optional[HuggingFaceEmbedding] = None

    @property
    def model_name(self) -> str:
        return EmbeddingModelName.HARRIER_OSS_V1.value

    @property
    def embedding_dimension(self) -> int:
        return 1024

    def get_llama_index_embedding(self) -> BaseEmbedding:
        """Lazy-load and return the HuggingFace embedding model."""
        if self._embed_model is None:
            model_kwargs: Dict[str, object] = {}
            if self._quantize:
                model_kwargs["load_in_4bit"] = True
            elif self._device != "cpu":
                model_kwargs["dtype"] = "float16"
            self._embed_model = HuggingFaceEmbedding(
                model_name=self._model_path,
                device=self._device,
                embed_batch_size=self._batch_size,
                model_kwargs=model_kwargs,
            )
        return self._embed_model

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embedding vectors."""
        model = self.get_llama_index_embedding()
        return [model.get_text_embedding(t) for t in texts]
