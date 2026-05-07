"""Embedding adapter for Alibaba-NLP/gte-multilingual-base."""

import gc
from typing import List, Optional

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.models.enums import EmbeddingModelName


class GteMultilingualEmbeddingAdapter:
    """Adapter for Alibaba-NLP/gte-multilingual-base.

    Compact 768-dim multilingual model (~305 MB fp16). Requires trust_remote_code=True.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 4,
        models_dir: str = "models",
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._batch_size = batch_size
        self._models_dir = models_dir
        self._embed_model: Optional[HuggingFaceEmbedding] = None

    @property
    def model_name(self) -> str:
        return EmbeddingModelName.GTE_MULTILINGUAL_BASE.value

    @property
    def embedding_dimension(self) -> int:
        return 768

    def prefetch(self) -> None:
        """Download model weights to local cache without loading to GPU memory."""
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=self._model_path, cache_dir=self._models_dir)

    def get_llama_index_embedding(self) -> BaseEmbedding:
        """Lazy-load and return the HuggingFace embedding model."""
        if self._embed_model is None:
            model_kwargs = {} if self._device == "cpu" else {"dtype": "float16"}
            self._embed_model = HuggingFaceEmbedding(
                model_name=self._model_path,
                device=self._device,
                embed_batch_size=self._batch_size,
                model_kwargs=model_kwargs,
                cache_folder=self._models_dir,
                trust_remote_code=True,
            )
        return self._embed_model

    def unload(self) -> None:
        """Release MPS/CUDA memory held by the GTE model."""
        if self._embed_model is not None:
            del self._embed_model
            self._embed_model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embedding vectors."""
        model = self.get_llama_index_embedding()
        return [model.get_text_embedding(t) for t in texts]
