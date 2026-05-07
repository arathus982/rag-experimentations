"""Embedding adapter for Qwen3-Embedding-8B."""

import gc
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
        return EmbeddingModelName.QWEN3_EMBEDDING_8B.value

    @property
    def embedding_dimension(self) -> int:
        return 4096

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
                # Qwen3-Embedding uses instruction-based encoding for retrieval tasks
                query_instruction="Instruct: Retrieve relevant passages for the query.\nQuery: ",
                text_instruction="",
                trust_remote_code=True,
            )
        return self._embed_model

    def unload(self) -> None:
        """Release MPS/CUDA memory held by the Qwen model."""
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
