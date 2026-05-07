"""Factory for creating embedding model adapters by name."""

from src.config.settings import EmbeddingSettings
from src.embedding.base import EmbeddingModelAdapter
from src.embedding.bge_m3 import BgeM3EmbeddingAdapter
from src.embedding.e5_small import E5SmallEmbeddingAdapter
from src.embedding.gte_multilingual import GteMultilingualEmbeddingAdapter
from src.embedding.qwen import QwenEmbeddingAdapter
from src.models.enums import EmbeddingModelName


class EmbeddingModelFactory:
    """Creates embedding model adapters by enum name."""

    @staticmethod
    def create(
        model_name: EmbeddingModelName,
        settings: EmbeddingSettings,
    ) -> EmbeddingModelAdapter:
        """Create an adapter instance for the specified model.

        Args:
            model_name: Which embedding model to instantiate.
            settings: Embedding configuration (device, paths, quantization).

        Returns:
            An initialized adapter conforming to EmbeddingModelAdapter protocol.
        """
        match model_name:
            case EmbeddingModelName.GTE_MULTILINGUAL_BASE:
                return GteMultilingualEmbeddingAdapter(
                    model_path=settings.gte_multilingual_model_path,
                    device=settings.device,
                    batch_size=settings.batch_size,
                    models_dir=settings.models_dir,
                )
            case EmbeddingModelName.MULTILINGUAL_E5_SMALL:
                return E5SmallEmbeddingAdapter(
                    model_path=settings.e5_small_model_path,
                    device=settings.device,
                    batch_size=settings.batch_size,
                    models_dir=settings.models_dir,
                )
            case EmbeddingModelName.BGE_M3:
                return BgeM3EmbeddingAdapter(
                    model_path=settings.bge_m3_model_path,
                    device=settings.device,
                    batch_size=settings.batch_size,
                    models_dir=settings.models_dir,
                )
            case EmbeddingModelName.QWEN3_EMBEDDING_8B:
                return QwenEmbeddingAdapter(
                    model_path=settings.qwen_model_path,
                    device=settings.device,
                    batch_size=settings.batch_size,
                    models_dir=settings.models_dir,
                )
