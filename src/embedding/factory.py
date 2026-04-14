"""Factory for creating embedding model adapters by name."""

from src.config.settings import EmbeddingSettings
from src.embedding.base import EmbeddingModelAdapter
from src.embedding.bge_m3 import BgeM3EmbeddingAdapter
from src.embedding.harrier import HarrierEmbeddingAdapter
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
            case EmbeddingModelName.HARRIER_OSS_V1:
                return HarrierEmbeddingAdapter(
                    model_path=settings.harrier_model_path,
                    device=settings.device,
                    quantize=settings.quantize_harrier,
                )
            case EmbeddingModelName.QWEN3_EMBEDDING_8B:
                return QwenEmbeddingAdapter(
                    model_path=settings.qwen_model_path,
                    device=settings.device,
                )
            case EmbeddingModelName.BGE_M3:
                return BgeM3EmbeddingAdapter(
                    model_path=settings.bge_m3_model_path,
                    device=settings.device,
                )
