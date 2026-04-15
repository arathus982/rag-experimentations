"""Factory for constructing reranker instances by model type."""

from src.config.settings import RerankerSettings
from src.models.enums import RerankerModel
from src.retrieval.reranker import BaseReranker, BgeReranker, Qwen3Reranker


class RerankerFactory:
    """Maps RerankerModel enum values to concrete reranker instances."""

    @staticmethod
    def create(model: RerankerModel, settings: RerankerSettings) -> BaseReranker:
        match model:
            case RerankerModel.BGE:
                return BgeReranker(settings.bge_model_path, settings.device)
            case RerankerModel.QWEN3:
                return Qwen3Reranker(settings.qwen3_model_path, settings.device)
