"""RAGAS evaluation for retrieval quality metrics."""

import statistics
import uuid
from datetime import datetime
from typing import List

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.evaluation import EvaluationResult as RagasEvaluationResult
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_entity_recall, context_precision, context_recall

from src.config.settings import OpenRouterSettings
from src.models.schemas import EvaluationResult


class RagasEvaluator:
    """Runs RAGAS evaluation on retrieval results using OpenRouter as the judge LLM.

    Evaluates three key metrics:
    - Context Precision: ranking quality (best stuff at top?)
    - Context Recall: completeness (whole answer present?)
    - Context Entities Recall: relational quality (entities found?)
    """

    def __init__(self, settings: OpenRouterSettings) -> None:
        judge_llm = ChatOpenAI(
            model=settings.eval_model,
            base_url=settings.base_url,
            api_key=settings.api_key,
        )
        self._llm = LangchainLLMWrapper(judge_llm)
        self._metrics = [context_precision, context_recall, context_entity_recall]

    def evaluate(
        self,
        questions: List[str],
        ground_truths: List[str],
        retrieved_contexts: List[List[str]],
        embedding_model: str = "",
        chunking_strategy: str = "",
    ) -> EvaluationResult:
        """Run RAGAS evaluation and return structured results.

        Args:
            questions: Test questions in Hungarian.
            ground_truths: Expected answers.
            retrieved_contexts: List of retrieved context lists per question.
            embedding_model: Name of the embedding model used.
            chunking_strategy: Name of the chunking strategy used.

        Returns:
            EvaluationResult with all three metric scores.
        """
        dataset = self._prepare_dataset(questions, ground_truths, retrieved_contexts)
        raw_result = evaluate(dataset=dataset, metrics=self._metrics, llm=self._llm)

        if not isinstance(raw_result, RagasEvaluationResult):
            raise RuntimeError(
                f"RAGAS evaluate() returned unexpected type: {type(raw_result)}. "
                "Pass return_executor=False (the default) to get an EvaluationResult."
            )

        context_precision_scores: List[float] = raw_result["context_precision"]
        context_recall_scores: List[float] = raw_result["context_recall"]
        entity_recall_scores: List[float] = raw_result["context_entity_recall"]

        return EvaluationResult(
            run_id=str(uuid.uuid4()),
            embedding_model=embedding_model,
            chunking_strategy=chunking_strategy,
            context_precision=statistics.mean(context_precision_scores),
            context_recall=statistics.mean(context_recall_scores),
            context_entities_recall=statistics.mean(entity_recall_scores),
            avg_indexing_time_seconds=0.0,
            total_documents=0,
            total_chunks=0,
            timestamp=datetime.utcnow(),
        )

    def _prepare_dataset(
        self,
        questions: List[str],
        ground_truths: List[str],
        retrieved_contexts: List[List[str]],
    ) -> Dataset:
        """Convert inputs to a RAGAS-compatible HuggingFace Dataset."""
        return Dataset.from_dict(
            {
                "question": questions,
                "ground_truth": ground_truths,
                "contexts": retrieved_contexts,
            }
        )
