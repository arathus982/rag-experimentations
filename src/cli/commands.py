"""CLI commands for the RAG evaluation pipeline."""

from pathlib import Path
from statistics import mean
from typing import List, Optional

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from rich.console import Console

from src.config.settings import AppSettings
from src.database.connection import DatabaseConnection
from src.embedding.factory import EmbeddingModelFactory
from src.evaluation.ragas_evaluator import RagasEvaluator
from src.evaluation.reporter import EvaluationReporter
from src.evaluation.result_store import EvaluationResultStore
from src.indexing.indexer import Indexer
from src.indexing.timer import IndexingTimer
from src.ingestion.pipeline import IngestionPipeline
from src.models.enums import ChunkingStrategy, EmbeddingModelName
from src.models.schemas import EvaluationComparison, GoldenQADataset

console = Console()


class CLICommands:
    """Top-level CLI commands orchestrating the RAG evaluation pipeline."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._data_dir = Path(settings.data_dir)
        self._db = DatabaseConnection(settings.database)
        self._result_store = EvaluationResultStore(self._db)
        self._reporter = EvaluationReporter()

    def ingest(self) -> None:
        """Download and convert Confluence docs to Markdown."""
        pipeline = IngestionPipeline(self._settings)
        pipeline.run()

    def index(
        self,
        models: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
    ) -> None:
        """Build indices for specified models x strategies.

        Defaults to all models and strategies if not specified.
        """
        selected_models = self._resolve_models(models)
        selected_strategies = self._resolve_strategies(strategies)
        documents = self._load_documents()

        for model_name in selected_models:
            adapter = EmbeddingModelFactory.create(model_name, self._settings.embedding)
            console.print(f"\n[bold blue]Model: {model_name.value}[/bold blue]")

            for strategy in selected_strategies:
                console.print(f"  Strategy: [cyan]{strategy.value}[/cyan]")
                timer = IndexingTimer()
                indexer = Indexer(self._settings.database, timer)

                indexer.index_documents(
                    documents=documents,
                    adapter=adapter,
                    strategy=strategy,
                )

                self._result_store.save_timing_records(timer.records)
                self._reporter.print_timing_summary(timer.records)

    def evaluate(
        self,
        models: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
    ) -> None:
        """Run RAGAS evaluation for specified models x strategies.

        Requires golden Q/A pairs in data/evaluation/golden_qa.json.
        """
        selected_models = self._resolve_models(models)
        selected_strategies = self._resolve_strategies(strategies)
        documents = self._load_documents()
        qa_dataset = self._load_golden_qa()

        if not qa_dataset.pairs:
            console.print(
                "[red]No golden Q/A pairs found. "
                "Create data/evaluation/golden_qa.json first.[/red]"
            )
            return

        questions = [p.question for p in qa_dataset.pairs]
        ground_truths = [p.ground_truth for p in qa_dataset.pairs]
        ragas_evaluator = RagasEvaluator()
        all_results = []

        for model_name in selected_models:
            adapter = EmbeddingModelFactory.create(model_name, self._settings.embedding)
            console.print(f"\n[bold blue]Evaluating: {model_name.value}[/bold blue]")

            for strategy in selected_strategies:
                console.print(f"  Strategy: [cyan]{strategy.value}[/cyan]")
                timer = IndexingTimer()
                indexer = Indexer(self._settings.database, timer)

                # Build index
                index = indexer.index_documents(
                    documents=documents,
                    adapter=adapter,
                    strategy=strategy,
                )

                # Retrieve contexts for each question
                retriever = index.as_retriever(similarity_top_k=10)
                retrieved_contexts: List[List[str]] = []
                for q in questions:
                    nodes = retriever.retrieve(q)
                    retrieved_contexts.append([n.text for n in nodes])

                # Run RAGAS evaluation
                result = ragas_evaluator.evaluate(
                    questions=questions,
                    ground_truths=ground_truths,
                    retrieved_contexts=retrieved_contexts,
                    embedding_model=model_name.value,
                    chunking_strategy=strategy.value,
                )

                # Update timing info
                if timer.records:
                    result.avg_indexing_time_seconds = mean(
                        r.indexing_duration_seconds for r in timer.records
                    )
                result.total_documents = len(documents)
                result.total_chunks = sum(r.num_chunks_produced for r in timer.records)

                self._result_store.save_result(result)
                self._result_store.save_timing_records(timer.records)
                all_results.append(result)

                console.print(
                    f"    Precision: {result.context_precision:.3f} | "
                    f"Recall: {result.context_recall:.3f} | "
                    f"Entities: {result.context_entities_recall:.3f}"
                )

        # Final comparison
        comparison = EvaluationComparison(results=all_results)
        comparison.identify_best()
        self._reporter.print_comparison(comparison)

    def report(self) -> None:
        """Print Rich-formatted comparison of all stored evaluation results."""
        comparison = self._result_store.get_comparison()
        self._reporter.print_comparison(comparison)

    def full_pipeline(self) -> None:
        """Run ingest -> index -> evaluate -> report sequentially."""
        console.print("[bold green]Starting full pipeline...[/bold green]\n")
        self.ingest()
        self.evaluate()
        self.report()

    def _load_documents(self) -> List[Document]:
        """Load all Markdown documents from the Confluence data directory."""
        confluence_dir = self._data_dir / "confluence"
        if not confluence_dir.exists():
            console.print("[red]No documents found. Run 'ingest' first.[/red]")
            return []

        reader = SimpleDirectoryReader(
            input_dir=str(confluence_dir),
            recursive=True,
            required_exts=[".md"],
        )
        return reader.load_data()

    def _load_golden_qa(self) -> GoldenQADataset:
        """Load golden Q/A pairs from JSON file."""
        qa_path = self._data_dir / "evaluation" / "golden_qa.json"
        if not qa_path.exists():
            return GoldenQADataset()
        raw = qa_path.read_text(encoding="utf-8")
        return GoldenQADataset.model_validate_json(raw)

    def _resolve_models(self, names: Optional[List[str]]) -> List[EmbeddingModelName]:
        """Resolve model name strings to enum values."""
        if not names:
            return list(EmbeddingModelName)
        return [EmbeddingModelName(n) for n in names]

    def _resolve_strategies(self, names: Optional[List[str]]) -> List[ChunkingStrategy]:
        """Resolve strategy name strings to enum values."""
        if not names:
            return list(ChunkingStrategy)
        return [ChunkingStrategy(n) for n in names]
