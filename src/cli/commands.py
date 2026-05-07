"""CLI commands for the RAG evaluation pipeline."""

from pathlib import Path
from typing import List, Optional

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from rich.console import Console

from src.config.settings import AppSettings
from src.database.connection import DatabaseConnection
from src.embedding.factory import EmbeddingModelFactory
from src.evaluation.dataset_generator import DatasetGenerator
from src.evaluation.ragas_evaluator import RagasEvaluator
from src.evaluation.reporter import EvaluationReporter
from src.evaluation.result_store import EvaluationResultStore
from src.indexing.indexer import Indexer
from src.indexing.timer import IndexingTimer
from src.ingestion.pipeline import IngestionPipeline
from src.models.enums import ChunkingStrategy, EmbeddingModelName, RerankerModel
from src.models.schemas import EvaluationComparison, EvaluationResult, GoldenQADataset
from src.retrieval.ensemble import EnsembleRetriever
from src.retrieval.reranker_factory import RerankerFactory
from src.visualization.dashboard import render as render_dashboard
from src.visualization.metrics_collector import MetricsCollector

console = Console()

# Fixed order from smallest to largest for prefetch and indexing phases.
_MODEL_SIZE_ORDER: List[EmbeddingModelName] = [
    EmbeddingModelName.GTE_MULTILINGUAL_BASE,   # ~305 MB fp16
    EmbeddingModelName.MULTILINGUAL_E5_SMALL,   # ~471 MB fp16
    EmbeddingModelName.BGE_M3,                  # ~1.2 GB fp16
    EmbeddingModelName.QWEN3_EMBEDDING_8B,      # ~16 GB fp16
]


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

    def generate_qa(self, force_regenerate: bool = False) -> None:
        """Generate Hungarian retrieval questions from ingested documents via Gemini."""
        generator = DatasetGenerator(self._settings.openrouter, self._data_dir)
        generator.generate(force_regenerate=force_regenerate)

    def generate_answers(self) -> None:
        """Generate ground truth answers for all questions that don't have one yet."""
        generator = DatasetGenerator(self._settings.openrouter, self._data_dir)
        generator.generate_answers()

    def index(
        self,
        models: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
    ) -> None:
        """Build indices for specified models x strategies.

        Two-phase execution:
          1. Prefetch weights for all models (smallest → largest, disk only, no GPU).
          2. Load and index with each model in the same order.

        Defaults to all models and strategies if not specified.
        """
        selected_models = self._sort_models_by_size(self._resolve_models(models))
        selected_strategies = self._resolve_strategies(strategies)
        documents = self._load_documents()

        # Create all adapters once — reused across both phases.
        adapters = {
            m: EmbeddingModelFactory.create(m, self._settings.embedding)
            for m in selected_models
        }

        # Phase 1: Download weights (no GPU loading).
        console.print("\n[bold yellow]Phase 1: Prefetching model weights[/bold yellow]")
        for model_name in selected_models:
            console.print(f"  [cyan]{model_name.value}[/cyan]")
            try:
                adapters[model_name].prefetch()
                console.print(f"  [green]✓ {model_name.value}[/green]")
            except Exception as e:
                console.print(f"  [red]Prefetch failed for {model_name.value}: {e}[/red]")

        # Phase 2: Load each model to GPU and run all strategies.
        console.print("\n[bold yellow]Phase 2: Building indices[/bold yellow]")
        check_indexer = Indexer(self._settings.database, IndexingTimer())

        for model_name in selected_models:
            adapter = adapters[model_name]

            pending = [
                s for s in selected_strategies
                if not check_indexer.is_indexed(adapter.model_name, s)
            ]
            already_done = [s for s in selected_strategies if s not in pending]

            console.print(f"\n[bold blue]Model: {model_name.value}[/bold blue]")

            if already_done:
                done_names = ", ".join(s.value for s in already_done)
                console.print(f"  [dim]Already indexed, skipping: {done_names}[/dim]")

            if not pending:
                continue

            try:
                for strategy in pending:
                    console.print(f"  Strategy: [cyan]{strategy.value}[/cyan]")
                    timer = IndexingTimer()
                    indexer = Indexer(self._settings.database, timer)

                    try:
                        indexer.index_documents(
                            documents=documents,
                            adapter=adapter,
                            strategy=strategy,
                        )
                    except Exception as e:
                        console.print(f"  [red]Indexing failed for {strategy.value}: {e}[/red]")
                        continue

                    self._result_store.save_timing_records(timer.records)
                    self._reporter.print_timing_summary(timer.records)
            finally:
                console.print(f"  [dim]Unloading {model_name.value} from device memory[/dim]")
                adapter.unload()

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

        ready_pairs = [p for p in qa_dataset.pairs if p.ground_truth]
        if not ready_pairs:
            console.print(
                "[red]No pairs with ground truth found. "
                "Run 'generate-answers' first.[/red]"
            )
            return

        if len(ready_pairs) < len(qa_dataset.pairs):
            console.print(
                f"[yellow]Using {len(ready_pairs)}/{len(qa_dataset.pairs)} pairs "
                f"(rest have no ground truth yet).[/yellow]"
            )

        questions = [p.question for p in ready_pairs]
        ground_truths = [p.ground_truth for p in ready_pairs if p.ground_truth is not None]
        ragas_evaluator = RagasEvaluator(self._settings.openrouter)
        all_results = []
        indexer = Indexer(self._settings.database, IndexingTimer())

        for model_name in selected_models:
            adapter = EmbeddingModelFactory.create(model_name, self._settings.embedding)
            embed_model = adapter.get_llama_index_embedding()
            console.print(f"\n[bold blue]Evaluating: {model_name.value}[/bold blue]")

            for strategy in selected_strategies:
                console.print(f"  Strategy: [cyan]{strategy.value}[/cyan]")

                try:
                    index = indexer.load_vector_index(
                        model_name=adapter.model_name,
                        strategy=strategy,
                        embed_model=embed_model,
                        embed_dim=adapter.embedding_dimension,
                    )
                except RuntimeError as e:
                    console.print(f"  [red]{e}[/red]")
                    continue

                # Retrieve contexts for each question using the saved index
                retriever = index.as_retriever(similarity_top_k=10)
                retrieved_contexts: List[List[str]] = []
                for q in questions:
                    nodes = retriever.retrieve(q)
                    retrieved_contexts.append([n.text for n in nodes])

                # Pull avg indexing time from DB records saved during `poe index`
                avg_time = self._result_store.get_avg_indexing_time(
                    model_name=model_name.value,
                    strategy=strategy.value,
                )

                result = ragas_evaluator.evaluate(
                    questions=questions,
                    ground_truths=ground_truths,
                    retrieved_contexts=retrieved_contexts,
                    embedding_model=model_name.value,
                    chunking_strategy=strategy.value,
                )

                result.avg_indexing_time_seconds = avg_time
                result.total_documents = len(documents)

                self._result_store.save_result(result)
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

    def evaluate_ensemble(self, reranker_model: RerankerModel) -> None:
        """Run RAGAS evaluation using BGE-M3 semantic + hierarchical ensemble with reranking."""
        qa_dataset = self._load_golden_qa()
        ready_pairs = [p for p in qa_dataset.pairs if p.ground_truth]

        if not ready_pairs:
            console.print("[red]No pairs with ground truth. Run 'generate-answers' first.[/red]")
            return

        questions = [p.question for p in ready_pairs]
        ground_truths = [p.ground_truth for p in ready_pairs if p.ground_truth is not None]

        adapter = EmbeddingModelFactory.create(EmbeddingModelName.BGE_M3, self._settings.embedding)
        embed_model = adapter.get_llama_index_embedding()
        indexer = Indexer(self._settings.database, IndexingTimer())

        indices = []
        for strategy in [ChunkingStrategy.SEMANTIC, ChunkingStrategy.HIERARCHICAL]:
            try:
                index = indexer.load_vector_index(
                    model_name=adapter.model_name,
                    strategy=strategy,
                    embed_model=embed_model,
                    embed_dim=adapter.embedding_dimension,
                )
                indices.append(index)
            except RuntimeError as e:
                console.print(f"  [red]{e}[/red]")

        if not indices:
            console.print("[red]No BGE-M3 indices found. Run 'index' first.[/red]")
            return

        console.print(f"\n[bold blue]Ensemble ({len(indices)} indices) + {reranker_model.value} reranker[/bold blue]")

        reranker = RerankerFactory.create(reranker_model, self._settings.reranker)
        ensemble = EnsembleRetriever(
            indices=indices,
            reranker=reranker,
            initial_top_k=self._settings.reranker.initial_top_k,
            final_top_k=self._settings.reranker.final_top_k,
        )

        retrieved_contexts: List[List[str]] = []
        for q in questions:
            nodes = ensemble.retrieve(q)
            retrieved_contexts.append([n.node.get_content() for n in nodes])

        strategy_label = f"ensemble_{reranker_model.value}_reranker"
        ragas_evaluator = RagasEvaluator(self._settings.openrouter)
        result: EvaluationResult = ragas_evaluator.evaluate(
            questions=questions,
            ground_truths=ground_truths,
            retrieved_contexts=retrieved_contexts,
            embedding_model=EmbeddingModelName.BGE_M3.value,
            chunking_strategy=strategy_label,
        )
        result.total_documents = len(self._load_documents())

        self._result_store.save_result(result)
        console.print(
            f"  Precision: {result.context_precision:.3f} | "
            f"Recall: {result.context_recall:.3f} | "
            f"Entities: {result.context_entities_recall:.3f}"
        )
        self._reporter.print_comparison(EvaluationComparison(results=[result]))

    def visualize(self, force_refresh: bool = False) -> None:
        """Compute (or load cached) document metrics and open the Plotly dashboard."""
        collector = MetricsCollector(self._data_dir)
        report = collector.collect(force_refresh=force_refresh)
        render_dashboard(report)

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

    def _sort_models_by_size(
        self, models: List[EmbeddingModelName]
    ) -> List[EmbeddingModelName]:
        """Sort models smallest to largest using the fixed size order."""
        order = {m: i for i, m in enumerate(_MODEL_SIZE_ORDER)}
        return sorted(models, key=lambda m: order.get(m, 999))

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
