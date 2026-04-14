"""Rich-formatted comparison tables for evaluation results."""

from typing import List

from rich.console import Console
from rich.table import Table

from src.models.enums import ChunkingStrategy, EmbeddingModelName
from src.models.schemas import EvaluationComparison, IndexingTimingRecord


class EvaluationReporter:
    """Generates Rich-formatted comparison tables for the terminal."""

    def __init__(self) -> None:
        self._console = Console()

    def print_comparison(self, comparison: EvaluationComparison) -> None:
        """Print the 3x3 evaluation matrix with metrics.

        Rows: embedding models
        Columns grouped by: chunking strategy
        Cells: precision / recall / entities_recall / avg_indexing_time
        Highlights the best value per metric in green.
        """
        if not comparison.results:
            self._console.print("[yellow]No evaluation results to display.[/yellow]")
            return

        table = Table(
            title="RAG Evaluation Results: Model x Strategy",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Model", style="bold")

        strategies = [s.value for s in ChunkingStrategy]
        for strategy in strategies:
            table.add_column(f"{strategy}\nP / R / ER / Time", justify="center")

        # Build a lookup: (model, strategy) -> result
        lookup = {(r.embedding_model, r.chunking_strategy): r for r in comparison.results}

        for model in EmbeddingModelName:
            row = [model.value]
            for strategy in strategies:
                result = lookup.get((model.value, strategy))
                if result:
                    cell = (
                        f"{result.context_precision:.3f} / "
                        f"{result.context_recall:.3f} / "
                        f"{result.context_entities_recall:.3f} / "
                        f"{result.avg_indexing_time_seconds:.2f}s"
                    )
                else:
                    cell = "-"
                row.append(cell)
            table.add_row(*row)

        self._console.print(table)

        # Print best combos
        if comparison.best_precision_combo:
            self._console.print(
                f"\n[green]Best Precision:[/green] {comparison.best_precision_combo}"
            )
            self._console.print(f"[green]Best Recall:[/green] {comparison.best_recall_combo}")
            self._console.print(
                f"[green]Best Entities Recall:[/green] " f"{comparison.best_entities_recall_combo}"
            )

    def print_timing_summary(self, records: List[IndexingTimingRecord]) -> None:
        """Print average indexing time per model x strategy combination."""
        if not records:
            self._console.print("[yellow]No timing records to display.[/yellow]")
            return

        table = Table(
            title="Indexing Time Summary (avg seconds per document)",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Model", style="bold")
        table.add_column("Strategy")
        table.add_column("Avg Time (s)", justify="right")
        table.add_column("Avg Chunks", justify="right")
        table.add_column("Docs", justify="right")

        # Group by (model, strategy)
        from collections import defaultdict

        groups: dict[tuple[str, str], list[IndexingTimingRecord]] = defaultdict(list)
        for r in records:
            groups[(r.embedding_model, r.chunking_strategy)].append(r)

        for (model, strategy), group_records in sorted(groups.items()):
            avg_time = sum(r.indexing_duration_seconds for r in group_records) / len(group_records)
            avg_chunks = sum(r.num_chunks_produced for r in group_records) / len(group_records)
            table.add_row(
                model,
                strategy,
                f"{avg_time:.4f}",
                f"{avg_chunks:.1f}",
                str(len(group_records)),
            )

        self._console.print(table)
