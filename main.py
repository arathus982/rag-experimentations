"""CLI entry point for the Hungarian RAG evaluation framework."""

from typing import List, Optional

import typer
from rich.console import Console

from src.cli.commands import CLICommands
from src.config.settings import AppSettings
from src.models.enums import RerankerModel

app = typer.Typer(
    name="rag-eval",
    help="Hungarian RAG evaluation framework: compare embedding models with triple-threat chunking.",
)
console = Console()


def _get_commands() -> CLICommands:
    """Initialize CLI commands with settings from .env."""
    settings = AppSettings()
    return CLICommands(settings)


@app.command()
def ingest() -> None:
    """Download Confluence docs to local Markdown with images."""
    _get_commands().ingest()


@app.command(name="generate-qa")
def generate_qa(
    force_regenerate: bool = typer.Option(
        False, "--force-regenerate", "-f", help="Ignore existing dataset and regenerate from scratch"
    ),
) -> None:
    """Generate Hungarian retrieval questions from ingested documents via Gemini."""
    _get_commands().generate_qa(force_regenerate=force_regenerate)


@app.command(name="generate-answers")
def generate_answers() -> None:
    """Generate ground truth answers for questions that don't have one yet."""
    _get_commands().generate_answers()


@app.command()
def index(
    models: Optional[List[str]] = typer.Option(
        None, "--model", "-m", help="Embedding model(s) to index"
    ),
    strategies: Optional[List[str]] = typer.Option(
        None, "--strategy", "-s", help="Chunking strategy(ies) to use"
    ),
) -> None:
    """Build indices for specified models x strategies."""
    _get_commands().index(models=models, strategies=strategies)


@app.command()
def evaluate(
    models: Optional[List[str]] = typer.Option(
        None, "--model", "-m", help="Embedding model(s) to evaluate"
    ),
    strategies: Optional[List[str]] = typer.Option(
        None, "--strategy", "-s", help="Chunking strategy(ies) to evaluate"
    ),
) -> None:
    """Run RAGAS evaluation for specified models x strategies."""
    _get_commands().evaluate(models=models, strategies=strategies)


@app.command()
def visualize(
    force_refresh: bool = typer.Option(
        False, "--force-refresh", "-f", help="Recompute metrics even if cache exists"
    ),
) -> None:
    """Compute document metrics and open the Plotly dashboard in a browser."""
    _get_commands().visualize(force_refresh=force_refresh)


@app.command(name="evaluate-ensemble")
def evaluate_ensemble(
    reranker: RerankerModel = typer.Option(
        RerankerModel.BGE, "--reranker", "-r", help="Reranker model to use (bge or qwen3)"
    ),
) -> None:
    """Run RAGAS evaluation using BGE-M3 ensemble retrieval with reranking."""
    _get_commands().evaluate_ensemble(reranker_model=reranker)


@app.command()
def report() -> None:
    """Print comparison of all stored evaluation results."""
    _get_commands().report()


@app.command(name="run-all")
def run_all() -> None:
    """Run full pipeline: ingest -> evaluate -> report."""
    _get_commands().full_pipeline()


if __name__ == "__main__":
    app()
