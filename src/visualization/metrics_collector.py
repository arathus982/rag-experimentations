"""Collect and cache per-document metrics (tokens, references, images)."""

import re
from datetime import datetime
from pathlib import Path
from typing import List

import tiktoken
from rich.console import Console

from src.ingestion.metadata_manager import MetadataManager
from src.models.schemas import DocumentMetrics, MetricsReport

console = Console()

# Gemini 2.5 Flash uses a SentencePiece tokenizer not available in tiktoken.
# cl100k_base (GPT-4) is the closest publicly available approximation.
_TOKENIZER_ENCODING = "cl100k_base"
_TOKENIZER_LABEL = f"{_TOKENIZER_ENCODING} (Gemini 2.5 Flash approximation)"

# Matches markdown links: [text](url) — image links excluded separately
_LINK_RE = re.compile(r"(?<!!)\[(?:[^\[\]]+)\]\(([^)]+)\)")
# Matches markdown images: ![alt](path)
_IMAGE_RE = re.compile(r"!\[(?:[^\[\]]*)\]\(([^)]+)\)")


class MetricsCollector:
    """Computes per-document metrics from downloaded Markdown files.

    Results are cached to data/metrics/document_metrics.json and reused
    on subsequent runs unless force_refresh=True is passed.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._cache_path = data_dir / "metrics" / "document_metrics.json"
        self._enc = tiktoken.get_encoding(_TOKENIZER_ENCODING)
        self._metadata_manager = MetadataManager(data_dir)

    def collect(self, force_refresh: bool = False) -> MetricsReport:
        """Return metrics report, loading from cache when available."""
        if not force_refresh and self._cache_path.exists():
            console.print("[dim]Loading metrics from cache...[/dim]")
            return MetricsReport.model_validate_json(
                self._cache_path.read_text(encoding="utf-8")
            )

        return self._compute_and_cache()

    def _compute_and_cache(self) -> MetricsReport:
        console.print("Computing document metrics...")
        manifest = self._metadata_manager.load_manifest()
        metrics: List[DocumentMetrics] = []

        for page_id, page in manifest.pages.items():
            if not page.local_path:
                continue
            md_path = Path(page.local_path)
            if not md_path.exists():
                continue

            text = md_path.read_text(encoding="utf-8")
            doc_metrics = self._compute_document_metrics(page_id, page.title, text, md_path, page.images)
            metrics.append(doc_metrics)

        report = MetricsReport(
            generated_at=datetime.utcnow(),
            tokenizer=_TOKENIZER_LABEL,
            total_documents=len(metrics),
            documents=metrics,
        )

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"Metrics cached to [bold]{self._cache_path}[/bold]")

        return report

    def _compute_document_metrics(
        self,
        page_id: str,
        title: str,
        text: str,
        md_path: Path,
        manifest_images: List[str],
    ) -> DocumentMetrics:
        token_count = len(self._enc.encode(text))
        reference_count = len(_LINK_RE.findall(text))
        # Prefer manifest image list (already downloaded); fall back to inline refs
        image_count = len(manifest_images) if manifest_images else len(_IMAGE_RE.findall(text))

        return DocumentMetrics(
            page_id=page_id,
            title=title,
            token_count=token_count,
            reference_count=reference_count,
            image_count=image_count,
            local_path=str(md_path),
        )
