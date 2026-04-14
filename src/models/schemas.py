"""Pydantic models for all structured data in the project."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ConfluencePage(BaseModel):
    """Represents a single Confluence page with metadata."""

    page_id: str
    title: str
    space_key: str
    url: str
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    local_path: Optional[str] = None
    images: List[str] = Field(default_factory=list)


class ConfluenceManifest(BaseModel):
    """Complete mapping of all downloaded Confluence pages."""

    space_key: str
    download_timestamp: datetime
    pages: Dict[str, ConfluencePage] = Field(default_factory=dict)


class IndexingTimingRecord(BaseModel):
    """Timing measurement for a single document indexing operation."""

    document_id: str
    document_title: str
    embedding_model: str
    chunking_strategy: str
    num_chunks_produced: int
    indexing_duration_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EvaluationResult(BaseModel):
    """Single evaluation run result for one (model, strategy) combination."""

    run_id: str
    embedding_model: str
    chunking_strategy: str
    context_precision: float
    context_recall: float
    context_entities_recall: float
    avg_indexing_time_seconds: float
    total_documents: int
    total_chunks: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, str] = Field(default_factory=dict)


class EvaluationComparison(BaseModel):
    """Aggregated comparison across all model x strategy combinations."""

    results: List[EvaluationResult]
    best_precision_combo: str = ""
    best_recall_combo: str = ""
    best_entities_recall_combo: str = ""

    def identify_best(self) -> None:
        """Identify the best (model, strategy) for each metric."""
        if not self.results:
            return

        best_p = max(self.results, key=lambda r: r.context_precision)
        best_r = max(self.results, key=lambda r: r.context_recall)
        best_e = max(self.results, key=lambda r: r.context_entities_recall)

        self.best_precision_combo = f"{best_p.embedding_model}+{best_p.chunking_strategy}"
        self.best_recall_combo = f"{best_r.embedding_model}+{best_r.chunking_strategy}"
        self.best_entities_recall_combo = f"{best_e.embedding_model}+{best_e.chunking_strategy}"


class GoldenQAPair(BaseModel):
    """A single manually-curated Q/A pair for evaluation."""

    question: str
    ground_truth: str
    source_page_id: Optional[str] = None


class GoldenQADataset(BaseModel):
    """Collection of golden Q/A pairs for RAGAS evaluation."""

    pairs: List[GoldenQAPair] = Field(default_factory=list)
