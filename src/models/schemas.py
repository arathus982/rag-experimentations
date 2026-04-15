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


class DocumentMetrics(BaseModel):
    """Token, reference, and image counts for a single downloaded page."""

    page_id: str
    title: str
    token_count: int
    reference_count: int
    image_count: int
    local_path: str


class MetricsReport(BaseModel):
    """Aggregated metrics for all downloaded documents, used for visualization."""

    generated_at: datetime
    tokenizer: str
    total_documents: int
    documents: List[DocumentMetrics]


class GoldenQAPair(BaseModel):
    """A single question (and optional ground truth) for retrieval evaluation."""

    question: str
    ground_truth: Optional[str] = None
    source_page_id: Optional[str] = None
    source_title: Optional[str] = None


class GoldenQADataset(BaseModel):
    """Collection of generated questions for RAGAS retrieval evaluation."""

    pairs: List[GoldenQAPair] = Field(default_factory=list)
