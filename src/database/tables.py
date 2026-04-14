"""SQLAlchemy table definitions for evaluation tracking.

Note: LlamaIndex manages its own tables (PGVectorStore, PropertyGraphStore).
These tables are only for custom evaluation and timing data.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class EvaluationResultTable(Base):
    """Stores RAGAS evaluation results per (model, strategy) run."""

    __tablename__ = "evaluation_results"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    embedding_model: Mapped[str] = mapped_column(String(64), nullable=False)
    chunking_strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    context_precision: Mapped[float] = mapped_column(Float, nullable=False)
    context_recall: Mapped[float] = mapped_column(Float, nullable=False)
    context_entities_recall: Mapped[float] = mapped_column(Float, nullable=False)
    avg_indexing_time_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    total_documents: Mapped[int] = mapped_column(Integer, nullable=False)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=True)


class IndexingTimingTable(Base):
    """Stores per-document indexing time measurements."""

    __tablename__ = "indexing_timings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(128), nullable=False)
    document_title: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(64), nullable=False)
    chunking_strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    num_chunks_produced: Mapped[int] = mapped_column(Integer, nullable=False)
    indexing_duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
