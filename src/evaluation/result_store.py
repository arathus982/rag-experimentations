"""Persists evaluation results and timing records in PostgreSQL."""

import json
from typing import List

from src.database.connection import DatabaseConnection
from src.database.tables import EvaluationResultTable, IndexingTimingTable
from src.models.schemas import (
    EvaluationComparison,
    EvaluationResult,
    IndexingTimingRecord,
)


class EvaluationResultStore:
    """Saves and loads evaluation results from PostgreSQL."""

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    def save_result(self, result: EvaluationResult) -> None:
        """Persist a single evaluation result."""
        for session in self._db.get_session():
            row = EvaluationResultTable(
                run_id=result.run_id,
                embedding_model=result.embedding_model,
                chunking_strategy=result.chunking_strategy,
                context_precision=result.context_precision,
                context_recall=result.context_recall,
                context_entities_recall=result.context_entities_recall,
                avg_indexing_time_seconds=result.avg_indexing_time_seconds,
                total_documents=result.total_documents,
                total_chunks=result.total_chunks,
                metadata_json=json.dumps(result.metadata),
            )
            session.add(row)

    def save_timing_records(self, records: List[IndexingTimingRecord]) -> None:
        """Persist indexing timing records in bulk."""
        for session in self._db.get_session():
            for record in records:
                row = IndexingTimingTable(
                    document_id=record.document_id,
                    document_title=record.document_title,
                    embedding_model=record.embedding_model,
                    chunking_strategy=record.chunking_strategy,
                    num_chunks_produced=record.num_chunks_produced,
                    indexing_duration_seconds=record.indexing_duration_seconds,
                )
                session.add(row)

    def get_all_results(self) -> List[EvaluationResult]:
        """Load all evaluation results from the database."""
        results: List[EvaluationResult] = []
        for session in self._db.get_session():
            rows = session.query(EvaluationResultTable).all()
            for row in rows:
                results.append(
                    EvaluationResult(
                        run_id=row.run_id,
                        embedding_model=row.embedding_model,
                        chunking_strategy=row.chunking_strategy,
                        context_precision=row.context_precision,
                        context_recall=row.context_recall,
                        context_entities_recall=row.context_entities_recall,
                        avg_indexing_time_seconds=row.avg_indexing_time_seconds,
                        total_documents=row.total_documents,
                        total_chunks=row.total_chunks,
                    )
                )
        return results

    def get_comparison(self) -> EvaluationComparison:
        """Load all results and identify best per metric."""
        results = self.get_all_results()
        comparison = EvaluationComparison(results=results)
        comparison.identify_best()
        return comparison
