"""Measures indexing time per document per model."""

import time
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, List

from src.models.schemas import IndexingTimingRecord


class IndexingTimer:
    """Records wall-clock indexing time per document using perf_counter."""

    def __init__(self) -> None:
        self._records: List[IndexingTimingRecord] = []

    @contextmanager
    def measure(
        self,
        document_id: str,
        document_title: str,
        embedding_model: str,
        chunking_strategy: str,
    ) -> Generator[None, None, None]:
        """Context manager that records indexing duration.

        Usage:
            with timer.measure("doc1", "Title", "harrier-oss-v1", "semantic"):
                # ... indexing happens here ...
        """
        start = time.perf_counter()
        yield
        duration = time.perf_counter() - start
        record = IndexingTimingRecord(
            document_id=document_id,
            document_title=document_title,
            embedding_model=embedding_model,
            chunking_strategy=chunking_strategy,
            num_chunks_produced=0,
            indexing_duration_seconds=duration,
            timestamp=datetime.utcnow(),
        )
        self._records.append(record)

    def update_last_chunk_count(self, count: int) -> None:
        """Set the chunk count on the most recent timing record."""
        if self._records:
            self._records[-1].num_chunks_produced = count

    @property
    def records(self) -> List[IndexingTimingRecord]:
        """All collected timing records."""
        return self._records

    def reset(self) -> None:
        """Clear all collected records."""
        self._records.clear()
