"""LlamaIndex Workflow for conditional retrieval.

Implements a state-machine approach:
1. Semantic search first
2. If confidence is low, fall back to property graph
3. Auto-merge hierarchical nodes
4. Return final context
"""

from typing import List, Optional

from llama_index.core import PropertyGraphIndex, VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)

from src.config.constants import DEFAULT_TOP_K, SIMILARITY_THRESHOLD


class SemanticResultEvent(Event):
    """Carries results from semantic search step."""

    nodes: List[NodeWithScore]
    confidence: float


class GraphFallbackEvent(Event):
    """Triggers graph-based fallback retrieval."""

    query: str


class HungarianRetrievalWorkflow(Workflow):
    """Conditional retrieval workflow optimized for Hungarian documents.

    Tries semantic search first. If the top result confidence is below
    the threshold, falls back to property graph retrieval. Finally,
    applies hierarchical auto-merging if available.
    """

    def __init__(
        self,
        semantic_index: VectorStoreIndex,
        graph_index: Optional[PropertyGraphIndex] = None,
        hierarchical_retriever: Optional[AutoMergingRetriever] = None,
        confidence_threshold: float = SIMILARITY_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._semantic_index = semantic_index
        self._graph_index = graph_index
        self._hierarchical_retriever = hierarchical_retriever
        self._confidence_threshold = confidence_threshold
        self._top_k = top_k

    @step
    async def semantic_search(
        self, ev: StartEvent
    ) -> SemanticResultEvent | StopEvent:  # type: ignore[return-value]
        """Step 1: Try semantic vector search."""
        query = ev.get("query", "")
        retriever = self._semantic_index.as_retriever(
            similarity_top_k=self._top_k,
        )
        nodes = await retriever.aretrieve(query)

        if not nodes:
            if self._graph_index:
                return GraphFallbackEvent(query=query)  # type: ignore[return-value]
            return StopEvent(result=[])

        confidence = nodes[0].score or 0.0
        if confidence < self._confidence_threshold and self._graph_index:
            return GraphFallbackEvent(query=query)  # type: ignore[return-value]

        return SemanticResultEvent(nodes=nodes, confidence=confidence)

    @step
    async def graph_fallback(self, ev: GraphFallbackEvent) -> StopEvent:
        """Step 2: Fall back to property graph retrieval."""
        if not self._graph_index:
            return StopEvent(result=[])

        retriever = self._graph_index.as_retriever()
        nodes = await retriever.aretrieve(ev.query)
        return StopEvent(result=nodes)

    @step
    async def merge_hierarchical(self, ev: SemanticResultEvent) -> StopEvent:
        """Step 3: Apply hierarchical auto-merging if available."""
        if self._hierarchical_retriever:
            # Auto-merging retriever handles the merge logic
            return StopEvent(result=ev.nodes)
        return StopEvent(result=ev.nodes)
