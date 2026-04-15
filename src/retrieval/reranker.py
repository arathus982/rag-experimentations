"""Reranker implementations for BGE and Qwen3 models."""

from abc import ABC, abstractmethod
from typing import Any, List

import torch
from llama_index.core.schema import NodeWithScore
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseReranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    def rerank(self, query: str, nodes: List[NodeWithScore], top_k: int) -> List[NodeWithScore]:
        """Score and re-sort nodes by relevance to the query, return top_k."""
        ...


class BgeReranker(BaseReranker):
    """Cross-encoder reranker using BAAI/bge-reranker-v2-m3 via FlagEmbedding."""

    def __init__(self, model_path: str, device: str) -> None:
        from FlagEmbedding import FlagReranker

        self._reranker = FlagReranker(model_path, use_fp16=device != "cpu")

    def rerank(self, query: str, nodes: List[NodeWithScore], top_k: int) -> List[NodeWithScore]:
        if not nodes:
            return []

        pairs = [[query, n.node.get_content()] for n in nodes]
        raw_scores = self._reranker.compute_score(pairs, normalize=True)
        scores: List[float] = [raw_scores] if isinstance(raw_scores, float) else list(raw_scores)

        ranked = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)
        results = []
        for node, score in ranked[:top_k]:
            node.score = score
            results.append(node)
        return results


# Qwen3-Reranker prompt constants
_QWEN3_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)
_QWEN3_INSTRUCTION = "Given a search query, retrieve relevant passages that answer the query"


class Qwen3Reranker(BaseReranker):
    """Generative reranker using Qwen/Qwen3-Reranker-4B.

    Scores relevance by extracting yes/no token logits from the last
    position of a prompted causal LM forward pass.
    """

    def __init__(self, model_path: str, device: str) -> None:
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        # AutoModelForCausalLM stubs are typed as _Wrapped with a broken __call__ signature;
        # routing through Any sidesteps the false positive without suppressing real errors.
        _loader: Any = AutoModelForCausalLM
        self._model: Any = (
            _loader.from_pretrained(model_path, dtype=torch.float16).to(device).eval()
        )
        self._true_id: int = self._tokenizer.convert_tokens_to_ids("yes")
        self._false_id: int = self._tokenizer.convert_tokens_to_ids("no")

    def _build_input(self, query: str, document: str) -> str:
        return (
            f"<|im_start|>system\n{_QWEN3_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"<Instruct>: {_QWEN3_INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    @torch.no_grad()
    def _score_batch(self, inputs: List[str]) -> List[float]:
        encoded = self._tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        ).to(self._device)
        logits = self._model(**encoded).logits[:, -1, :]
        true_log = logits[:, self._true_id]
        false_log = logits[:, self._false_id]
        scores = torch.softmax(torch.stack([false_log, true_log], dim=1), dim=1)[:, 1]
        return scores.cpu().tolist()

    def rerank(self, query: str, nodes: List[NodeWithScore], top_k: int) -> List[NodeWithScore]:
        if not nodes:
            return []

        inputs = [self._build_input(query, n.node.get_content()) for n in nodes]
        scores = self._score_batch(inputs)

        ranked = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)
        results = []
        for node, score in ranked[:top_k]:
            node.score = score
            results.append(node)
        return results
