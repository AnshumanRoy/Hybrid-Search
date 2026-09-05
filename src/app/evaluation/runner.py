"""Evaluation runner.

Loads an evaluation dataset (JSON lines) of the shape::

    {"question": "...", "expected_answer": "...", "relevant_chunk_ids": ["..."]}

Runs hybrid retrieval per question, then computes retrieval metrics and, when an
LLM response is available, generation metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.bootstrap import build_container
from app.evaluation.metrics import evaluate_retrieval, faithfulness, answer_relevance, context_relevance

logger = logging.getLogger(__name__)


class EvaluationRunner:
    def __init__(self, container) -> None:
        self.container = container

    def load_dataset(self, path: str | Path) -> list[dict]:
        path = Path(path)
        if path.suffix == ".json":
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(data, dict) and "questions" in data:
                data = data["questions"]
            return list(data)
        # Default: JSONL
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

    def _resolve_relevance(self, sample: dict) -> set[str]:
        """Relevant chunk ids from ``relevant_chunk_ids`` and/or ``relevant_docs``."""
        relevant = set(sample.get("relevant_chunk_ids", []))
        filenames = set(sample.get("relevant_docs", []))
        if not filenames:
            return relevant
        try:
            docs = self.container.repository.list_documents()
        except AttributeError:
            return relevant
        by_name = {d.filename.lower(): d.document_id for d in docs}
        for name in filenames:
            document_id = by_name.get(name.lower())
            if document_id is None:
                continue
            relevant.update(c.chunk_id for c in self.container.repository.get_chunks_by_document(document_id))
        return relevant

    def run(self, dataset_path: str | Path, *, generate: bool = False, top_k: int = 10) -> dict[str, Any]:
        samples = self.load_dataset(dataset_path)
        ranked_lists: list[list[str]] = []
        relevant_sets: list[set[str]] = []
        faithfulness_scores: list[float] = []
        relevance_scores: list[float] = []
        context_scores: list[float] = []

        for sample in samples:
            question = sample["question"]
            relevant = self._resolve_relevance(sample)
            ranked = self.container.retrieval_service.retrieve(question, top_k=top_k)
            ranked_ids = [c.chunk_id for c in ranked]

            ranked_lists.append(ranked_ids)
            relevant_sets.append(relevant)
            context_scores.append(context_relevance(ranked_ids, relevant))

            if generate:
                rag = self.container.rag_service.answer(question, top_k=top_k)
                context = self.container.context_builder.build(rag.retrieved_chunks)
                faithfulness_scores.append(faithfulness(rag.answer, context))
                relevance_scores.append(answer_relevance(rag.answer, question))

        metrics = evaluate_retrieval(ranked_lists, relevant_sets, self.container.settings.eval_top_ks)

        report: dict[str, Any] = {"retrieval": metrics, "questions": len(samples)}
        if generate:
            report["generation"] = {
                "faithfulness": _mean(faithfulness_scores),
                "answer_relevance": _mean(relevance_scores),
                "context_relevance": _mean(context_scores),
            }
        return report


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0