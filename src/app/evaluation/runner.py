"""Evaluation runner.

Loads an evaluation dataset (JSON lines) of the shape::

    {"question": "...", "expected_answer": "...", "relevant_chunk_ids": ["..."]}

Runs hybrid retrieval per question, then computes retrieval metrics and, when an
LLM response is available, generation metrics. Supports A/B comparisons of query
rewriting (single-query vs. multi-query expansion) and of LLM backends, plus
latency/cost telemetry so rewriting is measured, not assumed.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from app.bootstrap import build_container
from app.evaluation.metrics import evaluate_retrieval, faithfulness, answer_relevance, context_relevance

logger = logging.getLogger(__name__)

# Rough chars-per-token for the cost estimate (provider-reported usage wins when
# the backend exposes it, e.g. NIM).
_CHARS_PER_TOKEN = 4


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

    def run(
        self,
        dataset_path: str | Path,
        *,
        generate: bool = False,
        top_k: int = 10,
        rewrite: bool = False,
        max_variants: int = 3,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate a dataset under one configuration.

        ``rewrite`` fans out over ``max_variants`` expanded queries; without it,
        each question is retrieved as a single query. Latency is always
        recorded (mean retrieval/generation/total). With ``generate=True`` the
        report also carries token-cost estimates (provider-reported usage wins,
        e.g. NIM).
        """
        samples = self.load_dataset(dataset_path)
        if (
            generate
            and rewrite
            and getattr(getattr(self.container, "rag_service", None), "rewrite_service", None) is None
        ):
            logger.warning(
                "generate+rewrite requested but QUERY_REWRITE_ENABLED is off — the RAG rewrite "
                "gate degrades to single-query. Set QUERY_REWRITE_ENABLED=true for a true A/B."
            )
        ranked_lists: list[list[str]] = []
        relevant_sets: list[set[str]] = []
        faithfulness_scores: list[float] = []
        relevance_scores: list[float] = []
        context_scores: list[float] = []
        retrieval_ms: list[float] = []
        generation_ms: list[float] = []
        total_ms: list[float] = []
        retrieval_counts: list[int] = []
        usage_samples: list[dict] = []
        estimated_prompt_tokens: list[float] = []
        estimated_output_tokens: list[float] = []

        for sample in samples:
            question = sample["question"]
            relevant = self._resolve_relevance(sample)

            if generate:
                rag = self.container.rag_service.answer(
                    question,
                    top_k=top_k,
                    model=model,
                    rewrite=bool(rewrite),
                    max_variants=max_variants if rewrite else None,
                )
                ranked = rag.retrieved_chunks
                md = rag.metadata
                retrieval_ms.append(md.get("retrieval_ms", 0.0))
                generation_ms.append(md.get("generation_ms", 0.0))
                total_ms.append(md.get("total_ms", 0.0))
                retrieval_counts.append(md.get("retrieval_count", len(ranked)))
                context_chars = md.get("context_chars", 0)
                estimated_prompt_tokens.append(context_chars / _CHARS_PER_TOKEN)
                estimated_output_tokens.append(len(rag.answer) / _CHARS_PER_TOKEN)
                usage = md.get("llm_usage")
                if isinstance(usage, dict):
                    usage_samples.append(usage)
            else:
                started = time.perf_counter()
                if rewrite:
                    rewritten = self.container.rewrite_service.rewrite(question, max_variants=max_variants)
                    ranked = self.container.retrieval_service.retrieve_multi(rewritten, top_k=top_k)
                else:
                    ranked = self.container.retrieval_service.retrieve(question, top_k=top_k)
                retrieval_ms.append((time.perf_counter() - started) * 1000)
                retrieval_counts.append(len(ranked))

            ranked_ids = [c.chunk_id for c in ranked]
            ranked_lists.append(ranked_ids)
            relevant_sets.append(relevant)
            context_scores.append(context_relevance(ranked_ids, relevant))

            if generate:
                context = self.container.context_builder.build(rag.retrieved_chunks)
                faithfulness_scores.append(faithfulness(rag.answer, context))
                relevance_scores.append(answer_relevance(rag.answer, question))

        metrics = evaluate_retrieval(ranked_lists, relevant_sets, self.container.settings.eval_top_ks)

        report: dict[str, Any] = {
            "config": {
                "generate": generate,
                "top_k": top_k,
                "rewrite": rewrite,
                "max_variants": max_variants if rewrite else None,
                "model": model,
            },
            "retrieval": metrics,
            "questions": len(samples),
            "latency_ms": {
                "retrieval": _mean(retrieval_ms),
                "generation": _mean(generation_ms) if generate else None,
                "total": _mean(total_ms) if generate else None,
            },
            "retrieval_count_avg": _mean([float(n) for n in retrieval_counts]),
        }
        if generate:
            prompt_tokens = [u.get("prompt_tokens") for u in usage_samples]
            output_tokens = [u.get("completion_tokens") for u in usage_samples]
            report["generation"] = {
                "faithfulness": _mean(faithfulness_scores),
                "answer_relevance": _mean(relevance_scores),
                "context_relevance": _mean(context_scores),
            }
            report["cost"] = {
                "provider_usage_reported": bool(usage_samples),
                "prompt_tokens_avg": _mean(
                    [float(t) for t in prompt_tokens if t is not None] or estimated_prompt_tokens
                ),
                "completion_tokens_avg": _mean(
                    [float(t) for t in output_tokens if t is not None] or estimated_output_tokens
                ),
            }
        return report


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0