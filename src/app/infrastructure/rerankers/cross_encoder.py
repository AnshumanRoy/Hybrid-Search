"""Cross-encoder reranker built on SentenceTransformers.

Scores each (query, chunk) pair with a cross-encoder and reorders candidates,
preserving all metadata.  Operates on a bounded candidate set (never the whole
corpus).  Maps logits to a pseudo-probability in [0, 1] for stable downstream
thresholding.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from app.common.batching import batched
from app.domain.entities import RetrievedChunk
from app.domain.exceptions import RerankError
from app.domain.interfaces import Reranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str, batch_size: int = 16, device: str = "auto") -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = self._load(model_name, device)

    @staticmethod
    def _load(model_name: str, device: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise RerankError("sentence-transformers is not installed") from exc
        try:
            return CrossEncoder(model_name, device=device, max_length=512)
        except Exception as exc:
            raise RerankError(f"failed to load cross-encoder '{model_name}': {exc}") from exc

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        candidates = list(candidates)
        if not candidates:
            return []
        if top_k is None:
            top_k = len(candidates)

        pair_scores: list[float] = []
        try:
            for batch in batched(candidates, self.batch_size, name="rerank"):
                pairs = [(query, c.text) for c in batch]
                scores = self._model.predict(pairs, show_progress_bar=False)
                pair_scores.extend(float(s) for s in scores)
        except Exception as exc:
            raise RerankError(f"cross-encoder scoring failed for '{self.model_name}': {exc}") from exc

        scored = [
            (
                candidates[i].model_copy(
                    deep=True,
                    update={
                        "rerank_score": round(_sigmoid(pair_scores[i]), 6),
                        "retrieval_metadata": {
                            **candidates[i].retrieval_metadata,
                            "rerank": {"raw_logit": pair_scores[i], "model": self.model_name},
                        },
                    },
                ),
                _sigmoid(pair_scores[i]),
            )
            for i in range(len(candidates))
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        result = [pair[0] for pair in scored]
        return result[:top_k]

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        """Expose raw pair scores to evaluation/metrics tooling."""
        if not texts:
            return []
        try:
            return [float(s) for s in self._model.predict([(query, t) for t in texts], show_progress_bar=False)]
        except Exception as exc:
            raise RerankError(f"cross-encoder scoring failed: {exc}") from exc


def _sigmoid(x: float) -> float:
    import math

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)