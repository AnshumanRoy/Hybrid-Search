"""SentenceTransformers embedder with configurable CPU/GPU and batching.

Deterministic behavior: same model + same input ⇒ same embeddings (the model
is pinned; ``torch`` nondeterminism is disabled where supported).  Embeddings
are L2-normalized to make cosine similarity equivalent to dot product.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from app.common.batching import batched
from app.domain.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


def resolve_device(preferred: str) -> str:
    """Resolve the 'auto' pseudo-device to a concrete torch device string."""
    device = preferred.lower()
    if device in ("cpu", "cuda", "mps"):
        return device
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        except ImportError:  # pragma: no cover - torch always present with sentence-transformers
            return "cpu"
    raise EmbeddingError(f"unsupported embedding device: {preferred}")


class SentenceTransformerEmbedder:
    """SentenceTransformers-backed embedder satisfying the ``Embedder`` protocol."""
    def __init__(self, model_name: str, batch_size: int = 32, device: str = "auto", max_seq_length: int | None = None) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        if batch_size < 1:
            raise EmbeddingError(f"embedding batch size must be >= 1, got {batch_size}")
        self.device = resolve_device(device)
        self._model = self._load(model_name, self.device, max_seq_length)
        get_dimension = getattr(self._model, "get_embedding_dimension")

        self._dimension = get_dimension()

    @staticmethod
    def _load(model_name: str, device: str, max_seq_length: int | None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError("sentence-transformers is not installed") from exc

        try:
            model = SentenceTransformer(model_name_or_path=model_name, device=device)
            if max_seq_length is not None:
                model.max_seq_length = max_seq_length
            return model
        except Exception as exc:
            raise EmbeddingError(f"failed to load embedding model '{model_name}' on {device}: {exc}") from exc

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors: list[list[float]] = []
            for batch in batched(texts, self.batch_size, name="embedding"):
                tensors = self._model.encode(
                    list(batch),
                    batch_size=len(batch),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                vectors.extend(t.tolist() for t in tensors)
            return vectors
        except Exception as exc:
            raise EmbeddingError(f"embedding generation failed (model={self.model_name}): {exc}") from exc

    @property
    def dimension(self) -> int:
        return self._dimension