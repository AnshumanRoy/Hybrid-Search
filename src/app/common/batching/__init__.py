"""Batching utilities — the single source of truth for chunk/batch iteration."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Generic, TypeVar

T = TypeVar("T")


def batched(iterable: Iterable[T], size: int, *, name: str = "items") -> Iterator[tuple[T, ...]]:
    """Yield successive ``size``-sized chunks from ``iterable`` (like
    ``itertools.batched``, which is unavailable on older stdlibs).

    Raises ``ValueError`` for a non-positive ``size`` so callers fail fast
    instead of silently looping forever.
    """
    if size < 1:
        raise ValueError(f"batch size for '{name}' must be >= 1, got {size}")

    buffer: list[T] = []
    consumed = 0
    for item in iterable:
        buffer.append(item)
        consumed += 1
        if len(buffer) == size:
            yield tuple(buffer)
            buffer.clear()
    if buffer:
        yield tuple(buffer)


def batch_by_size(items: Sequence[str], max_chars: int) -> list[tuple[int, list[str]]]:
    """Greedy-pack ``items`` into groups whose total size fits ``max_chars``.

    Returns ``(estimated_chars, indices)`` tuples.  Unit over-sized item is its
    own batch.  Never changes item order.
    """
    groups: list[tuple[int, list[str]]] = []
    current: list[str] = []
    current_size = 0
    for item in items:
        item_size = len(item)
        if current and current_size + item_size > max_chars:
            groups.append((current_size, current))
            current = []
            current_size = 0
        current.append(item)
        current_size += item_size
    if current:
        groups.append((current_size, current))
    return groups


__all__ = ["batched", "batch_by_size"]