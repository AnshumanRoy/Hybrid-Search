"""Batching utility tests."""

import pytest

from app.common.batching import batch_by_size, batched


def test_batched_yields_equal_sized_groups() -> None:
    groups = list(batched(range(10), 3))
    assert groups == [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]


def test_batched_with_exact_multiple() -> None:
    assert list(batched(range(6), 3)) == [(0, 1, 2), (3, 4, 5)]


def test_batched_empty() -> None:
    assert list(batched([], 3)) == []


def test_batched_size_less_than_one_raises() -> None:
    with pytest.raises(ValueError):
        list(batched([1, 2], 0))


def test_batched_size_one() -> None:
    assert list(batched([1, 2, 3], 1)) == [(1,), (2,), (3,)]


def test_batch_by_size_packs_items() -> None:
    items = ["aa", "bb", "cccc"]
    groups = batch_by_size(items, max_chars=4)
    flattened = [item for _, group in groups for item in group]
    assert flattened == items
    assert all(size <= 4 for size, _ in groups)


def test_batch_by_size_oversized_item_alone() -> None:
    items = ["x" * 100, "ab"]
    groups = batch_by_size(items, max_chars=50)
    assert [len(g) for _, g in groups] == [1, 1]