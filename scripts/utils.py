"""Filesystem size sanity / smoke-checking helpers for ops."""

from __future__ import annotations


def humanize_bytes(num: int) -> str:
    if num < 1024:
        return f"{num} B"
    for unit in ("KB", "MB", "GB"):
        num /= 1024.0
        if num < 1024:
            return f"{num:.1f} {unit}"
    return f"{num:.1f} TB"