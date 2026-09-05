"""Local filesystem document source (strictly path-traversal-safe)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from app.domain.entities import FileReference
from app.domain.exceptions import DocumentSourceError

_LOCAL_SCHEME = "file"


class LocalFileSource:
    """Enumerates and reads files under a root directory.

    ``root`` must be an absolute path.  Paths are resolved and validated
    against the root so ``..``-style path traversal is rejected.
    """

    def __init__(self, root: Path, extensions: Iterable[str] | None = None) -> None:
        self.root = Path(root).resolve()
        if not self.root.exists():
            # A data directory is a configuration target — create it.
            self.root.mkdir(parents=True, exist_ok=True)
        self.extensions = [e.lower().lstrip(".") for e in (extensions or [])] or None

    def list_files(self) -> Iterable[FileReference]:
        for child in sorted(self.root.rglob("*")):
            if not child.is_file():
                continue
            if self.extensions and child.suffix.lower().lstrip(".") not in self.extensions:
                continue
            yield self._to_reference(child)

    def read(self, file: FileReference) -> bytes:
        path = self._resolve(file)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise DocumentSourceError(f"failed to read '{path}'") from exc

    def _to_reference(self, path: Path) -> FileReference:
        return FileReference(
            source_id=f"{_LOCAL_SCHEME}://{self.root}",
            path=str(path.relative_to(self.root)),
            uri=path.as_uri(),
            size_bytes=path.stat().st_size,
            metadata={"root": str(self.root)},
        )

    def _resolve(self, file: FileReference) -> Path:
        if file.source_id and not file.source_id.startswith(f"{_LOCAL_SCHEME}://"):
            raise DocumentSourceError(f"FileReference does not belong to this source: {file.source_id}")
        candidate = (self.root / file.path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise DocumentSourceError(f"path escapes source root: {file.path}") from None
        if not candidate.is_file():
            raise DocumentSourceError(f"file not found: {file.path}")
        return candidate