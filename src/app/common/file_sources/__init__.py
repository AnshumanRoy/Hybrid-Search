"""File-source infrastructure.

``FileSource`` implementations yield ``FileReference`` objects and produce raw
bytes on demand.  The parsing layer is source-agnostic; a future S3/Blob/Drive
source only needs to satisfy the ``FileSource`` protocol.
"""

from app.common.file_sources.local import LocalFileSource

__all__ = ["LocalFileSource"]