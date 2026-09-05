"""API layer.

FastAPI-specific import lives here; DTOs/domain stay clean.
Keep API models separate from domain models (mapping happens in this layer).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import FastAPI, File, HTTPException, UploadFile


@dataclass
class APIConfig:
    container: object


def create_app(container=None) -> FastAPI:
    """App factory, wired to a bootstrap Container (tests inject their own)."""
    from app.bootstrap import build_container

    if container is None:
        container = build_container()

    app = FastAPI(title="Hybrid Search RAG", version="0.1.0")
    app.state.container = container

    from app.api.routes import auth, documents, health, query

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(query.router)
    app.include_router(documents.router)

    return app