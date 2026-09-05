"""App entrypoint — ``__main__`` is the CLI router; ``app:app`` is the ASGI app."""

from app.api.app import create_app

app = create_app()