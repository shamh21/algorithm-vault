"""Production WSGI entrypoint for Gunicorn."""

from app import create_app

app = create_app()
