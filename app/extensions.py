"""Shared Flask extensions."""

from flask_sqlalchemy import SQLAlchemy

try:
    from flask_migrate import Migrate
except Exception:  # pragma: no cover - optional until dependencies are installed
    Migrate = None  # type: ignore[assignment]

db = SQLAlchemy()
migrate = Migrate() if Migrate is not None else None
