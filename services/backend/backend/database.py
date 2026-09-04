"""
Database session factory.
Uses SQLite for local dev — swap DATABASE_URL in .env for PostgreSQL (+PostGIS) in prod.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.config import get_settings

settings = get_settings()

# Use robust connection pooling for handling 1000+ camera concurrent requests
engine_kwargs = {}
if "sqlite" in settings.DATABASE_URL:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
    # SQLite uses SingletonThreadPool or NullPool by default; increase if needed
    engine_kwargs["pool_size"] = 20
    engine_kwargs["max_overflow"] = 40
else:
    # PostgreSQL/MySQL production settings
    engine_kwargs["pool_size"] = 50
    engine_kwargs["max_overflow"] = 100
    engine_kwargs["pool_timeout"] = 30
    engine_kwargs["pool_recycle"] = 1800

engine = create_engine(settings.DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Create all tables (called on startup).

    Models are auto-discovered by importing every module in backend/models/
    rather than listing them here. SQLAlchemy only knows about a table once
    its module has been imported, and the previous hardcoded import list was
    a standing trap: adding a model meant remembering to edit this function,
    and forgetting produced a confusing "no such table" at query time rather
    than an error at startup.
    """
    import importlib
    import pkgutil

    from backend import models

    for module_info in pkgutil.iter_modules(models.__path__):
        importlib.import_module(f"{models.__name__}.{module_info.name}")

    Base.metadata.create_all(bind=engine)
