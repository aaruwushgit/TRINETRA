"""
Database session factory.
Uses SQLite for local dev — swap DATABASE_URL in .env for PostgreSQL (+PostGIS) in prod.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.config import get_settings

settings = get_settings()

# How long a SQLite writer waits for the lock before giving up. Generous
# because the alternative is a lost row, and every writer here holds the lock
# for milliseconds.
SQLITE_BUSY_TIMEOUT_MS = 30_000

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

if "sqlite" in settings.DATABASE_URL:
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _sqlite_busy_timeout(dbapi_connection, connection_record):
        """Wait for a competing writer instead of failing the insert outright.

        SQLite allows a single writer, and this deployment has several: the API,
        the ANPR job runner and the live feeder all write concurrently. Without
        a timeout the loser of any overlap raises 'database is locked'
        immediately and the row is simply lost.

        Set on the connection rather than per call site, because the individual
        writers kept forgetting — the ingestion paths each set it by hand and
        the feeder's ORM path did not, which is exactly where the dropped
        inserts showed up.
        """
        cursor = dbapi_connection.cursor()
        cursor.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        cursor.close()


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
