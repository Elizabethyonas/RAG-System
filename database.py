"""Shared SQLAlchemy models and engine for chat history + pgvector knowledge base."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./models/chat_history.db")

_log = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def is_postgres_url(url: str) -> bool:
    """True for Postgres URLs SQLAlchemy accepts, including legacy ``postgres://`` (common on Render/Heroku)."""
    if not url:
        return False
    if url.startswith("sqlite:"):
        return False
    head = url.split("://", 1)[0].lower()
    # postgres:// and postgresql://*, postgresql+psycopg://*, etc.
    return head == "postgres" or head.startswith("postgresql")


class Base(DeclarativeBase):
    pass


class ChatSessionModel(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = (
        Index("ix_chat_sessions_user_updated", "user_id", "updated_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str] = mapped_column(String(255), default="New chat")
    car_context: Mapped[str] = mapped_column(String(255), default="")
    # Rolling chat summary to keep long conversations coherent without sending full history every time.
    chat_summary: Mapped[str] = mapped_column(Text, default="")
    # Links chat to Prisma Vehicle.id so every message can load maintenance health without client resending it.
    vehicle_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class ChatMessageModel(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        Index("ix_chat_messages_session_created_id", "session_id", "created_at", "id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


# --- Knowledge base (PostgreSQL + pgvector only; tables created only on Postgres) ---

from pgvector.sqlalchemy import Vector


EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


class RagKbChunkModel(Base):
    """Chunks + embeddings. Align Express uploads with this table or ETL into it."""

    __tablename__ = "rag_kb_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scope: Mapped[str] = mapped_column(String(16), index=True)  # "global" | "user"
    owner_user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    manual_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128), default="all-MiniLM-L6-v2")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RagUserProfileModel(Base):
    """Optional vehicle hint per user (replaces models/<user>/meta.txt when using Postgres)."""

    __tablename__ = "rag_user_profile"

    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    vehicle_meta: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class RagUserModel(Base):
    """Accounts for the Next.js web app. ``id`` is the JWT ``sub`` used by chat_sessions.user_id."""

    __tablename__ = "rag_users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

# rag_users lives on the primary DB when possible. If that CREATE fails (e.g. Neon
# disk-full), accounts fall back to a local SQLite file so register/login still work.
_auth_engine = engine
AuthSessionLocal = SessionLocal

# Optional second Postgres URL for Prisma tables (e.g. EducationContent) when the RAG stack
# uses a different database than driver-garage-backend (Docker pgvector vs local Express DB).
PRISMA_DATABASE_URL = os.environ.get("PRISMA_DATABASE_URL", "").strip()
_prisma_engine = None
PrismaSessionLocal: sessionmaker | None = None


def _ensure_prisma_engine() -> None:
    global _prisma_engine, PrismaSessionLocal
    if not PRISMA_DATABASE_URL or _prisma_engine is not None:
        return
    if not is_postgres_url(PRISMA_DATABASE_URL):
        raise ValueError("PRISMA_DATABASE_URL must be a PostgreSQL URL when set.")
    _prisma_engine = create_engine(
        PRISMA_DATABASE_URL,
        future=True,
        pool_pre_ping=True,
    )
    PrismaSessionLocal = sessionmaker(
        bind=_prisma_engine, autoflush=False, autocommit=False, expire_on_commit=False
    )


def session_for_prisma_reads():
    """DB session for Prisma-managed tables. Uses PRISMA_DATABASE_URL if set, else DATABASE_URL."""
    if PRISMA_DATABASE_URL:
        _ensure_prisma_engine()
        assert PrismaSessionLocal is not None
        return PrismaSessionLocal()
    return SessionLocal()


@event.listens_for(engine, "connect")
def _sqlite_enable_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
    if DATABASE_URL.startswith("sqlite:"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


@event.listens_for(engine, "connect")
def _postgres_enable_vector(dbapi_connection: Any, _connection_record: Any) -> None:
    if not is_postgres_url(DATABASE_URL):
        return
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        dbapi_connection.commit()
    except Exception:
        dbapi_connection.rollback()
        raise
    finally:
        cursor.close()


def create_all_tables() -> None:
    """Create chat tables always; KB tables only when Postgres + pgvector is available."""
    chat_tables = [
        ChatSessionModel.__table__,
        ChatMessageModel.__table__,
    ]
    try:
        if is_postgres_url(DATABASE_URL) and Vector is not None:
            Base.metadata.create_all(
                engine,
                tables=[
                    *chat_tables,
                    RagKbChunkModel.__table__,
                    RagUserProfileModel.__table__,
                ],
            )
        else:
            Base.metadata.create_all(engine, tables=chat_tables)
    except Exception as exc:
        _log.warning("create_all_tables (chat/kb) failed: %s", exc)
    _ensure_rag_users_table()


def _ensure_rag_users_table() -> None:
    """Create rag_users on the primary engine, or SQLite if the primary DB cannot grow."""
    global _auth_engine, AuthSessionLocal
    try:
        Base.metadata.create_all(engine, tables=[RagUserModel.__table__])
        _auth_engine = engine
        AuthSessionLocal = SessionLocal
        return
    except Exception as exc:
        _log.warning(
            "Could not create rag_users on primary DB (%s); using local SQLite for accounts.",
            exc,
        )

    auth_path = Path(__file__).resolve().parent / "models" / "auth.db"
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    _auth_engine = create_engine(f"sqlite:///{auth_path}", future=True, pool_pre_ping=True)
    AuthSessionLocal = sessionmaker(
        bind=_auth_engine, autoflush=False, autocommit=False, expire_on_commit=False
    )
    Base.metadata.create_all(_auth_engine, tables=[RagUserModel.__table__])


def ensure_chat_schema() -> None:
    """
    SQLAlchemy create_all() does NOT add new columns to existing tables.

    Deployed DBs created before `vehicle_id` / composite indexes will 500 on every query unless we ALTER.
    Safe to run repeatedly (best-effort; we swallow "already exists" type errors).
    """
    if DATABASE_URL.startswith("sqlite:"):
        # SQLite dev DBs also need a best-effort ALTER when schema drifts.
        ddl = [
            "ALTER TABLE chat_sessions ADD COLUMN chat_summary TEXT DEFAULT ''",
            "ALTER TABLE chat_sessions ADD COLUMN vehicle_id VARCHAR(36)",
        ]
        for stmt in ddl:
            try:
                with engine.begin() as conn:
                    conn.execute(text(stmt))
            except Exception:
                # Older SQLite doesn't support IF NOT EXISTS; ignore duplicates/errors.
                continue
        return
    if not is_postgres_url(DATABASE_URL):
        return
    ddl = [
        "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS chat_summary TEXT DEFAULT ''",
        "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS vehicle_id VARCHAR(36)",
        "CREATE INDEX IF NOT EXISTS ix_chat_sessions_vehicle_id ON chat_sessions (vehicle_id)",
        "CREATE INDEX IF NOT EXISTS ix_chat_sessions_user_updated ON chat_sessions (user_id, updated_at DESC)",
        "CREATE INDEX IF NOT EXISTS ix_chat_messages_session_created_id ON chat_messages (session_id, created_at DESC, id DESC)",
    ]
    for stmt in ddl:
        try:
            with engine.begin() as conn:
                conn.execute(text(stmt))
        except Exception as exc:
            _log.warning("ensure_chat_schema failed (%s): %s", stmt[:96], exc)


create_all_tables()
ensure_chat_schema()
