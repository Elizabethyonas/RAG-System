from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import func, select, text
from sqlalchemy.exc import OperationalError

warnings.filterwarnings(
    "ignore",
    message=".*clean_up_tokenization_spaces.*",
    category=FutureWarning,
)

import psycopg
from pgvector.psycopg import register_vector

from database import DATABASE_URL, RagKbChunkModel, SessionLocal, engine
from rag_kb import delete_global_chunks, kb_pgvector_enabled

DATA_PATH = "data/processed/unified_docs.txt"
INDEX_PATH = "models/faiss_index"
DOCS_OUT_PATH = "models/docs.txt"
EMBEDDINGS_CACHE_PATH = Path("models/global_embeddings.npy")
HNSW_INDEX_NAME = "ix_rag_kb_chunks_embedding_hnsw"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 200
INSERT_RETRIES = 5
NEON_MAX_BATCH_SIZE = 250

_COPY_SQL = (
    "COPY rag_kb_chunks "
    "(id, scope, owner_user_id, manual_id, chunk_index, content, embedding, embedding_model, created_at) "
    "FROM STDIN WITH (FORMAT BINARY)"
)


def _write_faiss(docs: list[str], embeddings: np.ndarray) -> None:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_OUT_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d + "\n")


def _is_neon_url() -> bool:
    return "neon.tech" in (DATABASE_URL or "").lower()


def _max_global_chunk_index() -> int:
    with SessionLocal() as db:
        n = db.scalar(
            select(func.max(RagKbChunkModel.chunk_index)).where(
                RagKbChunkModel.scope == "global",
                RagKbChunkModel.owner_user_id.is_(None),
                RagKbChunkModel.manual_id.is_(None),
            )
        )
    return int(n) if n is not None else -1


def _psycopg_dsn() -> str:
    url = DATABASE_URL or ""
    for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://"):
        if url.startswith(prefix):
            return "postgresql://" + url.split("://", 1)[1]
    return url


def _insert_batch(rows: list[dict]) -> None:
    """Stream a batch with COPY on a fresh TLS connection (avoids Neon SSL 'bad length')."""
    last_err: BaseException | None = None
    for attempt in range(1, INSERT_RETRIES + 1):
        try:
            with psycopg.connect(_psycopg_dsn(), autocommit=False, connect_timeout=30) as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    with cur.copy(_COPY_SQL) as copy:
                        copy.set_types(
                            [
                                "text",
                                "text",
                                "text",
                                "text",
                                "int4",
                                "text",
                                "vector",
                                "text",
                                "timestamptz",
                            ]
                        )
                        for r in rows:
                            copy.write_row(
                                (
                                    r["id"],
                                    r["scope"],
                                    r["owner_user_id"],
                                    r["manual_id"],
                                    r["chunk_index"],
                                    r["content"],
                                    r["embedding"],
                                    r["embedding_model"],
                                    r["created_at"],
                                )
                            )
                conn.commit()
            return
        except (OperationalError, psycopg.Error) as exc:
            last_err = exc
            try:
                engine.dispose()
            except Exception:
                pass
            wait = min(2**attempt, 30)
            brief = str(getattr(exc, "orig", exc)).split("\n", 1)[0][:180]
            print(f"Insert failed (attempt {attempt}/{INSERT_RETRIES}): {brief}", flush=True)
            print(f"Retrying in {wait}s...", flush=True)
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def _sync_global_to_postgres(
    docs: list[str],
    embeddings: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
    *,
    resume: bool = False,
) -> None:
    """Insert in batches. Drop HNSW during load so Postgres is not OOM-killed."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if _is_neon_url() and batch_size > NEON_MAX_BATCH_SIZE:
        print(
            f"Neon detected; capping batch size {batch_size} -> {NEON_MAX_BATCH_SIZE}."
        )
        batch_size = NEON_MAX_BATCH_SIZE
    total = len(docs)
    start_at = 0
    if resume:
        start_at = _max_global_chunk_index() + 1
        print(f"Resuming from chunk_index={start_at}", flush=True)

    with SessionLocal() as db:
        db.execute(text(f"DROP INDEX IF EXISTS {HNSW_INDEX_NAME}"))
        if not resume:
            delete_global_chunks(db)
        db.commit()

    for start in range(start_at, total, batch_size):
        end = min(start + batch_size, total)
        rows = [
            {
                "id": str(uuid4()),
                "scope": "global",
                "owner_user_id": None,
                "manual_id": None,
                "chunk_index": i,
                "content": docs[i],
                "embedding": np.asarray(embeddings[i], dtype=np.float32).reshape(-1),
                "embedding_model": EMBEDDING_MODEL,
                "created_at": datetime.now(timezone.utc),
            }
            for i in range(start, end)
        ]
        _insert_batch(rows)
        print(f"Inserted {end}/{total}", flush=True)

    print("Creating HNSW index (this can take a few minutes)...", flush=True)
    with SessionLocal() as db:
        db.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS {HNSW_INDEX_NAME} "
                "ON rag_kb_chunks USING hnsw (embedding vector_l2_ops)"
            )
        )
        db.commit()


def _use_postgres_for_kb() -> bool:
    if os.environ.get("RAG_KB_BACKEND", "auto").lower().strip() == "faiss":
        return False
    return kb_pgvector_enabled()


def _load_or_compute_embeddings(docs: list[str], *, force_reembed: bool) -> np.ndarray:
    cache = EMBEDDINGS_CACHE_PATH
    if not force_reembed and cache.exists():
        cached = np.load(cache)
        if cached.ndim == 2 and cached.shape[0] >= len(docs) and cached.shape[1] == 384:
            print(f"Reusing cached embeddings {cache} rows={cached.shape[0]} (using first {len(docs)})")
            return cached[: len(docs)].astype(np.float32)
        print(
            f"Ignoring embedding cache {cache} shape={getattr(cached, 'shape', None)}; "
            f"need at least ({len(docs)}, 384)."
        )

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Embedding {len(docs)} documents...")
    embeddings = model.encode(
        docs,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    embeddings = embeddings.astype(np.float32)
    os.makedirs(cache.parent, exist_ok=True)
    np.save(cache, embeddings)
    print(f"Saved embeddings cache: {cache}")
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed unified_docs.txt and write the global KB (Postgres pgvector or FAISS)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Index only the first N documents (faster smoke test).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Postgres COPY batch size (default 200). Capped at 250 on Neon.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Do not wipe existing global chunks; continue after the current max chunk_index.",
    )
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="Ignore models/global_embeddings.npy and recompute embeddings.",
    )
    args = parser.parse_args()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Missing dataset file: {DATA_PATH}. Run `python scripts/build_knowledge_base.py` "
            f"(place CSVs under data/raw/; uses DelucionQA unless --skip-qa) or update DATA_PATH."
        )

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]

    if not docs:
        raise ValueError(f"No documents found in {DATA_PATH}.")

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        docs = docs[: args.limit]
        print(f"Limiting to first {len(docs)} documents.")

    embeddings = _load_or_compute_embeddings(docs, force_reembed=args.reembed)

    if _use_postgres_for_kb():
        print("Writing global KB to PostgreSQL (rag_kb_chunks, scope=global)...")
        _sync_global_to_postgres(
            docs, embeddings, batch_size=args.batch_size, resume=args.resume
        )
        print("PostgreSQL sync OK.")
    else:
        print("Writing FAISS index + models/docs.txt ...")
        _write_faiss(docs, embeddings)
        print("Index built successfully!")
        print(f"- FAISS index: {INDEX_PATH}")
        print(f"- Stored docs : {DOCS_OUT_PATH}")


if __name__ == "__main__":
    main()
