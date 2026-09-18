from __future__ import annotations

import argparse
import os
import sys
import warnings
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
from sqlalchemy import text

warnings.filterwarnings(
    "ignore",
    message=".*clean_up_tokenization_spaces.*",
    category=FutureWarning,
)

from database import RagKbChunkModel, SessionLocal
from rag_kb import delete_global_chunks, kb_pgvector_enabled

DATA_PATH = "data/processed/unified_docs.txt"
INDEX_PATH = "models/faiss_index"
DOCS_OUT_PATH = "models/docs.txt"
EMBEDDINGS_CACHE_PATH = Path("models/global_embeddings.npy")
HNSW_INDEX_NAME = "ix_rag_kb_chunks_embedding_hnsw"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 500


def _write_faiss(docs: list[str], embeddings: np.ndarray) -> None:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_OUT_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d + "\n")


def _sync_global_to_postgres(
    docs: list[str],
    embeddings: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Insert in batches. Drop HNSW during load so Docker Postgres is not OOM-killed."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    total = len(docs)
    with SessionLocal() as db:
        db.execute(text(f"DROP INDEX IF EXISTS {HNSW_INDEX_NAME}"))
        delete_global_chunks(db)
        db.commit()

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            rows = [
                {
                    "id": str(uuid4()),
                    "scope": "global",
                    "owner_user_id": None,
                    "manual_id": None,
                    "chunk_index": i,
                    "content": docs[i],
                    "embedding": embeddings[i].astype(np.float64).flatten().tolist(),
                    "embedding_model": EMBEDDING_MODEL,
                }
                for i in range(start, end)
            ]
            db.bulk_insert_mappings(RagKbChunkModel, rows)
            db.commit()
            print(f"Inserted {end}/{total}", flush=True)

        print("Creating HNSW index (this can take a few minutes)...", flush=True)
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
        help="Postgres insert batch size (default 500). Smaller is safer on low-memory Docker.",
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
        _sync_global_to_postgres(docs, embeddings, batch_size=args.batch_size)
        print("PostgreSQL sync OK.")
    else:
        print("Writing FAISS index + models/docs.txt ...")
        _write_faiss(docs, embeddings)
        print("Index built successfully!")
        print(f"- FAISS index: {INDEX_PATH}")
        print(f"- Stored docs : {DOCS_OUT_PATH}")


if __name__ == "__main__":
    main()
