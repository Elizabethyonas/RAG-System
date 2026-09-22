"""Load a curated slice of unified_docs.txt into the global KB.

`build_index.py --limit N` keeps the first N lines, which are all used-car sales
listings; the manual Q&A block sits at the end of the corpus. This script keeps the
whole Q&A block and fills the remaining budget with an evenly spread sample of the
spec block, so a size-capped database (e.g. Neon free tier, 512 MB) still answers
troubleshooting questions.

Embeddings come from models/global_embeddings.npy, indexed by line number, so no
re-embedding is needed.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sqlalchemy import text

from database import SessionLocal
from rag_kb import delete_global_chunks
from scripts.build_index import (
    DATA_PATH,
    DEFAULT_BATCH_SIZE,
    EMBEDDINGS_CACHE_PATH,
    EMBEDDING_MODEL,
    HNSW_INDEX_NAME,
    NEON_MAX_BATCH_SIZE,
    _insert_batch,
    _is_neon_url,
)

LISTING_PREFIX = "The car "
SPEC_DOC_START = re.compile(r"^The .+ \(\d+(?:\.\d+)?–\d+(?:\.\d+)?\)")

# Below this row count an exact scan beats HNSW on both latency and recall. The corpus is
# dominated by near-identical spec vectors, so graph search strands the small manual Q&A
# cluster: measured at 20k rows, ef_search=400 found none of the correct chunks in 219 ms
# while a seq scan found them in 51 ms.
HNSW_MIN_ROWS = 50_000


def split_blocks(lines: list[str]) -> list[tuple[str, int, int]]:
    """Group lines into runs of sales listings vs everything else. Returns (kind, start, end_exclusive)."""
    blocks: list[tuple[str, int, int]] = []
    if not lines:
        return blocks
    kind = "listing" if lines[0].startswith(LISTING_PREFIX) else "other"
    start = 0
    for i, line in enumerate(lines[1:], start=1):
        cur = "listing" if line.startswith(LISTING_PREFIX) else "other"
        if cur != kind:
            blocks.append((kind, start, i))
            kind, start = cur, i
    blocks.append((kind, start, len(lines)))
    return blocks


def group_docs(lines: list[str], start: int, end: int) -> list[tuple[int, int]]:
    """Split a line range into whole documents so sampling never cuts a doc in half."""
    starts = [i for i in range(start, end) if SPEC_DOC_START.match(lines[i])]
    if not starts:
        return [(i, i + 1) for i in range(start, end)]
    if starts[0] > start:
        starts.insert(0, start)
    return [(s, starts[j + 1] if j + 1 < len(starts) else end) for j, s in enumerate(starts)]


def pick_spread(docs: list[tuple[int, int]], budget: int) -> list[int]:
    """Pick whole docs spread evenly across the range until `budget` lines are used.

    Walks the range in strided passes, so the first pass spans the whole corpus and later
    passes fill the gaps; a short first pass therefore still reaches the budget.
    """
    if budget <= 0 or not docs:
        return []
    avg = sum(e - s for s, e in docs) / len(docs)
    stride = max(1, int(len(docs) * avg / budget))

    picked: list[int] = []
    for offset in range(stride):
        for j in range(offset, len(docs), stride):
            s, e = docs[j]
            if len(picked) + (e - s) > budget:
                continue
            picked.extend(range(s, e))
            if len(picked) == budget:
                return sorted(picked)
    return sorted(picked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=20000, help="Total chunks to load (default 20000).")
    parser.add_argument("--data", type=Path, default=Path(DATA_PATH))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="Report the selection without writing.")
    parser.add_argument(
        "--hnsw",
        choices=("auto", "always", "never"),
        default="auto",
        help=f"Build the vector index. auto = only above {HNSW_MIN_ROWS} rows (default).",
    )
    args = parser.parse_args()

    lines = args.data.read_text(encoding="utf-8").splitlines()
    lines = [ln.strip() for ln in lines]
    if not lines:
        raise SystemExit(f"No documents in {args.data}")

    blocks = split_blocks(lines)
    others = [b for b in blocks if b[0] == "other"]
    if not others:
        raise SystemExit("No non-listing content found; nothing worth curating.")

    qa_kind, qa_start, qa_end = others[-1]
    spec_kind, spec_start, spec_end = max(others, key=lambda b: b[2] - b[1])

    qa_idx = list(range(qa_start, qa_end))
    if (spec_start, spec_end) == (qa_start, qa_end):
        spec_idx: list[int] = []
    else:
        docs = group_docs(lines, spec_start, spec_end)
        spec_idx = pick_spread(docs, args.total - len(qa_idx))

    selected = qa_idx + spec_idx
    print(f"Manual Q&A : {len(qa_idx)} chunks (lines {qa_start + 1}-{qa_end})")
    print(f"Specs      : {len(spec_idx)} chunks sampled from lines {spec_start + 1}-{spec_end}")
    print(f"Total      : {len(selected)} chunks")
    if args.dry_run:
        return

    embeddings = np.load(EMBEDDINGS_CACHE_PATH, mmap_mode="r")
    if embeddings.shape[0] < len(lines):
        raise SystemExit(
            f"Embedding cache {EMBEDDINGS_CACHE_PATH} has {embeddings.shape[0]} rows but the corpus "
            f"has {len(lines)} lines. Rebuild with `python scripts/build_index.py --reembed`."
        )

    batch_size = args.batch_size
    if _is_neon_url() and batch_size > NEON_MAX_BATCH_SIZE:
        print(f"Neon detected; capping batch size {batch_size} -> {NEON_MAX_BATCH_SIZE}.")
        batch_size = NEON_MAX_BATCH_SIZE

    with SessionLocal() as db:
        db.execute(text(f"DROP INDEX IF EXISTS {HNSW_INDEX_NAME}"))
        delete_global_chunks(db)
        db.commit()

    total = len(selected)
    for start in range(0, total, batch_size):
        batch = selected[start : start + batch_size]
        rows = [
            {
                "id": str(uuid4()),
                "scope": "global",
                "owner_user_id": None,
                "manual_id": None,
                "chunk_index": start + offset,
                "content": lines[line_no],
                "embedding": np.asarray(embeddings[line_no], dtype=np.float32).reshape(-1),
                "embedding_model": EMBEDDING_MODEL,
                "created_at": datetime.now(timezone.utc),
            }
            for offset, line_no in enumerate(batch)
        ]
        _insert_batch(rows)
        print(f"Inserted {min(start + batch_size, total)}/{total}", flush=True)

    build_hnsw = args.hnsw == "always" or (args.hnsw == "auto" and total >= HNSW_MIN_ROWS)
    if build_hnsw:
        print("Creating HNSW index (this can take a few minutes)...", flush=True)
        with SessionLocal() as db:
            db.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {HNSW_INDEX_NAME} "
                    "ON rag_kb_chunks USING hnsw (embedding vector_l2_ops)"
                )
            )
            db.commit()
    else:
        print(f"Skipping HNSW index ({total} rows < {HNSW_MIN_ROWS}); exact scan is faster here.")
    print("PostgreSQL sync OK.")


if __name__ == "__main__":
    main()
