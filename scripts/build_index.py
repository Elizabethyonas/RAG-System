from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/processed/car_docs.txt"
INDEX_PATH = "models/faiss_index"
DOCS_OUT_PATH = "models/docs.txt"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Missing dataset file: {DATA_PATH}. Create it or update DATA_PATH."
        )

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]

    if not docs:
        raise ValueError(f"No documents found in {DATA_PATH}.")

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

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_OUT_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d + "\n")

    print("Index built successfully!")
    print(f"- FAISS index: {INDEX_PATH}")
    print(f"- Stored docs : {DOCS_OUT_PATH}")


if __name__ == "__main__":
    main()

