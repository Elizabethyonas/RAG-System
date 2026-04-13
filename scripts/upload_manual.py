from __future__ import annotations

import argparse
import re
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL = "all-MiniLM-L6-v2"


def detect_car_model(text: str) -> str:
    patterns = (
        r"\b(Toyota\s+[A-Za-z0-9\-]+)\b",
        r"\b(Honda\s+[A-Za-z0-9\-]+)\b",
        r"\b(Hyundai\s+[A-Za-z0-9\-]+)\b",
        r"\b(Ford\s+[A-Za-z0-9\-]+)\b",
        r"\b(BMW\s+[A-Za-z0-9\-]+)\b",
        r"\b(Mercedes[- ]Benz\s+[A-Za-z0-9\-]+)\b",
        r"\b(Maruti(?:\s+Suzuki)?\s+[A-Za-z0-9\-]+)\b",
        r"\b(Kia\s+[A-Za-z0-9\-]+)\b",
        r"\b(Tata\s+[A-Za-z0-9\-]+)\b",
        r"\b(Mahindra\s+[A-Za-z0-9\-]+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return "Unknown Model"


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def chunk_text(text: str, size: int = 200) -> list[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def build_user_index(
    pdf_path: Path, user_id: str = "user1", chunk_words: int = 200
) -> dict[str, str | int]:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Extracting text from PDF: {pdf_path}")
    text = extract_text(pdf_path)
    if not text:
        raise ValueError("No extractable text found in PDF. Try another PDF or add OCR later.")

    chunks = chunk_text(text, size=chunk_words)
    if not chunks:
        raise ValueError("No chunks produced from PDF text.")
    print(f"Total chunks: {len(chunks)}")
    model_name = detect_car_model(text)
    print(f"Detected car model: {model_name}")

    model = SentenceTransformer(MODEL)
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    user_dir = Path("models") / user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(user_dir / "faiss_index"))
    with open(user_dir / "docs.txt", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c + "\n")
    with open(user_dir / "meta.txt", "w", encoding="utf-8") as f:
        f.write(model_name + "\n")

    print(f"User manual indexed successfully for user_id='{user_id}'")
    print(f"- Index: {user_dir / 'faiss_index'}")
    print(f"- Docs : {user_dir / 'docs.txt'}")
    print(f"- Meta : {user_dir / 'meta.txt'}")
    return {
        "user_id": user_id,
        "model_name": model_name,
        "chunks": len(chunks),
        "index_path": str(user_dir / "faiss_index"),
        "docs_path": str(user_dir / "docs.txt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a user-specific FAISS index from a manual PDF.")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to manual PDF")
    parser.add_argument("--user-id", default="user1", help="User id (writes to models/<user-id>/...)")
    parser.add_argument("--chunk-words", type=int, default=200, help="Words per chunk")
    args = parser.parse_args()

    build_user_index(args.pdf, user_id=args.user_id, chunk_words=args.chunk_words)


if __name__ == "__main__":
    main()
