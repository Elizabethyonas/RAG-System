from __future__ import annotations

import argparse
import os
import warnings
from typing import List

# Must run before importing torch/numpy/faiss: duplicate OpenMP runtimes often segfault on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings(
    "ignore",
    message=".*clean_up_tokenization_spaces.*",
    category=FutureWarning,
)

INDEX_PATH = "models/faiss_index"
DOCS_PATH = "models/docs.txt"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_faiss_and_docs() -> tuple[faiss.Index, List[str]]:
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"Missing FAISS index: {INDEX_PATH}. Run `python scripts/build_index.py` first."
        )
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(
            f"Missing docs file: {DOCS_PATH}. Run `python scripts/build_index.py` first."
        )

    index = faiss.read_index(INDEX_PATH)
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]

    return index, docs


def get_llm_device() -> str:
    """Pick device for the causal LM. Apple GPU off by default — TinyLlama + MPS often segfaults."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        if os.environ.get("RAG_USE_MPS", "").lower() in ("1", "true", "yes"):
            return "mps"
        return "cpu"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG chat over car_docs FAISS index.")
    parser.add_argument(
        "--car",
        default="",
        help="Optional car context (model, year, trim). Also set RAG_CAR_CONTEXT env var.",
    )
    args = parser.parse_args()

    car_context = (args.car or os.environ.get("RAG_CAR_CONTEXT", "") or "").strip()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = get_llm_device()
    if torch.backends.mps.is_available() and device == "cpu":
        print("Using device: cpu (LLM; MPS off by default — set RAG_USE_MPS=1 to try Apple GPU)")
    else:
        print(f"Using device: {device}")

    if car_context:
        print(f"Car context: {car_context}")

    # Load LLM before SentenceTransformer + FAISS to reduce native-library init clashes on macOS.
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL,
        clean_up_tokenization_spaces=False,
    )

    if device == "cuda":
        torch_dtype = torch.float16
    elif device == "mps":
        torch_dtype = torch.float32
    else:
        torch_dtype = None

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch_dtype,
    ).to(device)
    llm.eval()

    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    index, docs = load_faiss_and_docs()

    def retrieve(query: str, k: int = 2, car_context: str = "") -> List[str]:
        search_query = f"{car_context} {query}".strip() if car_context else query
        q_emb = embed_model.encode(
            [search_query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)

        # FAISS expects shape (n_queries, dim)
        _, indices = index.search(q_emb, k)

        results: List[str] = []
        for i in indices[0].tolist():
            if 0 <= i < len(docs):
                results.append(docs[i])
        return results

    def generate_answer(query: str, car_context: str = "") -> str:
        context_chunks = retrieve(query, k=2, car_context=car_context)
        context = "\n".join(context_chunks) if context_chunks else "(no relevant context found)"
        if car_context.strip():
            context = f"(Vehicle focus: {car_context.strip()})\n\n{context}"

        prompt = f"""
You are a car assistant.

Answer the question using ONLY the information from the context below.

- Do NOT repeat the context
- Do NOT generate extra questions
- Give a short, clear answer
- If the context does not contain the answer, say you do not have that information in the retrieved documents

Context:
{context}

Question:
{query}

Final Answer:
""".strip()

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = llm.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Final Answer:" in full_output:
            return full_output.split("Final Answer:")[-1].strip()
        return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    # CLI loop
    while True:
        q = input("\nAsk about your car (type 'exit' to quit): ").strip()
        if not q:
            continue
        if q.lower() == "exit":
            break

        answer = generate_answer(q, car_context=car_context)
        print("\n" + answer)


if __name__ == "__main__":
    main()

