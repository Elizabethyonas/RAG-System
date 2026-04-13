from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
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


def load_faiss_and_docs(index_path: str, docs_path: str) -> tuple[faiss.Index, List[str]]:
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Missing FAISS index: {index_path}. Run the index builder first."
        )
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"Missing docs file: {docs_path}. Run the index builder first."
        )

    index = faiss.read_index(index_path)
    with open(docs_path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]
    return index, docs


def load_global_index() -> tuple[faiss.Index, List[str]]:
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"Missing FAISS index: {INDEX_PATH}. Run `python scripts/build_index.py` first."
        )
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(
            f"Missing docs file: {DOCS_PATH}. Run `python scripts/build_index.py` first."
        )

    return load_faiss_and_docs(INDEX_PATH, DOCS_PATH)


def load_user_index(user_id: str) -> tuple[faiss.Index | None, List[str]]:
    user_index_path = os.path.join("models", user_id, "faiss_index")
    user_docs_path = os.path.join("models", user_id, "docs.txt")
    if not os.path.exists(user_index_path) or not os.path.exists(user_docs_path):
        return None, []
    return load_faiss_and_docs(user_index_path, user_docs_path)


def load_user_meta(user_id: str) -> str:
    meta_path = os.path.join("models", user_id, "meta.txt")
    if not os.path.exists(meta_path):
        return ""
    with open(meta_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_llm_device() -> str:
    """Pick device for the causal LM. Apple GPU off by default — TinyLlama + MPS often segfaults."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        if os.environ.get("RAG_USE_MPS", "").lower() in ("1", "true", "yes"):
            return "mps"
        return "cpu"
    return "cpu"


@dataclass
class RetrievalConfig:
    k_user: int = 2
    k_global: int = 1


class RAGAssistant:
    def __init__(self, user_id: str = "user1", car_context: str = "") -> None:
        self.user_id = user_id
        self.car_context = car_context.strip()
        self.user_model = load_user_meta(user_id)

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.device = get_llm_device()

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL,
            clean_up_tokenization_spaces=False,
        )
        if self.device == "cuda":
            torch_dtype = torch.float16
        elif self.device == "mps":
            torch_dtype = torch.float32
        else:
            torch_dtype = None
        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.llm.eval()

        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        self.global_index, self.global_docs = load_global_index()
        self.user_index, self.user_docs = load_user_index(user_id)

    @property
    def has_user_index(self) -> bool:
        return self.user_index is not None and bool(self.user_docs)

    def _search(self, index: faiss.Index, docs: List[str], q_emb: np.ndarray, k: int) -> List[str]:
        _, indices = index.search(q_emb, k)
        results: List[str] = []
        for i in indices[0].tolist():
            if 0 <= i < len(docs):
                results.append(docs[i])
        return results

    def retrieve(
        self, query: str, config: RetrievalConfig | None = None, car_context: str = ""
    ) -> List[str]:
        config = config or RetrievalConfig()
        ctx = car_context.strip() if car_context.strip() else self.car_context
        if self.user_model and not ctx:
            ctx = self.user_model

        search_query = f"{ctx} {query}".strip() if ctx else query
        q_emb = self.embed_model.encode(
            [search_query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)

        user_results: List[str] = []
        if self.user_index is not None and self.user_docs:
            user_results = self._search(self.user_index, self.user_docs, q_emb, config.k_user)
        global_results = self._search(self.global_index, self.global_docs, q_emb, config.k_global)

        merged: List[str] = []
        for c in user_results + global_results:
            if c and c not in merged:
                merged.append(c)
        return merged

    def _answer_mode(self, query: str) -> str:
        q = query.lower()
        simple_hints = ("explain", "what is", "what's", "meaning", "why", "how does")
        technical_hints = ("torque", "horsepower", "engine", "spec", "diagnostic", "dtc")
        if any(h in q for h in technical_hints):
            return "technical"
        if any(h in q for h in simple_hints):
            return "simple"
        return "simple"

    def _build_prompt(self, context: str, query: str, mode: str) -> str:
        if mode == "technical":
            return f"""
You are an expert automotive assistant.

Provide a precise and technical answer based on the context.

Guidelines:
- Use correct automotive terminology
- Be concise but informative
- Do NOT repeat the question
- Do NOT generate extra questions
- If unsure, say you are not certain

Context:
{context}

Question:
{query}

Answer:
""".strip()

        return f"""
You are a friendly and knowledgeable car assistant helping drivers understand their vehicles.

Answer the question using ONLY the context provided.

Guidelines:
- Speak naturally like a human expert
- Be clear and concise
- Do NOT repeat the question
- Do NOT generate extra questions
- If unsure, say you are not certain

Context:
{context}

Question:
{query}

Answer:
""".strip()

    def _format_answer(self, text: str) -> str:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            return "I am not certain based on the retrieved context."
        if cleaned.endswith(("?", "!", ".")):
            return cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
        return (cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()) + "."

    def generate_answer(self, query: str, car_context: str = "") -> str:
        context_chunks = self.retrieve(query, config=RetrievalConfig(k_user=2, k_global=1), car_context=car_context)
        context = "\n".join(context_chunks) if context_chunks else "(no relevant context found)"
        active_context = car_context.strip() if car_context.strip() else self.car_context
        if self.user_model and self.has_user_index:
            active_context = f"{active_context} {self.user_model}".strip()
        if active_context:
            context = f"(Vehicle focus: {active_context})\n\n{context}"
        mode = self._answer_mode(query)
        prompt = self._build_prompt(context=context, query=query, mode=mode)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in full_output:
            answer = full_output.split("Answer:")[-1].strip()
        elif "Final Answer:" in full_output:
            answer = full_output.split("Final Answer:")[-1].strip()
        else:
            answer = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        answer = answer.split("Question:")[0].split("Context:")[0].strip()
        return self._format_answer(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG chat over car_docs FAISS index.")
    parser.add_argument(
        "--car",
        default="",
        help="Optional car context (model, year, trim). Also set RAG_CAR_CONTEXT env var.",
    )
    parser.add_argument(
        "--user-id",
        default="user1",
        help="Optional user id for manual-aware retrieval (expects models/<user-id>/faiss_index).",
    )
    args = parser.parse_args()

    car_context = (args.car or os.environ.get("RAG_CAR_CONTEXT", "") or "").strip()
    assistant = RAGAssistant(user_id=args.user_id, car_context=car_context)

    if torch.backends.mps.is_available() and assistant.device == "cpu":
        print("Using device: cpu (LLM; MPS off by default — set RAG_USE_MPS=1 to try Apple GPU)")
    else:
        print(f"Using device: {assistant.device}")
    if assistant.has_user_index:
        model_note = f" ({assistant.user_model})" if assistant.user_model else ""
        print(f"Loaded user manual index: models/{args.user_id}/faiss_index{model_note}")
    else:
        print(f"No user manual index found for '{args.user_id}', using global index only.")
    if car_context:
        print(f"Car context: {car_context}")

    # CLI loop
    while True:
        q = input("\nAsk about your car (type 'exit' to quit): ").strip()
        if not q:
            continue
        if q.lower() == "exit":
            break

        answer = assistant.generate_answer(q, car_context=car_context)
        print("\n" + answer)


if __name__ == "__main__":
    main()

