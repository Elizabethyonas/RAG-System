"""all-MiniLM-L6-v2 query embeddings without PyTorch.

The knowledge base was embedded with SentenceTransformer. That pipeline mean-pools
token states and then L2-normalizes them. Render's free instance has 512 MB;
importing torch to run the same model gets the process killed. ONNX Runtime stays
small enough to answer a query.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = REPO_ROOT / "models" / "minilm-onnx"
MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 256


def model_dir() -> Path:
    override = os.environ.get("RAG_MINILM_DIR", "").strip()
    return Path(override).expanduser() if override else DEFAULT_DIR


def ensure_minilm_files() -> Path:
    """Download the ONNX weights and tokenizer once. Safe to call repeatedly."""
    dest = model_dir()
    model_path = dest / "onnx" / "model.onnx"
    tokenizer_path = dest / "tokenizer.json"
    if model_path.is_file() and tokenizer_path.is_file():
        return dest

    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    for filename in ("onnx/model.onnx", "tokenizer.json"):
        hf_hub_download(repo_id=MODEL_REPO, filename=filename, local_dir=str(dest))
    if not model_path.is_file() or not tokenizer_path.is_file():
        raise RuntimeError(f"MiniLM ONNX download did not produce files under {dest}")
    return dest


class OnnxMiniLM:
    """Drop-in for the SentenceTransformer.encode call used at query time."""

    def __init__(self, directory: Path | None = None) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        dest = directory or ensure_minilm_files()
        model_path = dest / "onnx" / "model.onnx"
        tokenizer_path = dest / "tokenizer.json"
        print(f"Loading MiniLM ONNX embedder from {model_path}", flush=True)

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        pad_id = tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            pad_id = 0
        tokenizer.enable_truncation(max_length=MAX_LENGTH)
        tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
        self._tokenizer = tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {item.name for item in self._session.get_inputs()}
        print("MiniLM ONNX embedder ready", flush=True)

    def encode(self, texts: str | list[str], **_options: object) -> np.ndarray:
        """Return unit-length float32 vectors, matching SentenceTransformer's MiniLM pipeline."""
        if isinstance(texts, str):
            batch = [texts]
        else:
            batch = list(texts)
        if not batch:
            return np.zeros((0, 384), dtype=np.float32)

        encodings = self._tokenizer.encode_batch(batch)
        input_ids = np.asarray([item.ids for item in encodings], dtype=np.int64)
        attention_mask = np.asarray([item.attention_mask for item in encodings], dtype=np.int64)
        token_type_ids = np.asarray([item.type_ids for item in encodings], dtype=np.int64)

        feeds: dict[str, np.ndarray] = {}
        if "input_ids" in self._input_names:
            feeds["input_ids"] = input_ids
        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = attention_mask
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = token_type_ids

        hidden = self._session.run(None, feeds)[0]
        if hidden.ndim == 3:
            mask = attention_mask.astype(np.float32)[..., None]
            summed = np.sum(hidden * mask, axis=1)
            counts = np.maximum(mask.sum(axis=1), 1e-9)
            embeddings = summed / counts
        elif hidden.ndim == 2:
            embeddings = hidden
        else:
            raise RuntimeError(f"Unexpected MiniLM ONNX output shape {hidden.shape}")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        return embeddings
