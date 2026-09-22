from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import re

import requests

# Repo root on sys.path (so `database` / `rag_kb` resolve when running as `python scripts/chat.py`).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Must run before importing torch/numpy/faiss: duplicate OpenMP runtimes often segfault on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)

import faiss
import numpy as np
from gradio_client import Client
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from database import RagUserProfileModel, SessionLocal


def _patch_transformers_mps_isin() -> None:
    """Torch 2.2 on Apple GPU crashes inside transformers when the pad token is a scalar.

    ``isin_mps_friendly`` indexes ``test_elements.shape[0]``. A 0-dim pad token has no
    dimension 0, so ``generate()`` raises IndexError and the chat request returns 500.
    """
    import transformers.generation.utils as gen_utils
    import transformers.pytorch_utils as pt_utils

    original = pt_utils.isin_mps_friendly
    if getattr(original, "_rag_mps_patch", False):
        return

    def isin_mps_friendly(elements: torch.Tensor, test_elements: torch.Tensor | int) -> torch.Tensor:
        if isinstance(test_elements, torch.Tensor) and test_elements.ndim == 0:
            test_elements = test_elements.reshape(1)
        return original(elements, test_elements)

    isin_mps_friendly._rag_mps_patch = True  # type: ignore[attr-defined]
    pt_utils.isin_mps_friendly = isin_mps_friendly
    gen_utils.isin_mps_friendly = isin_mps_friendly


_patch_transformers_mps_isin()

from rag_kb import count_user_chunks, kb_pgvector_enabled, search_kb_l2

warnings.filterwarnings(
    "ignore",
    message=".*clean_up_tokenization_spaces.*",
    category=FutureWarning,
)

INDEX_PATH = "models/faiss_index"
DOCS_PATH = "models/docs.txt"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# One shared model/client for all RAGAssistant instances — critical on low-RAM hosts (e.g. Render Free),
# where caching assistants per user/context would load SentenceTransformer repeatedly and OOM → 502.
_shared_sentence_transformer: SentenceTransformer | None = None
_hf_space_client_lock_target: str | None = None
_hf_space_client_singleton: Client | None = None


def _get_shared_sentence_transformer() -> SentenceTransformer:
    global _shared_sentence_transformer
    if _shared_sentence_transformer is None:
        _shared_sentence_transformer = SentenceTransformer(EMBEDDING_MODEL)
    return _shared_sentence_transformer


def _get_hf_space_client(target: str) -> Client:
    global _hf_space_client_lock_target, _hf_space_client_singleton
    if _hf_space_client_singleton is None or _hf_space_client_lock_target != target:
        _hf_space_client_lock_target = target
        _hf_space_client_singleton = Client(target)
    return _hf_space_client_singleton


LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct").strip()
FALLBACK_LLM_MODEL = os.environ.get(
    "RAG_FALLBACK_LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
).strip()

REMOTE_LLM_URL = os.environ.get("RAG_REMOTE_LLM_URL", "").strip()
REMOTE_LLM_SECRET = os.environ.get("RAG_REMOTE_LLM_SECRET", "").strip()
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "").strip().lower()
HF_SPACE_ID = os.environ.get("HF_SPACE_ID", "").strip()
HF_SPACE_URL = os.environ.get("HF_SPACE_URL", "").strip()
HF_SPACE_API_NAME = os.environ.get("HF_SPACE_API_NAME", "/predict").strip() or "/predict"

# Increase Hub network timeouts to reduce transient download failures.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")


def use_pgvector_for_kb() -> bool:
    """auto: Postgres URL -> pgvector; faiss: local FAISS files; pgvector: force DB (must be Postgres)."""
    backend = os.environ.get("RAG_KB_BACKEND", "auto").lower().strip()
    if backend == "faiss":
        return False
    if backend == "pgvector":
        if not kb_pgvector_enabled():
            raise RuntimeError(
                "RAG_KB_BACKEND=pgvector requires DATABASE_URL to be a PostgreSQL URL "
                "(same DB as Express + pgvector extension)."
            )
        return True
    return kb_pgvector_enabled()


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
    if use_pgvector_for_kb():
        with SessionLocal() as db:
            row = db.get(RagUserProfileModel, user_id)
            if row and (row.vehicle_meta or "").strip():
                return row.vehicle_meta.strip()
        return ""
    meta_path = os.path.join("models", user_id, "meta.txt")
    if not os.path.exists(meta_path):
        return ""
    with open(meta_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_llm_device() -> str:
    """Pick device for the causal LM.

    Apple GPU is the default on Macs. CPU inference of the 3B model is
    single-threaded (OpenMP is pinned to 1 to avoid a macOS crash) and takes
    minutes per reply. Set RAG_USE_MPS=0 to force CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        flag = os.environ.get("RAG_USE_MPS", "1").strip().lower()
        if flag not in ("0", "false", "no", "off"):
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return "mps"
        return "cpu"
    return "cpu"


_llm_load_lock = threading.Lock()
_shared_local_llm: tuple[str, str, object, object] | None = None


def _get_shared_local_llm(device: str) -> tuple[str, object, object]:
    """Load causal-LM weights once and reuse them across RAGAssistant instances."""
    global _shared_local_llm
    with _llm_load_lock:
        if _shared_local_llm is not None and _shared_local_llm[1] == device:
            name, _, tokenizer, model = _shared_local_llm
            return name, tokenizer, model

        if device == "cuda":
            torch_dtype = torch.float16
        elif device == "mps":
            torch_dtype = torch.float16
        else:
            torch_dtype = None

        load_errors: List[str] = []
        for candidate_model in [LLM_MODEL, FALLBACK_LLM_MODEL]:
            if not candidate_model:
                continue
            try:
                print(f"Loading LLM {candidate_model} on {device}...", flush=True)
                tokenizer = AutoTokenizer.from_pretrained(
                    candidate_model,
                    clean_up_tokenization_spaces=False,
                )
                if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                model = AutoModelForCausalLM.from_pretrained(
                    candidate_model,
                    torch_dtype=torch_dtype,
                ).to(device)
                model.eval()
                gen_cfg = model.generation_config
                gen_cfg.do_sample = False
                gen_cfg.top_k = None
                gen_cfg.top_p = None
                _shared_local_llm = (candidate_model, device, tokenizer, model)
                print(f"LLM ready: {candidate_model} on {device}", flush=True)
                return candidate_model, tokenizer, model
            except Exception as e:  # pragma: no cover - runtime/network dependent
                load_errors.append(f"{candidate_model}: {e}")

        msg = " | ".join(load_errors) if load_errors else "Unknown model loading failure"
        raise RuntimeError(f"Failed to load any LLM model. {msg}")


@dataclass
class RetrievalConfig:
    k_user: int = 2
    k_global: int = 3


CARCARE_PERSONA_PROMPT = """
You are CarCare AI, a practical and safety-first assistant for drivers.

ROLE
- Help users with vehicle maintenance, basic troubleshooting, safe driving guidance, and understanding car features/manuals.
- Use clear, simple language for non-experts.
- Be concise, structured, and actionable.

TONE
- Calm, professional, friendly.
- Never judgmental.
- Prefer short sections and bullet points over long paragraphs.

RESPONSE STYLE (ALWAYS)
1) Quick answer (1–2 lines)
2) Steps to follow (numbered)
3) What to check/prepare (bullets)
4) When to seek a mechanic (if relevant)
5) Safety warning (if relevant)

SAFETY RULES
- Prioritize human safety over convenience or cost.
- If there is risk (brake failure, fuel leak smell, overheating, smoke, warning lights with severe symptoms), advise the user to stop driving and seek professional help.
- Do not provide instructions that bypass safety systems or legal requirements.
- If unsure, say uncertainty clearly and suggest safe next steps.

BOUNDARIES
- Do not claim real-time sensor access unless explicitly provided in user context.
- Do not invent specs; if data is missing, ask a short clarifying question.
- Do not present guesses as facts.
- Avoid complex jargon unless user asks for technical detail.

VEHICLE-AWARE BEHAVIOR
- If vehicle details are provided, tailor answers to that vehicle.
- If key details are missing, ask for only what is necessary (symptom first; vehicle details only if not already provided in the app context).
- If manual context exists, prefer manual-consistent guidance.

FORMATTING RULES
- Keep answers compact and readable on mobile.
- Use markdown headings and numbered steps.
- Keep each step short and concrete.
- Avoid huge blocks of text.

ESCALATION TRIGGERS (URGENT)
Immediately include “Do not continue driving” when user mentions:
- brake not responding / severe steering issues
- engine overheating warning + steam/smell
- fuel leak smell
- smoke/fire signs
- battery/electrical burning smell
- sudden loss of power in traffic

OUTPUT QUALITY
- Give practical checks users can do safely.
- Include common causes ranked by likelihood when troubleshooting.
- End with one clear “next best action”.
""".strip()


def _normalize_chat_query(query: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", query.lower()).strip().split())


def _is_identity_or_meta_query(query: str) -> bool:
    """True for questions about the assistant itself — never run manual retrieval on these."""
    q = _normalize_chat_query(query)
    if len(q) > 140:
        return False
    patterns = (
        r"^who\s+are\s+you\b",
        r"^who\s+r\s+u\b",
        r"^what\s+are\s+you\s*(\?|$)",
        r"^what\s*(?:'s|s|is)\s+your\s+name\b",
        r"^what\s+should\s+i\s+call\s+you\b",
        r"^introduce\s+yourself\b",
        r"^tell\s+me\s+about\s+yourself\b",
        r"^what\s+(?:do\s+you\s+do|can\s+you\s+do|can\s+you\s+help(?:\s+with)?|are\s+you\s+for)\b",
        r"^how\s+can\s+you\s+help\b",
        r"^are\s+you\s+(?:a\s+)?(?:bot|ai|assistant|chatgpt)\b",
        r"^what\s+is\s+your\s+purpose\b",
        r"^do\s+you\s+work\s+for\b",
    )
    return any(re.search(p, q) for p in patterns)


class RAGAssistant:
    def __init__(
        self,
        user_id: str = "user1",
        car_context: str = "",
        *,
        use_user_manual: bool = True,
    ) -> None:
        self.user_id = user_id
        self.car_context = car_context.strip()
        self._use_user_manual = use_user_manual
        self._use_pgvector = use_pgvector_for_kb()

        if use_user_manual:
            self.user_model = load_user_meta(user_id)
        else:
            self.user_model = ""

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.remote_llm_url = REMOTE_LLM_URL
        self.remote_llm_secret = REMOTE_LLM_SECRET
        self.llm_provider = LLM_PROVIDER
        self.hf_space_id = HF_SPACE_ID
        self.hf_space_url = HF_SPACE_URL
        self.hf_space_api_name = HF_SPACE_API_NAME
        self.device = get_llm_device()

        if self.llm_provider == "hf_space":
            target = self.hf_space_id or self.hf_space_url or "(missing HF_SPACE_ID/HF_SPACE_URL)"
            self.active_llm_model = f"hf_space:{target}"
            self.llm = None
            self.tokenizer = None
        elif self.remote_llm_url:
            # Defer generation to a remote service (e.g. Colab). This avoids loading local
            # transformer weights on constrained hosts (Render free/CPU instances).
            self.active_llm_model = f"remote:{self.remote_llm_url}"
            self.llm = None
            self.tokenizer = None
        else:
            self.active_llm_model, self.tokenizer, self.llm = _get_shared_local_llm(self.device)

        self.embed_model = _get_shared_sentence_transformer()

        if self._use_pgvector:
            self.global_index = None
            self.global_docs = []
            self.user_index = None
            self.user_docs = []
        else:
            self.global_index, self.global_docs = load_global_index()
            if use_user_manual:
                self.user_index, self.user_docs = load_user_index(user_id)
            else:
                self.user_index, self.user_docs = None, []

    @property
    def has_user_index(self) -> bool:
        if self._use_pgvector:
            with SessionLocal() as db:
                return count_user_chunks(db, self.user_id) > 0
        return self.user_index is not None and bool(self.user_docs)

    def _search(self, index: faiss.Index, docs: List[str], q_emb: np.ndarray, k: int) -> List[str]:
        _, indices = index.search(q_emb, k)
        results: List[str] = []
        for i in indices[0].tolist():
            if 0 <= i < len(docs):
                results.append(docs[i])
        return results

    def _embed_query(self, text: str) -> np.ndarray:
        return self.embed_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)

    def _retrieve_once(
        self,
        q_emb: np.ndarray,
        config: RetrievalConfig,
        *,
        manual_ids: Sequence[str] | None = None,
    ) -> tuple[List[str], List[str]]:
        user_results: List[str] = []
        global_results: List[str] = []
        if self._use_pgvector:
            with SessionLocal() as db:
                if self._use_user_manual:
                    user_results = search_kb_l2(
                        db,
                        scope="user",
                        owner_user_id=self.user_id,
                        query_embedding=q_emb,
                        k=config.k_user,
                    )
                global_results = search_kb_l2(
                    db,
                    scope="global",
                    owner_user_id=None,
                    query_embedding=q_emb,
                    k=config.k_global,
                    manual_ids=manual_ids,
                )
        else:
            if self.user_index is not None and self.user_docs:
                user_results = self._search(self.user_index, self.user_docs, q_emb, config.k_user)
            global_results = self._search(self.global_index, self.global_docs, q_emb, config.k_global)
        return user_results, global_results

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
        car_context: str = "",
        priority_context: str = "",
        manual_ids: Sequence[str] | None = None,
    ) -> List[str]:
        config = config or RetrievalConfig()
        ctx = car_context.strip() if car_context.strip() else self.car_context
        if self.user_model and not ctx:
            ctx = self.user_model

        # Priority order:
        # 1) Health + vehicle context + query (diagnostic first)
        # 2) Vehicle context + query (manual/model-targeted retrieval)
        # 3) Raw query only (general fallback)
        candidate_queries: List[str] = []
        if priority_context.strip():
            candidate_queries.append(f"{priority_context.strip()} {ctx} {query}".strip())
        if ctx:
            candidate_queries.append(f"{ctx} {query}".strip())
        candidate_queries.append(query.strip())

        merged: List[str] = []
        seen_queries: set[str] = set()
        target_limit = config.k_user + config.k_global
        for q in candidate_queries:
            if not q or q in seen_queries:
                continue
            seen_queries.add(q)
            q_emb = self._embed_query(q)
            user_results: List[str] = []
            global_results: List[str] = []
            if manual_ids and self._use_pgvector:
                # Step 1: strict manual-targeted retrieval (exact/near vehicle manual candidates).
                user_results, global_results = self._retrieve_once(
                    q_emb,
                    RetrievalConfig(k_user=config.k_user, k_global=max(1, config.k_global - 1)),
                    manual_ids=manual_ids,
                )
                if len(user_results) + len(global_results) < max(2, target_limit // 2):
                    # Step 2: bounded fallback to full global set when targeted subset is sparse.
                    _, global_fallback = self._retrieve_once(q_emb, RetrievalConfig(k_user=0, k_global=2))
                    for c in global_fallback:
                        if c and c not in global_results:
                            global_results.append(c)
            else:
                user_results, global_results = self._retrieve_once(q_emb, config)
            for c in user_results + global_results:
                if c and c not in merged:
                    merged.append(c)
            if len(merged) >= target_limit:
                break

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

    def _quick_smalltalk(self, query: str) -> str | None:
        q = query.strip().lower()
        q_alpha = re.sub(r"[^a-z\s]", "", q)
        greetings = {"hi", "hello", "hey", "yo", "hii", "helo"}
        thanks = {"thanks", "thank you", "thx", "ty"}
        bye = {"bye", "goodbye", "see you"}

        if q in greetings or q_alpha in greetings:
            return (
                "Hi — I’m **CarCare AI**, your safety-first vehicle assistant.\n\n"
                "## Steps to follow\n"
                "1. Tell me the **symptom** (noise, smell, warning light, leak, vibration).\n"
                "2. Tell me **when it happens** (cold start, braking, turning, highway, bumps).\n\n"
                "## What to check/prepare\n"
                "- Any **dashboard warning lights** (which ones?)\n"
                "- Any **recent work** (battery, brakes, oil change)\n\n"
                "## Next best action\n"
                "Describe what you’re seeing and I’ll guide you step-by-step."
            )
        if q in thanks or q_alpha in thanks:
            return (
                "Glad to help.\n\n"
                "## Next best action\n"
                "If anything else comes up, tell me the **symptom** and **when it happens**."
            )
        if q in bye or q_alpha in bye:
            return (
                "Take care — drive safe.\n\n"
                "## Safety warning\n"
                "If you notice **smoke**, **fuel smell**, **overheating**, or **brake/steering problems**, pull over when safe and get professional help."
            )
        return None

    def _identity_intro_reply(self) -> str:
        return (
            "I’m **CarCare AI** — a practical, safety-first assistant for drivers.\n\n"
            "## What I help with\n"
            "1. Maintenance basics and service intervals\n"
            "2. Safe troubleshooting when something feels wrong\n"
            "3. Understanding dashboard warnings and owner-manual-style guidance (when available)\n\n"
            "## What to tell me\n"
            "- The **symptom** (noise, smell, light, leak, vibration)\n"
            "- **When** it happens\n\n"
            "## Important\n"
            "- I use the vehicle details your app sends; I won’t ask for year/make/model unless it’s missing.\n"
            "- I **don’t** have live sensor/OBD data unless your app sends it in context.\n"
            "- For **brake/steering failures**, **strong burning smells**, **smoke**, **overheating**, or **fuel odor**: "
            "**Do not continue driving** — get professional help.\n\n"
            "## Next best action\n"
            "What’s going on with your car today?"
        )

    def _wants_vehicle_status(self, query: str) -> bool:
        q = " ".join(re.sub(r"[^\w\s]", " ", query.lower()).split())
        phrases = (
            "tell me about my car",
            "about my car",
            "my car status",
            "car status",
            "vehicle status",
            "how is my car",
            "health of my car",
            "overall health",
            "maintenance status",
            "condition of my car",
            "how my car is doing",
            "status of my car",
            "vehicle health",
        )
        return any(p in q for p in phrases)

    def _vehicle_status_missing_snapshot_message(self, vehicle_focus: str) -> str:
        label = vehicle_focus.strip() or "this vehicle"
        return (
            f"I don’t have a maintenance-health snapshot linked for **{label}** in this chat session.\n\n"
            "Make sure you **start the assistant from your vehicle screen** (or send **vehicle_id**) so the "
            "server can load **Vehicle** + **VehicleMaintenanceHealth** from the database.\n\n"
            "After that, ask again for status—I will report overall health and each component percentage."
        )

    def _vehicle_status_reply_from_priority(self, priority_context: str, vehicle_focus: str) -> str:
        lines_raw = [ln.strip() for ln in priority_context.splitlines() if ln.strip()]
        label = vehicle_focus.strip() or "your vehicle"
        for ln in lines_raw:
            if ln.lower().startswith("vehicle:"):
                label = ln.split(":", 1)[1].strip()
                break
        out: List[str] = [
            f"**Vehicle status — {label}**",
            "",
            "Summary from your **saved vehicle profile** (latest maintenance-health snapshot). "
            "This reflects calculated maintenance scores, not live OBD readings.",
            "",
        ]
        for ln in lines_raw:
            out.append(f"• {ln}")
        out.extend(
            [
                "",
                "**Disclaimer:** Values are planning aids only. For safety-critical faults or warning lamps, "
                "follow your owner manual and consult a qualified technician.",
            ]
        )
        return "\n".join(out)

    def _build_prompt(self, context: str, query: str, mode: str) -> str:
        if mode == "technical":
            return f"""
{CARCARE_PERSONA_PROMPT}

ADDITIONAL TECHNICAL MODE RULES
- If the user asks who you are, what you are, or your name: answer as **CarCare AI**; **do not** dump unrelated manual excerpts.
- Use correct automotive terminology and stay factual.
- If a Vehicle profile snapshot lists maintenance-health percentages, report them accurately.
- Do NOT invent DTCs, measurements, or specs.
- If information is missing, state what is missing and ask a brief clarifying question.
- Do not promote the manuals brand or name; focus on the content.

Context:
{context}

Question:
{query}

Answer:
""".strip()

        return f"""
{CARCARE_PERSONA_PROMPT}

ADDITIONAL RESPONSE RULES
- If the user asks who you are, what you are, or your name: answer as **CarCare AI** in plain language; **do not**
  paste unrelated numbered lists from manuals.
- Avoid repetition: do not repeat the same sentence/bullet more than once.
- Answer using the context below.
- When context includes a Vehicle profile snapshot, treat those vehicle facts and percentages as authoritative.
- Use manual excerpts as supporting detail for procedures and warnings.
- If manual excerpts are missing, still answer from available vehicle/profile context.
- Do NOT repeat the user question verbatim.
- Do NOT invent sensors, DTC codes, or measurements not present in context.
- Do not promote the manuals brand or name; focus on the content.

Context:
{context}

Question:
{query}

Answer:
""".strip()

    def _format_recent_messages(self, recent_messages: Sequence[Dict[str, str]] | None) -> str:
        if not recent_messages:
            return ""
        lines: List[str] = []
        for m in list(recent_messages)[-6:]:
            role = (m.get("role") or "").strip().lower()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role not in {"user", "assistant", "system"}:
                role = "user"
            clipped = content.replace("\n", " ").strip()
            clipped = re.sub(r"\s+", " ", clipped)
            if len(clipped) > 320:
                clipped = clipped[:317].rstrip() + "..."
            lines.append(f"{role}: {clipped}")
        return "\n".join(lines).strip()

    def _compose_conversation_prefix(
        self,
        *,
        chat_summary: str = "",
        recent_messages: Sequence[Dict[str, str]] | None = None,
    ) -> str:
        summary = (chat_summary or "").strip()
        recent = self._format_recent_messages(recent_messages)
        blocks: List[str] = []
        if summary:
            blocks.append(f"(Conversation summary so far)\n{summary[:1400]}")
        if recent:
            blocks.append(f"(Most recent messages)\n{recent}")
        return "\n\n".join(blocks).strip()

    def _heuristic_chat_summary(self, prev: str, user_text: str, assistant_text: str) -> str:
        parts: List[str] = []
        if prev:
            parts.append(prev)
        u = re.sub(r"\s+", " ", user_text).strip()
        a = re.sub(r"\s+", " ", assistant_text).strip()
        if u:
            parts.append(f"User: {u[:180]}")
        if a:
            parts.append(f"Assistant: {a[:220]}")
        updated = "\n".join(parts)
        if len(updated) > 1200:
            updated = updated[-1200:]
        return updated.strip()

    def update_chat_summary(self, prev_summary: str, user_text: str, assistant_text: str) -> str:
        """
        Rolling summary for long chats.

        Remote providers still use the LLM. A local model does not: a second
        generation was running before the HTTP response and added minutes.
        """
        prev = (prev_summary or "").strip()
        u = (user_text or "").strip()
        a = (assistant_text or "").strip()
        if not u and not a:
            return prev[:1400]
        if self.llm is not None:
            return self._heuristic_chat_summary(prev, u, a)

        prompt = f"""
You maintain a rolling conversation summary for a car-care assistant.

Update the summary using the latest turn. Keep it short, factual, and useful for future replies.
Preserve: vehicle identity (if mentioned), symptoms, warning lights, what was tried, conclusions, and next steps.
Do NOT repeat yourself. Do NOT include long lists.

Return plain text only (no markdown headings).
Target length: 6-12 lines, <= 1200 characters.

Previous summary:
{prev if prev else "(empty)"}

Latest turn:
user: {u[:800]}
assistant: {a[:1000]}

Updated summary:
""".strip()

        try:
            if self.llm_provider == "hf_space":
                updated = self._hf_space_generate(prompt)
            elif self.remote_llm_url:
                updated = self._remote_generate(prompt)
            else:
                assert self.llm is not None
                assert self.tokenizer is not None
                inputs, input_len = self._tokenize_for_generation(prompt)
                with torch.no_grad():
                    outputs = self.llm.generate(
                        **inputs,
                        max_new_tokens=160,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "Updated summary:" in full_output:
                    updated = full_output.split("Updated summary:")[-1].strip()
                else:
                    updated = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        except Exception:
            updated = self._heuristic_chat_summary(prev, u, a)

        updated = re.sub(r"\s+\n", "\n", (updated or "").strip())
        updated = "\n".join([ln.strip() for ln in updated.splitlines() if ln.strip()])
        if len(updated) > 1200:
            updated = updated[-1200:]
        return updated.strip()

    def _is_repetitive_output(self, text: str) -> bool:
        lines = [re.sub(r"\s+", " ", ln.strip().lower()) for ln in text.splitlines() if ln.strip()]
        if len(lines) < 6:
            return False
        counts: dict[str, int] = {}
        for ln in lines:
            counts[ln] = counts.get(ln, 0) + 1
        top = max(counts.values(), default=1)
        return top >= 4 or (top / max(1, len(lines))) >= 0.45

    def _dedupe_bullets_and_lines(self, text: str) -> str:
        out_lines: List[str] = []
        seen: set[str] = set()
        for ln in text.splitlines():
            raw = ln.rstrip()
            if not raw.strip():
                # Keep at most one consecutive blank line
                if out_lines and out_lines[-1].strip():
                    out_lines.append("")
                continue
            key = re.sub(r"^\s*[-•\d]+\s*[\.\)]?\s*", "", raw).strip().lower()
            key = re.sub(r"\s+", " ", key)
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            out_lines.append(raw)
            if len(out_lines) >= 60:
                break
        return "\n".join(out_lines).strip()

    def _burning_clarify_reply(self) -> str:
        return (
            "If you mean a **burning smell**, treat this as potentially urgent.\n\n"
            "## Steps to follow\n"
            "1. If you smell **burning**, see **smoke**, or any warning light: **pull over safely** and turn the engine off.\n"
            "2. Keep the hood closed for a few minutes; then check **under the car** for leaks and look for smoke.\n"
            "3. Don’t keep trying to drive it.\n\n"
            "## What to check/prepare\n"
            "- Is it a **burning smell** or the engine **overheating**?\n"
            "- Does the engine **crank** (tries to start) or is it completely dead?\n"
            "- Any **smoke** or **fluid leak**?\n\n"
            "## When to seek a mechanic\n"
            "- Immediately if there’s **smoke**, **fuel smell**, or overheating.\n\n"
            "## Safety warning\n"
            "**Do not continue driving** if there’s burning smell/smoke/overheating.\n\n"
            "## Next best action\n"
            "Reply with: **burning smell vs overheating**, and whether it **cranks** or **starts then stalls**."
        )

    def _format_answer(self, text: str) -> str:
        raw = text.strip()
        if not raw:
            return "I am not certain based on the retrieved context."
        raw = self._dedupe_bullets_and_lines(raw)
        if self._is_repetitive_output(raw):
            # Fallback: model is looping; ask a single clarifying question with safety-first guidance.
            return self._burning_clarify_reply()
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        cleaned = "\n".join(lines).strip()
        if not cleaned:
            return "I am not certain based on the retrieved context."
        capitalized = False
        formatted_lines: List[str] = []
        for line in lines:
            if line.strip() and not capitalized and line[0].isalpha():
                line = line[0].upper() + line[1:]
                capitalized = True
            formatted_lines.append(line)
        out = "\n".join(formatted_lines).strip()
        if "\n" not in out and not out.endswith(("?", "!", ".")):
            out += "."
        return out

    def _tokenize_for_generation(self, prompt: str) -> tuple[dict[str, torch.Tensor], int]:
        if self.remote_llm_url or self.llm_provider == "hf_space":
            raise RuntimeError("Tokenization is not used when a remote LLM provider is configured.")
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                tokenized = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                if isinstance(tokenized, dict) and "input_ids" in tokenized:
                    tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
                    return tokenized, tokenized["input_ids"].shape[1]
            except Exception:
                pass

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs, inputs["input_ids"].shape[1]

    def _remote_generate(self, prompt: str) -> str:
        if not self.remote_llm_url:
            raise RuntimeError("Remote LLM URL not configured.")
        headers = {"Content-Type": "application/json"}
        if self.remote_llm_secret:
            headers["X-LLM-Secret"] = self.remote_llm_secret
        payload = {
            "prompt": prompt,
            "max_new_tokens": 120,
        }
        try:
            resp = requests.post(self.remote_llm_url, json=payload, headers=headers, timeout=90)
        except requests.RequestException as e:
            raise RuntimeError(f"Remote LLM request failed: {e}") from e
        if resp.status_code != 200:
            body = resp.text[:500]
            raise RuntimeError(f"Remote LLM error {resp.status_code}: {body}")
        data = resp.json()
        answer = (data.get("answer") or "").strip()
        return answer

    def _hf_space_generate(self, prompt: str) -> str:
        target = self.hf_space_id or self.hf_space_url
        if not target:
            raise RuntimeError("LLM_PROVIDER=hf_space requires HF_SPACE_ID or HF_SPACE_URL.")
        client = _get_hf_space_client(target)
        try:
            result = client.predict(prompt=prompt, api_name=self.hf_space_api_name)
        except Exception as e:  # pragma: no cover - network/runtime dependent
            raise RuntimeError(f"HF Space generation failed: {e}") from e

        if isinstance(result, str):
            text = result.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return str(parsed.get("answer") or text).strip()
                except json.JSONDecodeError:
                    pass
            return text
        if isinstance(result, dict):
            return str(result.get("answer") or "").strip()
        return str(result).strip()

    def generate_answer(
        self,
        query: str,
        car_context: str = "",
        priority_context: str = "",
        manual_ids: Sequence[str] | None = None,
        chat_summary: str = "",
        recent_messages: Sequence[Dict[str, str]] | None = None,
    ) -> str:
        quick = self._quick_smalltalk(query)
        if quick is not None:
            return quick

        if _is_identity_or_meta_query(query):
            return self._identity_intro_reply()

        active_context = car_context.strip() if car_context.strip() else self.car_context
        if self.user_model and self.has_user_index:
            active_context = f"{active_context} {self.user_model}".strip()

        snap = priority_context.strip()
        wants_status = self._wants_vehicle_status(query)

        if wants_status:
            if snap:
                context_chunks = self.retrieve(
                    query,
                    config=RetrievalConfig(k_user=2, k_global=3),
                    car_context=car_context,
                    priority_context=priority_context,
                    manual_ids=manual_ids,
                )
                reply = self._vehicle_status_reply_from_priority(snap, active_context)
                if context_chunks:
                    reply += "\n\n—\n**Owner manual excerpts**\n"
                    for i, ch in enumerate(context_chunks[:3], 1):
                        excerpt = " ".join(ch.split())[:480]
                        reply += f"\n{i}. {excerpt}"
                return self._format_answer(reply)
            return self._format_answer(self._vehicle_status_missing_snapshot_message(active_context))

        context_chunks = self.retrieve(
            query,
            config=RetrievalConfig(k_user=2, k_global=3),
            car_context=car_context,
            priority_context=priority_context,
            manual_ids=manual_ids,
        )
        manual_block = (
            "No matching owner-manual excerpts were retrieved."
            if not context_chunks
            else "\n".join(context_chunks)
        )
        context = manual_block[:2500]
        convo_prefix = self._compose_conversation_prefix(
            chat_summary=chat_summary,
            recent_messages=recent_messages,
        )
        if convo_prefix:
            context = f"{convo_prefix}\n\n{context}".strip()
        if snap:
            context = (
                f"(Vehicle profile snapshot — factual vehicle state)\n{snap[:2400]}\n\n"
                f"(Owner manual excerpts)\n{context}"
            )
        if active_context:
            context = f"(Vehicle focus: {active_context[:300]})\n\n{context}"
        mode = self._answer_mode(query)
        prompt = self._build_prompt(context=context, query=query, mode=mode)

        if self.llm_provider == "hf_space":
            answer = self._hf_space_generate(prompt)
        elif self.remote_llm_url:
            answer = self._remote_generate(prompt)
        else:
            assert self.llm is not None
            assert self.tokenizer is not None
            inputs, input_len = self._tokenize_for_generation(prompt)
            max_tokens = 140 if snap else 96

            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Answer:" in full_output:
                answer = full_output.split("Answer:")[-1].strip()
            elif "Final Answer:" in full_output:
                answer = full_output.split("Final Answer:")[-1].strip()
            else:
                answer = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        answer = answer.split("Question:")[0].split("Context:")[0].strip()
        answer_lower = answer.lower()
        if (not answer or "no relevant context" in answer_lower) and snap:
            blended = self._vehicle_status_reply_from_priority(snap, active_context)
            if context_chunks:
                blended += "\n\n—\n**Owner manual excerpts**\n"
                for i, ch in enumerate(context_chunks[:2], 1):
                    blended += f"\n{i}. {' '.join(ch.split())[:400]}"
            return self._format_answer(blended)
        return self._format_answer(answer)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG chat: pgvector (Postgres) or local FAISS — see RAG_KB_BACKEND."
    )
    parser.add_argument(
        "--car",
        default="",
        help="Optional car context (model, year, trim). Also set RAG_CAR_CONTEXT env var.",
    )
    parser.add_argument(
        "--user-id",
        default="user1",
        help="User id for user-scoped manual chunks (Postgres rag_kb_chunks or models/<id>/).",
    )
    args = parser.parse_args()

    car_context = (args.car or os.environ.get("RAG_CAR_CONTEXT", "") or "").strip()
    assistant = RAGAssistant(
        user_id=args.user_id,
        car_context=car_context,
        use_user_manual=True,
    )

    print(f"Using device: {assistant.device}")
    print(f"LLM model requested: {LLM_MODEL}")
    print(f"LLM fallback model: {FALLBACK_LLM_MODEL}")
    print(f"LLM model loaded: {assistant.active_llm_model}")
    if assistant._use_pgvector:
        print("Knowledge retrieval: PostgreSQL + pgvector (rag_kb_chunks).")
    else:
        print("Knowledge retrieval: local FAISS + models/docs.txt.")
    if assistant.has_user_index:
        model_note = f" ({assistant.user_model})" if assistant.user_model else ""
        if assistant._use_pgvector:
            print(f"User manual chunks in DB for user_id='{args.user_id}'{model_note}")
        else:
            print(f"Loaded user manual index: models/{args.user_id}/faiss_index{model_note}")
    else:
        print(f"No user manual chunks for '{args.user_id}', using global KB only.")
    if car_context:
        print(f"Car context: {car_context}")

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
