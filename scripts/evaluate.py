from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.chat import RAGAssistant

EMBED_MODEL = "all-MiniLM-L6-v2"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def normalize_qa_item(item: dict[str, Any]) -> tuple[str, str]:
    if "question" in item and "answer" in item:
        return str(item["question"]).strip(), str(item["answer"]).strip()
    if "query" in item:
        gt = item.get("ground_truth") or item.get("answer") or ""
        return str(item["query"]).strip(), str(gt).strip()
    return "", ""


def evaluate(
    sample_size: int,
    split: str | None,
    user_id: str,
    car_context: str,
    low_threshold: float,
    include_user_manual: bool,
) -> dict[str, Any]:
    ds_all = load_dataset("corvicai/delucionqa")
    if split and split not in ds_all:
        raise ValueError(f"Split '{split}' not found. Available: {list(ds_all.keys())}")
    split_name = split if split else list(ds_all.keys())[0]
    ds = ds_all[split_name]

    n = min(sample_size, len(ds))
    print(f"[eval] split={split_name}, samples={n}", flush=True)
    print(
        f"[eval] user_manual={'on' if include_user_manual else 'off'} (user_id={user_id})",
        flush=True,
    )
    print("[eval] Loading RAG assistant (LLM + retriever)...", flush=True)
    assistant = RAGAssistant(
        user_id=user_id,
        car_context=car_context,
        use_user_manual=include_user_manual,
    )
    print("[eval] Loading embedding model for scoring...", flush=True)
    embed_model = SentenceTransformer(EMBED_MODEL)

    rows: list[dict[str, Any]] = []
    scores: list[float] = []

    for i in range(n):
        item = ds[i]
        question, ground_truth = normalize_qa_item(item)
        if not question or not ground_truth:
            continue

        prediction = assistant.generate_answer(question, car_context=car_context)
        pred_emb = embed_model.encode(prediction, convert_to_tensor=True)
        gt_emb = embed_model.encode(ground_truth, convert_to_tensor=True)
        score = float(util.cos_sim(pred_emb, gt_emb).item())

        scores.append(score)
        rows.append(
            {
                "idx": i,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "semantic_similarity": score,
            }
        )

        if (i + 1) % 5 == 0 or (i + 1) == n:
            running_mean = mean(scores)
            print(
                f"[eval] {i + 1}/{n} done | running mean similarity={running_mean:.3f}",
                flush=True,
            )

    if not scores:
        raise RuntimeError("No valid QA pairs were evaluated.")

    low_cases = sorted(
        [r for r in rows if r["semantic_similarity"] < low_threshold],
        key=lambda x: x["semantic_similarity"],
    )

    return {
        "dataset": "corvicai/delucionqa",
        "split": split_name,
        "samples_requested": sample_size,
        "samples_evaluated": len(scores),
        "include_user_manual": include_user_manual,
        "user_id": user_id,
        "mean_similarity": mean(scores),
        "median_similarity": median(scores),
        "p10_similarity": percentile(scores, 0.10),
        "p90_similarity": percentile(scores, 0.90),
        "low_similarity_threshold": low_threshold,
        "low_similarity_count": len(low_cases),
        "low_similarity_examples": low_cases[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG answer quality on DelucionQA with semantic similarity."
    )
    parser.add_argument("--samples", type=int, default=50, help="Number of examples to evaluate")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split (default: first available split, usually 'test')",
    )
    parser.add_argument("--user-id", type=str, default="user1", help="User id for optional manual")
    parser.add_argument("--car", type=str, default="", help="Optional car context")
    parser.add_argument(
        "--include-user-manual",
        action="store_true",
        help="Include models/<user-id>/ manual index during eval (default: global KB only)",
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.60,
        help="Threshold below which examples are listed as low quality",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/eval_report.json"),
        help="Where to save JSON report",
    )
    args = parser.parse_args()

    report = evaluate(
        sample_size=args.samples,
        split=args.split,
        user_id=args.user_id,
        car_context=args.car,
        low_threshold=args.low_threshold,
        include_user_manual=args.include_user_manual,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Evaluation complete")
    print(f"- Dataset split      : {report['split']}")
    print(f"- Samples evaluated  : {report['samples_evaluated']}")
    print(f"- Mean similarity    : {report['mean_similarity']:.3f}")
    print(f"- Median similarity  : {report['median_similarity']:.3f}")
    print(f"- P10 / P90          : {report['p10_similarity']:.3f} / {report['p90_similarity']:.3f}")
    print(f"- Low-score examples : {report['low_similarity_count']} (< {report['low_similarity_threshold']})")
    print(f"- Report saved       : {args.out}")


if __name__ == "__main__":
    main()
