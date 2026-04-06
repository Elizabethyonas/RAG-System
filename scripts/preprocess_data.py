from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("data/raw/car data.csv")
DEFAULT_OUTPUT = Path("data/processed/car_docs.txt")


def row_to_line(row: pd.Series) -> str:
    """One retrieval unit per row; single line (build_index reads line-by-line)."""
    parts: list[str] = []
    for col in row.index:
        val = row[col]
        if pd.isna(val):
            continue
        s = str(val).strip()
        if s:
            parts.append(f"{col}: {s}")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turn a car CSV (e.g. Kaggle CarDekho car data.csv) into car_docs.txt."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"CSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output text path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Missing CSV: {args.input.resolve()}\n"
            "Download a Kaggle used-car dataset (e.g. search “CarDekho car data”) "
            f"and save it as {args.input}, or pass --input /path/to/your.csv"
        )

    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError(f"No rows in {args.input}")

    lines = [row_to_line(row) for _, row in df.iterrows()]
    lines = [ln for ln in lines if ln]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

    print(f"Wrote {len(lines)} documents to {args.output}")


if __name__ == "__main__":
    main()
