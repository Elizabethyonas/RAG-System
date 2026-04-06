from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

DEFAULT_OUT = Path("data/processed/unified_docs.txt")
RAW = Path("data/raw")

CARDEKHO_CANDIDATES = (
    RAW / "CAR DETAILS FROM CAR DEKHO.csv",
    RAW / "car details v4.csv",
    RAW / "cardekho.csv",
    RAW / "car data.csv",
)
SPECS_CANDIDATES = (RAW / "Car Dataset 1945-2020.csv", RAW / "specs.csv")
PAKWHEELS_CSV = RAW / "PakWheelsDataSet.csv"


def _existing_in_order(paths: tuple[Path, ...]) -> list[Path]:
    return [p for p in paths if p.is_file()]


def _first_existing(paths: tuple[Path, ...]) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def _unique_existing_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        if not p.is_file():
            continue
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def first_present(row: pd.Series, candidates: tuple[str, ...]):
    cols = {str(c).strip().lower(): c for c in row.index}
    for cand in candidates:
        key = cand.lower()
        if key in cols:
            val = row[cols[key]]
            if pd.notna(val):
                return val
    return None


def chunk_words(text: str, size: int) -> list[str]:
    words = (text or "").split()
    if not words:
        return []
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def _norm_cols(df: pd.DataFrame) -> set[str]:
    return {str(c).strip().lower() for c in df.columns}


def detect_auto_file_kind(df: pd.DataFrame) -> str:
    """Kaggle `auto.csv` is often a used-car listing; UCI Auto has mpg / cylinders."""
    nc = _norm_cols(df)
    if "mpg" in nc:
        return "uci_auto"
    if "cylinders" in nc and "displacement" in nc and "acceleration" in nc:
        return "uci_auto"
    if "kms_driven" in nc or "km_driven" in nc or "present_price" in nc:
        return "cardekho"
    return "cardekho"


def _listing_vehicle_label(row: pd.Series) -> str | None:
    """Full car line: single name column, or Make+Model (e.g. v4), or Make+Name (e.g. PakWheels)."""
    make = first_present(row, ("make", "brand", "manufacturer"))
    single = first_present(row, ("name", "car_name", "car name", "carname"))
    model = first_present(row, ("model", "modle"))
    if single is not None and make is not None:
        sm = str(single).strip()
        mk = str(make).strip()
        if sm.lower().startswith(mk.lower()):
            return sm
        return f"{mk} {sm}".strip()
    if single is not None:
        return str(single).strip()
    if make is not None and model is not None:
        return f"{str(make).strip()} {str(model).strip()}"
    if make is not None:
        return str(make).strip()
    if model is not None:
        return str(model).strip()
    return None


def process_cardekho(df: pd.DataFrame) -> list[str]:
    docs: list[str] = []
    for _, row in df.iterrows():
        name = _listing_vehicle_label(row)
        if name is None:
            continue
        year = first_present(row, ("year",))
        fuel = first_present(
            row,
            ("fuel", "fuel_type", "fuel type", "engine type", "engine_type"),
        )
        transmission = first_present(row, ("transmission",))
        km = first_present(
            row,
            (
                "km_driven",
                "km driven",
                "kms_driven",
                "kms driven",
                "kilometer",
                "kilometre",
                "mileage(kms)",
                "mileage (kms)",
                "mileage",
            ),
        )
        price = first_present(
            row,
            (
                "selling_price",
                "selling price",
                "present_price",
                "present price",
                "price",
            ),
        )
        owner = first_present(row, ("owner",))
        seller = first_present(row, ("seller_type", "seller type"))
        city = first_present(row, ("city", "location"))
        engine_cc = first_present(
            row,
            ("engine capacity(cc)", "engine capacity (cc)", "engine capacity", "engine_cc", "engine cc"),
        )

        head = f"The car {name}"
        if year is not None:
            head += f" from year {year}"

        clauses: list[str] = []
        if fuel is not None:
            clauses.append(f"has {fuel} fuel type")
        if transmission is not None:
            clauses.append(f"uses {transmission} transmission")
        if km is not None:
            clauses.append(f"has driven {km} km")
        if price is not None:
            clauses.append(f"is listed around {price}")
        if seller is not None:
            clauses.append(f"seller type is {seller}")
        if owner is not None:
            clauses.append(f"seller/owner category is {owner}")
        if city is not None:
            clauses.append(f"located in {city}")
        if engine_cc is not None:
            clauses.append(f"engine capacity around {engine_cc} cc")

        if clauses:
            if len(clauses) == 1:
                text = f"{head} {clauses[0]}."
            else:
                text = f"{head} {', '.join(clauses[:-1])}, and {clauses[-1]}."
        else:
            text = f"{head}."
        docs.append(text)
    return docs


def process_specs(df: pd.DataFrame) -> list[str]:
    docs: list[str] = []
    for _, row in df.iterrows():
        make = first_present(row, ("make", "manufacturer", "brand"))
        model = first_present(row, ("model", "modle", "Model", "Modle"))
        y_from = first_present(row, ("year_from", "year from"))
        y_to = first_present(row, ("year_to", "year to"))
        year_single = first_present(row, ("year", "model_year", "model year", "Year"))
        if y_from is not None and y_to is not None and str(y_from) != str(y_to):
            year_disp = f"{y_from}–{y_to}"
        elif y_from is not None:
            year_disp = y_from
        elif y_to is not None:
            year_disp = y_to
        else:
            year_disp = year_single

        engine_l = first_present(
            row,
            ("engine_size", "engine size", "displacement", "Engine"),
        )
        engine_cc = first_present(row, ("capacity_cm3", "capacity cm3", "capacity_cc"))
        hp = first_present(
            row,
            ("horsepower", "horse_power", "Horsepower", "hp", "engine_hp", "engine hp"),
        )
        torque = first_present(
            row,
            ("torque", "Torque", "maximum_torque_n_m", "maximum torque n m"),
        )
        drivetrain = first_present(
            row,
            ("drivetrain", "drive_type", "drive type", "Drivetrain", "drive_wheels", "drive wheels"),
        )
        fuel = first_present(row, ("fuel_type", "fuel type", "fuel", "Fuel", "fuel_grade", "fuel grade"))
        trans = first_present(row, ("transmission", "Transmission"))
        body = first_present(row, ("body_type", "body type", "body style", "body", "Body"))

        opener: list[str] = []
        if make is not None or model is not None:
            who = " ".join(str(x) for x in (make, model) if x is not None).strip()
            seg = f"The {who}" if who else "This vehicle"
            if year_disp is not None:
                seg += f" ({year_disp})"
            opener.append(seg)

        detail: list[str] = []
        if engine_l is not None:
            s = str(engine_l).strip()
            detail.append(f"a {s}L engine" if s.replace(".", "", 1).isdigit() else f"a {s} engine")
        elif engine_cc is not None:
            detail.append(f"a {engine_cc} cc engine displacement")
        if hp is not None:
            detail.append(f"{hp} horsepower")
        if torque is not None:
            detail.append(f"peak torque around {torque} N·m")
        if fuel is not None:
            detail.append(f"{fuel} fuel")
        if trans is not None:
            detail.append(f"{trans} transmission")
        if drivetrain is not None:
            detail.append(f"{drivetrain} drivetrain")
        if body is not None:
            detail.append(f"{body} body style")

        spec_cols_used = frozenset(
            {
                "make",
                "manufacturer",
                "brand",
                "model",
                "modle",
                "year",
                "year_from",
                "year from",
                "year_to",
                "year to",
                "model_year",
                "model year",
                "engine_size",
                "engine size",
                "engine",
                "displacement",
                "capacity_cm3",
                "capacity cm3",
                "capacity_cc",
                "horsepower",
                "horse_power",
                "hp",
                "engine_hp",
                "engine hp",
                "torque",
                "maximum_torque_n_m",
                "maximum torque n m",
                "drivetrain",
                "drive_type",
                "drive type",
                "drive_wheels",
                "drive wheels",
                "fuel_type",
                "fuel type",
                "fuel",
                "fuel_grade",
                "fuel grade",
                "transmission",
                "body_type",
                "body type",
                "body style",
                "body",
            }
        )
        extras: list[str] = []
        cols_lower = {str(c).strip().lower(): c for c in row.index}
        for lk, orig in cols_lower.items():
            if lk in spec_cols_used:
                continue
            val = row[orig]
            if pd.isna(val) or str(val).strip() == "":
                continue
            label = str(orig).replace("_", " ")
            extras.append(f"{label}: {val}")

        if opener and detail:
            text = f"{opener[0]} has " + ", ".join(detail) + "."
        elif opener:
            text = opener[0] + "."
        elif detail:
            text = "This vehicle has " + ", ".join(detail) + "."
        else:
            text = ""

        if extras:
            text = (text + " " if text else "") + "Other specifications: " + "; ".join(extras) + "."

        text = text.strip()
        if text:
            docs.append(text)
    return docs


def process_automobile(df: pd.DataFrame) -> list[str]:
    docs: list[str] = []
    for _, row in df.iterrows():
        name = first_present(row, ("name", "car name", "car_name"))
        mpg = first_present(row, ("mpg",))
        cyl = first_present(row, ("cylinders",))
        disp = first_present(row, ("displacement",))
        hp = first_present(row, ("horsepower",))
        weight = first_present(row, ("weight",))
        accel = first_present(row, ("acceleration",))
        my = first_present(row, ("model year", "model_year", "year"))

        if name is None and mpg is None:
            continue

        parts = []
        if name is not None:
            opener = f"The car {name}"
            if my is not None:
                opener += f" (model year {my})"
            parts.append(opener)
        else:
            parts.append("This vehicle")

        stats = []
        if mpg is not None:
            stats.append(f"fuel efficiency of {mpg} mpg")
        if hp is not None:
            stats.append(f"{hp} horsepower")
        if cyl is not None:
            stats.append(f"{cyl} cylinders")
        if disp is not None:
            stats.append(f"{disp} displacement")
        if weight is not None:
            stats.append(f"weight {weight}")
        if accel is not None:
            stats.append(f"0–60 acceleration {accel}")

        if stats:
            text = parts[0] + " has " + ", ".join(stats) + "."
        else:
            text = parts[0] + "."
        docs.append(text)
    return docs


def process_delucionqa(split: str | None) -> list[str]:
    from datasets import load_dataset

    ds_all = load_dataset("corvicai/delucionqa")
    splits = [split] if split and split in ds_all else list(ds_all.keys())

    docs: list[str] = []
    for sp in splits:
        for item in ds_all[sp]:
            if "question" in item and "answer" in item:
                a = item["answer"]
                ctx = (item.get("context") or "").strip()
            elif "query" in item:
                a = item.get("ground_truth") or item.get("answer") or ""
                ctx = (item.get("context") or "").strip()
            else:
                continue
            a = str(a).strip()
            if not a:
                continue
            if ctx:
                text = f"{ctx}\nAnswer: {a}"
            else:
                text = f"Answer: {a}"
            docs.append(text.strip())
    return docs


def collect_csv(path: Path | None, label: str) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.is_file():
        print(f"[skip] {label}: missing file {path}")
        return None
    return pd.read_csv(path, low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge car CSVs + DelucionQA into chunked lines in unified_docs.txt for RAG indexing."
    )
    parser.add_argument(
        "--cardekho",
        type=Path,
        default=None,
        help=f"Use only this listing CSV (overrides default multi-file load under {RAW})",
    )
    parser.add_argument(
        "--specs",
        type=Path,
        default=None,
        help="Use only this spec CSV (overrides loading all of Car Dataset + specs.csv)",
    )
    parser.add_argument(
        "--auto",
        type=Path,
        default=None,
        help="Third table: UCI Auto (mpg) or another CarDekho-style CSV (default: data/raw/auto.csv)",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help=(
            "Load only the first CarDekho file and first spec file found; skip PakWheels. "
            "Default is to merge every listing/spec file that exists (incl. v4, specs.csv, PakWheels)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output text (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=200,
        help="Max words per line (one embedding unit per line)",
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Do not download corvicai/delucionqa (CSV sources only)",
    )
    parser.add_argument(
        "--qa-split",
        type=str,
        default=None,
        help="Split name for DelucionQA (default: all splits in the dataset card)",
    )
    args = parser.parse_args()

    cardekho_paths: list[Path] = []
    if args.cardekho is not None:
        cardekho_paths.append(args.cardekho)
    elif args.primary_only:
        fe = _first_existing(CARDEKHO_CANDIDATES)
        if fe is not None:
            cardekho_paths.append(fe)
    else:
        cardekho_paths.extend(_existing_in_order(CARDEKHO_CANDIDATES))
        if PAKWHEELS_CSV.is_file():
            cardekho_paths.append(PAKWHEELS_CSV)
    cardekho_paths = _unique_existing_paths(cardekho_paths)

    specs_paths: list[Path] = []
    if args.specs is not None:
        specs_paths.append(args.specs)
    elif args.primary_only:
        fs = _first_existing(SPECS_CANDIDATES)
        if fs is not None:
            specs_paths.append(fs)
    else:
        specs_paths.extend(_existing_in_order(SPECS_CANDIDATES))
    specs_paths = _unique_existing_paths(specs_paths)

    auto_path = args.auto if args.auto is not None else RAW / "auto.csv"

    if not cardekho_paths:
        print(
            "[skip] cardekho: no listing CSV found ("
            + ", ".join(p.name for p in CARDEKHO_CANDIDATES)
            + ")"
        )
    if not specs_paths and args.specs is None:
        print("[skip] specs: no Car Dataset 1945-2020.csv or specs.csv")
    if cardekho_paths:
        print("[load] listings: " + ", ".join(p.name for p in cardekho_paths))
    if specs_paths:
        print("[load] specs: " + ", ".join(p.name for p in specs_paths))

    all_docs: list[str] = []

    for path in cardekho_paths:
        df_cd = collect_csv(path, "cardekho")
        if df_cd is not None:
            all_docs.extend(process_cardekho(df_cd))

    for path in specs_paths:
        df_sp = collect_csv(path, "specs")
        if df_sp is not None:
            all_docs.extend(process_specs(df_sp))

    df_au = collect_csv(auto_path, "auto / UCI")
    if df_au is not None:
        if detect_auto_file_kind(df_au) == "uci_auto":
            all_docs.extend(process_automobile(df_au))
        else:
            all_docs.extend(process_cardekho(df_au))

    if not args.skip_qa:
        try:
            all_docs.extend(process_delucionqa(args.qa_split))
        except Exception as e:
            warnings.warn(f"Could not load DelucionQA: {e}", stacklevel=1)

    chunked: list[str] = []
    for doc in all_docs:
        for piece in chunk_words(doc, args.chunk_words):
            if piece:
                chunked.append(piece)

    if not chunked:
        raise SystemExit(
            "No documents produced. Place CSVs under data/raw/ (see defaults) "
            "and ensure network access for DelucionQA, or pass --skip-qa only if CSVs exist."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for line in chunked:
            f.write(line + "\n")

    print(f"Total raw docs: {len(all_docs)}")
    print(f"Total chunked lines: {len(chunked)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
