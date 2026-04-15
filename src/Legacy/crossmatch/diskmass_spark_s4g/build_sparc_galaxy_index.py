from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

def normalize_basic(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())

def catalog_prefix(name: str) -> str:
    upper = name.upper()
    for prefix in ("NGC", "UGC", "PGC", "IC", "DDO", "ESO", "F", "UGCA", "KK", "CAMB"):
        if upper.startswith(prefix):
            return prefix
    return "OTHER"

def extract_catalog_number(name: str) -> str:
    m = re.search(r"(\d[\dA-Z\-]*)", name.upper())
    return m.group(1) if m else ""

def iter_sparc_files(input_dir: Path) -> Iterable[Path]:
    for file_path in sorted(input_dir.glob("*.csv")):
        if file_path.is_file() and file_path.name.lower() != "normalization_summary.csv":
            yield file_path

def count_data_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)

def build_index(input_dir: Path, output_file: Path) -> Path:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "galaxy_id", "filename", "filepath", "catalog_prefix",
        "catalog_number", "name_normalized", "row_count", "source_expected"
    ]
    rows = []
    for file_path in iter_sparc_files(input_dir):
        galaxy_id = file_path.stem
        rows.append({
            "galaxy_id": galaxy_id,
            "filename": file_path.name,
            "filepath": str(file_path).replace("\\", "/"),
            "catalog_prefix": catalog_prefix(galaxy_id),
            "catalog_number": extract_catalog_number(galaxy_id),
            "name_normalized": normalize_basic(galaxy_id),
            "row_count": count_data_rows(file_path),
            "source_expected": "SPARC",
        })
    if not rows:
        raise ValueError(f"No SPARC galaxy CSV files found in: {input_dir}")
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_file

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SPARC galaxy index for crossmatch work.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/derived/sparc"))
    parser.add_argument("--output-file", type=Path, default=Path("data/derived/crossmatch/sparc_galaxy_index.csv"))
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    output_path = build_index(args.input_dir, args.output_file)
    print(f"[OK] SPARC galaxy index written: {output_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
