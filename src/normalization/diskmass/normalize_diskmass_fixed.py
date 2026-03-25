from __future__ import annotations

# -----
# normalize_diskmass_fixed.py
#
# Normalize DiskMass Vizier XML+CSV exports using file-specific fixed rules.
#
# Why fixed rules:
# - The DiskMass files in this project do not share the exact same XML wrapper layout.
# - Generic header detection was not reliable enough.
# - We already identified approximate true header/data regions for the current files.
#
# This script:
# 1) Opens each raw DiskMass file
# 2) Keeps only lines between the real CSV header and the real data block
# 3) Skips XML/footer/separator/unit rows
# 4) Writes clean comma-separated CSV files
#
# Recommended usage:
#   python src/normalization/normalize_diskmass_fixed.py ^
#       --input-dir data/raw/diskmass ^
#       --output-dir data/derived/diskmass ^
#       --overwrite
#
# -----

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple


# -----
# Section: fixed per-file rules
# -----

# Rules are based on the currently downloaded DiskMass raw files.
# We use filename keywords rather than exact full filenames to remain robust to small renames.
FILE_RULES: Dict[str, Dict[str, int]] = {
    "sample": {
        "header_line_1based": 158,
        "data_line_1based": 161,
    },
    "table7": {
        "header_line_1based": 110,
        "data_line_1based": 113,
    },
    "survey_i": {
        "header_line_1based": 126,
        "data_line_1based": 129,
    },
    "716_198": {   # fallback alias for Survey I
        "header_line_1based": 126,
        "data_line_1based": 129,
    },
}


def infer_rule_key(filename: str) -> str:
    lower = filename.lower()

    if "table7" in lower:
        return "table7"
    if "sample" in lower:
        return "sample"
    if "survey_i" in lower:
        return "survey_i"
    if "716_198" in lower:
        return "716_198"

    raise ValueError(
        f"No fixed normalization rule matched this filename: {filename}"
    )


# -----
# Section: helpers
# -----

def read_all_lines(file_path: Path) -> List[str]:
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n\r") for line in f]


def split_line(line: str) -> List[str]:
    """
    DiskMass exported rows may still be semicolon/tab/comma separated.
    Choose the strongest delimiter for the current line.
    """
    counts = {
        ";": line.count(";"),
        "\t": line.count("\t"),
        ",": line.count(","),
    }

    if counts[";"] > 0 and counts[";"] >= counts["\t"] and counts[";"] >= counts[","]:
        return [cell.strip() for cell in line.split(";")]
    if counts["\t"] > 0 and counts["\t"] >= counts[","]:
        return [cell.strip() for cell in line.split("\t")]
    if counts[","] > 0:
        return [cell.strip() for cell in line.split(",")]

    return [cell.strip() for cell in line.split()]


def clean_header_cells(cells: List[str]) -> List[str]:
    used = {}
    result = []

    for idx, raw in enumerate(cells, start=1):
        name = raw.strip()
        if not name:
            name = f"col_{idx}"

        name = re.sub(r"\s+", "_", name)
        name = name.replace("/", "_")
        name = name.replace("(", "")
        name = name.replace(")", "")
        name = name.replace("[", "")
        name = name.replace("]", "")
        name = name.replace("-", "_")
        name = name.strip("_")

        if not name:
            name = f"col_{idx}"

        if name in used:
            used[name] += 1
            name = f"{name}_{used[name]}"
        else:
            used[name] = 1

        result.append(name)

    return result


def looks_like_footer_or_xml(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    upper = s.upper()
    if s.startswith("<") and s.endswith(">"):
        return True
    if "]]></CSV>" in upper or "</CSV>" in upper or "</DATA>" in upper:
        return True
    if s.startswith("<?XML"):
        return True
    return False


def looks_like_separator_row(cells: List[str]) -> bool:
    joined = "".join(cell.strip() for cell in cells)
    if not joined:
        return True
    return set(joined) <= set("-=+.:_~")


def pad_or_trim(cells: List[str], target_len: int) -> List[str]:
    if len(cells) < target_len:
        return cells + [""] * (target_len - len(cells))
    if len(cells) > target_len:
        return cells[:target_len]
    return cells


# -----
# Section: core normalization
# -----

def normalize_one_file(input_file: Path, output_file: Path) -> dict:
    rule_key = infer_rule_key(input_file.name)
    rule = FILE_RULES[rule_key]

    all_lines = read_all_lines(input_file)

    header_idx = rule["header_line_1based"] - 1
    data_idx = rule["data_line_1based"] - 1

    if header_idx >= len(all_lines):
        raise ValueError(
            f"Configured header line {rule['header_line_1based']} exceeds file length for {input_file.name}"
        )
    if data_idx >= len(all_lines):
        raise ValueError(
            f"Configured data line {rule['data_line_1based']} exceeds file length for {input_file.name}"
        )

    header_raw = all_lines[header_idx]
    header_cells = clean_header_cells(split_line(header_raw))
    ncols = len(header_cells)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    skipped_count = 0

    with output_file.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(header_cells)

        for line in all_lines[data_idx:]:
            if looks_like_footer_or_xml(line):
                # If footer starts, we can safely stop
                upper = line.strip().upper()
                if "]]></CSV>" in upper or "</CSV>" in upper or "</DATA>" in upper:
                    break
                skipped_count += 1
                continue

            cells = split_line(line)

            if looks_like_separator_row(cells):
                skipped_count += 1
                continue

            cells = pad_or_trim(cells, ncols)
            writer.writerow(cells)
            row_count += 1

    if row_count == 0:
        raise ValueError(f"No data rows written for {input_file.name}")

    return {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "rule_key": rule_key,
        "header_line_1based": rule["header_line_1based"],
        "data_line_1based": rule["data_line_1based"],
        "column_count": ncols,
        "row_count": row_count,
        "skipped_count": skipped_count,
    }


def write_summary(results: List[dict], output_dir: Path) -> Path:
    summary_path = output_dir / "diskmass_fixed_normalization_summary.csv"
    fields = [
        "input_file",
        "output_file",
        "rule_key",
        "header_line_1based",
        "data_line_1based",
        "column_count",
        "row_count",
        "skipped_count",
    ]

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    return summary_path


# -----
# Section: CLI
# -----

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize DiskMass raw files with file-specific fixed rules."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw DiskMass files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where cleaned CSV files will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files if present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in args.input_dir.iterdir() if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No files found in: {args.input_dir}")

    results: List[dict] = []

    for input_file in files:
        suffix = input_file.suffix.lower()
        if suffix not in {".tsv", ".txt", ".dat", ".csv"}:
            continue

        output_file = args.output_dir / f"{input_file.stem}.csv"
        if output_file.exists() and not args.overwrite:
            print(f"[SKIP] {output_file.name} already exists")
            continue

        try:
            result = normalize_one_file(input_file, output_file)
            results.append(result)
            print(
                f"[OK] {input_file.name} -> {output_file.name} | "
                f"rule={result['rule_key']} | cols={result['column_count']} | rows={result['row_count']}"
            )
        except Exception as exc:
            print(f"[FAILED] {input_file.name} -> {exc}")

    if results:
        summary_path = write_summary(results, args.output_dir)
        print(f"[SUMMARY] wrote {len(results)} normalized CSV files")
        print(f"[SUMMARY] {summary_path}")
    else:
        print("[INFO] No files were normalized.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
