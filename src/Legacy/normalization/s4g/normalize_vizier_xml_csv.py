from __future__ import annotations

# -----
# normalize_vizier_xml_csv.py
#
# Purpose:
#   Normalize Vizier "XML+CSV" exports into clean comma-separated CSV files.
#
# Why this exists:
#   Vizier downloads often contain a long XML/VOTable wrapper first, and only later
#   the real table content appears inside a CSV block. A generic TSV/CSV converter
#   will often fail because it reads the XML header instead of the actual table.
#
# What this script does:
#   1) Find the real "<CSV" or "<CSV ...>" block inside the file.
#   2) Start reading only after that block.
#   3) Use the first real non-empty line as the header.
#   4) Skip common unit / separator rows after the header.
#   5) Stop at the closing "]]></CSV>" or "</CSV>" area.
#   6) Write a clean .csv file for later matching work.
#
# Recommended usage:
#
#   S4G only:
#   python src/normalization/normalize_vizier_xml_csv.py ^
#       --input-dir data/raw/s4g ^
#       --output-dir data/derived/s4g ^
#       --overwrite
#
#   DiskMass only:
#   python src/normalization/normalize_vizier_xml_csv.py ^
#       --input-dir data/raw/diskmass ^
#       --output-dir data/derived/diskmass ^
#       --overwrite
#
#   Both in one CMD line:
#   python src/normalization/normalize_vizier_xml_csv.py --input-dir data/raw/s4g --output-dir data/derived/s4g --overwrite ^
#   && python src/normalization/normalize_vizier_xml_csv.py --input-dir data/raw/diskmass --output-dir data/derived/diskmass --overwrite
# -----

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, List


# -----
# Section: file discovery
# -----

SUPPORTED_PATTERNS = ("*.tsv", "*.txt", "*.dat", "*.csv")


def iter_input_files(input_dir: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in SUPPORTED_PATTERNS:
        for file_path in sorted(input_dir.glob(pattern)):
            if file_path.is_file() and file_path not in seen:
                seen.add(file_path)
                yield file_path


# -----
# Section: helpers
# -----

def normalize_header_cells(cells: List[str]) -> List[str]:
    used: dict[str, int] = {}
    result: List[str] = []

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


def looks_like_separator_row(cells: List[str]) -> bool:
    joined = "".join(cell.strip() for cell in cells)
    if not joined:
        return True
    return set(joined) <= set("-=+.:_")


def looks_like_unit_row(cells: List[str]) -> bool:
    # Common Vizier unit rows often contain tokens like deg, mag, km/s, etc.
    # We skip them only near the header region.
    joined = " ".join(cell.strip().lower() for cell in cells if cell.strip())
    if not joined:
        return False

    unit_hints = (
        "deg", "mag", "km/s", "km s", "m/s", "arcsec", "arcmin",
        "kpc", "pc", "msun", "jy", "dex", "log", "yr"
    )

    # If the row is mostly alphabetic units and short tokens, it is likely a unit row.
    alpha_count = sum(bool(re.search(r"[a-zA-Z]", c)) for c in cells if c.strip())
    numeric_count = sum(bool(re.fullmatch(r"[-+]?\d+(\.\d+)?", c.strip())) for c in cells if c.strip())

    if any(h in joined for h in unit_hints) and alpha_count >= numeric_count:
        return True

    return False


def pad_or_trim_row(cells: List[str], target_len: int) -> List[str]:
    if len(cells) < target_len:
        return cells + [""] * (target_len - len(cells))
    if len(cells) > target_len:
        return cells[:target_len]
    return cells


# -----
# Section: core parsing
# -----

def extract_csv_block_lines(file_path: Path) -> List[str]:
    """
    Extract only the content lines that live inside the Vizier CSV block.
    """
    inside_csv_block = False
    collected: List[str] = []

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n\r")
            stripped = line.strip()

            if not inside_csv_block:
                # Start after any line containing <CSV
                if "<CSV" in stripped.upper():
                    inside_csv_block = True
                continue

            # Stop at the CSV closing region
            upper = stripped.upper()
            if "]]></CSV>" in upper or "</CSV>" in upper or "</DATA>" in upper:
                break

            # Ignore empty XML-ish lines inside the block
            if stripped.startswith("<") and stripped.endswith(">"):
                continue

            collected.append(line)

    return collected


def parse_semicolon_or_tab_or_comma(line: str) -> List[str]:
    """
    Vizier CSV block can be semicolon-separated, tab-separated, or comma-separated.
    We choose the strongest visible delimiter for the line.
    """
    stripped = line.rstrip("\n\r")

    counts = {
        ";": stripped.count(";"),
        "\t": stripped.count("\t"),
        ",": stripped.count(","),
    }

    if counts[";"] > 0 and counts[";"] >= counts["\t"] and counts[";"] >= counts[","]:
        return [cell.strip() for cell in stripped.split(";")]

    if counts["\t"] > 0 and counts["\t"] >= counts[","]:
        return [cell.strip() for cell in stripped.split("\t")]

    if counts[","] > 0:
        return [cell.strip() for cell in stripped.split(",")]

    # fallback
    return [cell.strip() for cell in stripped.split()]


def normalize_vizier_file(input_file: Path, output_file: Path) -> dict:
    block_lines = extract_csv_block_lines(input_file)
    if not block_lines:
        raise ValueError("No Vizier CSV block found. The file may not be an XML+CSV export.")

    # Find first real non-empty line = header candidate
    header_idx = None
    for idx, line in enumerate(block_lines):
        if line.strip():
            header_idx = idx
            break

    if header_idx is None:
        raise ValueError("CSV block exists, but no non-empty header line was found.")

    header_cells_raw = parse_semicolon_or_tab_or_comma(block_lines[header_idx])
    header = normalize_header_cells(header_cells_raw)

    data_lines = block_lines[header_idx + 1:]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    skipped_after_header = 0

    with output_file.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(header)

        for raw_line in data_lines:
            if not raw_line.strip():
                skipped_after_header += 1
                continue

            cells = parse_semicolon_or_tab_or_comma(raw_line)

            if looks_like_separator_row(cells):
                skipped_after_header += 1
                continue

            if looks_like_unit_row(cells) and row_count == 0:
                skipped_after_header += 1
                continue

            cells = pad_or_trim_row(cells, len(header))
            writer.writerow(cells)
            row_count += 1

    if row_count == 0:
        raise ValueError("Header was found, but no data rows were written.")

    return {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "header_column_count": len(header),
        "row_count": row_count,
        "skipped_after_header": skipped_after_header,
    }


# -----
# Section: summary
# -----

def write_summary(results: List[dict], output_dir: Path) -> Path:
    summary_path = output_dir / "vizier_xml_csv_normalization_summary.csv"
    fields = [
        "input_file",
        "output_file",
        "header_column_count",
        "row_count",
        "skipped_after_header",
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
        description="Normalize Vizier XML+CSV exports into clean CSV files."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw Vizier exports.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for normalized CSV files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = list(iter_input_files(args.input_dir))
    if not files:
        raise FileNotFoundError(f"No supported files found in: {args.input_dir}")

    results: List[dict] = []

    for input_file in files:
        output_file = args.output_dir / f"{input_file.stem}.csv"

        if output_file.exists() and not args.overwrite:
            print(f"[SKIP] {output_file.name} already exists")
            continue

        try:
            result = normalize_vizier_file(input_file, output_file)
            results.append(result)
            print(
                f"[OK] {input_file.name} -> {output_file.name} | "
                f"cols={result['header_column_count']} | rows={result['row_count']}"
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
