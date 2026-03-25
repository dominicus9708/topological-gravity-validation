from __future__ import annotations

# -----
# normalize_sparc.py
#
# Convert SPARC rotmod .dat files into standardized CSV files.
#
# Default behavior:
# - input directory : data/raw/sparc
# - output directory: data/derived/sparc
#
# Supported input columns (SPARC rotmod standard):
#   radius, v_obs, v_err, v_gas, v_disk, v_bul
#
# Output columns:
#   galaxy_id, source, radius_kpc, v_obs_kms, v_err_kms,
#   v_gas_kms, v_disk_kms, v_bulge_kms, v_bar_kms
#
# The script keeps raw data untouched and creates one CSV per galaxy.
# -----

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, List


# -----
# Section: configuration
# -----

DEFAULT_INPUT_DIR = Path("data/raw/sparc")
DEFAULT_OUTPUT_DIR = Path("data/derived/sparc")
DEFAULT_SOURCE_NAME = "SPARC"

EXPECTED_COLUMN_COUNT = 6
OUTPUT_COLUMNS = [
    "galaxy_id",
    "source",
    "radius_kpc",
    "v_obs_kms",
    "v_err_kms",
    "v_gas_kms",
    "v_disk_kms",
    "v_bulge_kms",
    "v_bar_kms",
]


# -----
# Section: helpers
# -----


def infer_galaxy_id(file_path: Path) -> str:
    """Infer galaxy identifier from a SPARC rotmod filename."""
    stem = file_path.stem
    if stem.endswith("_rotmod"):
        stem = stem[:-7]
    return stem



def parse_numeric_row(line: str) -> List[float] | None:
    """Parse one numeric data row from a SPARC rotmod file.

    Returns None for blank lines, comments, or malformed lines that should be skipped.
    """
    stripped = line.strip()
    if not stripped:
        return None

    # Common comment/header cases
    if stripped.startswith("#"):
        return None
    if re.search(r"[A-Za-z]", stripped):
        return None

    parts = stripped.split()
    if len(parts) < EXPECTED_COLUMN_COUNT:
        return None

    try:
        values = [float(parts[i]) for i in range(EXPECTED_COLUMN_COUNT)]
    except ValueError:
        return None

    return values



def baryonic_speed(v_gas: float, v_disk: float, v_bulge: float) -> float:
    """Compute total baryonic circular speed from component speeds.

    SPARC component curves are combined in quadrature.
    """
    total_sq = max(v_gas, 0.0) ** 2 + max(v_disk, 0.0) ** 2 + max(v_bulge, 0.0) ** 2
    return total_sq ** 0.5



def iter_rotmod_files(input_dir: Path) -> Iterable[Path]:
    """Yield SPARC rotmod files in sorted order."""
    yield from sorted(input_dir.glob("*_rotmod.dat"))


# -----
# Section: core normalization
# -----


def normalize_one_file(file_path: Path, output_dir: Path, source_name: str) -> dict:
    """Normalize a single SPARC rotmod file into one CSV."""
    galaxy_id = infer_galaxy_id(file_path)
    output_path = output_dir / f"{galaxy_id}.csv"

    rows_written = 0
    skipped_lines = 0

    with file_path.open("r", encoding="utf-8", errors="ignore") as infile, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for line_number, line in enumerate(infile, start=1):
            parsed = parse_numeric_row(line)
            if parsed is None:
                skipped_lines += 1
                continue

            radius, v_obs, v_err, v_gas, v_disk, v_bulge = parsed
            v_bar = baryonic_speed(v_gas=v_gas, v_disk=v_disk, v_bulge=v_bulge)

            writer.writerow(
                {
                    "galaxy_id": galaxy_id,
                    "source": source_name,
                    "radius_kpc": f"{radius:.10g}",
                    "v_obs_kms": f"{v_obs:.10g}",
                    "v_err_kms": f"{v_err:.10g}",
                    "v_gas_kms": f"{v_gas:.10g}",
                    "v_disk_kms": f"{v_disk:.10g}",
                    "v_bulge_kms": f"{v_bulge:.10g}",
                    "v_bar_kms": f"{v_bar:.10g}",
                }
            )
            rows_written += 1

    if rows_written == 0:
        try:
            output_path.unlink(missing_ok=True)
        except TypeError:
            if output_path.exists():
                output_path.unlink()
        raise ValueError(f"No usable numeric rows found in {file_path}")

    return {
        "galaxy_id": galaxy_id,
        "input_file": str(file_path),
        "output_file": str(output_path),
        "rows_written": rows_written,
        "skipped_lines": skipped_lines,
    }



def build_summary_csv(results: List[dict], output_dir: Path) -> Path:
    """Write a summary CSV for normalization results."""
    summary_path = output_dir / "normalization_summary.csv"
    summary_columns = [
        "galaxy_id",
        "input_file",
        "output_file",
        "rows_written",
        "skipped_lines",
    ]

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(results)

    return summary_path


# -----
# Section: command line interface
# -----


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize SPARC rotmod .dat files into standardized CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing SPARC *_rotmod.dat files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write normalized CSV files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default=DEFAULT_SOURCE_NAME,
        help=f"Source label written into CSV rows (default: {DEFAULT_SOURCE_NAME})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()



def validate_directories(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    existing_csvs = list(output_dir.glob("*.csv"))
    if existing_csvs and not overwrite:
        raise FileExistsError(
            "Output directory already contains CSV files. "
            "Use --overwrite if you intentionally want to replace them. "
            f"Output directory: {output_dir}"
        )



def main() -> int:
    args = parse_args()
    validate_directories(args.input_dir, args.output_dir, args.overwrite)

    rotmod_files = list(iter_rotmod_files(args.input_dir))
    if not rotmod_files:
        raise FileNotFoundError(
            f"No SPARC rotmod files were found in: {args.input_dir}"
        )

    results: List[dict] = []
    failures: List[str] = []

    for file_path in rotmod_files:
        try:
            result = normalize_one_file(
                file_path=file_path,
                output_dir=args.output_dir,
                source_name=args.source_name,
            )
            results.append(result)
            print(f"[OK] {result['galaxy_id']} -> {result['output_file']}")
        except Exception as exc:  # pragma: no cover
            failures.append(f"{file_path.name}: {exc}")
            print(f"[FAILED] {file_path.name} -> {exc}")

    if results:
        summary_path = build_summary_csv(results, args.output_dir)
        print(f"[SUMMARY] wrote {len(results)} normalized files")
        print(f"[SUMMARY] summary CSV: {summary_path}")

    if failures:
        print("[ERROR] Some files failed during normalization:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
