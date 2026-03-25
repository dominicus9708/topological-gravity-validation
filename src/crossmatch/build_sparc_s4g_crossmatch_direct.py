from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path
from typing import Optional

import pandas as pd


def normalize_name(name: object) -> str:
    if name is None:
        return ""
    text = str(name).strip().upper()
    if text == "" or text == "NAN":
        return ""
    return re.sub(r"[^A-Z0-9]", "", text)


def choose_s4g_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "Name", "RAJ2000", "DEJ2000", "amaj", "PA", "ell",
        "3.6", "e_3.6", "4.5", "e_4.5", "c31", "c42",
        "M3.6", "M4.5", "logM*", "Dmean", "e_Dmean",
        "TT", "e_TT", "Vr", "e_Vr"
    ]
    return [c for c in preferred if c in df.columns]


def build_match(
    sparc_index_file: Path,
    s4g_catalog_file: Path,
    output_file: Path,
    unmatched_file: Optional[Path] = None,
) -> tuple[Path, Optional[Path], int, int]:
    sparc = pd.read_csv(sparc_index_file)
    s4g = pd.read_csv(s4g_catalog_file)

    if "galaxy_id" not in sparc.columns or "name_normalized" not in sparc.columns:
        raise ValueError("SPARC index must contain 'galaxy_id' and 'name_normalized' columns.")
    if "Name" not in s4g.columns:
        raise ValueError("S4G catalog must contain 'Name' column.")

    s4g = s4g.copy()
    sparc = sparc.copy()

    s4g["s4g_name_normalized"] = s4g["Name"].map(normalize_name)
    sparc["name_normalized"] = sparc["name_normalized"].map(normalize_name)

    keep_s4g_cols = choose_s4g_columns(s4g)
    s4g_small = s4g[["s4g_name_normalized"] + keep_s4g_cols].copy()

    merged = sparc.merge(
        s4g_small,
        left_on="name_normalized",
        right_on="s4g_name_normalized",
        how="left",
    )

    merged["match_status"] = merged["Name"].notna().map(lambda x: "matched" if x else "unmatched")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)

    unmatched_count = int((merged["match_status"] == "unmatched").sum())
    matched_count = int((merged["match_status"] == "matched").sum())

    unmatched_path = None
    if unmatched_file is not None:
        unmatched_path = unmatched_file
        unmatched_file.parent.mkdir(parents=True, exist_ok=True)
        merged.loc[merged["match_status"] == "unmatched", [
            "galaxy_id", "filename", "catalog_prefix", "catalog_number", "name_normalized", "row_count"
        ]].to_csv(unmatched_file, index=False)

    return output_file, unmatched_path, matched_count, unmatched_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a first-pass direct-name SPARC↔S4G crossmatch.")
    parser.add_argument(
        "--sparc-index",
        type=Path,
        default=Path("data/derived/crossmatch/sparc_galaxy_index.csv"),
        help="Path to SPARC galaxy index CSV.",
    )
    parser.add_argument(
        "--s4g-catalog",
        type=Path,
        default=Path("data/derived/s4g/S4G_catalog.csv"),
        help="Path to normalized S4G catalog CSV.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/derived/crossmatch/sparc_s4g_crossmatch_direct.csv"),
        help="Output CSV for direct normalized-name matches.",
    )
    parser.add_argument(
        "--unmatched-file",
        type=Path,
        default=Path("data/derived/crossmatch/sparc_s4g_unmatched_direct.csv"),
        help="Output CSV for currently unmatched SPARC galaxies.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_file, unmatched_file, matched_count, unmatched_count = build_match(
        args.sparc_index,
        args.s4g_catalog,
        args.output_file,
        args.unmatched_file,
    )
    print(f"[OK] direct SPARC↔S4G crossmatch written: {output_file}")
    print(f"[OK] matched={matched_count} unmatched={unmatched_count}")
    if unmatched_file is not None:
        print(f"[OK] unmatched list written: {unmatched_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
