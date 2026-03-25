from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


# -----
# build_sparc_diskmass_same_name_direct.py
#
# Second-pass SPARC ↔ DiskMass crossmatch by same-name direct matching
# for non-UGC catalog names.
#
# Purpose:
#   1) Keep the UGC-only direct pass separate.
#   2) Try direct matching for names that are already the same after
#      simple normalization (NGC, IC, DDO, ESO, F, UGCA, etc.).
#   3) Do NOT force-convert non-UGC names into fake UGC identifiers.
#
# Recommended usage:
#
#   python src/crossmatch/build_sparc_diskmass_same_name_direct.py ^
#       --sparc-unmatched data/derived/crossmatch/sparc_diskmass_unmatched_direct.csv ^
#       --diskmass-sample "data/derived/diskmass/J_ApJS_276_59_sample_The_DiskMass_Survey_(DMS)_XI_Full_Ha_sample.csv" ^
#       --diskmass-survey1 "data/derived/diskmass/J_ApJ_716_198_The_ DiskMass_survey_I.csv" ^
#       --output-file data/derived/crossmatch/sparc_diskmass_crossmatch_same_name.csv ^
#       --still-unmatched-file data/derived/crossmatch/sparc_diskmass_unmatched_same_name.csv
# -----

def normalize_name(name: object) -> str:
    if name is None:
        return ""
    text = str(name).strip().upper()
    if text == "" or text == "NAN":
        return ""
    return re.sub(r"[^A-Z0-9]", "", text)


def is_true_ugc_name(name: object) -> bool:
    text = normalize_name(name)
    return text.startswith("UGC")


def choose_sample_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "UGC", "Name", "_RAJ2000", "_DEJ2000", "RAJ2000", "DEJ2000",
        "Type", "Dist", "i", "PA", "R25", "hR", "mu0", "Vsys", "Vrot", "MHI"
    ]
    return [c for c in preferred if c in df.columns]


def choose_survey1_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "UGC", "Name", "_RAJ2000", "_DEJ2000", "RAJ2000", "DEJ2000",
        "Type", "Dist", "PA", "i", "hR", "mu0", "Vsys", "Vrot", "sigma", "MHI"
    ]
    return [c for c in preferred if c in df.columns]


def build_reference_name(row: pd.Series) -> str:
    # Priority:
    # 1) explicit Name column if present and non-empty
    # 2) fallback to UGC column
    if "Name" in row and pd.notna(row["Name"]) and str(row["Name"]).strip():
        return normalize_name(row["Name"])
    if "UGC" in row and pd.notna(row["UGC"]) and str(row["UGC"]).strip():
        return normalize_name(f"UGC{row['UGC']}")
    return ""


def build_crossmatch(
    sparc_unmatched_file: Path,
    diskmass_sample_file: Path,
    diskmass_survey1_file: Path,
    output_file: Path,
    still_unmatched_file: Optional[Path] = None,
) -> tuple[Path, Optional[Path], int, int]:
    sparc = pd.read_csv(sparc_unmatched_file)
    sample = pd.read_csv(diskmass_sample_file)
    survey1 = pd.read_csv(diskmass_survey1_file)

    if "galaxy_id" not in sparc.columns:
        raise ValueError("SPARC unmatched input must contain 'galaxy_id' column.")

    sparc = sparc.copy()
    sample = sample.copy()
    survey1 = survey1.copy()

    # This pass is only for non-UGC names.
    sparc["name_same_direct"] = sparc["galaxy_id"].map(normalize_name)
    sparc["eligible_same_name"] = ~sparc["galaxy_id"].map(is_true_ugc_name)

    sample["diskmass_name_direct"] = sample.apply(build_reference_name, axis=1)
    survey1["diskmass_name_direct"] = survey1.apply(build_reference_name, axis=1)

    sample_cols = choose_sample_columns(sample)
    survey1_cols = choose_survey1_columns(survey1)

    sample_small = sample[["diskmass_name_direct"] + sample_cols].copy()
    survey1_small = survey1[["diskmass_name_direct"] + survey1_cols].copy()

    sample_small = sample_small.drop_duplicates(subset=["diskmass_name_direct"])
    survey1_small = survey1_small.drop_duplicates(subset=["diskmass_name_direct"])

    merged = sparc.merge(
        sample_small,
        left_on="name_same_direct",
        right_on="diskmass_name_direct",
        how="left",
        suffixes=("", "_sample"),
    )

    merged = merged.merge(
        survey1_small,
        left_on="name_same_direct",
        right_on="diskmass_name_direct",
        how="left",
        suffixes=("", "_survey1"),
    )

    sample_hit = merged["diskmass_name_direct"].notna()
    survey1_hit = merged["diskmass_name_direct_survey1"].notna() if "diskmass_name_direct_survey1" in merged.columns else False
    any_hit = sample_hit | survey1_hit

    merged["match_status"] = merged.apply(
        lambda row: "not_applicable_ugc"
        if not bool(row["eligible_same_name"])
        else ("matched_same_name" if any_hit.loc[row.name] else "still_unmatched"),
        axis=1,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)

    matched_count = int((merged["match_status"] == "matched_same_name").sum())
    unmatched_count = int((merged["match_status"] == "still_unmatched").sum())

    out_still = None
    if still_unmatched_file is not None:
        out_still = still_unmatched_file
        still_unmatched_file.parent.mkdir(parents=True, exist_ok=True)
        merged.loc[merged["match_status"] == "still_unmatched", [
            "galaxy_id", "filename", "catalog_prefix", "catalog_number",
            "name_normalized", "row_count", "name_same_direct"
        ]].to_csv(still_unmatched_file, index=False)

    return output_file, out_still, matched_count, unmatched_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Second-pass SPARC↔DiskMass same-name direct crossmatch for non-UGC names."
    )
    parser.add_argument(
        "--sparc-unmatched",
        type=Path,
        default=Path("data/derived/crossmatch/sparc_diskmass_unmatched_direct.csv"),
        help="Path to unmatched SPARC list from UGC direct pass.",
    )
    parser.add_argument(
        "--diskmass-sample",
        type=Path,
        default=Path("data/derived/diskmass/J_ApJS_276_59_sample_The_DiskMass_Survey_(DMS)_XI_Full_Ha_sample.csv"),
        help="Path to DiskMass sample CSV.",
    )
    parser.add_argument(
        "--diskmass-survey1",
        type=Path,
        default=Path("data/derived/diskmass/J_ApJ_716_198_The_ DiskMass_survey_I.csv"),
        help="Path to DiskMass Survey I CSV.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/derived/crossmatch/sparc_diskmass_crossmatch_same_name.csv"),
        help="Output CSV for same-name direct matches.",
    )
    parser.add_argument(
        "--still-unmatched-file",
        type=Path,
        default=Path("data/derived/crossmatch/sparc_diskmass_unmatched_same_name.csv"),
        help="Output CSV for galaxies still unmatched after same-name pass.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_file, still_unmatched, matched_count, unmatched_count = build_crossmatch(
        args.sparc_unmatched,
        args.diskmass_sample,
        args.diskmass_survey1,
        args.output_file,
        args.still_unmatched_file,
    )
    print(f"[OK] same-name SPARC↔DiskMass crossmatch written: {output_file}")
    print(f"[OK] matched_same_name={matched_count} still_unmatched={unmatched_count}")
    if still_unmatched is not None:
        print(f"[OK] remaining unmatched list written: {still_unmatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
