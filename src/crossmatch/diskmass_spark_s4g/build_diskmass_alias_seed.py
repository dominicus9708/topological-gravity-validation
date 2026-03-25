from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


# -----
# build_diskmass_alias_seed.py
#
# Purpose:
#   Build a review table for the next matching stage:
#   SPARC ↔ DiskMass alias-based matching.
#
# Why this exists:
#   Direct-name matching and coordinate-bridge matching have reached their limit.
#   The remaining problem is usually "same galaxy, different catalog name"
#   such as NGC ↔ UGC, IC ↔ UGC, etc.
#
# What this script does:
#   1) Read DiskMass sample + Survey I
#   2) Merge them into one UGC-based reference table
#   3) Keep useful review columns (UGC, Name, coords, Dist, Type, etc.)
#   4) Add blank alias-review columns for manual or later assisted filling
#
# Recommended usage:
#   python src/crossmatch/build_diskmass_alias_seed.py ^
#       --sparc-index data/derived/crossmatch/sparc_galaxy_index.csv ^
#       --diskmass-sample "data/derived/diskmass/J_ApJS_276_59_sample_The_DiskMass_Survey_(DMS)_XI_Full_Ha_sample.csv" ^
#       --diskmass-survey1 "data/derived/diskmass/J_ApJ_716_198_The_ DiskMass_survey_I.csv" ^
#       --output-file data/derived/crossmatch/diskmass_alias_seed.csv
# -----

def normalize_name(name: object) -> str:
    if name is None:
        return ""
    text = str(name).strip().upper()
    if text == "" or text == "NAN":
        return ""
    return re.sub(r"[^A-Z0-9]", "", text)


def normalize_ugc(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if text == "" or text == "NAN":
        return ""
    text = text.replace("UGC", "")
    digits = re.sub(r"[^0-9]", "", text)
    if digits == "":
        return ""
    return f"UGC{int(digits)}"


def choose_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "UGC", "Name", "_RAJ2000", "_DEJ2000", "RAJ2000", "DEJ2000",
        "Type", "Dist", "PA", "i", "R25", "hR", "mu0", "Vsys", "Vrot", "sigma", "MHI"
    ]
    return [c for c in preferred if c in df.columns]


def merge_diskmass_tables(sample: pd.DataFrame, survey1: pd.DataFrame) -> pd.DataFrame:
    sample = sample.copy()
    survey1 = survey1.copy()

    if "UGC" not in sample.columns:
        raise ValueError("DiskMass sample CSV must contain 'UGC' column.")
    if "UGC" not in survey1.columns:
        raise ValueError("DiskMass Survey I CSV must contain 'UGC' column.")

    sample["ugc_norm"] = sample["UGC"].map(normalize_ugc)
    survey1["ugc_norm"] = survey1["UGC"].map(normalize_ugc)

    sample_cols = choose_columns(sample)
    survey1_cols = choose_columns(survey1)

    sample_small = sample[["ugc_norm"] + sample_cols].copy().drop_duplicates(subset=["ugc_norm"])
    survey1_small = survey1[["ugc_norm"] + survey1_cols].copy().drop_duplicates(subset=["ugc_norm"])

    merged = sample_small.merge(
        survey1_small,
        on="ugc_norm",
        how="outer",
        suffixes=("_sample", "_survey1"),
    )

    # Build a stable display name
    def pick_name(row) -> str:
        for col in ("Name_sample", "Name_survey1"):
            if col in row.index:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    return str(val).strip()
        if "UGC_sample" in row.index and pd.notna(row["UGC_sample"]):
            return f"UGC{int(float(row['UGC_sample']))}"
        if "UGC_survey1" in row.index and pd.notna(row["UGC_survey1"]):
            return f"UGC{int(float(row['UGC_survey1']))}"
        return row["ugc_norm"]

    merged["diskmass_display_name"] = merged.apply(pick_name, axis=1)
    merged["diskmass_name_normalized"] = merged["diskmass_display_name"].map(normalize_name)

    return merged


def build_seed(
    sparc_index_file: Path,
    diskmass_sample_file: Path,
    diskmass_survey1_file: Path,
    output_file: Path,
) -> Path:
    sparc = pd.read_csv(sparc_index_file)
    sample = pd.read_csv(diskmass_sample_file)
    survey1 = pd.read_csv(diskmass_survey1_file)

    if "galaxy_id" not in sparc.columns:
        raise ValueError("SPARC index must contain 'galaxy_id' column.")

    diskmass = merge_diskmass_tables(sample, survey1)

    # Keep a lightweight SPARC list for review
    sparc_ref = sparc[["galaxy_id", "catalog_prefix", "catalog_number", "name_normalized"]].copy()
    sparc_ref["sparc_display_name"] = sparc_ref["galaxy_id"]

    # Store SPARC list as a compact string reference to help manual review
    # This is not meant for direct machine matching, only for quick lookup context.
    sparc_names_joined = " | ".join(sparc_ref["sparc_display_name"].astype(str).tolist())

    out = diskmass.copy()
    out["alias_match_status"] = "pending_review"
    out["candidate_sparc_galaxy_id"] = ""
    out["candidate_sparc_name_normalized"] = ""
    out["alias_source"] = ""
    out["alias_confidence"] = ""
    out["review_notes"] = ""
    out["sparc_reference_note"] = sparc_names_joined

    # Add a simple exact normalized-name hint if any exists
    sparc_lookup = {normalize_name(v): v for v in sparc["galaxy_id"].astype(str)}
    out["auto_exact_name_hint"] = out["diskmass_name_normalized"].map(lambda x: sparc_lookup.get(x, ""))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DiskMass alias seed table for manual/assisted catalog bridging.")
    parser.add_argument(
        "--sparc-index",
        type=Path,
        default=Path("data/derived/crossmatch/sparc_galaxy_index.csv"),
        help="Path to SPARC galaxy index CSV.",
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
        default=Path("data/derived/crossmatch/diskmass_alias_seed.csv"),
        help="Output CSV path for alias review seed table.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = build_seed(
        args.sparc_index,
        args.diskmass_sample,
        args.diskmass_survey1,
        args.output_file,
    )
    print(f"[OK] DiskMass alias seed written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
