from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


# -----
# Section: normalization helpers
# -----

def normalize_basic(name: object) -> str:
    if name is None:
        return ""
    text = str(name).strip().upper()
    if text == "" or text == "NAN":
        return ""
    return re.sub(r"[^A-Z0-9]", "", text)


def normalize_ugc_only(name: object) -> str:
    """
    Normalize only true UGC identifiers.
    Non-UGC names return empty string.

    Examples:
      UGC00128 -> UGC128
      UGC 4305 -> UGC4305
      4305     -> UGC4305   # only when used on an actual DiskMass UGC column
    """
    if name is None:
        return ""
    text = str(name).strip().upper()
    if text == "" or text == "NAN":
        return ""

    # For SPARC galaxy names, only accept explicit UGC prefix
    if text.startswith("UGC"):
        digits = re.sub(r"[^0-9]", "", text.replace("UGC", ""))
        if digits == "":
            return ""
        return f"UGC{int(digits)}"

    # If this is already just a number-like UGC field, allow numeric-only
    if re.fullmatch(r"[0-9]+", text):
        return f"UGC{int(text)}"

    return ""


def choose_sample_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "UGC", "Name", "RAJ2000", "DEJ2000", "Type", "Dist",
        "i", "PA", "R25", "hR", "mu0", "Vsys", "Vrot", "MHI"
    ]
    return [c for c in preferred if c in df.columns]


def choose_survey1_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "UGC", "RAJ2000", "DEJ2000", "Type", "Dist", "PA", "i", "hR", "mu0",
        "Vsys", "Vrot", "sigma", "MHI"
    ]
    return [c for c in preferred if c in df.columns]


# -----
# Section: core builder
# -----

def build_crossmatch(
    sparc_index_file: Path,
    diskmass_sample_file: Path,
    diskmass_survey1_file: Path,
    output_file: Path,
    unmatched_file: Optional[Path] = None,
) -> tuple[Path, Optional[Path], int, int]:
    sparc = pd.read_csv(sparc_index_file)
    sample = pd.read_csv(diskmass_sample_file)
    survey1 = pd.read_csv(diskmass_survey1_file)

    required_sparc = {"galaxy_id", "name_normalized"}
    if not required_sparc.issubset(set(sparc.columns)):
        raise ValueError("SPARC index must contain 'galaxy_id' and 'name_normalized' columns.")

    if "UGC" not in sample.columns:
        raise ValueError("DiskMass sample file must contain 'UGC' column.")
    if "UGC" not in survey1.columns:
        raise ValueError("DiskMass Survey I file must contain 'UGC' column.")

    sparc = sparc.copy()
    sample = sample.copy()
    survey1 = survey1.copy()

    # IMPORTANT:
    # Only true UGC-prefixed SPARC galaxies are eligible in this direct pass.
    sparc["sparc_ugc_normalized"] = sparc["galaxy_id"].map(
        lambda x: normalize_ugc_only(x) if str(x).strip().upper().startswith("UGC") else ""
    )

    sample["diskmass_ugc_normalized"] = sample["UGC"].map(normalize_ugc_only)
    survey1["diskmass_ugc_normalized"] = survey1["UGC"].map(normalize_ugc_only)

    sample_cols = choose_sample_columns(sample)
    survey1_cols = choose_survey1_columns(survey1)

    sample_small = sample[["diskmass_ugc_normalized"] + sample_cols].copy()
    survey1_small = survey1[["diskmass_ugc_normalized"] + survey1_cols].copy()

    sample_small = sample_small.drop_duplicates(subset=["diskmass_ugc_normalized"])
    survey1_small = survey1_small.drop_duplicates(subset=["diskmass_ugc_normalized"])

    merged = sparc.merge(
        sample_small,
        left_on="sparc_ugc_normalized",
        right_on="diskmass_ugc_normalized",
        how="left",
        suffixes=("", "_sample"),
    )

    merged = merged.merge(
        survey1_small,
        left_on="sparc_ugc_normalized",
        right_on="diskmass_ugc_normalized",
        how="left",
        suffixes=("", "_survey1"),
    )

    sample_hit = merged["diskmass_ugc_normalized"].notna()
    survey1_hit = merged["diskmass_ugc_normalized_survey1"].notna() if "diskmass_ugc_normalized_survey1" in merged.columns else False
    merged["match_status"] = (sample_hit | survey1_hit).map(lambda x: "matched" if x else "unmatched")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)

    matched_count = int((merged["match_status"] == "matched").sum())
    unmatched_count = int((merged["match_status"] == "unmatched").sum())

    out_unmatched = None
    if unmatched_file is not None:
        out_unmatched = unmatched_file
        unmatched_file.parent.mkdir(parents=True, exist_ok=True)
        merged.loc[merged["match_status"] == "unmatched", [
            "galaxy_id", "filename", "catalog_prefix", "catalog_number",
            "name_normalized", "sparc_ugc_normalized", "row_count"
        ]].to_csv(unmatched_file, index=False)

    return output_file, out_unmatched, matched_count, unmatched_count


# -----
# Section: CLI
# -----

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a first-pass SPARC↔DiskMass crossmatch using true UGC-only matching.")
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
        default=Path("data/derived/crossmatch/sparc_diskmass_crossmatch_direct.csv"),
        help="Output CSV for first-pass SPARC↔DiskMass matches.",
    )
    parser.add_argument(
        "--unmatched-file",
        type=Path,
        default=Path("data/derived/crossmatch/sparc_diskmass_unmatched_direct.csv"),
        help="Output CSV for unmatched SPARC galaxies after first-pass DiskMass matching.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_file, unmatched_file, matched_count, unmatched_count = build_crossmatch(
        args.sparc_index,
        args.diskmass_sample,
        args.diskmass_survey1,
        args.output_file,
        args.unmatched_file,
    )
    print(f"[OK] direct SPARC↔DiskMass crossmatch written: {output_file}")
    print(f"[OK] matched={matched_count} unmatched={unmatched_count}")
    if unmatched_file is not None:
        print(f"[OK] unmatched list written: {unmatched_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
