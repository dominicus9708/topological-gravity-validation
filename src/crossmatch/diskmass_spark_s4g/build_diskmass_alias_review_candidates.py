from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Optional

import pandas as pd


# -----
# build_diskmass_alias_review_candidates.py
#
# Purpose:
#   Enrich diskmass_alias_seed.csv with coordinate-based SPARC candidate suggestions.
#
# Why this exists:
#   Direct / same-name / bridge matching reached their limit.
#   The next practical step is alias review:
#   "Which SPARC galaxy is the most plausible counterpart of this DiskMass UGC entry?"
#
# Strategy:
#   1) Read diskmass_alias_seed.csv
#   2) Read sparc_s4g_crossmatch_direct.csv
#   3) Use S4G coordinates as SPARC-side proxy coordinates
#   4) For each DiskMass row, find the nearest SPARC candidates on the sky
#   5) Write the top candidate suggestions into review columns
#
# Recommended usage:
#   python src/crossmatch/build_diskmass_alias_review_candidates.py ^
#       --alias-seed data/derived/crossmatch/diskmass_alias_seed.csv ^
#       --sparc-s4g-direct data/derived/crossmatch/sparc_s4g_crossmatch_direct.csv ^
#       --output-file data/derived/crossmatch/diskmass_alias_review_candidates.csv
# -----

def to_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.upper() == "NAN":
            return None
        return float(text)
    except Exception:
        return None


def angular_separation_deg(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    ra1 = math.radians(ra1_deg)
    dec1 = math.radians(dec1_deg)
    ra2 = math.radians(ra2_deg)
    dec2 = math.radians(dec2_deg)

    cos_sep = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    cos_sep = min(1.0, max(-1.0, cos_sep))
    return math.degrees(math.acos(cos_sep))


def choose_diskmass_coord_cols(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    ra_col = "_RAJ2000_sample" if "_RAJ2000_sample" in df.columns else None
    dec_col = "_DEJ2000_sample" if "_DEJ2000_sample" in df.columns else None

    if ra_col is None and "_RAJ2000_survey1" in df.columns:
        ra_col = "_RAJ2000_survey1"
    if dec_col is None and "_DEJ2000_survey1" in df.columns:
        dec_col = "_DEJ2000_survey1"

    if ra_col is None and "RAJ2000_sample" in df.columns:
        ra_col = "RAJ2000_sample"
    if dec_col is None and "DEJ2000_sample" in df.columns:
        dec_col = "DEJ2000_sample"

    if ra_col is None and "RAJ2000_survey1" in df.columns:
        ra_col = "RAJ2000_survey1"
    if dec_col is None and "DEJ2000_survey1" in df.columns:
        dec_col = "DEJ2000_survey1"

    return ra_col, dec_col


def build_review_candidates(
    alias_seed_file: Path,
    sparc_s4g_direct_file: Path,
    output_file: Path,
    max_sep_deg: float,
) -> Path:
    alias_df = pd.read_csv(alias_seed_file)
    sparc_s4g = pd.read_csv(sparc_s4g_direct_file)

    required_s4g = {"galaxy_id", "RAJ2000", "DEJ2000", "match_status"}
    if not required_s4g.issubset(set(sparc_s4g.columns)):
        raise ValueError("SPARC↔S4G direct file must contain galaxy_id, RAJ2000, DEJ2000, match_status.")

    # Keep only SPARC galaxies that actually matched to S4G and therefore have usable coordinates
    sparc_coords = sparc_s4g.loc[sparc_s4g["match_status"] == "matched", [
        "galaxy_id", "catalog_prefix", "catalog_number", "name_normalized", "RAJ2000", "DEJ2000"
    ]].copy()
    sparc_coords["sp_ra_deg"] = sparc_coords["RAJ2000"].map(to_float)
    sparc_coords["sp_dec_deg"] = sparc_coords["DEJ2000"].map(to_float)
    sparc_coords = sparc_coords.loc[sparc_coords["sp_ra_deg"].notna() & sparc_coords["sp_dec_deg"].notna()].reset_index(drop=True)

    ra_col, dec_col = choose_diskmass_coord_cols(alias_df)
    if ra_col is None or dec_col is None:
        raise ValueError("Could not find DiskMass coordinate columns in alias seed file.")

    alias_df = alias_df.copy()
    alias_df["dm_ra_deg"] = alias_df[ra_col].map(to_float)
    alias_df["dm_dec_deg"] = alias_df[dec_col].map(to_float)

    # Review columns to fill
    for n in (1, 2, 3):
        alias_df[f"candidate_{n}_sparc_galaxy_id"] = ""
        alias_df[f"candidate_{n}_sep_deg"] = ""
        alias_df[f"candidate_{n}_catalog_prefix"] = ""
        alias_df[f"candidate_{n}_catalog_number"] = ""

    alias_df["coord_candidate_count"] = 0
    alias_df["best_coord_sep_deg"] = ""
    alias_df["coord_alias_recommendation"] = ""

    for idx, row in alias_df.iterrows():
        ra = row["dm_ra_deg"]
        dec = row["dm_dec_deg"]
        if pd.isna(ra) or pd.isna(dec):
            alias_df.at[idx, "coord_alias_recommendation"] = "no_diskmass_coords"
            continue

        candidates = []
        for _, sp in sparc_coords.iterrows():
            sep = angular_separation_deg(float(ra), float(dec), float(sp["sp_ra_deg"]), float(sp["sp_dec_deg"]))
            if sep <= max_sep_deg:
                candidates.append((sep, sp["galaxy_id"], sp["catalog_prefix"], sp["catalog_number"]))

        candidates.sort(key=lambda x: x[0])
        alias_df.at[idx, "coord_candidate_count"] = len(candidates)

        if candidates:
            alias_df.at[idx, "best_coord_sep_deg"] = candidates[0][0]
            alias_df.at[idx, "coord_alias_recommendation"] = "review_top_candidate"
            for n, cand in enumerate(candidates[:3], start=1):
                sep, gid, prefix, number = cand
                alias_df.at[idx, f"candidate_{n}_sparc_galaxy_id"] = gid
                alias_df.at[idx, f"candidate_{n}_sep_deg"] = sep
                alias_df.at[idx, f"candidate_{n}_catalog_prefix"] = prefix
                alias_df.at[idx, f"candidate_{n}_catalog_number"] = number
        else:
            alias_df.at[idx, "coord_alias_recommendation"] = "no_candidate_within_threshold"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    alias_df.to_csv(output_file, index=False)
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build coordinate-based SPARC candidate suggestions for DiskMass alias review.")
    parser.add_argument(
        "--alias-seed",
        type=Path,
        default=Path("data/derived/crossmatch/diskmass_alias_seed.csv"),
        help="Path to diskmass alias seed CSV.",
    )
    parser.add_argument(
        "--sparc-s4g-direct",
        type=Path,
        default=Path("data/derived/crossmatch/sparc_s4g_crossmatch_direct.csv"),
        help="Path to SPARC↔S4G direct crossmatch CSV.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/derived/crossmatch/diskmass_alias_review_candidates.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--max-sep-deg",
        type=float,
        default=0.20,
        help="Maximum angular separation in degrees for candidate suggestions (default: 0.20 deg).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = build_review_candidates(
        args.alias_seed,
        args.sparc_s4g_direct,
        args.output_file,
        args.max_sep_deg,
    )
    print(f"[OK] DiskMass alias review candidates written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
