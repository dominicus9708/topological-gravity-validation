from __future__ import annotations
import argparse
import math
import re
from pathlib import Path
from typing import Optional

import pandas as pd


# -----
# build_sparc_diskmass_coord_bridge.py
#
# Third-pass SPARC ↔ DiskMass coordinate-based bridge matching.
#
# Strategy:
#   1) Use SPARC↔S4G direct crossmatch as a coordinate source for SPARC galaxies
#      that already matched to S4G.
#   2) Compare those SPARC proxy coordinates against DiskMass sample / Survey I
#      coordinates.
#   3) Match by small angular separation threshold.
#
# Why this step:
#   - Name-based direct matching has already been exhausted.
#   - Many galaxies use different catalog names across datasets.
#   - S4G already gave us coordinates for a useful subset of SPARC galaxies.
#
# Recommended usage:
#
#   python src/crossmatch/build_sparc_diskmass_coord_bridge.py ^
#       --sparc-s4g-direct data/derived/crossmatch/sparc_s4g_crossmatch_direct.csv ^
#       --sparc-diskmass-unmatched data/derived/crossmatch/sparc_diskmass_unmatched_same_name.csv ^
#       --diskmass-sample "data/derived/diskmass/J_ApJS_276_59_sample_The_DiskMass_Survey_(DMS)_XI_Full_Ha_sample.csv" ^
#       --diskmass-survey1 "data/derived/diskmass/J_ApJ_716_198_The_ DiskMass_survey_I.csv" ^
#       --output-file data/derived/crossmatch/sparc_diskmass_crossmatch_coord_bridge.csv ^
#       --still-unmatched-file data/derived/crossmatch/sparc_diskmass_unmatched_coord_bridge.csv
# -----

def normalize_name(name: object) -> str:
    if name is None:
        return ""
    text = str(name).strip().upper()
    if text == "" or text == "NAN":
        return ""
    return re.sub(r"[^A-Z0-9]", "", text)


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
    """
    Great-circle angular separation in degrees.
    """
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


def choose_diskmass_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "UGC", "Name", "_RAJ2000", "_DEJ2000", "RAJ2000", "DEJ2000",
        "Type", "Dist", "i", "PA", "R25", "hR", "mu0", "Vsys", "Vrot", "sigma", "MHI"
    ]
    return [c for c in preferred if c in df.columns]


def add_diskmass_coords(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Prefer underscore columns when present, otherwise fallback.
    ra_col = "_RAJ2000" if "_RAJ2000" in out.columns else ("RAJ2000" if "RAJ2000" in out.columns else None)
    dec_col = "_DEJ2000" if "_DEJ2000" in out.columns else ("DEJ2000" if "DEJ2000" in out.columns else None)

    if ra_col is None or dec_col is None:
        out["dm_ra_deg"] = None
        out["dm_dec_deg"] = None
        return out

    out["dm_ra_deg"] = out[ra_col].map(to_float)
    out["dm_dec_deg"] = out[dec_col].map(to_float)
    return out


def build_candidate_table(sample: pd.DataFrame, survey1: pd.DataFrame) -> pd.DataFrame:
    sample = add_diskmass_coords(sample)
    survey1 = add_diskmass_coords(survey1)

    sample_cols = choose_diskmass_columns(sample)
    survey1_cols = choose_diskmass_columns(survey1)

    sample_small = sample[["dm_ra_deg", "dm_dec_deg"] + sample_cols].copy()
    sample_small["diskmass_source"] = "sample"

    survey1_small = survey1[["dm_ra_deg", "dm_dec_deg"] + survey1_cols].copy()
    survey1_small["diskmass_source"] = "survey1"

    merged = pd.concat([sample_small, survey1_small], ignore_index=True)

    # Drop rows with no coordinates
    merged = merged.loc[merged["dm_ra_deg"].notna() & merged["dm_dec_deg"].notna()].copy()

    # Build a weak display name
    if "Name" in merged.columns:
        merged["diskmass_display_name"] = merged["Name"].fillna("")
    else:
        merged["diskmass_display_name"] = ""

    if "UGC" in merged.columns:
        merged["diskmass_display_name"] = merged.apply(
            lambda row: row["diskmass_display_name"] if str(row["diskmass_display_name"]).strip() else f"UGC{row['UGC']}",
            axis=1,
        )

    return merged.reset_index(drop=True)


def build_crossmatch(
    sparc_s4g_direct_file: Path,
    sparc_diskmass_unmatched_file: Path,
    diskmass_sample_file: Path,
    diskmass_survey1_file: Path,
    output_file: Path,
    still_unmatched_file: Optional[Path],
    max_sep_deg: float,
) -> tuple[Path, Optional[Path], int, int]:
    s4g_direct = pd.read_csv(sparc_s4g_direct_file)
    unmatched = pd.read_csv(sparc_diskmass_unmatched_file)
    sample = pd.read_csv(diskmass_sample_file)
    survey1 = pd.read_csv(diskmass_survey1_file)

    # Need SPARC galaxy_id plus S4G coords
    required_s4g = {"galaxy_id", "RAJ2000", "DEJ2000", "match_status"}
    if not required_s4g.issubset(set(s4g_direct.columns)):
        raise ValueError("SPARC↔S4G direct file must contain galaxy_id, RAJ2000, DEJ2000, match_status.")

    if "galaxy_id" not in unmatched.columns:
        raise ValueError("SPARC DiskMass unmatched file must contain galaxy_id.")

    # Keep only SPARC galaxies that still need DiskMass matching
    unmatched_ids = set(unmatched["galaxy_id"].astype(str))
    s4g_direct = s4g_direct.copy()
    s4g_direct["sp_ra_deg"] = s4g_direct["RAJ2000"].map(to_float)
    s4g_direct["sp_dec_deg"] = s4g_direct["DEJ2000"].map(to_float)

    # Only use SPARC galaxies that are matched to S4G and still unmatched in DiskMass
    sparc_coords = s4g_direct.loc[
        (s4g_direct["match_status"] == "matched")
        & (s4g_direct["galaxy_id"].astype(str).isin(unmatched_ids))
        & s4g_direct["sp_ra_deg"].notna()
        & s4g_direct["sp_dec_deg"].notna(),
        ["galaxy_id", "filename", "catalog_prefix", "catalog_number", "name_normalized", "row_count", "sp_ra_deg", "sp_dec_deg"]
    ].copy()

    candidates = build_candidate_table(sample, survey1)

    rows = []
    still_unmatched_rows = []

    for _, sp in sparc_coords.iterrows():
        best_idx = None
        best_sep = None

        for idx, dm in candidates.iterrows():
            sep = angular_separation_deg(
                float(sp["sp_ra_deg"]), float(sp["sp_dec_deg"]),
                float(dm["dm_ra_deg"]), float(dm["dm_dec_deg"])
            )
            if best_sep is None or sep < best_sep:
                best_sep = sep
                best_idx = idx

        if best_idx is not None and best_sep is not None and best_sep <= max_sep_deg:
            dm = candidates.loc[best_idx]
            row = {k: sp[k] for k in sp.index}
            for col in candidates.columns:
                row[col] = dm[col]
            row["angular_sep_deg"] = best_sep
            row["match_status"] = "matched_coord_bridge"
            rows.append(row)
        else:
            row = {k: sp[k] for k in sp.index}
            row["angular_sep_deg"] = best_sep
            row["match_status"] = "still_unmatched"
            still_unmatched_rows.append(row)

    output_df = pd.DataFrame(rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_df.empty:
        # write empty table with minimal columns for consistency
        output_df = pd.DataFrame(columns=[
            "galaxy_id", "sp_ra_deg", "sp_dec_deg",
            "diskmass_source", "diskmass_display_name",
            "dm_ra_deg", "dm_dec_deg", "angular_sep_deg", "match_status"
        ])
    output_df.to_csv(output_file, index=False)

    out_still = None
    if still_unmatched_file is not None:
        out_still = still_unmatched_file
        still_unmatched_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(still_unmatched_rows).to_csv(still_unmatched_file, index=False)

    return output_file, out_still, len(rows), len(still_unmatched_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Third-pass SPARC↔DiskMass coordinate bridge matching using S4G coordinates.")
    parser.add_argument(
        "--sparc-s4g-direct",
        type=Path,
        default=Path("data/derived/crossmatch/sparc_s4g_crossmatch_direct.csv"),
        help="Path to SPARC↔S4G direct crossmatch CSV.",
    )
    parser.add_argument(
        "--sparc-diskmass-unmatched",
        type=Path,
        default=Path("data/derived/crossmatch/sparc_diskmass_unmatched_same_name.csv"),
        help="Path to SPARC galaxies still unmatched after DiskMass same-name pass.",
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
        default=Path("data/derived/crossmatch/sparc_diskmass_crossmatch_coord_bridge.csv"),
        help="Output CSV for coordinate-bridge matches.",
    )
    parser.add_argument(
        "--still-unmatched-file",
        type=Path,
        default=Path("data/derived/crossmatch/sparc_diskmass_unmatched_coord_bridge.csv"),
        help="Output CSV for galaxies still unmatched after coordinate bridge pass.",
    )
    parser.add_argument(
        "--max-sep-deg",
        type=float,
        default=0.02,
        help="Maximum angular separation in degrees for a coordinate match (default: 0.02 deg).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_file, still_unmatched, matched_count, unmatched_count = build_crossmatch(
        args.sparc_s4g_direct,
        args.sparc_diskmass_unmatched,
        args.diskmass_sample,
        args.diskmass_survey1,
        args.output_file,
        args.still_unmatched_file,
        args.max_sep_deg,
    )
    print(f"[OK] coordinate-bridge SPARC↔DiskMass crossmatch written: {output_file}")
    print(f"[OK] matched_coord_bridge={matched_count} still_unmatched={unmatched_count}")
    if still_unmatched is not None:
        print(f"[OK] remaining unmatched list written: {still_unmatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
