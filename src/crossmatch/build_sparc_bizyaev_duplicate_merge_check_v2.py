#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(".")

MATCH_2014_FILE = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "crossmatch"
    / "sparc_bizyaev"
    / "recomputed_with_coords_2014"
    / "matched"
    / "sparc_bizyaev2014_with_coords_crossmatch_matched.csv"
)

MATCH_2002_FILE = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "crossmatch"
    / "sparc_bizyaev2002"
    / "working"
    / "matched"
    / "sparc_bizyaev2002_crossmatch_matched.csv"
)

SPARC_INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "crossmatch"
    / "sparc_diskmass_S4G"
    / "sparc_galaxy_index_with_coords.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "crossmatch"
    / "sparc_bizyaev"
    / "merged_future"
)

REVIEW_SEP_ARCSEC = 60.0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    norm_map = {normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_colname(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def strip_leading_zeros_after_prefix(s: str) -> str:
    """
    Convert names like:
      UGC00128 -> UGC128
      NGC0891  -> NGC891
      IC02574  -> IC2574
    while leaving names without a simple prefix+digits form untouched.
    """
    m = re.match(r"^([A-Z]+)(0+)(\d.*)$", s)
    if not m:
        return s
    prefix, _, rest = m.groups()
    return prefix + rest


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().upper()
    s = s.replace(" ", "")
    s = s.replace("_", "")
    s = s.replace("-", "")
    s = strip_leading_zeros_after_prefix(s)
    return s


def to_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def angular_sep_arcsec(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    ra1 = np.deg2rad(ra1_deg)
    dec1 = np.deg2rad(dec1_deg)
    ra2 = np.deg2rad(ra2_deg)
    dec2 = np.deg2rad(dec2_deg)

    cosang = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )
    cosang = float(np.clip(cosang, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosang)) * 3600.0)


def main() -> int:
    ensure_dir(OUTPUT_DIR)

    for needed in [MATCH_2014_FILE, MATCH_2002_FILE, SPARC_INDEX_FILE]:
        if not needed.exists():
            raise SystemExit(f"Missing required file: {needed}")

    df2014 = pd.read_csv(MATCH_2014_FILE)
    df2002 = pd.read_csv(MATCH_2002_FILE)
    sparc = pd.read_csv(SPARC_INDEX_FILE)

    s_name_col = find_first_existing_column(sparc, ["galaxy_id", "Name"])
    s_ra_col = find_first_existing_column(sparc, ["RA_deg", "_RAJ2000", "RAJ2000"])
    s_dec_col = find_first_existing_column(sparc, ["DEC_deg", "_DEJ2000", "DEJ2000"])

    if s_name_col is None or s_ra_col is None or s_dec_col is None:
        raise SystemExit(
            f"Missing required SPARC columns. name={s_name_col}, ra={s_ra_col}, dec={s_dec_col}"
        )

    def prep(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
        out = df.copy()
        out["source_catalog"] = source_label

        name_col = find_first_existing_column(out, ["matched_sparc_name", "sparc_name", "Name"])
        idx_col = find_first_existing_column(out, ["matched_sparc_index"])
        if name_col is None:
            raise SystemExit(f"{source_label}: missing matched SPARC name column")

        out["matched_sparc_name_std"] = out[name_col].astype(str).str.strip()
        out["matched_sparc_name_norm"] = out["matched_sparc_name_std"].map(normalize_name)

        if idx_col is not None:
            out["matched_sparc_index_std"] = pd.to_numeric(out[idx_col], errors="coerce")
        else:
            out["matched_sparc_index_std"] = np.nan

        source_name_col = find_first_existing_column(
            out, ["biz_name", "preferred_name", "AName", "NED", "kautsch_name"]
        )
        out["source_object_name"] = (
            out[source_name_col].astype(str).str.strip() if source_name_col else ""
        )

        return out

    df2014 = prep(df2014, "Bizyaev_2014")
    df2002 = prep(df2002, "Bizyaev_2002")

    sparc_ref = sparc.copy()
    sparc_ref["sparc_name_std"] = sparc_ref[s_name_col].astype(str).str.strip()
    sparc_ref["sparc_name_norm"] = sparc_ref["sparc_name_std"].map(normalize_name)
    sparc_ref["RA_deg_std"] = to_float_series(sparc_ref[s_ra_col])
    sparc_ref["DEC_deg_std"] = to_float_series(sparc_ref[s_dec_col])

    left = df2014.merge(
        sparc_ref[["sparc_name_std", "sparc_name_norm", "RA_deg_std", "DEC_deg_std"]],
        left_on="matched_sparc_name_norm",
        right_on="sparc_name_norm",
        how="left",
    )
    left = left.rename(
        columns={
            "matched_sparc_name_std": "matched_sparc_name_2014",
            "matched_sparc_index_std": "matched_sparc_index_2014",
            "source_object_name": "source_object_name_2014",
            "RA_deg_std": "RA_deg_2014",
            "DEC_deg_std": "DEC_deg_2014",
        }
    )

    right = df2002.merge(
        sparc_ref[["sparc_name_std", "sparc_name_norm", "RA_deg_std", "DEC_deg_std"]],
        left_on="matched_sparc_name_norm",
        right_on="sparc_name_norm",
        how="left",
    )
    right = right.rename(
        columns={
            "matched_sparc_name_std": "matched_sparc_name_2002",
            "matched_sparc_index_std": "matched_sparc_index_2002",
            "source_object_name": "source_object_name_2002",
            "RA_deg_std": "RA_deg_2002",
            "DEC_deg_std": "DEC_deg_2002",
        }
    )

    overlap = left.merge(
        right,
        on=["matched_sparc_name_norm"],
        how="outer",
        suffixes=("_2014raw", "_2002raw"),
        indicator=True,
    )

    overlap["duplicate_status"] = "unique_to_one_catalog"
    both_mask = overlap["_merge"] == "both"
    overlap.loc[both_mask, "duplicate_status"] = "confirmed_duplicate_by_sparc_name"
    overlap["sparc_sep_arcsec"] = np.nan

    for idx, row in overlap.loc[both_mask].iterrows():
        ra1 = row.get("RA_deg_2014")
        dec1 = row.get("DEC_deg_2014")
        ra2 = row.get("RA_deg_2002")
        dec2 = row.get("DEC_deg_2002")
        if pd.notna(ra1) and pd.notna(dec1) and pd.notna(ra2) and pd.notna(dec2):
            sep = angular_sep_arcsec(float(ra1), float(dec1), float(ra2), float(dec2))
            overlap.at[idx, "sparc_sep_arcsec"] = sep
            if sep > REVIEW_SEP_ARCSEC:
                overlap.at[idx, "duplicate_status"] = "review_needed"

    confirmed_duplicates = overlap[
        overlap["duplicate_status"] == "confirmed_duplicate_by_sparc_name"
    ].copy()
    review_needed = overlap[overlap["duplicate_status"] == "review_needed"].copy()

    rows = []
    used_norm_names = set()

    for _, row in overlap.iterrows():
        norm_name = row.get("matched_sparc_name_norm", "")
        if not norm_name or norm_name in used_norm_names:
            continue

        if row["_merge"] == "both":
            rows.append(
                {
                    "matched_sparc_name": row.get("matched_sparc_name_2014") or row.get("matched_sparc_name_2002"),
                    "matched_sparc_name_norm": norm_name,
                    "included_source": "Bizyaev_2014",
                    "secondary_source": "Bizyaev_2002",
                    "duplicate_flag": True,
                    "duplicate_status": row.get("duplicate_status"),
                    "source_object_name_primary": row.get("source_object_name_2014"),
                    "source_object_name_secondary": row.get("source_object_name_2002"),
                    "matched_sep_deg_primary": row.get("matched_sep_deg_2014raw"),
                    "matched_sep_deg_secondary": row.get("matched_sep_deg_2002raw"),
                    "sparc_sep_arcsec_between_catalog_matches": row.get("sparc_sep_arcsec"),
                }
            )
        elif row["_merge"] == "left_only":
            rows.append(
                {
                    "matched_sparc_name": row.get("matched_sparc_name_2014"),
                    "matched_sparc_name_norm": norm_name,
                    "included_source": "Bizyaev_2014",
                    "secondary_source": "",
                    "duplicate_flag": False,
                    "duplicate_status": "unique_to_2014",
                    "source_object_name_primary": row.get("source_object_name_2014"),
                    "source_object_name_secondary": "",
                    "matched_sep_deg_primary": row.get("matched_sep_deg_2014raw"),
                    "matched_sep_deg_secondary": np.nan,
                    "sparc_sep_arcsec_between_catalog_matches": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "matched_sparc_name": row.get("matched_sparc_name_2002"),
                    "matched_sparc_name_norm": norm_name,
                    "included_source": "Bizyaev_2002",
                    "secondary_source": "",
                    "duplicate_flag": False,
                    "duplicate_status": "unique_to_2002",
                    "source_object_name_primary": row.get("source_object_name_2002"),
                    "source_object_name_secondary": "",
                    "matched_sep_deg_primary": row.get("matched_sep_deg_2002raw"),
                    "matched_sep_deg_secondary": np.nan,
                    "sparc_sep_arcsec_between_catalog_matches": np.nan,
                }
            )

        used_norm_names.add(norm_name)

    final_unique = pd.DataFrame(rows)
    if len(final_unique):
        final_unique = final_unique.sort_values(
            by=["included_source", "matched_sparc_name"], kind="stable"
        ).reset_index(drop=True)

    all_overlap_path = OUTPUT_DIR / "sparc_bizyaev_2014_vs_2002_overlap_all.csv"
    dup_path = OUTPUT_DIR / "sparc_bizyaev_confirmed_duplicates.csv"
    review_path = OUTPUT_DIR / "sparc_bizyaev_duplicate_review_needed.csv"
    final_unique_path = OUTPUT_DIR / "sparc_bizyaev_final_unique_sample.csv"
    summary_path = OUTPUT_DIR / "sparc_bizyaev_duplicate_merge_summary.json"

    overlap.to_csv(all_overlap_path, index=False)
    confirmed_duplicates.to_csv(dup_path, index=False)
    review_needed.to_csv(review_path, index=False)
    final_unique.to_csv(final_unique_path, index=False)

    summary = {
        "rows_2014_matched": int(len(df2014)),
        "rows_2002_matched": int(len(df2002)),
        "candidate_rows_total_before_duplicate_screening": int(len(df2014) + len(df2002)),
        "confirmed_duplicates_count": int(len(confirmed_duplicates)),
        "review_needed_count": int(len(review_needed)),
        "final_unique_sample_count": int(len(final_unique)),
        "unique_to_2014_count": int((final_unique["duplicate_status"] == "unique_to_2014").sum()) if len(final_unique) else 0,
        "unique_to_2002_count": int((final_unique["duplicate_status"] == "unique_to_2002").sum()) if len(final_unique) else 0,
        "duplicate_kept_as_2014_primary_count": int((final_unique["duplicate_status"] == "confirmed_duplicate_by_sparc_name").sum()) if len(final_unique) else 0,
        "review_sep_arcsec_threshold": REVIEW_SEP_ARCSEC,
        "input_2014": str(MATCH_2014_FILE),
        "input_2002": str(MATCH_2002_FILE),
        "input_sparc_index": str(SPARC_INDEX_FILE),
        "output_dir": str(OUTPUT_DIR),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] 2014 matched rows: {len(df2014)}")
    print(f"[OK] 2002 matched rows: {len(df2002)}")
    print(f"[OK] confirmed duplicates: {len(confirmed_duplicates)}")
    print(f"[OK] review needed: {len(review_needed)}")
    print(f"[OK] final unique sample: {len(final_unique)}")
    print(f"[OK] output folder: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
