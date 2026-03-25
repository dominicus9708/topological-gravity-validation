#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a SPARC galaxy index WITH coordinates by merging:
1) existing SPARC galaxy index
2) a user-supplied coordinate table

Purpose
-------
Kautsch 2006 and some other edge-on catalogs rely heavily on coordinate-based matching.
The current project SPARC index appears to lack RA/DEC columns, so this script builds
a new coordinate-enriched SPARC index.

Recommended input for the coordinate table
------------------------------------------
CSV with at least:
- galaxy_id   (or name / Galaxy)
- RA_deg      (or RAJ2000 / ra_deg / RA)
- DEC_deg     (or DEJ2000 / dec_deg / DEC)

This script is intentionally conservative:
- exact/normalized name matching only
- no web lookup
- no automatic guessing beyond name normalization

Outputs
-------
- data/derived/crossmatch/sparc_diskmass_S4G/sparc_galaxy_index_with_coords.csv
- data/derived/crossmatch/sparc_diskmass_S4G/sparc_coordinate_merge_report.json
- data/derived/crossmatch/sparc_diskmass_S4G/sparc_coordinate_unmatched.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


PROJECT_ROOT = Path(".")
SPARC_INDEX_FILE = PROJECT_ROOT / "data" / "derived" / "crossmatch" / "sparc_diskmass_S4G" / "sparc_galaxy_index.csv"
# User should prepare this file separately.
SPARC_COORD_SOURCE_FILE = PROJECT_ROOT / "data" / "metadata" / "sparc_coordinates.csv"

OUTPUT_FILE = PROJECT_ROOT / "data" / "derived" / "crossmatch" / "sparc_diskmass_S4G" / "sparc_galaxy_index_with_coords.csv"
REPORT_FILE = PROJECT_ROOT / "data" / "derived" / "crossmatch" / "sparc_diskmass_S4G" / "sparc_coordinate_merge_report.json"
UNMATCHED_FILE = PROJECT_ROOT / "data" / "derived" / "crossmatch" / "sparc_diskmass_S4G" / "sparc_coordinate_unmatched.csv"


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    norm_map = {normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_colname(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().upper()
    s = s.replace(" ", "")
    s = s.replace("_", "")
    return s


def extract_catalog_token(name: object) -> str:
    s = normalize_name(name)
    if not s:
        return ""

    # Traditional catalog names
    m = re.match(r"^(NGC|UGC|IC|ESO|PGC|DDO|UGCA|F|KK|D)([-A-Z0-9]+)$", s)
    if m:
        prefix, rest = m.groups()
        if rest.isdigit():
            rest = str(int(rest))
        return f"{prefix}{rest}"

    return s


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def pick_name_column(df: pd.DataFrame) -> str:
    col = find_first_existing_column(df, ["galaxy_id", "Galaxy", "name", "Name", "filename", "id"])
    if col is None:
        raise SystemExit(f"Could not find a name column in: {list(df.columns)}")
    return col


def pick_ra_dec_columns(df: pd.DataFrame) -> tuple[str, str]:
    ra_col = find_first_existing_column(df, ["RA_deg", "RAJ2000_deg", "RAJ2000", "ra_deg", "RA", "ra"])
    dec_col = find_first_existing_column(df, ["DEC_deg", "DEJ2000_deg", "DEJ2000", "dec_deg", "DEC", "dec"])
    if ra_col is None or dec_col is None:
        raise SystemExit(
            "Coordinate source file must contain RA/DEC columns. "
            f"Detected columns={list(df.columns)}, picked ra={ra_col}, dec={dec_col}"
        )
    return ra_col, dec_col


def main() -> int:
    if not SPARC_INDEX_FILE.exists():
        raise SystemExit(f"Missing SPARC index file: {SPARC_INDEX_FILE}")
    if not SPARC_COORD_SOURCE_FILE.exists():
        raise SystemExit(
            "Missing coordinate source file.\n"
            f"Expected: {SPARC_COORD_SOURCE_FILE}\n"
            "Create this CSV first, then run again."
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    sparc = pd.read_csv(SPARC_INDEX_FILE)
    coord = pd.read_csv(SPARC_COORD_SOURCE_FILE)

    sparc_name_col = pick_name_column(sparc)
    coord_name_col = pick_name_column(coord)
    coord_ra_col, coord_dec_col = pick_ra_dec_columns(coord)

    sparc = sparc.copy()
    coord = coord.copy()

    sparc["__name_norm__"] = sparc[sparc_name_col].map(normalize_name)
    sparc["__token__"] = sparc[sparc_name_col].map(extract_catalog_token)

    coord["__name_norm__"] = coord[coord_name_col].map(normalize_name)
    coord["__token__"] = coord[coord_name_col].map(extract_catalog_token)
    coord["__ra__"] = to_float(coord[coord_ra_col])
    coord["__dec__"] = to_float(coord[coord_dec_col])

    # Build lookup maps
    coord_by_name = {}
    coord_by_token = {}

    for _, row in coord.iterrows():
        name_norm = row["__name_norm__"]
        token = row["__token__"]
        payload = (row["__ra__"], row["__dec__"], row.get(coord_name_col))

        if name_norm and name_norm not in coord_by_name:
            coord_by_name[name_norm] = payload
        if token and token not in coord_by_token:
            coord_by_token[token] = payload

    merged_ra = []
    merged_dec = []
    matched_source_name = []
    matched_reason = []

    for _, row in sparc.iterrows():
        name_norm = row["__name_norm__"]
        token = row["__token__"]

        ra = None
        dec = None
        src_name = None
        reason = "unmatched"

        if name_norm in coord_by_name:
            ra, dec, src_name = coord_by_name[name_norm]
            reason = "name_exact"
        elif token in coord_by_token:
            ra, dec, src_name = coord_by_token[token]
            reason = "catalog_exact"

        merged_ra.append(ra)
        merged_dec.append(dec)
        matched_source_name.append(src_name)
        matched_reason.append(reason)

    sparc["RA_deg"] = merged_ra
    sparc["DEC_deg"] = merged_dec
    sparc["coord_source_name"] = matched_source_name
    sparc["coord_match_reason"] = matched_reason

    sparc.to_csv(OUTPUT_FILE, index=False)

    unmatched = sparc[sparc["coord_match_reason"] == "unmatched"].copy()
    unmatched.to_csv(UNMATCHED_FILE, index=False)

    report = {
        "sparc_rows_total": int(len(sparc)),
        "coord_rows_total": int(len(coord)),
        "matched_rows": int((sparc["coord_match_reason"] != "unmatched").sum()),
        "unmatched_rows": int((sparc["coord_match_reason"] == "unmatched").sum()),
        "match_reason_counts": {
            str(k): int(v) for k, v in sparc["coord_match_reason"].value_counts(dropna=False).to_dict().items()
        },
        "detected_columns": {
            "sparc_name": sparc_name_col,
            "coord_name": coord_name_col,
            "coord_ra": coord_ra_col,
            "coord_dec": coord_dec_col,
        },
        "output_file": str(OUTPUT_FILE),
        "unmatched_file": str(UNMATCHED_FILE),
        "coord_source_file": str(SPARC_COORD_SOURCE_FILE),
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] matched={report['matched_rows']} unmatched={report['unmatched_rows']}")
    print(f"[OK] output file: {OUTPUT_FILE}")
    print(f"[OK] report file: {REPORT_FILE}")
    print(f"[OK] unmatched file: {UNMATCHED_FILE}")
    print()
    print("Next step:")
    print(f"1) Prepare {SPARC_COORD_SOURCE_FILE}")
    print("2) Re-run this script")
    print("3) Point Kautsch/Bizyaev coordinate crossmatch scripts to sparc_galaxy_index_with_coords.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
