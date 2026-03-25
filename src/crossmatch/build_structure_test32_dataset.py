#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a dedicated test dataset for the final 32 structure-linked SPARC galaxies.

Goal
----
Take the final unique sample produced by the duplicate-merge step and create a new
derived/crossmatch folder for the next structure-vs-rotation test.

Design principle
----------------
Use SPARC galaxy data as the base table, then append columns that do NOT exist in SPARC
(thickness / structure information from Bizyaev 2014 or Bizyaev 2002).

Outputs
-------
data/derived/crossmatch/sparc_structure_test32/
    structure_test32_summary.json
    structure_test32_master_catalog.csv
    missing_source_files.csv
    per_galaxy/
        <GALAXY>_structure_enriched.csv

What each per-galaxy file contains
----------------------------------
- all columns from the corresponding SPARC normalized CSV
- appended metadata columns describing:
  - matched SPARC name
  - source catalog used (Bizyaev_2014 or Bizyaev_2002)
  - secondary source if duplicate-merged
  - source object names
  - structure-related columns from the chosen external catalog row
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


PROJECT_ROOT = Path(".")

FINAL_UNIQUE_FILE = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "crossmatch"
    / "sparc_bizyaev"
    / "merged_future"
    / "sparc_bizyaev_final_unique_sample.csv"
)

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

SPARC_NORMALIZED_DIR = PROJECT_ROOT / "data" / "processed" / "sparc_normalized"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "crossmatch"
    / "sparc_structure_test32"
)

PER_GALAXY_DIR = OUTPUT_DIR / "per_galaxy"


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


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().upper()
    s = s.replace(" ", "")
    s = s.replace("_", "")
    s = s.replace("-", "")
    m = re.match(r"^([A-Z]+)0+(\d.*)$", s)
    if m:
        s = m.group(1) + m.group(2)
    return s


def to_safe_filename(name: str) -> str:
    s = str(name).strip().replace(" ", "")
    s = s.replace("/", "_").replace("\\", "_")
    return s


def build_match_lookup(df: pd.DataFrame, source_label: str) -> dict[str, dict]:
    """
    Build lookup keyed by normalized matched SPARC name.
    Keep the first row for each normalized name.
    """
    if df.empty:
        return {}

    matched_name_col = find_first_existing_column(df, ["matched_sparc_name", "Name", "galaxy_id"])
    if matched_name_col is None:
        raise SystemExit(f"{source_label}: could not find matched SPARC name column")

    df = df.copy()
    df["__matched_norm__"] = df[matched_name_col].map(normalize_name)
    out: dict[str, dict] = {}

    for _, row in df.iterrows():
        key = row["__matched_norm__"]
        if not key or key in out:
            continue
        payload = row.to_dict()
        payload["__source_catalog__"] = source_label
        out[key] = payload

    return out


def prefix_external_columns(payload: dict, prefix: str, excluded: set[str]) -> dict:
    out = {}
    for key, value in payload.items():
        if key in excluded:
            continue
        out[f"{prefix}{key}"] = value
    return out


def main() -> int:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PER_GALAXY_DIR)

    for needed in [FINAL_UNIQUE_FILE, MATCH_2014_FILE, MATCH_2002_FILE, SPARC_NORMALIZED_DIR]:
        if not needed.exists():
            raise SystemExit(f"Missing required input: {needed}")

    final_unique = pd.read_csv(FINAL_UNIQUE_FILE)
    match2014 = pd.read_csv(MATCH_2014_FILE)
    match2002 = pd.read_csv(MATCH_2002_FILE)

    lookup2014 = build_match_lookup(match2014, "Bizyaev_2014")
    lookup2002 = build_match_lookup(match2002, "Bizyaev_2002")

    if final_unique.empty:
        raise SystemExit("Final unique sample is empty. Nothing to export.")

    master_rows = []
    missing_rows = []

    excluded_external_cols = {
        "__matched_norm__",
    }

    included_source_col = find_first_existing_column(final_unique, ["included_source"])
    secondary_source_col = find_first_existing_column(final_unique, ["secondary_source"])
    sparc_name_col = find_first_existing_column(final_unique, ["matched_sparc_name"])
    sparc_norm_col = find_first_existing_column(final_unique, ["matched_sparc_name_norm"])
    duplicate_flag_col = find_first_existing_column(final_unique, ["duplicate_flag"])
    duplicate_status_col = find_first_existing_column(final_unique, ["duplicate_status"])
    source_primary_name_col = find_first_existing_column(final_unique, ["source_object_name_primary"])
    source_secondary_name_col = find_first_existing_column(final_unique, ["source_object_name_secondary"])

    for _, row in final_unique.iterrows():
        sparc_name = str(row[sparc_name_col]).strip()
        sparc_norm = normalize_name(row[sparc_norm_col] if sparc_norm_col else sparc_name)
        included_source = str(row[included_source_col]).strip() if included_source_col else ""
        secondary_source = str(row[secondary_source_col]).strip() if secondary_source_col else ""

        normalized_file = SPARC_NORMALIZED_DIR / f"{sparc_name}_normalized.csv"
        if not normalized_file.exists():
            missing_rows.append(
                {
                    "matched_sparc_name": sparc_name,
                    "expected_file": str(normalized_file),
                    "included_source": included_source,
                    "reason": "normalized_sparc_file_missing",
                }
            )
            continue

        base_df = pd.read_csv(normalized_file)

        external_payload = {}
        if included_source == "Bizyaev_2014":
            external_payload = lookup2014.get(sparc_norm, {})
        elif included_source == "Bizyaev_2002":
            external_payload = lookup2002.get(sparc_norm, {})

        if not external_payload:
            missing_rows.append(
                {
                    "matched_sparc_name": sparc_name,
                    "expected_file": str(normalized_file),
                    "included_source": included_source,
                    "reason": "external_match_row_missing",
                }
            )
            continue

        chosen_prefix = "structure_2014_" if included_source == "Bizyaev_2014" else "structure_2002_"
        ext_cols = prefix_external_columns(external_payload, chosen_prefix, excluded_external_cols)

        # metadata columns to append to every row of the SPARC table
        metadata = {
            "matched_sparc_name": sparc_name,
            "matched_sparc_name_norm": sparc_norm,
            "included_source_catalog": included_source,
            "secondary_source_catalog": secondary_source,
            "duplicate_flag": row[duplicate_flag_col] if duplicate_flag_col else "",
            "duplicate_status": row[duplicate_status_col] if duplicate_status_col else "",
            "source_object_name_primary": row[source_primary_name_col] if source_primary_name_col else "",
            "source_object_name_secondary": row[source_secondary_name_col] if source_secondary_name_col else "",
        }
        metadata.update(ext_cols)

        enriched = base_df.copy()
        for key, value in metadata.items():
            enriched[key] = value

        out_name = f"{to_safe_filename(sparc_name)}_structure_enriched.csv"
        out_path = PER_GALAXY_DIR / out_name
        enriched.to_csv(out_path, index=False)

        master_row = {
            "matched_sparc_name": sparc_name,
            "matched_sparc_name_norm": sparc_norm,
            "included_source_catalog": included_source,
            "secondary_source_catalog": secondary_source,
            "duplicate_flag": row[duplicate_flag_col] if duplicate_flag_col else "",
            "duplicate_status": row[duplicate_status_col] if duplicate_status_col else "",
            "source_object_name_primary": row[source_primary_name_col] if source_primary_name_col else "",
            "source_object_name_secondary": row[source_secondary_name_col] if source_secondary_name_col else "",
            "normalized_sparc_file": str(normalized_file),
            "output_enriched_file": str(out_path),
            "base_row_count": int(len(base_df)),
            "exported_column_count": int(len(enriched.columns)),
        }

        # Keep a compact subset of structure values in the master catalog
        preferred_structure_cols = [
            "z0_over_h",
            "h_over_z0",
            "scale_height",
            "scale_length",
            "axial_ratio_r",
            "edge_likelihood_r",
            "matched_sep_deg",
            "match_reason",
        ]
        for col in preferred_structure_cols:
            if col in external_payload:
                master_row[f"external_{col}"] = external_payload[col]

        master_rows.append(master_row)

    master_df = pd.DataFrame(master_rows).sort_values(
        by=["included_source_catalog", "matched_sparc_name"], kind="stable"
    ).reset_index(drop=True)

    missing_df = pd.DataFrame(missing_rows)

    master_path = OUTPUT_DIR / "structure_test32_master_catalog.csv"
    missing_path = OUTPUT_DIR / "missing_source_files.csv"
    summary_path = OUTPUT_DIR / "structure_test32_summary.json"

    master_df.to_csv(master_path, index=False)
    missing_df.to_csv(missing_path, index=False)

    summary = {
        "final_unique_input_rows": int(len(final_unique)),
        "exported_galaxy_files": int(len(master_df)),
        "missing_or_skipped_rows": int(len(missing_df)),
        "included_source_counts": (
            {str(k): int(v) for k, v in master_df["included_source_catalog"].value_counts().to_dict().items()}
            if len(master_df) else {}
        ),
        "duplicate_status_counts": (
            {str(k): int(v) for k, v in master_df["duplicate_status"].value_counts().to_dict().items()}
            if len(master_df) else {}
        ),
        "input_final_unique_file": str(FINAL_UNIQUE_FILE),
        "input_match_2014_file": str(MATCH_2014_FILE),
        "input_match_2002_file": str(MATCH_2002_FILE),
        "input_sparc_normalized_dir": str(SPARC_NORMALIZED_DIR),
        "output_dir": str(OUTPUT_DIR),
        "per_galaxy_dir": str(PER_GALAXY_DIR),
        "master_catalog_file": str(master_path),
        "missing_rows_file": str(missing_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] final_unique_input_rows={len(final_unique)}")
    print(f"[OK] exported_galaxy_files={len(master_df)}")
    print(f"[OK] missing_or_skipped_rows={len(missing_df)}")
    print(f"[OK] output_dir={OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
