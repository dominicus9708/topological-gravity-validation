#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build initial raw mass candidate registry for WISE H II targets.

Placement
---------
data/raw/Validation of Structural Contrast Baseline/script/

Input
-----
data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/candidate_lists/

Expected input files
--------------------
- wise_hii_shortlist_known_group_simple_shell.csv
- wise_hii_shortlist_candidate_simple_shell.csv
Optional supportive files:
- wise_hii_sorted_by_radius.csv
- wise_hii_normalized_full.csv

Output
------
data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/mass/

Purpose
-------
This script does NOT infer stellar masses automatically.
It prepares the raw workbench for mass identification by:

1) merging shortlist targets,
2) deduplicating by wise_name,
3) preserving source-identification columns,
4) creating a mass-search registry CSV,
5) creating an empty mass-results CSV to be filled after literature search,
6) creating an empty literature-source registry CSV.

The resulting files are meant for the next practical step:
manual or semi-manual identification of stellar-mass-related papers and values.

Windows example
---------------
python "data\raw\Validation of Structural Contrast Baseline\script\build_wise_hii_mass_candidate_registry_001.py"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Root of topological_gravity_project")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()

    raw_base = root / "data" / "raw" / "Validation of Structural Contrast Baseline"
    input_dir = raw_base / "wise_hii_catalog" / "candidate_lists"
    output_dir = raw_base / "wise_hii_catalog" / "mass"
    output_dir.mkdir(parents=True, exist_ok=True)

    known_file = input_dir / "wise_hii_shortlist_known_group_simple_shell.csv"
    candidate_file = input_dir / "wise_hii_shortlist_candidate_simple_shell.csv"
    sorted_file = input_dir / "wise_hii_sorted_by_radius.csv"
    full_file = input_dir / "wise_hii_normalized_full.csv"

    if not known_file.exists():
        raise SystemExit(f"Missing required input: {known_file}")
    if not candidate_file.exists():
        raise SystemExit(f"Missing required input: {candidate_file}")

    known_df = load_csv(known_file)
    cand_df = load_csv(candidate_file)

    base_cols = [
        "wise_name",
        "catalog_class",
        "glon",
        "glat",
        "ra",
        "dec",
        "radius_arcsec",
        "hii_region_name",
        "membership",
        "dist_kpc",
        "w3_jy",
        "w4_jy",
        "mips24_jy",
        "priority_rank",
        "size_bucket",
    ]

    known_df = ensure_columns(known_df, base_cols).copy()
    cand_df = ensure_columns(cand_df, base_cols).copy()

    known_df["input_group"] = "known_group_shortlist"
    cand_df["input_group"] = "candidate_shortlist"

    merged = pd.concat([known_df[base_cols + ["input_group"]], cand_df[base_cols + ["input_group"]]],
                       ignore_index=True)

    merged["wise_name"] = merged["wise_name"].astype(str).str.strip()
    merged = merged[merged["wise_name"] != ""].copy()

    merged["mass_search_status"] = "pending_search"
    merged["mass_value_msun"] = pd.NA
    merged["mass_value_type"] = pd.NA
    merged["mass_value_notes"] = pd.NA
    merged["mass_source_key"] = pd.NA
    merged["mass_source_match_method"] = pd.NA
    merged["radio_proxy_available"] = pd.NA
    merged["log_nly"] = pd.NA
    merged["spectral_type"] = pd.NA
    merged["fits_download_status"] = "not_checked"
    merged["fits_local_path"] = pd.NA
    merged["notes"] = pd.NA

    dedup = (
        merged
        .sort_values(["priority_rank", "catalog_class", "wise_name"], ascending=[True, True, True])
        .drop_duplicates(subset=["wise_name"], keep="first")
        .reset_index(drop=True)
    )

    dedup["search_query_primary"] = dedup["wise_name"]
    dedup["search_query_secondary"] = dedup["hii_region_name"]
    dedup["search_query_membership"] = dedup["membership"]

    mass_search_registry = dedup[
        [
            "wise_name",
            "catalog_class",
            "input_group",
            "priority_rank",
            "size_bucket",
            "glon",
            "glat",
            "ra",
            "dec",
            "radius_arcsec",
            "hii_region_name",
            "membership",
            "dist_kpc",
            "search_query_primary",
            "search_query_secondary",
            "search_query_membership",
            "mass_search_status",
            "notes",
        ]
    ].copy()

    mass_results_template = dedup[
        [
            "wise_name",
            "catalog_class",
            "input_group",
            "priority_rank",
            "glon",
            "glat",
            "ra",
            "dec",
            "radius_arcsec",
            "hii_region_name",
            "membership",
            "dist_kpc",
            "mass_value_msun",
            "mass_value_type",
            "mass_value_notes",
            "mass_source_key",
            "mass_source_match_method",
            "radio_proxy_available",
            "log_nly",
            "spectral_type",
            "fits_download_status",
            "fits_local_path",
            "notes",
        ]
    ].copy()

    source_registry = pd.DataFrame(
        columns=[
            "source_key",
            "wise_name",
            "matched_object_name",
            "paper_title",
            "authors",
            "year",
            "journal",
            "doi",
            "ads_url",
            "arxiv_url",
            "source_type",
            "mass_field_description",
            "mass_value_msun",
            "mass_range_lower_msun",
            "mass_range_upper_msun",
            "log_nly",
            "spectral_type",
            "distance_kpc",
            "matching_notes",
        ]
    )

    manifest_lines = []
    manifest_lines.append("WISE H II mass raw initialization")
    manifest_lines.append("")
    manifest_lines.append(f"Project root: {root}")
    manifest_lines.append(f"Input directory: {input_dir}")
    manifest_lines.append(f"Output directory: {output_dir}")
    manifest_lines.append("")
    manifest_lines.append(f"Known shortlist rows: {len(known_df)}")
    manifest_lines.append(f"Candidate shortlist rows: {len(cand_df)}")
    manifest_lines.append(f"Merged rows before dedup: {len(merged)}")
    manifest_lines.append(f"Unique wise_name rows: {len(dedup)}")
    manifest_lines.append("")
    manifest_lines.append("Generated files:")
    manifest_lines.append("- wise_hii_mass_search_registry.csv")
    manifest_lines.append("- wise_hii_mass_candidates_initial.csv")
    manifest_lines.append("- wise_hii_mass_source_registry.csv")
    manifest_lines.append("- wise_hii_mass_init_manifest.txt")
    manifest_lines.append("")
    manifest_lines.append("Operational note:")
    manifest_lines.append("This step initializes raw mass-identification work.")
    manifest_lines.append("No stellar masses are inferred here.")
    manifest_lines.append("Mass values and source rows should be filled only after literature confirmation.")

    mass_search_registry.to_csv(output_dir / "wise_hii_mass_search_registry.csv", index=False)
    mass_results_template.to_csv(output_dir / "wise_hii_mass_candidates_initial.csv", index=False)
    source_registry.to_csv(output_dir / "wise_hii_mass_source_registry.csv", index=False)
    (output_dir / "wise_hii_mass_init_manifest.txt").write_text("\n".join(manifest_lines), encoding="utf-8")

    print("\n".join(manifest_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
