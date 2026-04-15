#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Initialize raw FITS-check registry for WISE H II targets.

Placement
---------
data/raw/Validation of Structural Contrast Baseline/script/

Input
-----
data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/candidate_lists/

Expected input files
--------------------
Required:
- wise_hii_shortlist_known_group_simple_shell.csv
- wise_hii_shortlist_candidate_simple_shell.csv

Optional supportive files:
- wise_hii_sorted_by_radius.csv
- wise_hii_normalized_full.csv

Output
------
data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/fits/

Purpose
-------
This script does NOT download FITS files yet.
It prepares the raw workbench for FITS availability checking by:

1) merging shortlist targets,
2) deduplicating by wise_name,
3) preserving FITS query coordinates and search radius,
4) creating a FITS-check registry CSV,
5) creating an empty FITS-results CSV,
6) creating an empty FITS-source registry CSV.

The resulting files are meant for the next practical step:
manual or semi-manual FITS availability confirmation and later downloader linkage.

Windows example
---------------
python "data\raw\Validation of Structural Contrast Baseline\script\build_wise_hii_fits_check_registry_001.py"
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


def suggest_cutout_radius_arcmin(radius_arcsec):
    try:
        r = float(radius_arcsec)
    except Exception:
        return pd.NA
    # Conservative image cutout size: roughly 3x region radius, floor at 2 arcmin.
    arcmin = max((3.0 * r) / 60.0, 2.0)
    return round(arcmin, 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Root of topological_gravity_project")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()

    raw_base = root / "data" / "raw" / "Validation of Structural Contrast Baseline"
    input_dir = raw_base / "wise_hii_catalog" / "candidate_lists"
    output_dir = raw_base / "wise_hii_catalog" / "fits"
    output_dir.mkdir(parents=True, exist_ok=True)

    known_file = input_dir / "wise_hii_shortlist_known_group_simple_shell.csv"
    candidate_file = input_dir / "wise_hii_shortlist_candidate_simple_shell.csv"

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

    merged = pd.concat(
        [known_df[base_cols + ["input_group"]], cand_df[base_cols + ["input_group"]]],
        ignore_index=True
    )

    merged["wise_name"] = merged["wise_name"].astype(str).str.strip()
    merged = merged[merged["wise_name"] != ""].copy()

    dedup = (
        merged
        .sort_values(["priority_rank", "catalog_class", "wise_name"], ascending=[True, True, True])
        .drop_duplicates(subset=["wise_name"], keep="first")
        .reset_index(drop=True)
    )

    dedup["fits_check_status"] = "pending_check"
    dedup["fits_source_service"] = pd.NA
    dedup["fits_band"] = pd.NA
    dedup["fits_url"] = pd.NA
    dedup["fits_downloadable"] = pd.NA
    dedup["fits_local_path"] = pd.NA
    dedup["fits_local_verified"] = pd.NA
    dedup["fits_image_plane_verified"] = pd.NA
    dedup["fits_notes"] = pd.NA
    dedup["cutout_radius_arcmin_suggested"] = dedup["radius_arcsec"].apply(suggest_cutout_radius_arcmin)

    fits_check_registry = dedup[
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
            "cutout_radius_arcmin_suggested",
            "hii_region_name",
            "membership",
            "dist_kpc",
            "fits_check_status",
            "fits_notes",
        ]
    ].copy()

    fits_results_template = dedup[
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
            "cutout_radius_arcmin_suggested",
            "hii_region_name",
            "membership",
            "dist_kpc",
            "fits_source_service",
            "fits_band",
            "fits_url",
            "fits_downloadable",
            "fits_local_path",
            "fits_local_verified",
            "fits_image_plane_verified",
            "fits_check_status",
            "fits_notes",
        ]
    ].copy()

    fits_source_registry = pd.DataFrame(
        columns=[
            "source_key",
            "wise_name",
            "source_service",
            "survey_name",
            "band_name",
            "ra",
            "dec",
            "cutout_radius_arcmin",
            "product_url",
            "access_notes",
            "download_status",
            "local_path",
            "fits_verified",
            "image_plane_verified",
            "matching_notes",
        ]
    )

    manifest_lines = []
    manifest_lines.append("WISE H II FITS raw initialization")
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
    manifest_lines.append("- wise_hii_fits_check_registry.csv")
    manifest_lines.append("- wise_hii_fits_candidates_initial.csv")
    manifest_lines.append("- wise_hii_fits_source_registry.csv")
    manifest_lines.append("- wise_hii_fits_init_manifest.txt")
    manifest_lines.append("")
    manifest_lines.append("Operational note:")
    manifest_lines.append("This step initializes raw FITS-availability work.")
    manifest_lines.append("No FITS files are downloaded here.")
    manifest_lines.append("FITS availability, URL rows, and local verification should be filled after source confirmation.")

    fits_check_registry.to_csv(output_dir / "wise_hii_fits_check_registry.csv", index=False)
    fits_results_template.to_csv(output_dir / "wise_hii_fits_candidates_initial.csv", index=False)
    fits_source_registry.to_csv(output_dir / "wise_hii_fits_source_registry.csv", index=False)
    (output_dir / "wise_hii_fits_init_manifest.txt").write_text("\n".join(manifest_lines), encoding="utf-8")

    print("\n".join(manifest_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
