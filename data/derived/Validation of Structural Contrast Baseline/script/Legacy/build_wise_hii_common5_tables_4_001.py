#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a fast-priority common comparison shortlist of 5 targets
for both standard and topological validation.

Version intent
--------------
This script is intended as the next step after tables_3-like staging work.
It creates a tables_4-style candidate selection output.

Placement
---------
data/derived/Validation of Structural Contrast Baseline/script/

Inputs
------
Mass side preference:
1) data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/mass/wise_hii_mass_dual_candidates_working.csv
2) fallback: .../wise_hii_mass_dual_candidates_initial.csv

Mass search support:
- data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/mass/wise_hii_mass_dual_search_input.csv

FITS side preference:
1) data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/fits/wise_hii_fits_candidates_working.csv
2) fallback: .../wise_hii_fits_candidates_initial.csv

Optional reference:
- data/derived/Validation of Structural Contrast Baseline/wise_hii_catalog/tables_n/wise_hii_tables_n_full_merged.csv
  or prior staging tables if present, but not required.

Outputs
-------
data/derived/Validation of Structural Contrast Baseline/wise_hii_catalog/tables_4/

Core purpose
------------
- rank the 2506 shared targets
- prioritize objects likely to support BOTH:
  * standard comparison
  * topological comparison
- choose a fast common-comparison shortlist of 5 targets

Important
---------
This script does NOT create final input.
It creates a focused common-comparison candidate set for accelerated source gathering.

Windows example
---------------
python "data\\derived\\Validation of Structural Contrast Baseline\\script\\build_wise_hii_common5_tables_4_001.py"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required input: {path}")
    return pd.read_csv(path)


def read_csv_prefer(primary: Path, fallback: Path) -> tuple[pd.DataFrame, str]:
    if primary.exists():
        return pd.read_csv(primary), str(primary)
    if fallback.exists():
        return pd.read_csv(fallback), str(fallback)
    raise SystemExit(f"Missing both preferred and fallback inputs:\n- {primary}\n- {fallback}")


def coalesce(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.where(a.notna(), b)


def has_text(v) -> bool:
    return pd.notna(v) and str(v).strip() != ""


def truthy(v) -> bool:
    if pd.isna(v):
        return False
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {
        "1", "true", "t", "yes", "y", "verified", "downloaded", "ok", "ready",
        "api_xml_ready", "source_row_applied", "yes"
    }


def normalize_priority(v):
    try:
        return float(v)
    except Exception:
        return 999999.0


def score_row(row) -> dict:
    # Mass readiness
    mass_direct = 1 if has_text(row.get("mass_value_msun")) else 0
    mass_source = 1 if has_text(row.get("mass_source_key")) else 0
    proxy_kind = str(row.get("proxy_kind", "")).strip().lower()
    proxy_log_nly = 1 if has_text(row.get("log_nly")) or proxy_kind == "log_nly" else 0
    proxy_spt = 1 if has_text(row.get("spectral_type")) or proxy_kind == "spectral_type" else 0
    proxy_radio = 1 if truthy(row.get("radio_proxy_available")) or proxy_kind == "radio_continuum" else 0
    ionizing_ref = 1 if has_text(row.get("ionizing_source_reference")) else 0

    # FITS readiness
    fits_url = 1 if has_text(row.get("fits_url")) else 0
    fits_service = 1 if has_text(row.get("fits_source_service")) else 0
    fits_downloadable = 1 if truthy(row.get("fits_downloadable")) else 0
    fits_verified = 1 if truthy(row.get("fits_local_verified")) or truthy(row.get("fits_image_plane_verified")) else 0

    # Object descriptiveness / likely literature match quality
    has_region = 1 if has_text(row.get("hii_region_name")) else 0
    has_membership = 1 if has_text(row.get("membership")) else 0
    membership_len = len(str(row.get("membership")).strip()) if has_membership else 0
    has_distance = 1 if has_text(row.get("dist_kpc")) else 0
    known_group = 1 if str(row.get("input_group", "")).strip().lower() == "known_group_shortlist" else 0
    high_class = 1 if str(row.get("catalog_class", "")).strip().upper() in {"K", "G"} else 0

    # Search-track hints from dual search input
    search_track_direct = 1 if truthy(row.get("search_track_direct")) else 0
    search_track_proxy = 1 if truthy(row.get("search_track_proxy")) else 0
    priority_rank = normalize_priority(row.get("priority_rank"))

    # Composite score
    # Strongly reward objects that are:
    # - standard-usable now (fits ready-ish)
    # - likely to become topological-usable quickly (source/proxy hints)
    composite = (
        40 * fits_downloadable +
        30 * fits_url +
        10 * fits_service +
        60 * mass_direct +
        45 * mass_source +
        30 * proxy_log_nly +
        24 * proxy_spt +
        20 * proxy_radio +
        12 * ionizing_ref +
        12 * has_region +
        10 * has_membership +
        6 * has_distance +
        8 * known_group +
        6 * high_class +
        4 * search_track_direct +
        4 * search_track_proxy +
        min(membership_len, 60) * 0.1
    )

    return {
        "score_total": composite,
        "score_fits_ready": 40 * fits_downloadable + 30 * fits_url + 10 * fits_service + 15 * fits_verified,
        "score_mass_ready": 60 * mass_direct + 45 * mass_source,
        "score_proxy_ready": 30 * proxy_log_nly + 24 * proxy_spt + 20 * proxy_radio + 12 * ionizing_ref,
        "flag_mass_direct": mass_direct,
        "flag_mass_source": mass_source,
        "flag_proxy_log_nly": proxy_log_nly,
        "flag_proxy_spectral_type": proxy_spt,
        "flag_proxy_radio": proxy_radio,
        "flag_ionizing_ref": ionizing_ref,
        "flag_fits_url": fits_url,
        "flag_fits_downloadable": fits_downloadable,
        "flag_fits_verified": fits_verified,
        "flag_known_group": known_group,
        "flag_has_region_name": has_region,
        "flag_has_membership": has_membership,
        "priority_rank_numeric": priority_rank,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Root of topological_gravity_project")
    parser.add_argument("--top-k", default=5, type=int, help="Number of common comparison candidates to select")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()

    raw_base = root / "data" / "raw" / "Validation of Structural Contrast Baseline" / "wise_hii_catalog"
    mass_dir = raw_base / "mass"
    fits_dir = raw_base / "fits"

    out_dir = root / "data" / "derived" / "Validation of Structural Contrast Baseline" / "wise_hii_catalog" / "tables_4"
    out_dir.mkdir(parents=True, exist_ok=True)

    mass_df, mass_used = read_csv_prefer(
        mass_dir / "wise_hii_mass_dual_candidates_working.csv",
        mass_dir / "wise_hii_mass_dual_candidates_initial.csv",
    )
    search_df = read_csv_required(mass_dir / "wise_hii_mass_dual_search_input.csv")
    fits_df, fits_used = read_csv_prefer(
        fits_dir / "wise_hii_fits_candidates_working.csv",
        fits_dir / "wise_hii_fits_candidates_initial.csv",
    )

    for df in [mass_df, search_df, fits_df]:
        df["wise_name"] = df["wise_name"].astype(str).str.strip()

    # Merge mass + search input
    mass = pd.merge(
        search_df,
        mass_df,
        on="wise_name",
        how="outer",
        suffixes=("_search", "_mass"),
    )

    # Coalesce shared identity fields
    for col in ["catalog_class", "input_group", "priority_rank", "glon", "glat", "ra", "dec",
                "radius_arcsec", "hii_region_name", "membership", "dist_kpc"]:
        a = f"{col}_search"
        b = f"{col}_mass"
        if a in mass.columns or b in mass.columns:
            mass[col] = coalesce(mass.get(a), mass.get(b))

    keep_mass_cols = [
        "wise_name", "catalog_class", "input_group", "priority_rank", "glon", "glat", "ra", "dec",
        "radius_arcsec", "hii_region_name", "membership", "dist_kpc",
        "search_track_direct", "search_track_proxy",
        "direct_query_primary", "direct_query_secondary",
        "proxy_query_log_nly", "proxy_query_spectral_type", "proxy_query_radio",
        "proxy_priority_type", "proxy_candidate_type", "recommended_bridge_field", "review_status",
        "mass_source_key", "mass_value_msun", "mass_value_type", "mass_value_notes",
        "proxy_kind", "proxy_value", "proxy_value_unit", "log_nly", "spectral_type",
        "radio_proxy_available", "ionizing_source_reference", "mass_search_status", "notes"
    ]
    mass = mass[[c for c in keep_mass_cols if c in mass.columns]].copy()

    # Merge with fits
    merged = pd.merge(
        mass,
        fits_df,
        on="wise_name",
        how="outer",
        suffixes=("_mass", "_fits")
    )

    for col in ["catalog_class", "input_group", "priority_rank", "glon", "glat", "ra", "dec",
                "radius_arcsec", "hii_region_name", "membership", "dist_kpc"]:
        a = f"{col}_mass"
        b = f"{col}_fits"
        if a in merged.columns or b in merged.columns:
            merged[col] = coalesce(merged.get(a), merged.get(b))

    # Basic shared-target filter
    merged["shared_basic_geometry"] = (
        merged["wise_name"].notna()
        & merged["ra"].notna()
        & merged["dec"].notna()
        & merged["radius_arcsec"].notna()
    )

    score_records = [score_row(row) for _, row in merged.iterrows()]
    score_df = pd.DataFrame(score_records)
    merged = pd.concat([merged.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)

    # Ranking
    merged = merged.sort_values(
        ["score_total", "score_fits_ready", "score_proxy_ready", "score_mass_ready", "flag_known_group", "flag_has_region_name", "flag_has_membership", "priority_rank_numeric"],
        ascending=[False, False, False, False, False, False, False, True]
    ).reset_index(drop=True)

    merged["tables_4_role"] = "ranked_candidate_pool"
    merged["common_comparison_shortlist"] = False
    merged.loc[merged.index < int(args.top_k), "common_comparison_shortlist"] = True

    topk = merged[merged["common_comparison_shortlist"]].copy()

    # Save outputs
    merged.to_csv(out_dir / "wise_hii_tables_4_ranked_pool.csv", index=False)
    topk.to_csv(out_dir / "wise_hii_tables_4_common5_shortlist.csv", index=False)

    summary = pd.DataFrame([
        {"metric": "rows_total", "value": int(len(merged))},
        {"metric": "shared_basic_geometry", "value": int(merged["shared_basic_geometry"].sum())},
        {"metric": "flag_fits_downloadable", "value": int(merged["flag_fits_downloadable"].sum())},
        {"metric": "flag_mass_source", "value": int(merged["flag_mass_source"].sum())},
        {"metric": "flag_proxy_log_nly", "value": int(merged["flag_proxy_log_nly"].sum())},
        {"metric": "flag_proxy_spectral_type", "value": int(merged["flag_proxy_spectral_type"].sum())},
        {"metric": "flag_proxy_radio", "value": int(merged["flag_proxy_radio"].sum())},
        {"metric": "common_comparison_shortlist", "value": int(len(topk))},
    ])
    summary.to_csv(out_dir / "wise_hii_tables_4_summary.csv", index=False)

    manifest_lines = [
        "WISE H II tables_4 common-comparison candidate selection",
        "",
        f"Project root: {root}",
        f"Mass candidate source used: {mass_used}",
        f"FITS candidate source used: {fits_used}",
        f"Output dir: {out_dir}",
        "",
        "Purpose:",
        "Select a fast common-comparison shortlist for fair standard/topological comparison.",
        "This output is still NOT final input.",
        "",
        f"Total ranked rows: {len(merged)}",
        f"Top shortlist size: {len(topk)}",
        "",
        "Interpretation:",
        "- ranked_pool: all shared candidates ranked by fast readiness potential",
        "- common5_shortlist: accelerated common comparison target set",
        "",
        "Operational note:",
        "Use this shortlist to focus direct/proxy source gathering on the same 5 targets for both standard and topological tracks.",
    ]
    (out_dir / "wise_hii_tables_4_manifest.txt").write_text("\n".join(manifest_lines), encoding="utf-8")

    print("\n".join(manifest_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
