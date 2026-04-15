#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Initialize dual-track mass search workspace for WISE H II targets.

Goal
----
Prepare raw-layer inputs so that direct-mass search and proxy search
(log Nly / spectral type / radio continuum / ionizing-source literature)
can be tracked simultaneously from the beginning.

Recommended placement
---------------------
data/raw/Validation of Structural Contrast Baseline/script/

Expected existing input
-----------------------
data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/mass/wise_hii_mass_search_registry.csv

Outputs
-------
data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/mass/
- wise_hii_mass_dual_search_input.csv
- wise_hii_mass_dual_source_registry.csv
- wise_hii_mass_dual_candidates_initial.csv
- wise_hii_mass_dual_init_manifest.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required input: {path}")
    return pd.read_csv(path)


def ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Root of topological_gravity_project")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    mass_dir = root / "data" / "raw" / "Validation of Structural Contrast Baseline" / "wise_hii_catalog" / "mass"
    mass_dir.mkdir(parents=True, exist_ok=True)

    search_registry_path = mass_dir / "wise_hii_mass_search_registry.csv"
    df = read_csv_required(search_registry_path)

    required_base = [
        "wise_name", "catalog_class", "input_group", "priority_rank", "size_bucket",
        "glon", "glat", "ra", "dec", "radius_arcsec",
        "hii_region_name", "membership", "dist_kpc",
        "search_query_primary", "search_query_secondary", "search_query_membership",
        "mass_search_status", "notes",
    ]
    df = ensure_columns(df, required_base)

    # Prepare dual search input
    dual = df[required_base].copy()
    dual["wise_name"] = dual["wise_name"].astype(str).str.strip()

    dual["search_track_direct"] = "pending"
    dual["search_track_proxy"] = "pending"
    dual["direct_query_primary"] = dual["wise_name"].astype(str) + " stellar mass HII region"
    dual["direct_query_secondary"] = dual["membership"].fillna("").astype(str) + " ionizing star mass"
    dual["proxy_query_log_nly"] = dual["wise_name"].astype(str) + " log Nly HII region"
    dual["proxy_query_spectral_type"] = dual["membership"].fillna("").astype(str) + " spectral type HII"
    dual["proxy_query_radio"] = dual["hii_region_name"].fillna("").astype(str) + " radio continuum HII region"
    dual["proxy_priority_type"] = "direct_mass_first_then_proxy"
    dual["proxy_candidate_type"] = "unreviewed"
    dual["direct_source_found"] = pd.NA
    dual["proxy_source_found"] = pd.NA
    dual["direct_source_key"] = pd.NA
    dual["proxy_source_key"] = pd.NA
    dual["direct_match_quality"] = pd.NA
    dual["proxy_match_quality"] = pd.NA
    dual["recommended_bridge_field"] = pd.NA
    dual["review_status"] = "pending_review"
    dual["review_notes"] = pd.NA

    # Initial candidate application table
    candidates = df[[
        "wise_name", "catalog_class", "input_group", "priority_rank",
        "glon", "glat", "ra", "dec", "radius_arcsec",
        "hii_region_name", "membership", "dist_kpc"
    ]].copy()
    candidates["mass_source_key"] = pd.NA
    candidates["mass_value_msun"] = pd.NA
    candidates["mass_value_type"] = pd.NA
    candidates["mass_value_notes"] = pd.NA
    candidates["proxy_kind"] = pd.NA
    candidates["proxy_value"] = pd.NA
    candidates["proxy_value_unit"] = pd.NA
    candidates["log_nly"] = pd.NA
    candidates["spectral_type"] = pd.NA
    candidates["radio_proxy_available"] = pd.NA
    candidates["ionizing_source_reference"] = pd.NA
    candidates["mass_search_status"] = "pending_dual_search"
    candidates["notes"] = pd.NA

    # Empty dual source registry
    source_registry = pd.DataFrame(columns=[
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
        "source_track",                # direct_mass / proxy_log_nly / proxy_spectral_type / proxy_radio / proxy_ionizing_source
        "source_type",                 # direct / proxy / mixed
        "match_method",
        "match_quality",
        "mass_field_description",
        "mass_value_msun",
        "mass_range_lower_msun",
        "mass_range_upper_msun",
        "proxy_kind",
        "proxy_value",
        "proxy_value_unit",
        "log_nly",
        "spectral_type",
        "radio_proxy_available",
        "distance_kpc",
        "matching_notes",
    ])

    dual_out = mass_dir / "wise_hii_mass_dual_search_input.csv"
    cand_out = mass_dir / "wise_hii_mass_dual_candidates_initial.csv"
    src_out = mass_dir / "wise_hii_mass_dual_source_registry.csv"
    manifest_out = mass_dir / "wise_hii_mass_dual_init_manifest.txt"

    dual.to_csv(dual_out, index=False)
    candidates.to_csv(cand_out, index=False)
    source_registry.to_csv(src_out, index=False)

    lines = [
        "WISE H II dual-track mass search initialization",
        "",
        f"Project root: {root}",
        f"Input registry: {search_registry_path}",
        f"Output directory: {mass_dir}",
        "",
        f"Rows initialized: {len(dual)}",
        "",
        "Generated files:",
        "- wise_hii_mass_dual_search_input.csv",
        "- wise_hii_mass_dual_candidates_initial.csv",
        "- wise_hii_mass_dual_source_registry.csv",
        "- wise_hii_mass_dual_init_manifest.txt",
        "",
        "Operational rule:",
        "Run direct-mass search and proxy search in parallel from the raw layer.",
        "Do not wait for direct-mass failure before starting proxy search.",
        "Use source_track to distinguish direct and proxy literature rows.",
    ]
    manifest_out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
