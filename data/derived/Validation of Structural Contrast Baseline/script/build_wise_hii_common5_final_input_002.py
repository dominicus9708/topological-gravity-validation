#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        "api_xml_ready", "source_row_applied", "dual_source_row_applied"
    }


def classify_mass_status(row) -> str:
    has_direct_mass = has_text(row.get("mass_value_msun"))
    has_proxy = any(
        has_text(row.get(c))
        for c in ["proxy_kind", "proxy_value", "log_nly", "spectral_type", "radio_proxy_available", "mass_source_key"]
    )
    has_source = has_text(row.get("mass_source_key"))

    if has_direct_mass and has_source:
        return "mass_ready_direct"
    if has_proxy and has_source:
        return "mass_ready_proxy"
    if has_direct_mass or has_proxy or has_source:
        return "mass_partial"
    return "mass_not_ready"


def classify_fits_status(row) -> str:
    verified = truthy(row.get("fits_local_verified")) or truthy(row.get("fits_image_plane_verified"))
    downloadable = truthy(row.get("fits_downloadable"))
    has_url = has_text(row.get("fits_url"))
    has_source = has_text(row.get("fits_source_service"))

    if verified and (has_url or has_source):
        return "fits_ready_verified"
    if downloadable and (has_url or has_source):
        return "fits_ready_downloadable"
    if has_url or has_source or downloadable:
        return "fits_partial"
    return "fits_not_ready"


def extract_shortlist_identity(shortlist: pd.DataFrame) -> pd.DataFrame:
    shortlist = shortlist.copy()
    shortlist["wise_name"] = shortlist["wise_name"].astype(str).str.strip()

    # Build a clean shortlist identity table from whatever columns are present
    clean = pd.DataFrame()
    clean["wise_name"] = shortlist["wise_name"]

    identity_map = {
        "catalog_class": ["catalog_class", "catalog_class_mass", "catalog_class_fits"],
        "input_group": ["input_group", "input_group_mass", "input_group_fits"],
        "priority_rank": ["priority_rank", "priority_rank_mass", "priority_rank_fits"],
        "glon": ["glon", "glon_mass", "glon_fits"],
        "glat": ["glat", "glat_mass", "glat_fits"],
        "ra": ["ra", "ra_mass", "ra_fits"],
        "dec": ["dec", "dec_mass", "dec_fits"],
        "radius_arcsec": ["radius_arcsec", "radius_arcsec_mass", "radius_arcsec_fits"],
        "hii_region_name": ["hii_region_name", "hii_region_name_mass", "hii_region_name_fits"],
        "membership": ["membership", "membership_mass", "membership_fits"],
        "dist_kpc": ["dist_kpc", "dist_kpc_mass", "dist_kpc_fits"],
        "region_bucket": ["region_bucket"],
        "score_total": ["score_total"],
        "common_comparison_shortlist": ["common_comparison_shortlist"],
        "tables_5_role": ["tables_5_role"],
    }

    for out_col, candidates in identity_map.items():
        vals = None
        for c in candidates:
            if c in shortlist.columns:
                vals = shortlist[c] if vals is None else coalesce(vals, shortlist[c])
        if vals is not None:
            clean[out_col] = vals

    # Dedup just in case
    clean = clean.drop_duplicates(subset=["wise_name"], keep="first").reset_index(drop=True)
    return clean


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Root of topological_gravity_project")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()

    shortlist_path = root / "data" / "derived" / "Validation of Structural Contrast Baseline" / "wise_hii_catalog" / "tables_5" / "wise_hii_tables_5_common5_shortlist.csv"
    mass_dir = root / "data" / "raw" / "Validation of Structural Contrast Baseline" / "wise_hii_catalog" / "mass"
    fits_dir = root / "data" / "raw" / "Validation of Structural Contrast Baseline" / "wise_hii_catalog" / "fits"

    standard_out = root / "data" / "derived" / "Validation of Structural Contrast Baseline" / "input" / "standard" / "wise_hii_common5"
    topological_out = root / "data" / "derived" / "Validation of Structural Contrast Baseline" / "input" / "topological" / "wise_hii_common5"
    standard_out.mkdir(parents=True, exist_ok=True)
    topological_out.mkdir(parents=True, exist_ok=True)

    shortlist_raw = read_csv_required(shortlist_path)
    shortlist = extract_shortlist_identity(shortlist_raw)

    mass_df, mass_used = read_csv_prefer(
        mass_dir / "wise_hii_mass_dual_candidates_working.csv",
        mass_dir / "wise_hii_mass_dual_candidates_initial.csv",
    )
    fits_df, fits_used = read_csv_prefer(
        fits_dir / "wise_hii_fits_candidates_working.csv",
        fits_dir / "wise_hii_fits_candidates_initial.csv",
    )

    for df in [shortlist, mass_df, fits_df]:
        df["wise_name"] = df["wise_name"].astype(str).str.strip()
        df.drop_duplicates(subset=["wise_name"], keep="first", inplace=True)

    merged = pd.merge(shortlist, mass_df, on="wise_name", how="left", suffixes=("_short", "_mass"))
    merged = pd.merge(merged, fits_df, on="wise_name", how="left", suffixes=("", "_fits"))

    # identity coalesce
    shared_fields = ["catalog_class", "input_group", "priority_rank", "glon", "glat", "ra", "dec",
                     "radius_arcsec", "hii_region_name", "membership", "dist_kpc"]
    for col in shared_fields:
        candidates = [c for c in merged.columns if c == col or c.startswith(col + "_")]
        if not candidates:
            continue
        base = merged[candidates[0]]
        for c in candidates[1:]:
            base = coalesce(base, merged[c])
        merged[col] = base

    merged["mass_status"] = merged.apply(classify_mass_status, axis=1)
    merged["fits_status"] = merged.apply(classify_fits_status, axis=1)
    merged["topological_ready_flag"] = merged["mass_status"].isin({"mass_ready_direct", "mass_ready_proxy"}) & merged["fits_status"].isin({"fits_ready_verified", "fits_ready_downloadable"})

    standard_cols = [
        "wise_name", "catalog_class", "input_group", "priority_rank", "glon", "glat", "ra", "dec",
        "radius_arcsec", "hii_region_name", "membership", "dist_kpc",
        "region_bucket", "score_total",
        "fits_source_service", "fits_band", "fits_url", "fits_downloadable",
        "fits_local_path", "fits_local_verified", "fits_image_plane_verified",
        "fits_check_status", "fits_notes", "cutout_radius_arcmin_suggested",
    ]
    standard_final = merged[[c for c in standard_cols if c in merged.columns]].copy()
    standard_final["final_input_track"] = "standard"
    standard_final["final_input_group"] = "wise_hii_common5"
    standard_final["baseline_mass_policy"] = "mass_free"
    standard_final["baseline_ready_flag"] = standard_final.apply(
        lambda r: truthy(r.get("fits_downloadable")) or truthy(r.get("fits_local_verified")) or truthy(r.get("fits_image_plane_verified")),
        axis=1,
    )

    topo_cols = [
        "wise_name", "catalog_class", "input_group", "priority_rank", "glon", "glat", "ra", "dec",
        "radius_arcsec", "hii_region_name", "membership", "dist_kpc",
        "region_bucket", "score_total",
        "fits_source_service", "fits_band", "fits_url", "fits_downloadable",
        "fits_local_path", "fits_local_verified", "fits_image_plane_verified",
        "fits_check_status", "fits_notes", "cutout_radius_arcmin_suggested",
        "mass_source_key", "mass_value_msun", "mass_value_type", "mass_value_notes",
        "proxy_kind", "proxy_value", "proxy_value_unit", "log_nly", "spectral_type",
        "radio_proxy_available", "ionizing_source_reference", "mass_search_status", "notes",
        "mass_status", "fits_status", "topological_ready_flag",
    ]
    topological_final = merged[[c for c in topo_cols if c in merged.columns]].copy()
    topological_final["final_input_track"] = "topological"
    topological_final["final_input_group"] = "wise_hii_common5"

    standard_csv = standard_out / "wise_hii_common5_standard_final_input.csv"
    topological_csv = topological_out / "wise_hii_common5_topological_final_input.csv"
    manifest_txt = topological_out / "wise_hii_common5_final_input_manifest.txt"

    standard_final.to_csv(standard_csv, index=False)
    topological_final.to_csv(topological_csv, index=False)

    manifest_lines = [
        "WISE H II common5 final input build",
        "",
        f"Project root: {root}",
        f"Shortlist used: {shortlist_path}",
        f"Mass candidate source used: {mass_used}",
        f"FITS candidate source used: {fits_used}",
        "",
        f"Standard output: {standard_csv}",
        f"Topological output: {topological_csv}",
        "",
        f"Number of common5 targets: {len(shortlist)}",
        f"Standard baseline-ready rows: {int(standard_final['baseline_ready_flag'].sum())}",
        f"Topological ready rows: {int(topological_final['topological_ready_flag'].sum())}",
        "",
        "Policy:",
        "- Standard final input is mass-free baseline input.",
        "- Topological final input uses the same 5 targets and carries mass/proxy fields plus readiness metadata.",
        "- Fair comparison is preserved by using the same shortlist for both tracks.",
    ]
    manifest_txt.write_text("\n".join(manifest_lines), encoding="utf-8")

    print("\n".join(manifest_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
