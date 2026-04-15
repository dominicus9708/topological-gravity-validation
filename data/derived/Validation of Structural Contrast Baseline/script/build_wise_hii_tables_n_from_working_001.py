#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build WISE H II derived tables_n from raw initial + working tables.

Placement
---------
data/derived/Validation of Structural Contrast Baseline/script/

Output
------
data/derived/Validation of Structural Contrast Baseline/wise_hii_catalog/tables_n/

Design
------
- tables_n is still a derived staging/evaluation area, NOT final input.
- Prefer *_working.csv when present.
- Fall back to *_initial.csv or base registry when working files are absent.
- Re-evaluate mass/fits readiness after working updates.
- Separate:
  * full merged staging
  * intersection
  * provisional pool
  * standard-stage ready
  * topological-stage ready

Windows example
---------------
python "data\derived\Validation of Structural Contrast Baseline\script\build_wise_hii_tables_n_from_working_001.py"
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


def coalesce(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    return series_a.where(series_a.notna(), series_b)


def has_text(v) -> bool:
    return pd.notna(v) and str(v).strip() != ""


def truthy(v) -> bool:
    if pd.isna(v):
        return False
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {
        "1", "true", "t", "yes", "y", "verified", "downloaded", "ok", "ready",
        "api_xml_ready", "source_row_applied"
    }


def classify_mass_status(row) -> str:
    has_direct_mass = has_text(row.get("mass_value_msun"))
    has_proxy = any(
        has_text(row.get(c))
        for c in ["log_nly", "spectral_type", "radio_proxy_available", "mass_source_key"]
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
    has_local = has_text(row.get("fits_local_path"))
    has_source = has_text(row.get("fits_source_service"))

    if verified and (has_local or has_url or has_source):
        return "fits_ready_verified"
    if downloadable and (has_url or has_source):
        return "fits_ready_downloadable"
    if has_url or has_local or has_source or downloadable:
        return "fits_partial"
    return "fits_not_ready"


def classify_standard_stage(row) -> str:
    coords_ok = all(has_text(row.get(c)) for c in ["wise_name", "ra", "dec", "radius_arcsec"])
    fits_state = row.get("fits_status", "")
    if coords_ok and fits_state in {"fits_ready_verified", "fits_ready_downloadable"}:
        return "standard_stage_ready"
    if coords_ok and fits_state == "fits_partial":
        return "standard_stage_seed_only"
    if coords_ok:
        return "standard_stage_pending_fits"
    return "standard_stage_not_ready"


def classify_topological_stage(row) -> str:
    fits_state = row.get("fits_status", "")
    mass_state = row.get("mass_status", "")

    fits_ok = fits_state in {"fits_ready_verified", "fits_ready_downloadable"}
    mass_ok = mass_state in {"mass_ready_direct", "mass_ready_proxy"}

    if fits_ok and mass_ok:
        return "topological_stage_ready"
    if fits_ok and mass_state in {"mass_partial", "mass_not_ready"}:
        return "topological_stage_pending_mass"
    if mass_ok and fits_state in {"fits_partial", "fits_not_ready"}:
        return "topological_stage_pending_fits"
    if row.get("intersection_present", False):
        return "topological_stage_provisional"
    return "topological_stage_not_ready"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Root of topological_gravity_project")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()

    raw_base = root / "data" / "raw" / "Validation of Structural Contrast Baseline" / "wise_hii_catalog"
    mass_dir = raw_base / "mass"
    fits_dir = raw_base / "fits"

    out_dir = root / "data" / "derived" / "Validation of Structural Contrast Baseline" / "wise_hii_catalog" / "tables_n"
    out_dir.mkdir(parents=True, exist_ok=True)

    mass_search = read_csv_required(mass_dir / "wise_hii_mass_search_registry.csv")
    mass_candidates, mass_source_path = read_csv_prefer(
        mass_dir / "wise_hii_mass_candidates_working.csv",
        mass_dir / "wise_hii_mass_candidates_initial.csv",
    )
    fits_check = read_csv_required(fits_dir / "wise_hii_fits_check_registry.csv")
    fits_candidates, fits_source_path = read_csv_prefer(
        fits_dir / "wise_hii_fits_candidates_working.csv",
        fits_dir / "wise_hii_fits_candidates_initial.csv",
    )

    for df in [mass_search, mass_candidates, fits_check, fits_candidates]:
        df["wise_name"] = df["wise_name"].astype(str).str.strip()

    # Mass side
    mass = pd.merge(
        mass_search,
        mass_candidates,
        on="wise_name",
        how="outer",
        suffixes=("_search", "_cand"),
    )
    for col in ["catalog_class", "input_group", "priority_rank", "glon", "glat", "ra", "dec",
                "radius_arcsec", "hii_region_name", "membership", "dist_kpc"]:
        a = f"{col}_search"
        b = f"{col}_cand"
        if a in mass.columns or b in mass.columns:
            mass[col] = coalesce(mass.get(a), mass.get(b))

    keep_mass_cols = [
        "wise_name", "catalog_class", "input_group", "priority_rank", "glon", "glat", "ra", "dec",
        "radius_arcsec", "hii_region_name", "membership", "dist_kpc",
        "search_query_primary", "search_query_secondary", "search_query_membership",
        "mass_search_status", "mass_value_msun", "mass_value_type", "mass_value_notes",
        "mass_source_key", "mass_source_match_method", "radio_proxy_available",
        "log_nly", "spectral_type"
    ]
    mass = mass[[c for c in keep_mass_cols if c in mass.columns]].copy()
    mass["mass_row_present"] = True

    # FITS side
    fits = pd.merge(
        fits_check,
        fits_candidates,
        on="wise_name",
        how="outer",
        suffixes=("_check", "_cand"),
    )
    for col in ["catalog_class", "input_group", "priority_rank", "glon", "glat", "ra", "dec",
                "radius_arcsec", "cutout_radius_arcmin_suggested", "hii_region_name", "membership", "dist_kpc"]:
        a = f"{col}_check"
        b = f"{col}_cand"
        if a in fits.columns or b in fits.columns:
            fits[col] = coalesce(fits.get(a), fits.get(b))

    keep_fits_cols = [
        "wise_name", "catalog_class", "input_group", "priority_rank", "glon", "glat", "ra", "dec",
        "radius_arcsec", "cutout_radius_arcmin_suggested", "hii_region_name", "membership", "dist_kpc",
        "fits_source_service", "fits_band", "fits_url", "fits_downloadable", "fits_local_path",
        "fits_local_verified", "fits_image_plane_verified", "fits_check_status", "fits_notes"
    ]
    fits = fits[[c for c in keep_fits_cols if c in fits.columns]].copy()
    fits["fits_row_present"] = True

    merged = pd.merge(
        mass,
        fits,
        on="wise_name",
        how="outer",
        suffixes=("_mass", "_fits"),
        indicator=True
    )

    for col in ["catalog_class", "input_group", "priority_rank", "glon", "glat", "ra", "dec",
                "radius_arcsec", "hii_region_name", "membership", "dist_kpc"]:
        a = f"{col}_mass"
        b = f"{col}_fits"
        if a in merged.columns or b in merged.columns:
            merged[col] = coalesce(merged.get(a), merged.get(b))

    merged["mass_row_present"] = merged["mass_row_present"].fillna(False)
    merged["fits_row_present"] = merged["fits_row_present"].fillna(False)
    merged["intersection_present"] = merged["mass_row_present"] & merged["fits_row_present"]

    merged["mass_status"] = merged.apply(classify_mass_status, axis=1)
    merged["fits_status"] = merged.apply(classify_fits_status, axis=1)
    merged["strict_ready"] = (
        merged["mass_status"].isin({"mass_ready_direct", "mass_ready_proxy"})
        & merged["fits_status"].isin({"fits_ready_verified", "fits_ready_downloadable"})
    )
    merged["provisional_preselection"] = (
        merged["intersection_present"]
        & merged["wise_name"].notna()
        & merged["ra"].notna()
        & merged["dec"].notna()
        & merged["radius_arcsec"].notna()
    )

    merged["standard_stage_status"] = merged.apply(classify_standard_stage, axis=1)
    merged["topological_stage_status"] = merged.apply(classify_topological_stage, axis=1)

    merged["tables_n_role"] = "excluded"
    merged.loc[merged["provisional_preselection"], "tables_n_role"] = "staging_provisional"
    merged.loc[merged["standard_stage_status"] == "standard_stage_seed_only", "tables_n_role"] = "staging_standard_seed"
    merged.loc[merged["standard_stage_status"] == "standard_stage_ready", "tables_n_role"] = "staging_standard_ready"
    merged.loc[merged["topological_stage_status"] == "topological_stage_ready", "tables_n_role"] = "staging_topological_ready"

    merged["final_input_allowed"] = False
    merged["final_input_block_reason"] = "tables_n_is_staging_only_use_final_input_builder_later"

    merged = merged.sort_values(
        ["strict_ready", "provisional_preselection", "priority_rank", "wise_name"],
        ascending=[False, False, True, True]
    ).reset_index(drop=True)

    # Output files
    merged.to_csv(out_dir / "wise_hii_tables_n_full_merged.csv", index=False)
    merged[merged["intersection_present"]].copy().to_csv(out_dir / "wise_hii_tables_n_intersection_all.csv", index=False)
    merged[merged["provisional_preselection"]].copy().to_csv(out_dir / "wise_hii_tables_n_provisional_preselection.csv", index=False)
    merged[merged["strict_ready"]].copy().to_csv(out_dir / "wise_hii_tables_n_strict_ready.csv", index=False)
    merged[merged["standard_stage_status"].isin(["standard_stage_seed_only", "standard_stage_ready"])].copy().to_csv(out_dir / "wise_hii_tables_n_standard_stage_pool.csv", index=False)
    merged[merged["standard_stage_status"] == "standard_stage_ready"].copy().to_csv(out_dir / "wise_hii_tables_n_standard_ready.csv", index=False)
    merged[merged["topological_stage_status"] == "topological_stage_ready"].copy().to_csv(out_dir / "wise_hii_tables_n_topological_ready.csv", index=False)

    summary = pd.DataFrame([
        {"metric": "mass_rows", "value": int(mass["wise_name"].nunique())},
        {"metric": "fits_rows", "value": int(fits["wise_name"].nunique())},
        {"metric": "intersection_present", "value": int(merged["intersection_present"].sum())},
        {"metric": "provisional_preselection", "value": int(merged["provisional_preselection"].sum())},
        {"metric": "strict_ready", "value": int(merged["strict_ready"].sum())},
        {"metric": "standard_stage_seed_only", "value": int((merged["standard_stage_status"] == "standard_stage_seed_only").sum())},
        {"metric": "standard_stage_ready", "value": int((merged["standard_stage_status"] == "standard_stage_ready").sum())},
        {"metric": "topological_stage_ready", "value": int((merged["topological_stage_status"] == "topological_stage_ready").sum())},
        {"metric": "mass_ready_direct", "value": int((merged["mass_status"] == "mass_ready_direct").sum())},
        {"metric": "mass_ready_proxy", "value": int((merged["mass_status"] == "mass_ready_proxy").sum())},
        {"metric": "fits_ready_verified", "value": int((merged["fits_status"] == "fits_ready_verified").sum())},
        {"metric": "fits_ready_downloadable", "value": int((merged["fits_status"] == "fits_ready_downloadable").sum())},
        {"metric": "fits_partial", "value": int((merged["fits_status"] == "fits_partial").sum())},
    ])
    summary.to_csv(out_dir / "wise_hii_tables_n_summary.csv", index=False)

    manifest_lines = [
        "WISE H II tables_n derived rebuild from working tables",
        "",
        f"Project root: {root}",
        f"Mass candidates source used: {mass_source_path}",
        f"FITS candidates source used: {fits_source_path}",
        f"Output dir: {out_dir}",
        "",
        "Critical rule:",
        "tables_n is staging/evaluation only and must not be used as final input directly.",
        "Final input must be created later under data/derived/.../input/",
        "",
        f"Mass side rows: {mass['wise_name'].nunique()}",
        f"FITS side rows: {fits['wise_name'].nunique()}",
        f"Intersection present: {int(merged['intersection_present'].sum())}",
        f"Provisional preselection: {int(merged['provisional_preselection'].sum())}",
        f"Strict ready: {int(merged['strict_ready'].sum())}",
        f"Standard stage seed only: {int((merged['standard_stage_status'] == 'standard_stage_seed_only').sum())}",
        f"Standard stage ready: {int((merged['standard_stage_status'] == 'standard_stage_ready').sum())}",
        f"Topological stage ready: {int((merged['topological_stage_status'] == 'topological_stage_ready').sum())}",
        "",
        "Interpretation:",
        "- standard_stage_seed_only: FITS query/url seed exists but local verified FITS is not yet complete.",
        "- standard_stage_ready: FITS is evaluated as usable for later final standard input builder.",
        "- topological_stage_ready: both FITS usability and mass/proxy readiness are present.",
    ]
    (out_dir / "wise_hii_tables_n_manifest.txt").write_text("\n".join(manifest_lines), encoding="utf-8")

    print("\n".join(manifest_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
