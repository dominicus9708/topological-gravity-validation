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


def coalesce(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    return series_a.where(series_a.notna(), series_b)


def truthy(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y", "verified", "downloaded", "ready", "ok"}


def has_any_text(x) -> bool:
    return pd.notna(x) and str(x).strip() != ""


def classify_mass_status(row) -> str:
    has_direct_mass = has_any_text(row.get("mass_value_msun"))
    has_proxy = any(
        has_any_text(row.get(c))
        for c in ["log_nly", "spectral_type", "radio_proxy_available", "mass_source_key"]
    )
    has_source = has_any_text(row.get("mass_source_key"))

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
    has_url = has_any_text(row.get("fits_url"))
    has_local = has_any_text(row.get("fits_local_path"))
    has_source = has_any_text(row.get("fits_source_service"))

    if verified and (has_local or has_url or has_source):
        return "fits_ready_verified"
    if downloadable and (has_url or has_source):
        return "fits_ready_downloadable"
    if has_url or has_local or has_source or downloadable:
        return "fits_partial"
    return "fits_not_ready"


def classify_standard_stage(row) -> str:
    coords_ok = all(has_any_text(row.get(c)) for c in ["wise_name", "ra", "dec", "radius_arcsec"])
    fits_state = row.get("fits_status", "")
    if coords_ok and fits_state in {"fits_ready_verified", "fits_ready_downloadable"}:
        return "standard_stage_ready"
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
    if fits_ok and mass_state == "mass_partial":
        return "topological_stage_pending_mass"
    if mass_ok and fits_state == "fits_partial":
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

    out_dir = root / "data" / "derived" / "Validation of Structural Contrast Baseline" / "wise_hii_catalog" / "tables_2"
    out_dir.mkdir(parents=True, exist_ok=True)

    mass_search = read_csv_required(mass_dir / "wise_hii_mass_search_registry.csv")
    mass_cand = read_csv_required(mass_dir / "wise_hii_mass_candidates_initial.csv")
    fits_check = read_csv_required(fits_dir / "wise_hii_fits_check_registry.csv")
    fits_cand = read_csv_required(fits_dir / "wise_hii_fits_candidates_initial.csv")

    for df in [mass_search, mass_cand, fits_check, fits_cand]:
        df["wise_name"] = df["wise_name"].astype(str).str.strip()

    mass = pd.merge(mass_search, mass_cand, on="wise_name", how="outer", suffixes=("_search", "_cand"))
    mass["catalog_class"] = coalesce(mass.get("catalog_class_search"), mass.get("catalog_class_cand"))
    mass["input_group"] = coalesce(mass.get("input_group_search"), mass.get("input_group_cand"))
    mass["priority_rank"] = coalesce(mass.get("priority_rank_search"), mass.get("priority_rank_cand"))
    mass["glon"] = coalesce(mass.get("glon_search"), mass.get("glon_cand"))
    mass["glat"] = coalesce(mass.get("glat_search"), mass.get("glat_cand"))
    mass["ra"] = coalesce(mass.get("ra_search"), mass.get("ra_cand"))
    mass["dec"] = coalesce(mass.get("dec_search"), mass.get("dec_cand"))
    mass["radius_arcsec"] = coalesce(mass.get("radius_arcsec_search"), mass.get("radius_arcsec_cand"))
    mass["hii_region_name"] = coalesce(mass.get("hii_region_name_search"), mass.get("hii_region_name_cand"))
    mass["membership"] = coalesce(mass.get("membership_search"), mass.get("membership_cand"))
    mass["dist_kpc"] = coalesce(mass.get("dist_kpc_search"), mass.get("dist_kpc_cand"))
    mass["mass_row_present"] = True

    keep_mass_cols = [
        "wise_name", "catalog_class", "input_group", "priority_rank", "glon", "glat", "ra", "dec",
        "radius_arcsec", "hii_region_name", "membership", "dist_kpc",
        "search_query_primary", "search_query_secondary", "search_query_membership",
        "mass_search_status", "mass_value_msun", "mass_value_type", "mass_value_notes",
        "mass_source_key", "mass_source_match_method", "radio_proxy_available",
        "log_nly", "spectral_type", "mass_row_present"
    ]
    mass = mass[[c for c in keep_mass_cols if c in mass.columns]].copy()

    fits = pd.merge(fits_check, fits_cand, on="wise_name", how="outer", suffixes=("_check", "_cand"))
    fits["catalog_class"] = coalesce(fits.get("catalog_class_check"), fits.get("catalog_class_cand"))
    fits["input_group"] = coalesce(fits.get("input_group_check"), fits.get("input_group_cand"))
    fits["priority_rank"] = coalesce(fits.get("priority_rank_check"), fits.get("priority_rank_cand"))
    fits["glon"] = coalesce(fits.get("glon_check"), fits.get("glon_cand"))
    fits["glat"] = coalesce(fits.get("glat_check"), fits.get("glat_cand"))
    fits["ra"] = coalesce(fits.get("ra_check"), fits.get("ra_cand"))
    fits["dec"] = coalesce(fits.get("dec_check"), fits.get("dec_cand"))
    fits["radius_arcsec"] = coalesce(fits.get("radius_arcsec_check"), fits.get("radius_arcsec_cand"))
    fits["cutout_radius_arcmin_suggested"] = coalesce(
        fits.get("cutout_radius_arcmin_suggested_check"),
        fits.get("cutout_radius_arcmin_suggested_cand"),
    )
    fits["hii_region_name"] = coalesce(fits.get("hii_region_name_check"), fits.get("hii_region_name_cand"))
    fits["membership"] = coalesce(fits.get("membership_check"), fits.get("membership_cand"))
    fits["dist_kpc"] = coalesce(fits.get("dist_kpc_check"), fits.get("dist_kpc_cand"))
    fits["fits_check_status"] = coalesce(fits.get("fits_check_status_check"), fits.get("fits_check_status_cand"))
    fits["fits_notes"] = coalesce(fits.get("fits_notes_check"), fits.get("fits_notes_cand"))
    fits["fits_row_present"] = True

    keep_fits_cols = [
        "wise_name", "catalog_class", "input_group", "priority_rank", "glon", "glat", "ra", "dec",
        "radius_arcsec", "cutout_radius_arcmin_suggested", "hii_region_name", "membership", "dist_kpc",
        "fits_source_service", "fits_band", "fits_url", "fits_downloadable", "fits_local_path",
        "fits_local_verified", "fits_image_plane_verified", "fits_check_status", "fits_notes",
        "fits_row_present"
    ]
    fits = fits[[c for c in keep_fits_cols if c in fits.columns]].copy()

    merged = pd.merge(mass, fits, on="wise_name", how="outer", suffixes=("_mass", "_fits"), indicator=True)

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
    merged["strict_ready"] = merged["mass_status"].isin({"mass_ready_direct", "mass_ready_proxy"}) & merged["fits_status"].isin({"fits_ready_verified", "fits_ready_downloadable"})
    merged["provisional_preselection"] = merged["intersection_present"] & merged["wise_name"].notna() & merged["ra"].notna() & merged["dec"].notna() & merged["radius_arcsec"].notna()

    merged["standard_stage_status"] = merged.apply(classify_standard_stage, axis=1)
    merged["topological_stage_status"] = merged.apply(classify_topological_stage, axis=1)

    merged["tables_2_role"] = "excluded"
    merged.loc[merged["provisional_preselection"], "tables_2_role"] = "staging_provisional"
    merged.loc[merged["standard_stage_status"] == "standard_stage_ready", "tables_2_role"] = "staging_standard_ready"
    merged.loc[merged["topological_stage_status"] == "topological_stage_ready", "tables_2_role"] = "staging_topological_ready"

    merged["final_input_allowed"] = False
    merged["final_input_block_reason"] = "not_evaluated"
    merged.loc[~merged["provisional_preselection"], "final_input_block_reason"] = "missing_basic_geometry_or_missing_intersection"
    merged.loc[merged["provisional_preselection"], "final_input_block_reason"] = "tables_2_is_staging_only_use_final_input_builder_later"

    merged = merged.sort_values(
        ["strict_ready", "provisional_preselection", "priority_rank", "wise_name"],
        ascending=[False, False, True, True]
    ).reset_index(drop=True)

    merged.to_csv(out_dir / "wise_hii_tables_2_staging_full_merged.csv", index=False)
    merged[merged["intersection_present"]].copy().to_csv(out_dir / "wise_hii_tables_2_staging_intersection_all.csv", index=False)
    merged[merged["provisional_preselection"]].copy().to_csv(out_dir / "wise_hii_tables_2_staging_provisional_preselection.csv", index=False)
    merged[merged["strict_ready"]].copy().to_csv(out_dir / "wise_hii_tables_2_staging_strict_ready.csv", index=False)
    merged[merged["standard_stage_status"] == "standard_stage_ready"].copy().to_csv(out_dir / "wise_hii_tables_2_staging_standard_ready.csv", index=False)
    merged[merged["topological_stage_status"] == "topological_stage_ready"].copy().to_csv(out_dir / "wise_hii_tables_2_staging_topological_ready.csv", index=False)

    summary_rows = [
        {"metric": "mass_rows", "value": int(mass["wise_name"].nunique())},
        {"metric": "fits_rows", "value": int(fits["wise_name"].nunique())},
        {"metric": "intersection_present", "value": int(merged["intersection_present"].sum())},
        {"metric": "provisional_preselection", "value": int(merged["provisional_preselection"].sum())},
        {"metric": "strict_ready", "value": int(merged["strict_ready"].sum())},
        {"metric": "standard_stage_ready", "value": int((merged["standard_stage_status"] == "standard_stage_ready").sum())},
        {"metric": "topological_stage_ready", "value": int((merged["topological_stage_status"] == "topological_stage_ready").sum())},
        {"metric": "final_input_allowed_true", "value": int(merged["final_input_allowed"].sum())},
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "wise_hii_tables_2_staging_summary.csv", index=False)

    manifest = []
    manifest.append("WISE H II tables_2 staging build")
    manifest.append("")
    manifest.append(f"Project root: {root}")
    manifest.append(f"Raw mass dir: {mass_dir}")
    manifest.append(f"Raw fits dir: {fits_dir}")
    manifest.append(f"Output dir: {out_dir}")
    manifest.append("")
    manifest.append("Critical rule:")
    manifest.append("tables_2 is staging only and must not be used as final input directly.")
    manifest.append("Final input must be generated later under data/derived/.../input/")
    manifest.append("")
    manifest.append(f"Mass side rows: {mass['wise_name'].nunique()}")
    manifest.append(f"FITS side rows: {fits['wise_name'].nunique()}")
    manifest.append(f"Intersection present: {int(merged['intersection_present'].sum())}")
    manifest.append(f"Provisional preselection: {int(merged['provisional_preselection'].sum())}")
    manifest.append(f"Strict ready: {int(merged['strict_ready'].sum())}")
    manifest.append(f"Standard stage ready: {int((merged['standard_stage_status'] == 'standard_stage_ready').sum())}")
    manifest.append(f"Topological stage ready: {int((merged['topological_stage_status'] == 'topological_stage_ready').sum())}")
    manifest.append("")
    manifest.append("Interpretation:")
    manifest.append("- staging_provisional: shared target pool with basic geometry, still not final input.")
    manifest.append("- staging_standard_ready: looks usable for future standard input builder, but still not final input.")
    manifest.append("- staging_topological_ready: looks usable for future topological input builder, but still not final input.")
    manifest.append("")
    manifest.append("Operational note:")
    manifest.append("Even when strict_ready is nonzero, this script must not create input/ files.")
    manifest.append("Use a separate final-input builder after raw confirmation is sufficiently complete.")
    (out_dir / "wise_hii_tables_2_staging_manifest.txt").write_text("\n".join(manifest), encoding="utf-8")

    print("\n".join(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
