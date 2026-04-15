#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build derived tables_2 by merging raw mass/fits outputs and selecting
intersection candidates for future standard/topological validation input.

Placement
---------
data/derived/Validation of Structural Contrast Baseline/script/

Inputs
------
Raw mass:
- data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/mass/wise_hii_mass_search_registry.csv
- data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/mass/wise_hii_mass_candidates_initial.csv
- data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/mass/wise_hii_mass_source_registry.csv

Raw fits:
- data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/fits/wise_hii_fits_check_registry.csv
- data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/fits/wise_hii_fits_candidates_initial.csv
- data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/fits/wise_hii_fits_source_registry.csv

Outputs
-------
data/derived/Validation of Structural Contrast Baseline/wise_hii_catalog/tables_2/

Purpose
-------
1) Merge raw mass/fits registries by wise_name.
2) Keep only the shared target pool.
3) Compute both:
   - strict ready intersection
   - provisional preselection for future standard/topological input work
4) Save multi-level derived tables.

Important
---------
Because current raw mass/fits tables may still be mostly unfilled, strict_ready may be empty.
That is not an error. In that case, provisional_preselection becomes the main work table.

Windows example
---------------
python "data\derived\Validation of Structural Contrast Baseline\script\build_wise_hii_tables_2_intersection_001.py"
"""

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
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "verified", "downloaded", "ready", "ok"}


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


def classify_standard_eligibility(row) -> str:
    # Standard stage mainly needs target identity, coordinates, radius, and a plausible FITS path.
    coords_ok = all(has_any_text(row.get(c)) for c in ["wise_name", "ra", "dec", "radius_arcsec"])
    fits_state = row.get("fits_status", "")
    if coords_ok and fits_state in {"fits_ready_verified", "fits_ready_downloadable"}:
        return "standard_candidate"
    if coords_ok:
        return "standard_pending_fits"
    return "standard_not_ready"


def classify_topological_eligibility(row) -> str:
    # Topological stage should require both FITS usability and mass/bridge readiness.
    fits_state = row.get("fits_status", "")
    mass_state = row.get("mass_status", "")

    fits_ok = fits_state in {"fits_ready_verified", "fits_ready_downloadable"}
    mass_ok = mass_state in {"mass_ready_direct", "mass_ready_proxy"}

    if fits_ok and mass_ok:
        return "topological_candidate"
    if fits_ok and mass_state == "mass_partial":
        return "topological_pending_mass"
    if mass_ok and fits_state == "fits_partial":
        return "topological_pending_fits"
    if row.get("intersection_present", False):
        return "topological_provisional_only"
    return "topological_not_ready"


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

    # Read raw inputs
    mass_search = read_csv_required(mass_dir / "wise_hii_mass_search_registry.csv")
    mass_cand = read_csv_required(mass_dir / "wise_hii_mass_candidates_initial.csv")
    fits_check = read_csv_required(fits_dir / "wise_hii_fits_check_registry.csv")
    fits_cand = read_csv_required(fits_dir / "wise_hii_fits_candidates_initial.csv")

    # Minimal harmonization
    for df in [mass_search, mass_cand, fits_check, fits_cand]:
        df["wise_name"] = df["wise_name"].astype(str).str.strip()

    # Build mass side
    mass = pd.merge(
        mass_search,
        mass_cand,
        on="wise_name",
        how="outer",
        suffixes=("_search", "_cand"),
    )

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

    # Build fits side
    fits = pd.merge(
        fits_check,
        fits_cand,
        on="wise_name",
        how="outer",
        suffixes=("_check", "_cand"),
    )

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

    keep_fits_cols = [
        "wise_name", "catalog_class", "input_group", "priority_rank", "glon", "glat", "ra", "dec",
        "radius_arcsec", "cutout_radius_arcmin_suggested", "hii_region_name", "membership", "dist_kpc",
        "fits_source_service", "fits_band", "fits_url", "fits_downloadable", "fits_local_path",
        "fits_local_verified", "fits_image_plane_verified", "fits_check_status", "fits_notes"
    ]
    fits = fits[[c for c in keep_fits_cols if c in fits.columns]].copy()
    fits["fits_row_present"] = True

    # Outer merge to inspect intersection and non-intersection
    merged = pd.merge(
        mass, fits,
        on="wise_name",
        how="outer",
        suffixes=("_mass", "_fits"),
        indicator=True
    )

    # Coalesce shared descriptive fields
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

    merged["standard_eligibility"] = merged.apply(classify_standard_eligibility, axis=1)
    merged["topological_eligibility"] = merged.apply(classify_topological_eligibility, axis=1)

    merged["selection_tier"] = "excluded"
    merged.loc[merged["provisional_preselection"], "selection_tier"] = "provisional_preselection"
    merged.loc[merged["standard_eligibility"] == "standard_candidate", "selection_tier"] = "standard_candidate"
    merged.loc[merged["topological_eligibility"] == "topological_candidate", "selection_tier"] = "topological_candidate"

    merged = merged.sort_values(
        ["strict_ready", "provisional_preselection", "priority_rank", "wise_name"],
        ascending=[False, False, True, True]
    ).reset_index(drop=True)

    # Save outputs
    merged.to_csv(out_dir / "wise_hii_tables_2_full_merged.csv", index=False)

    intersection_all = merged[merged["intersection_present"]].copy()
    intersection_all.to_csv(out_dir / "wise_hii_tables_2_intersection_all.csv", index=False)

    provisional = merged[merged["provisional_preselection"]].copy()
    provisional.to_csv(out_dir / "wise_hii_tables_2_provisional_preselection.csv", index=False)

    strict = merged[merged["strict_ready"]].copy()
    strict.to_csv(out_dir / "wise_hii_tables_2_strict_ready.csv", index=False)

    standard_candidates = merged[merged["standard_eligibility"] == "standard_candidate"].copy()
    standard_candidates.to_csv(out_dir / "wise_hii_tables_2_standard_candidates.csv", index=False)

    topological_candidates = merged[merged["topological_eligibility"] == "topological_candidate"].copy()
    topological_candidates.to_csv(out_dir / "wise_hii_tables_2_topological_candidates.csv", index=False)

    summary_rows = [
        {"metric": "mass_rows", "value": int(mass["wise_name"].nunique())},
        {"metric": "fits_rows", "value": int(fits["wise_name"].nunique())},
        {"metric": "intersection_present", "value": int(intersection_all["wise_name"].nunique())},
        {"metric": "provisional_preselection", "value": int(provisional["wise_name"].nunique())},
        {"metric": "strict_ready", "value": int(strict["wise_name"].nunique())},
        {"metric": "standard_candidates", "value": int(standard_candidates["wise_name"].nunique())},
        {"metric": "topological_candidates", "value": int(topological_candidates["wise_name"].nunique())},
        {"metric": "mass_ready_direct", "value": int((merged["mass_status"] == "mass_ready_direct").sum())},
        {"metric": "mass_ready_proxy", "value": int((merged["mass_status"] == "mass_ready_proxy").sum())},
        {"metric": "mass_partial", "value": int((merged["mass_status"] == "mass_partial").sum())},
        {"metric": "fits_ready_verified", "value": int((merged["fits_status"] == "fits_ready_verified").sum())},
        {"metric": "fits_ready_downloadable", "value": int((merged["fits_status"] == "fits_ready_downloadable").sum())},
        {"metric": "fits_partial", "value": int((merged["fits_status"] == "fits_partial").sum())},
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "wise_hii_tables_2_summary.csv", index=False)

    manifest = []
    manifest.append("WISE H II tables_2 derived intersection build")
    manifest.append("")
    manifest.append(f"Project root: {root}")
    manifest.append(f"Raw mass dir: {mass_dir}")
    manifest.append(f"Raw fits dir: {fits_dir}")
    manifest.append(f"Output dir: {out_dir}")
    manifest.append("")
    manifest.append(f"Mass side rows: {mass['wise_name'].nunique()}")
    manifest.append(f"FITS side rows: {fits['wise_name'].nunique()}")
    manifest.append(f"Intersection present: {intersection_all['wise_name'].nunique()}")
    manifest.append(f"Provisional preselection: {provisional['wise_name'].nunique()}")
    manifest.append(f"Strict ready: {strict['wise_name'].nunique()}")
    manifest.append(f"Standard candidates: {standard_candidates['wise_name'].nunique()}")
    manifest.append(f"Topological candidates: {topological_candidates['wise_name'].nunique()}")
    manifest.append("")
    manifest.append("Interpretation rule:")
    manifest.append("- strict_ready: both mass and FITS are meaningfully ready.")
    manifest.append("- provisional_preselection: shared target pool with basic geometry, useful as first-pass workbench.")
    manifest.append("- standard_candidate: FITS appears usable for baseline input construction.")
    manifest.append("- topological_candidate: FITS and mass/bridge both appear usable.")
    manifest.append("")
    manifest.append("Operational note:")
    manifest.append("If strict_ready is empty, that is not an error.")
    manifest.append("It means raw mass/FITS confirmation has not yet matured enough.")
    manifest.append("In that case, provisional_preselection is the main table to continue filling.")

    (out_dir / "wise_hii_tables_2_manifest.txt").write_text("\n".join(manifest), encoding="utf-8")
    print("\n".join(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
