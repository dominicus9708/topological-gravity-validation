#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
WISE H II first-pass derived builder

Author: Kwon Dominicus
Purpose:
    Build first-pass derived tables from raw WISE H II catalog files and candidate lists.

Design principle:
    raw -> derived 1st pass -> derived/input
    This script stops at the "derived 1st pass" stage.
    It does NOT overwrite raw files.
    It does NOT decide the final scientific target automatically.
    It prepares stable working tables for manual review and later input fixing.

Recommended location:
    data/derived/Validation of Structural Contrast Baseline/script/

Recommended project root:
    C:\\Users\\mincu\\Desktop\\topological_gravity_project
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\mincu\Desktop\topological_gravity_project")

RAW_BASE = Path("data") / "raw" / "Validation of Structural Contrast Baseline"
DERIVED_BASE = Path("data") / "derived" / "Validation of Structural Contrast Baseline"

RAW_CATALOG_DIR = RAW_BASE / "wise_hii_catalog" / "catalog"
RAW_CANDIDATE_DIR = RAW_BASE / "wise_hii_catalog" / "candidate_lists"

DERIVED_DATA_DIR = DERIVED_BASE / "wise_hii_catalog"
DERIVED_REVIEW_DIR = DERIVED_DATA_DIR / "review"
DERIVED_TABLE_DIR = DERIVED_DATA_DIR / "tables"
DERIVED_LOG_DIR = DERIVED_DATA_DIR / "logs"

# ------------------------------------------------------------
# Raw input files
# ------------------------------------------------------------

RAW_V23 = RAW_CATALOG_DIR / "wise_hii_V2.3.csv"
RAW_V13 = RAW_CATALOG_DIR / "wise_hii_V1.3.csv"

RAW_NORMALIZED = RAW_CANDIDATE_DIR / "wise_hii_normalized_full.csv"
RAW_SHORTLIST_KG = RAW_CANDIDATE_DIR / "wise_hii_shortlist_known_group_simple_shell.csv"
RAW_SHORTLIST_C = RAW_CANDIDATE_DIR / "wise_hii_shortlist_candidate_simple_shell.csv"
RAW_RADIUS_SORTED = RAW_CANDIDATE_DIR / "wise_hii_sorted_by_radius.csv"

# ------------------------------------------------------------
# Derived output files
# ------------------------------------------------------------

OUT_MASTER = DERIVED_TABLE_DIR / "wise_hii_master_catalog_for_baseline_review.csv"
OUT_REVIEW_KG = DERIVED_REVIEW_DIR / "wise_hii_review_known_group_priority.csv"
OUT_REVIEW_C = DERIVED_REVIEW_DIR / "wise_hii_review_candidate_priority.csv"
OUT_REVIEW_TOP = DERIVED_REVIEW_DIR / "wise_hii_review_top_priority_first_pass.csv"
OUT_MANIFEST = DERIVED_LOG_DIR / "wise_hii_first_pass_manifest.txt"
OUT_SUMMARY = DERIVED_LOG_DIR / "wise_hii_first_pass_summary.csv"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def read_csv_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def ensure_dirs(project_root: Path) -> None:
    for d in [DERIVED_DATA_DIR, DERIVED_REVIEW_DIR, DERIVED_TABLE_DIR, DERIVED_LOG_DIR]:
        (project_root / d).mkdir(parents=True, exist_ok=True)


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def radius_band(radius: float) -> str:
    if pd.isna(radius):
        return "unknown"
    if 90 <= radius <= 180:
        return "preferred_core_band"
    if 60 <= radius < 90 or 180 < radius <= 220:
        return "usable_band"
    if 220 < radius <= 300:
        return "large_band"
    return "outside_first_pass_band"


def complexity_warning(name: str, membership: str, other_names: str) -> str:
    text = " ".join([clean_text(name), clean_text(membership), clean_text(other_names)]).upper()
    complex_tokens = [
        "W51", "W49", "W43", "W3", "W4", "W5",
        "M17", "OMEGA", "NGC6334", "RCW49", "RCW 49",
        "CARINA", "EAGLE", "SAGITTARIUS", "Sgr B", "Sgr D"
    ]
    for token in complex_tokens:
        if token.upper() in text:
            return "possible_complex_region"
    return ""


def build_master(v23: pd.DataFrame, v13: pd.DataFrame, normalized: pd.DataFrame) -> pd.DataFrame:
    v23 = v23.copy()
    v13 = v13.copy()
    normalized = normalized.copy()

    # Normalize key columns for merging
    v23["wise_name"] = v23["WISE Name"].astype(str).str.strip()
    v13["wise_name"] = v13["WISE Name"].astype(str).str.strip()

    # V1.3 backup columns
    v13_backup = v13.rename(columns={
        "Catalog": "catalog_v13",
        "GLong": "glon_v13",
        "GLat": "glat_v13",
        "RA": "ra_v13",
        "Dec": "dec_v13",
        "Radius": "radius_arcsec_v13",
        "HII Region": "hii_region_name_v13",
        "Membership": "membership_v13",
        "Dist.": "dist_kpc_v13",
    })[
        [
            "wise_name", "catalog_v13", "glon_v13", "glat_v13",
            "ra_v13", "dec_v13", "radius_arcsec_v13",
            "hii_region_name_v13", "membership_v13", "dist_kpc_v13"
        ]
    ]

    # V2.3 selected and renamed columns
    v23_sel = v23.rename(columns={
        "Catalog": "catalog_class_raw",
        "GLong": "glon_raw",
        "GLat": "glat_raw",
        "RA": "ra_raw",
        "Dec": "dec_raw",
        "Radius": "radius_arcsec_raw",
        "HII Region": "hii_region_name_raw",
        "Membership": "membership_raw",
        "Dist.<br>(kpc)": "dist_kpc_raw",
        "GLIMPSE 8um<br>(Jy)": "glimpse8_jy_raw",
        "WISE 12um<br>(Jy)": "w3_jy_raw",
        "WISE 22um<br>(Jy)": "w4_jy_raw",
        "MIPSGAL 24um<br>(Jy)": "mips24_jy_raw",
        "Hi-Gal 70um<br>(Jy)": "higal70_jy_raw",
        "Hi-Gal 160um<br>(Jy)": "higal160_jy_raw",
        "HRDS 3cm<br>(Jy)": "hrds3cm_jy_raw",
        "MAGPIS 20cm<br>(Jy)": "magpis20cm_jy_raw",
        "VGPS 21cm<br>(Jy)": "vgps21cm_jy_raw",
        "Other Names": "other_names_raw",
        "Author": "author_raw",
        "Dist. Author": "dist_author_raw",
    })

    v23_keep = [
        "wise_name", "catalog_class_raw", "glon_raw", "glat_raw", "ra_raw", "dec_raw",
        "radius_arcsec_raw", "hii_region_name_raw", "membership_raw", "dist_kpc_raw",
        "glimpse8_jy_raw", "w3_jy_raw", "w4_jy_raw", "mips24_jy_raw",
        "higal70_jy_raw", "higal160_jy_raw", "hrds3cm_jy_raw",
        "magpis20cm_jy_raw", "vgps21cm_jy_raw", "other_names_raw",
        "author_raw", "dist_author_raw"
    ]
    v23_sel = v23_sel[v23_keep]

    # Candidate normalized table
    norm_keep = [
        "wise_name", "catalog_class", "glon", "glat", "ra", "dec",
        "radius_arcsec", "hii_region_name", "membership", "dist_kpc",
        "w3_jy", "w4_jy", "mips24_jy", "priority_rank", "size_bucket"
    ]
    normalized = normalized[norm_keep]

    master = normalized.merge(v23_sel, on="wise_name", how="left")
    master = master.merge(v13_backup, on="wise_name", how="left")

    # Candidate source tags
    master["in_shortlist_known_group"] = False
    master["in_shortlist_candidate"] = False

    # Derived review-friendly columns
    master["radius_band"] = master["radius_arcsec"].apply(radius_band)
    master["has_hii_name"] = master["hii_region_name"].fillna("").astype(str).str.strip().ne("")
    master["has_membership"] = master["membership"].fillna("").astype(str).str.strip().ne("")
    master["has_other_names"] = master["other_names_raw"].fillna("").astype(str).str.strip().ne("")
    master["has_distance"] = master["dist_kpc"].notna() | master["dist_kpc_raw"].notna() | master["dist_kpc_v13"].notna()

    master["nonzero_w3"] = pd.to_numeric(master["w3_jy"], errors="coerce").fillna(0) > 0
    master["nonzero_w4"] = pd.to_numeric(master["w4_jy"], errors="coerce").fillna(0) > 0
    master["nonzero_mips24"] = pd.to_numeric(master["mips24_jy"], errors="coerce").fillna(0) > 0
    master["any_nonzero_mid_ir"] = master[["nonzero_w3", "nonzero_w4", "nonzero_mips24"]].any(axis=1)

    master["complexity_warning"] = master.apply(
        lambda r: complexity_warning(
            r.get("hii_region_name", ""),
            r.get("membership", ""),
            r.get("other_names_raw", "")
        ),
        axis=1
    )

    master["baseline_first_pass_score"] = (
        (master["priority_rank"].fillna(99).astype(float).rsub(5) * 10).clip(lower=0)
        + master["radius_arcsec"].apply(lambda x: 20 if pd.notna(x) and 90 <= x <= 180 else (10 if pd.notna(x) and 60 <= x <= 220 else 0))
        + master["has_hii_name"].astype(int) * 8
        + master["has_membership"].astype(int) * 6
        + master["has_other_names"].astype(int) * 4
        + master["any_nonzero_mid_ir"].astype(int) * 5
        - (master["complexity_warning"].ne("").astype(int) * 12)
    )

    # Human-readable flags
    master["flag_has_hii_name"] = master["has_hii_name"].map(yes_no)
    master["flag_has_membership"] = master["has_membership"].map(yes_no)
    master["flag_has_other_names"] = master["has_other_names"].map(yes_no)
    master["flag_has_distance"] = master["has_distance"].map(yes_no)
    master["flag_any_nonzero_mid_ir"] = master["any_nonzero_mid_ir"].map(yes_no)

    return master


def apply_shortlist_flags(master: pd.DataFrame, shortlist_kg: pd.DataFrame, shortlist_c: pd.DataFrame) -> pd.DataFrame:
    master = master.copy()
    kg_names = set(shortlist_kg["wise_name"].astype(str).str.strip())
    c_names = set(shortlist_c["wise_name"].astype(str).str.strip())

    master["in_shortlist_known_group"] = master["wise_name"].isin(kg_names)
    master["in_shortlist_candidate"] = master["wise_name"].isin(c_names)

    master["shortlist_source"] = "none"
    master.loc[master["in_shortlist_known_group"], "shortlist_source"] = "known_group"
    master.loc[master["in_shortlist_candidate"], "shortlist_source"] = "candidate"
    return master


def build_review_tables(master: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    base_cols = [
        "wise_name", "catalog_class", "priority_rank", "shortlist_source",
        "glon", "glat", "ra", "dec", "radius_arcsec", "radius_band",
        "hii_region_name", "membership", "other_names_raw",
        "flag_has_hii_name", "flag_has_membership", "flag_has_other_names",
        "flag_has_distance", "flag_any_nonzero_mid_ir",
        "w3_jy", "w4_jy", "mips24_jy",
        "size_bucket", "complexity_warning", "baseline_first_pass_score"
    ]

    known_group = master[master["in_shortlist_known_group"]].copy()
    known_group = known_group.sort_values(
        by=["baseline_first_pass_score", "radius_arcsec", "wise_name"],
        ascending=[False, True, True]
    )

    candidate = master[master["in_shortlist_candidate"]].copy()
    candidate = candidate.sort_values(
        by=["baseline_first_pass_score", "radius_arcsec", "wise_name"],
        ascending=[False, True, True]
    )

    top_priority = master[
        (master["in_shortlist_known_group"]) &
        (master["radius_arcsec"].between(90, 180, inclusive="both")) &
        (master["complexity_warning"] == "")
    ].copy()

    top_priority = top_priority.sort_values(
        by=["baseline_first_pass_score", "radius_arcsec", "wise_name"],
        ascending=[False, True, True]
    )

    return {
        "known_group": known_group[base_cols],
        "candidate": candidate[base_cols],
        "top_priority": top_priority[base_cols],
    }


def write_manifest(project_root: Path, master: pd.DataFrame, reviews: Dict[str, pd.DataFrame]) -> None:
    lines: List[str] = []
    lines.append("Validation of Structural Contrast Baseline")
    lines.append("WISE H II first-pass derived manifest")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append("This is the first derived-stage summary built from the raw WISE H II catalog and candidate lists.")
    lines.append("It is intended for manual target review before fixing final input targets.")
    lines.append("")
    lines.append("Key rule")
    lines.append("--------")
    lines.append("This stage prepares review tables.")
    lines.append("It does not yet define the final input target set automatically.")
    lines.append("")
    lines.append("Source files used")
    lines.append("-----------------")
    lines.append(str(project_root / RAW_V23))
    lines.append(str(project_root / RAW_V13))
    lines.append(str(project_root / RAW_NORMALIZED))
    lines.append(str(project_root / RAW_SHORTLIST_KG))
    lines.append(str(project_root / RAW_SHORTLIST_C))
    lines.append("")
    lines.append("Output files")
    lines.append("------------")
    lines.append(str(project_root / OUT_MASTER))
    lines.append(str(project_root / OUT_REVIEW_KG))
    lines.append(str(project_root / OUT_REVIEW_C))
    lines.append(str(project_root / OUT_REVIEW_TOP))
    lines.append("")
    lines.append("Summary")
    lines.append("-------")
    lines.append(f"master rows: {len(master)}")
    lines.append(f"known/group review rows: {len(reviews['known_group'])}")
    lines.append(f"candidate review rows: {len(reviews['candidate'])}")
    lines.append(f"top-priority first-pass rows: {len(reviews['top_priority'])}")
    lines.append("")
    lines.append("Recommended next step")
    lines.append("---------------------")
    lines.append("1. Open wise_hii_review_top_priority_first_pass.csv")
    lines.append("2. Manually inspect 5 to 10 targets by image cutout")
    lines.append("3. Keep 1 clean shell + 1 mildly asymmetric shell + 1 backup target")
    lines.append("4. Only after that, build data/derived/.../input/")
    lines.append("")

    out_path = project_root / OUT_MANIFEST
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_csv(project_root: Path, master: pd.DataFrame, reviews: Dict[str, pd.DataFrame]) -> None:
    rows = [
        {"metric": "master_rows", "value": len(master)},
        {"metric": "known_group_rows", "value": len(reviews["known_group"])},
        {"metric": "candidate_rows", "value": len(reviews["candidate"])},
        {"metric": "top_priority_rows", "value": len(reviews["top_priority"])},
        {"metric": "radius_90_180_in_known_group", "value": int(((master["in_shortlist_known_group"]) & (master["radius_arcsec"].between(90, 180, inclusive="both"))).sum())},
        {"metric": "known_group_with_complexity_warning", "value": int(((master["in_shortlist_known_group"]) & (master["complexity_warning"] != "")).sum())},
        {"metric": "known_group_with_other_names", "value": int(((master["in_shortlist_known_group"]) & (master["has_other_names"])).sum())},
    ]
    pd.DataFrame(rows).to_csv(project_root / OUT_SUMMARY, index=False, encoding="utf-8-sig")


def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()

    ensure_dirs(project_root)

    # Read raw files
    v23 = read_csv_any(project_root / RAW_V23)
    v13 = read_csv_any(project_root / RAW_V13)
    normalized = read_csv_any(project_root / RAW_NORMALIZED)
    shortlist_kg = read_csv_any(project_root / RAW_SHORTLIST_KG)
    shortlist_c = read_csv_any(project_root / RAW_SHORTLIST_C)

    # Build master
    master = build_master(v23, v13, normalized)
    master = apply_shortlist_flags(master, shortlist_kg, shortlist_c)

    # Review tables
    reviews = build_review_tables(master)

    # Save outputs
    master.to_csv(project_root / OUT_MASTER, index=False, encoding="utf-8-sig")
    reviews["known_group"].to_csv(project_root / OUT_REVIEW_KG, index=False, encoding="utf-8-sig")
    reviews["candidate"].to_csv(project_root / OUT_REVIEW_C, index=False, encoding="utf-8-sig")
    reviews["top_priority"].to_csv(project_root / OUT_REVIEW_TOP, index=False, encoding="utf-8-sig")

    write_manifest(project_root, master, reviews)
    write_summary_csv(project_root, master, reviews)

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - first-pass derived builder")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print("")
    print("[OK] Created:")
    print(project_root / OUT_MASTER)
    print(project_root / OUT_REVIEW_KG)
    print(project_root / OUT_REVIEW_C)
    print(project_root / OUT_REVIEW_TOP)
    print(project_root / OUT_MANIFEST)
    print(project_root / OUT_SUMMARY)
    print("")
    print("Recommended next file to open:")
    print(project_root / OUT_REVIEW_TOP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
