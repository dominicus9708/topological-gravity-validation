#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
WISE H II final input builder

Author: Kwon Dominicus

Purpose
-------
Automatically locate the previously generated derived-stage files and build
final input-ready CSV outputs under:

    data/derived/Validation of Structural Contrast Baseline/input/

This script is intended to be placed at:

    data/derived/Validation of Structural Contrast Baseline/script/

Pipeline position
-----------------
raw -> derived/tables + derived/review -> input

Input discovery rule
--------------------
The script automatically looks for the following four files:

1) data/derived/Validation of Structural Contrast Baseline/wise_hii_catalog/tables/
   - wise_hii_master_catalog_for_baseline_review.csv

2) data/derived/Validation of Structural Contrast Baseline/wise_hii_catalog/review/
   - wise_hii_review_known_group_priority.csv
   - wise_hii_review_candidate_priority.csv
   - wise_hii_review_top_priority_first_pass.csv

It then creates:
- an input candidate pool
- an input top shortlist
- an image-review queue
- a manual selection template

This script does NOT download images.
It prepares the final input-stage tables for human review and next-step cutout work.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_PROJECT_ROOT = Path(r"C:\Users\mincu\Desktop\topological_gravity_project")

DERIVED_BASE = Path("data") / "derived" / "Validation of Structural Contrast Baseline"
WISE_BASE = DERIVED_BASE / "wise_hii_catalog"

TABLES_DIR = WISE_BASE / "tables"
REVIEW_DIR = WISE_BASE / "review"

INPUT_DIR = DERIVED_BASE / "input"
INPUT_WISE_DIR = INPUT_DIR / "wise_hii_catalog"
INPUT_LOG_DIR = INPUT_DIR / "logs"

# Auto-discovered required files
MASTER_FILE = TABLES_DIR / "wise_hii_master_catalog_for_baseline_review.csv"
KNOWN_GROUP_FILE = REVIEW_DIR / "wise_hii_review_known_group_priority.csv"
CANDIDATE_FILE = REVIEW_DIR / "wise_hii_review_candidate_priority.csv"
TOP_PRIORITY_FILE = REVIEW_DIR / "wise_hii_review_top_priority_first_pass.csv"

# Output files
OUT_INPUT_POOL = INPUT_WISE_DIR / "wise_hii_input_candidate_pool.csv"
OUT_INPUT_TOP = INPUT_WISE_DIR / "wise_hii_input_top_shortlist.csv"
OUT_IMAGE_QUEUE = INPUT_WISE_DIR / "wise_hii_image_review_queue.csv"
OUT_MANUAL_TEMPLATE = INPUT_WISE_DIR / "wise_hii_manual_selection_template.csv"
OUT_MANIFEST = INPUT_LOG_DIR / "wise_hii_input_build_manifest.txt"
OUT_SUMMARY = INPUT_LOG_DIR / "wise_hii_input_build_summary.csv"


def ensure_dirs(project_root: Path) -> None:
    for d in [INPUT_DIR, INPUT_WISE_DIR, INPUT_LOG_DIR]:
        (project_root / d).mkdir(parents=True, exist_ok=True)


def must_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def build_input_pool(master: pd.DataFrame) -> pd.DataFrame:
    """
    Final input-stage candidate pool.
    Keep only the columns that are useful for next human review and target fixing.
    """
    df = master.copy()

    # Ensure expected columns exist even if earlier versions differ slightly
    expected_defaults = {
        "wise_name": "",
        "catalog_class": "",
        "priority_rank": "",
        "shortlist_source": "",
        "glon": "",
        "glat": "",
        "ra": "",
        "dec": "",
        "radius_arcsec": "",
        "radius_band": "",
        "hii_region_name": "",
        "membership": "",
        "other_names_raw": "",
        "flag_has_hii_name": "",
        "flag_has_membership": "",
        "flag_has_other_names": "",
        "flag_has_distance": "",
        "flag_any_nonzero_mid_ir": "",
        "w3_jy": "",
        "w4_jy": "",
        "mips24_jy": "",
        "size_bucket": "",
        "complexity_warning": "",
        "baseline_first_pass_score": "",
    }
    for c, v in expected_defaults.items():
        if c not in df.columns:
            df[c] = v

    # Final pool logic:
    # - include known_group and candidate sources
    # - keep bands useful for first validation
    # - keep outside band only if score is still strong
    shortlist_mask = df["shortlist_source"].isin(["known_group", "candidate"])
    band_mask = df["radius_band"].isin(["preferred_core_band", "usable_band", "large_band"])
    score_mask = pd.to_numeric(df["baseline_first_pass_score"], errors="coerce").fillna(0) >= 20

    pool = df[shortlist_mask & (band_mask | score_mask)].copy()

    pool["selection_stage"] = "input_pool"
    pool["recommended_review_order"] = (
        pool["baseline_first_pass_score"].rank(method="dense", ascending=False).astype(int)
    )

    cols = [
        "wise_name", "selection_stage", "recommended_review_order",
        "catalog_class", "priority_rank", "shortlist_source",
        "glon", "glat", "ra", "dec",
        "radius_arcsec", "radius_band",
        "hii_region_name", "membership", "other_names_raw",
        "flag_has_hii_name", "flag_has_membership", "flag_has_other_names",
        "flag_has_distance", "flag_any_nonzero_mid_ir",
        "w3_jy", "w4_jy", "mips24_jy",
        "size_bucket", "complexity_warning", "baseline_first_pass_score"
    ]
    pool = pool[cols].sort_values(
        by=["baseline_first_pass_score", "radius_arcsec", "wise_name"],
        ascending=[False, True, True]
    )
    return pool


def build_top_shortlist(top_priority: pd.DataFrame, n_keep: int = 150) -> pd.DataFrame:
    """
    Build a smaller input shortlist from top_priority.
    This is not yet the final physical target list.
    It is the final shortlist for image review and human choice.
    """
    df = top_priority.copy()

    # Standardize columns if needed
    if "wise_name" not in df.columns:
        raise ValueError("top priority file does not contain wise_name")

    if "baseline_first_pass_score" in df.columns:
        df["baseline_first_pass_score_num"] = pd.to_numeric(df["baseline_first_pass_score"], errors="coerce").fillna(0)
    else:
        df["baseline_first_pass_score_num"] = 0

    df["input_priority_tier"] = "tier_2"
    df.loc[df["baseline_first_pass_score_num"] >= 55, "input_priority_tier"] = "tier_1"

    shortlist = df.sort_values(
        by=["baseline_first_pass_score_num", "radius_arcsec", "wise_name"],
        ascending=[False, True, True]
    ).head(n_keep).copy()

    shortlist["selection_stage"] = "input_top_shortlist"
    shortlist["manual_cutout_check"] = "pending"
    shortlist["manual_shell_morphology"] = ""
    shortlist["manual_background_cleanliness"] = ""
    shortlist["manual_overlap_risk"] = ""
    shortlist["manual_keep_for_final_input"] = ""

    ordered_cols = list(shortlist.columns)
    first_cols = [
        "wise_name", "selection_stage", "input_priority_tier",
        "manual_cutout_check", "manual_shell_morphology",
        "manual_background_cleanliness", "manual_overlap_risk",
        "manual_keep_for_final_input"
    ]
    remaining = [c for c in ordered_cols if c not in first_cols]
    shortlist = shortlist[first_cols + remaining]
    return shortlist


def build_image_review_queue(shortlist: pd.DataFrame, n_queue: int = 30) -> pd.DataFrame:
    """
    Practical first image-review queue.
    Use top shortlist and reduce to a manageable queue for manual cutout inspection.
    """
    df = shortlist.copy()

    # Prefer tier_1 first, then smallest-to-mid radius within recommended band
    if "baseline_first_pass_score_num" not in df.columns:
        df["baseline_first_pass_score_num"] = pd.to_numeric(df.get("baseline_first_pass_score", 0), errors="coerce").fillna(0)

    if "radius_arcsec" in df.columns:
        df["radius_arcsec_num"] = pd.to_numeric(df["radius_arcsec"], errors="coerce")
    else:
        df["radius_arcsec_num"] = pd.NA

    queue = df.sort_values(
        by=["input_priority_tier", "baseline_first_pass_score_num", "radius_arcsec_num", "wise_name"],
        ascending=[True, False, True, True]
    ).head(n_queue).copy()

    queue["image_review_order"] = range(1, len(queue) + 1)
    queue["cutout_download_status"] = "pending"
    queue["visual_grade"] = ""
    queue["visual_comment"] = ""

    keep_cols = [
        "image_review_order", "wise_name", "input_priority_tier",
        "catalog_class", "shortlist_source",
        "glon", "glat", "ra", "dec", "radius_arcsec", "radius_band",
        "hii_region_name", "membership", "other_names_raw",
        "complexity_warning", "baseline_first_pass_score",
        "cutout_download_status", "visual_grade", "visual_comment"
    ]
    existing = [c for c in keep_cols if c in queue.columns]
    return queue[existing]


def build_manual_template(image_queue: pd.DataFrame) -> pd.DataFrame:
    """
    Create the template that the user can manually fill after image inspection.
    """
    df = image_queue.copy()
    template = pd.DataFrame({
        "wise_name": df["wise_name"],
        "manual_final_rank": "",
        "manual_keep": "",
        "manual_reason": "",
        "manual_shell_type": "",
        "manual_background_ring_defined": "",
        "manual_notes": "",
    })
    return template


def write_manifest(project_root: Path, rows: Dict[str, int]) -> None:
    lines: List[str] = []
    lines.append("Validation of Structural Contrast Baseline")
    lines.append("WISE H II final input builder manifest")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Auto-discovered source files")
    lines.append("----------------------------")
    lines.append(str(project_root / MASTER_FILE))
    lines.append(str(project_root / KNOWN_GROUP_FILE))
    lines.append(str(project_root / CANDIDATE_FILE))
    lines.append(str(project_root / TOP_PRIORITY_FILE))
    lines.append("")
    lines.append("Output files")
    lines.append("------------")
    lines.append(str(project_root / OUT_INPUT_POOL))
    lines.append(str(project_root / OUT_INPUT_TOP))
    lines.append(str(project_root / OUT_IMAGE_QUEUE))
    lines.append(str(project_root / OUT_MANUAL_TEMPLATE))
    lines.append("")
    lines.append("Row summary")
    lines.append("-----------")
    for key, value in rows.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Interpretation")
    lines.append("--------------")
    lines.append("input_candidate_pool.csv  : all workable input-stage candidates")
    lines.append("input_top_shortlist.csv   : narrowed shortlist for next review step")
    lines.append("image_review_queue.csv    : first manual cutout inspection queue")
    lines.append("manual_selection_template.csv : file to fill after visual review")
    lines.append("")
    lines.append("Recommended next step")
    lines.append("---------------------")
    lines.append("1. Open wise_hii_image_review_queue.csv")
    lines.append("2. Download or inspect cutouts for the first 10 to 30 rows")
    lines.append("3. Fill manual_selection_template.csv")
    lines.append("4. Then build the final fixed input file from the manual keep list")
    lines.append("")

    (project_root / OUT_MANIFEST).write_text("\n".join(lines), encoding="utf-8")


def write_summary(project_root: Path, rows: Dict[str, int]) -> None:
    summary = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in rows.items()]
    )
    summary.to_csv(project_root / OUT_SUMMARY, index=False, encoding="utf-8-sig")


def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()

    ensure_dirs(project_root)

    # Auto-discover the four files
    master = must_read_csv(project_root / MASTER_FILE)
    known_group = must_read_csv(project_root / KNOWN_GROUP_FILE)
    candidate = must_read_csv(project_root / CANDIDATE_FILE)
    top_priority = must_read_csv(project_root / TOP_PRIORITY_FILE)

    # Build outputs
    input_pool = build_input_pool(master)
    top_shortlist = build_top_shortlist(top_priority, n_keep=150)
    image_queue = build_image_review_queue(top_shortlist, n_queue=30)
    manual_template = build_manual_template(image_queue)

    # Save
    input_pool.to_csv(project_root / OUT_INPUT_POOL, index=False, encoding="utf-8-sig")
    top_shortlist.to_csv(project_root / OUT_INPUT_TOP, index=False, encoding="utf-8-sig")
    image_queue.to_csv(project_root / OUT_IMAGE_QUEUE, index=False, encoding="utf-8-sig")
    manual_template.to_csv(project_root / OUT_MANUAL_TEMPLATE, index=False, encoding="utf-8-sig")

    rows = {
        "master_rows": len(master),
        "known_group_rows": len(known_group),
        "candidate_rows": len(candidate),
        "top_priority_rows": len(top_priority),
        "input_pool_rows": len(input_pool),
        "top_shortlist_rows": len(top_shortlist),
        "image_review_queue_rows": len(image_queue),
        "manual_template_rows": len(manual_template),
    }

    write_manifest(project_root, rows)
    write_summary(project_root, rows)

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - final input builder")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print("")
    print("[OK] Created:")
    print(project_root / OUT_INPUT_POOL)
    print(project_root / OUT_INPUT_TOP)
    print(project_root / OUT_IMAGE_QUEUE)
    print(project_root / OUT_MANUAL_TEMPLATE)
    print(project_root / OUT_MANIFEST)
    print(project_root / OUT_SUMMARY)
    print("")
    print("Recommended next file to open:")
    print(project_root / OUT_IMAGE_QUEUE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
