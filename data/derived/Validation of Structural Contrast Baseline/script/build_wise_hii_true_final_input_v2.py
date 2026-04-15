#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
WISE H II true final input builder (revised)

- If wise_hii_manual_selection_filled.csv exists and is filled:
    build the true final input.
- If it does not exist:
    auto-create a starter file from the existing template or image review queue,
    then stop cleanly with instructions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\mincu\Desktop\topological_gravity_project")

DERIVED_BASE = Path("data") / "derived" / "Validation of Structural Contrast Baseline"
WISE_BASE = DERIVED_BASE / "wise_hii_catalog"

TABLES_DIR = WISE_BASE / "tables"
REVIEW_DIR = WISE_BASE / "review"

INPUT_DIR = DERIVED_BASE / "input"
INPUT_WISE_DIR = INPUT_DIR / "wise_hii_catalog"
INPUT_LOG_DIR = INPUT_DIR / "logs"

MASTER_FILE = TABLES_DIR / "wise_hii_master_catalog_for_baseline_review.csv"
TOP_PRIORITY_FILE = REVIEW_DIR / "wise_hii_review_top_priority_first_pass.csv"

IMAGE_QUEUE_FILE = INPUT_WISE_DIR / "wise_hii_image_review_queue.csv"
MANUAL_TEMPLATE_FILE = INPUT_WISE_DIR / "wise_hii_manual_selection_template.csv"

MANUAL_FILLED_FILE = REVIEW_DIR / "wise_hii_manual_selection_filled.csv"

OUT_FINAL_INPUT = INPUT_WISE_DIR / "wise_hii_final_input.csv"
OUT_FINAL_TARGETS = INPUT_WISE_DIR / "wise_hii_final_targets_summary.csv"
OUT_FINAL_MANIFEST = INPUT_LOG_DIR / "wise_hii_final_input_manifest.txt"
OUT_FINAL_SUMMARY = INPUT_LOG_DIR / "wise_hii_final_input_summary.csv"


def ensure_dirs(project_root: Path) -> None:
    for d in [INPUT_DIR, INPUT_WISE_DIR, INPUT_LOG_DIR, REVIEW_DIR]:
        (project_root / d).mkdir(parents=True, exist_ok=True)


def must_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, dtype=str, low_memory=False)


def normalize_keep_flag(value) -> str:
    text = str(value).strip().lower()
    if text in {"yes", "y", "1", "true", "keep"}:
        return "yes"
    return "no"


def validate_manual_file(df: pd.DataFrame) -> None:
    required = ["wise_name", "manual_keep"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Manual selection file is missing required columns: "
            + ", ".join(missing)
        )


def maybe_create_manual_filled_starter(project_root: Path) -> Path | None:
    target = project_root / MANUAL_FILLED_FILE
    if target.exists():
        return target

    source_template = project_root / MANUAL_TEMPLATE_FILE
    source_queue = project_root / IMAGE_QUEUE_FILE

    starter = None

    if source_template.exists():
        df = pd.read_csv(source_template, dtype=str, low_memory=False)
        if "manual_keep" not in df.columns:
            df["manual_keep"] = ""
        starter = df.copy()

    elif source_queue.exists():
        q = pd.read_csv(source_queue, dtype=str, low_memory=False)
        starter = pd.DataFrame({
            "wise_name": q["wise_name"] if "wise_name" in q.columns else [],
            "manual_final_rank": "",
            "manual_keep": "",
            "manual_reason": "",
            "manual_shell_type": "",
            "manual_background_ring_defined": "",
            "manual_notes": "",
        })

    if starter is None:
        return None

    target.parent.mkdir(parents=True, exist_ok=True)
    starter.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def build_final_input(master: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    manual = manual.copy()
    validate_manual_file(manual)

    manual["wise_name"] = manual["wise_name"].astype(str).str.strip()
    manual["manual_keep"] = manual["manual_keep"].apply(normalize_keep_flag)

    keep_df = manual[manual["manual_keep"] == "yes"].copy()
    if keep_df.empty:
        raise ValueError(
            "No rows marked as keep=yes in wise_hii_manual_selection_filled.csv"
        )

    master = master.copy()
    master["wise_name"] = master["wise_name"].astype(str).str.strip()

    final_df = keep_df.merge(master, on="wise_name", how="left", suffixes=("_manual", ""))

    unresolved = final_df["catalog_class"].isna().sum() if "catalog_class" in final_df.columns else len(final_df)
    if unresolved > 0:
        raise ValueError(
            f"{unresolved} selected target(s) could not be matched back to the master review catalog."
        )

    if "manual_final_rank" not in final_df.columns:
        final_df["manual_final_rank"] = ""
    if "baseline_first_pass_score" not in final_df.columns:
        final_df["baseline_first_pass_score"] = ""
    if "radius_arcsec" not in final_df.columns:
        final_df["radius_arcsec"] = ""

    final_df["manual_final_rank_num"] = pd.to_numeric(final_df["manual_final_rank"], errors="coerce")
    final_df["baseline_first_pass_score_num"] = pd.to_numeric(final_df["baseline_first_pass_score"], errors="coerce")
    final_df["radius_arcsec_num"] = pd.to_numeric(final_df["radius_arcsec"], errors="coerce")

    final_df = final_df.sort_values(
        by=["manual_final_rank_num", "baseline_first_pass_score_num", "radius_arcsec_num", "wise_name"],
        ascending=[True, False, True, True],
        na_position="last"
    ).copy()

    desired_cols = [
        "wise_name",
        "manual_final_rank",
        "manual_reason",
        "manual_shell_type",
        "manual_background_ring_defined",
        "manual_notes",
        "catalog_class",
        "shortlist_source",
        "glon",
        "glat",
        "ra",
        "dec",
        "radius_arcsec",
        "radius_band",
        "hii_region_name",
        "membership",
        "other_names_raw",
        "flag_has_hii_name",
        "flag_has_membership",
        "flag_has_other_names",
        "flag_has_distance",
        "flag_any_nonzero_mid_ir",
        "w3_jy",
        "w4_jy",
        "mips24_jy",
        "complexity_warning",
        "baseline_first_pass_score",
    ]

    for col in desired_cols:
        if col not in final_df.columns:
            final_df[col] = ""

    final_df = final_df[desired_cols].copy()
    final_df["target_status"] = "fixed_final_input"
    final_df["input_ready"] = "yes"

    ordered_front = [
        "wise_name",
        "target_status",
        "input_ready",
        "manual_final_rank",
        "manual_reason",
        "manual_shell_type",
        "manual_background_ring_defined",
        "manual_notes",
    ]
    remaining = [c for c in final_df.columns if c not in ordered_front]
    final_df = final_df[ordered_front + remaining]

    return final_df


def build_final_targets_summary(final_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "wise_name",
        "manual_final_rank",
        "manual_reason",
        "manual_shell_type",
        "manual_background_ring_defined",
        "radius_arcsec",
        "catalog_class",
        "shortlist_source",
        "baseline_first_pass_score",
    ]
    existing = [c for c in keep_cols if c in final_df.columns]
    return final_df[existing].copy()


def write_manifest(project_root: Path, final_df: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("Validation of Structural Contrast Baseline")
    lines.append("WISE H II true final input manifest")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Files used")
    lines.append("----------")
    lines.append(str(project_root / MASTER_FILE))
    lines.append(str(project_root / TOP_PRIORITY_FILE))
    lines.append(str(project_root / MANUAL_FILLED_FILE))
    lines.append("")
    lines.append("Files created")
    lines.append("-------------")
    lines.append(str(project_root / OUT_FINAL_INPUT))
    lines.append(str(project_root / OUT_FINAL_TARGETS))
    lines.append(str(project_root / OUT_FINAL_MANIFEST))
    lines.append(str(project_root / OUT_FINAL_SUMMARY))
    lines.append("")
    lines.append("Final target count")
    lines.append("------------------")
    lines.append(str(len(final_df)))
    lines.append("")
    for _, row in final_df.iterrows():
        lines.append(
            f"- {row.get('wise_name','')} | rank={row.get('manual_final_rank','')} | reason={row.get('manual_reason','')}"
        )
    lines.append("")
    (project_root / OUT_FINAL_MANIFEST).write_text("\n".join(lines), encoding="utf-8")


def write_summary(project_root: Path, final_df: pd.DataFrame) -> None:
    rows = [
        {"metric": "final_input_rows", "value": len(final_df)},
        {"metric": "unique_wise_names", "value": final_df["wise_name"].nunique()},
    ]
    if "manual_shell_type" in final_df.columns:
        for shell_type, count in final_df["manual_shell_type"].fillna("").value_counts().items():
            rows.append({"metric": f"manual_shell_type::{shell_type or 'blank'}", "value": int(count)})
    pd.DataFrame(rows).to_csv(project_root / OUT_FINAL_SUMMARY, index=False, encoding="utf-8-sig")


def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()

    ensure_dirs(project_root)

    master = must_read_csv(project_root / MASTER_FILE)
    _top_priority = must_read_csv(project_root / TOP_PRIORITY_FILE)

    manual_path = maybe_create_manual_filled_starter(project_root)
    if manual_path is None:
        raise FileNotFoundError(
            "Could not find or create wise_hii_manual_selection_filled.csv. "
            "Also could not find a template or image review queue to bootstrap from."
        )

    manual = must_read_csv(manual_path)

    if "manual_keep" in manual.columns and manual["manual_keep"].fillna("").astype(str).str.strip().eq("").all():
        print("=" * 72)
        print("Starter manual selection file created.")
        print("=" * 72)
        print("Please fill this file first, then rerun:")
        print(manual_path)
        print("")
        print("Required action:")
        print("- set manual_keep = yes or no")
        print("- optionally fill rank/reason/shell_type/background info")
        return 0

    final_df = build_final_input(master, manual)
    final_targets = build_final_targets_summary(final_df)

    final_df.to_csv(project_root / OUT_FINAL_INPUT, index=False, encoding="utf-8-sig")
    final_targets.to_csv(project_root / OUT_FINAL_TARGETS, index=False, encoding="utf-8-sig")

    write_manifest(project_root, final_df)
    write_summary(project_root, final_df)

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - true final input builder")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print("")
    print("[OK] Created:")
    print(project_root / OUT_FINAL_INPUT)
    print(project_root / OUT_FINAL_TARGETS)
    print(project_root / OUT_FINAL_MANIFEST)
    print(project_root / OUT_FINAL_SUMMARY)
    print("")
    print("This output is the real final input stage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
