#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
Skeleton pipeline for WISE H II first-pass targets

Author: Kwon Dominicus

Purpose
-------
This script is the skeleton-stage pipeline for:

    src/Validation of Structural Contrast Baseline/skeleton/

It reads the fixed final input file:

    data/derived/Validation of Structural Contrast Baseline/input/wise_hii_catalog/wise_hii_final_input.csv

and writes timestamped outputs under:

    results/Validation of Structural Contrast Baseline/output/skeleton/YYYYMMDD_HHMMSS/
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(r"C:\Users\mincu\Desktop\topological_gravity_project")

INPUT_FILE = Path("data") / "derived" / "Validation of Structural Contrast Baseline" / "input" / "wise_hii_catalog" / "wise_hii_final_input.csv"
RESULTS_BASE = Path("results") / "Validation of Structural Contrast Baseline" / "output" / "skeleton"


@dataclass
class TargetGuide:
    wise_name: str
    radius_arcsec: float
    shell_inner_arcsec: float
    shell_outer_arcsec: float
    bg_inner_arcsec: float
    bg_outer_arcsec: float


def ensure_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Final input is missing required columns: " + ", ".join(missing))


def sanitize_wise_name(name: str) -> str:
    text = str(name).strip()
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def create_timestamped_output_dir(project_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = project_root / RESULTS_BASE / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_final_input(project_root: Path) -> pd.DataFrame:
    path = project_root / INPUT_FILE
    if not path.exists():
        raise FileNotFoundError(f"Final input file not found: {path}")
    return pd.read_csv(path, dtype=str, low_memory=False)


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = [
        "wise_name",
        "manual_final_rank",
        "manual_reason",
        "manual_shell_type",
        "manual_background_ring_defined",
        "catalog_class",
        "glon",
        "glat",
        "ra",
        "dec",
        "radius_arcsec",
        "target_status",
        "input_ready",
    ]
    ensure_required_columns(df, required)

    df["radius_arcsec_num"] = pd.to_numeric(df["radius_arcsec"], errors="coerce")
    if df["radius_arcsec_num"].isna().any():
        bad = df.loc[df["radius_arcsec_num"].isna(), "wise_name"].tolist()
        raise ValueError("Invalid radius_arcsec values for: " + ", ".join(map(str, bad)))

    df["manual_final_rank_num"] = pd.to_numeric(df["manual_final_rank"], errors="coerce")
    df["wise_name"] = df["wise_name"].astype(str).str.strip()

    if df["wise_name"].duplicated().any():
        dup = df.loc[df["wise_name"].duplicated(), "wise_name"].tolist()
        raise ValueError("Duplicate wise_name entries found: " + ", ".join(dup))

    not_ready = df.loc[df["input_ready"].astype(str).str.lower().ne("yes"), "wise_name"].tolist()
    if not_ready:
        raise ValueError("Rows not marked input_ready=yes: " + ", ".join(not_ready))

    not_fixed = df.loc[df["target_status"].astype(str).ne("fixed_final_input"), "wise_name"].tolist()
    if not_fixed:
        raise ValueError("Rows not fixed_final_input: " + ", ".join(not_fixed))

    df = df.sort_values(
        by=["manual_final_rank_num", "radius_arcsec_num", "wise_name"],
        ascending=[True, True, True],
        na_position="last"
    ).reset_index(drop=True)

    return df


def build_guides(df: pd.DataFrame) -> List[TargetGuide]:
    guides: List[TargetGuide] = []
    for _, row in df.iterrows():
        r = float(row["radius_arcsec_num"])
        shell_inner = max(0.60 * r, 1.0)
        shell_outer = 1.00 * r
        bg_inner = 1.20 * r
        bg_outer = 1.80 * r
        guides.append(
            TargetGuide(
                wise_name=str(row["wise_name"]),
                radius_arcsec=r,
                shell_inner_arcsec=round(shell_inner, 3),
                shell_outer_arcsec=round(shell_outer, 3),
                bg_inner_arcsec=round(bg_inner, 3),
                bg_outer_arcsec=round(bg_outer, 3),
            )
        )
    return guides


def guides_to_dataframe(guides: List[TargetGuide]) -> pd.DataFrame:
    return pd.DataFrame([g.__dict__ for g in guides])


def build_target_summary(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "wise_name",
        "manual_final_rank",
        "manual_reason",
        "manual_shell_type",
        "manual_background_ring_defined",
        "catalog_class",
        "shortlist_source",
        "glon",
        "glat",
        "ra",
        "dec",
        "radius_arcsec",
        "radius_band",
        "baseline_first_pass_score",
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = ""
    return df[keep_cols].copy()


def save_plot_for_target(target: pd.Series, guide: TargetGuide, out_dir: Path) -> None:
    name = sanitize_wise_name(target["wise_name"])
    target_dir = out_dir / "targets" / name
    target_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvspan(guide.shell_inner_arcsec, guide.shell_outer_arcsec, alpha=0.25, label="shell guide")
    ax.axvspan(guide.bg_inner_arcsec, guide.bg_outer_arcsec, alpha=0.18, label="background guide")
    ax.axvline(guide.radius_arcsec, linestyle="--", label="catalog radius")
    ax.set_xlabel("Radius (arcsec)")
    ax.set_ylabel("Guide level")
    ax.set_title(f"{target['wise_name']} skeleton radial guide")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.legend()
    fig.tight_layout()
    fig.savefig(target_dir / f"{name}_skeleton_radial_guide.png", dpi=160)
    plt.close(fig)


def save_per_target_summary(target: pd.Series, guide: TargetGuide, out_dir: Path) -> None:
    name = sanitize_wise_name(target["wise_name"])
    target_dir = out_dir / "targets" / name
    target_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"wise_name: {target['wise_name']}",
        f"manual_final_rank: {target.get('manual_final_rank', '')}",
        f"manual_reason: {target.get('manual_reason', '')}",
        f"manual_shell_type: {target.get('manual_shell_type', '')}",
        f"manual_background_ring_defined: {target.get('manual_background_ring_defined', '')}",
        f"catalog_class: {target.get('catalog_class', '')}",
        f"shortlist_source: {target.get('shortlist_source', '')}",
        f"glon: {target.get('glon', '')}",
        f"glat: {target.get('glat', '')}",
        f"ra: {target.get('ra', '')}",
        f"dec: {target.get('dec', '')}",
        f"radius_arcsec: {target.get('radius_arcsec', '')}",
        "",
        "skeleton radial guide",
        f"shell_inner_arcsec: {guide.shell_inner_arcsec}",
        f"shell_outer_arcsec: {guide.shell_outer_arcsec}",
        f"bg_inner_arcsec: {guide.bg_inner_arcsec}",
        f"bg_outer_arcsec: {guide.bg_outer_arcsec}",
    ]
    (target_dir / f"{name}_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(project_root: Path, out_dir: Path, df: pd.DataFrame, guides_df: pd.DataFrame) -> None:
    lines = [
        "Validation of Structural Contrast Baseline",
        "Skeleton pipeline manifest",
        "=" * 60,
        "",
        f"project_root: {project_root}",
        f"input_file: {project_root / INPUT_FILE}",
        f"output_dir: {out_dir}",
        "",
        f"target_count: {len(df)}",
        "",
        "selected targets",
        "-" * 20,
    ]
    for _, row in df.iterrows():
        lines.append(f"- {row['wise_name']} | rank={row.get('manual_final_rank', '')} | radius_arcsec={row.get('radius_arcsec', '')}")
    lines += [
        "",
        "meaning",
        "-" * 20,
        "This is a skeleton-stage output only.",
        "The shell/background ranges are guide definitions for the next standard stage.",
    ]
    (out_dir / "run_manifest.txt").write_text("\n".join(lines), encoding="utf-8")


def write_summary_csv(out_dir: Path, df: pd.DataFrame, guides_df: pd.DataFrame) -> None:
    rows = [
        {"metric": "target_count", "value": len(df)},
        {"metric": "radius_min_arcsec", "value": float(df["radius_arcsec_num"].min())},
        {"metric": "radius_median_arcsec", "value": float(df["radius_arcsec_num"].median())},
        {"metric": "radius_max_arcsec", "value": float(df["radius_arcsec_num"].max())},
        {"metric": "shell_inner_min_arcsec", "value": float(guides_df["shell_inner_arcsec"].min())},
        {"metric": "shell_outer_max_arcsec", "value": float(guides_df["shell_outer_arcsec"].max())},
        {"metric": "bg_outer_max_arcsec", "value": float(guides_df["bg_outer_arcsec"].max())},
    ]
    pd.DataFrame(rows).to_csv(out_dir / "run_summary.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()

    out_dir = create_timestamped_output_dir(project_root)

    raw_df = load_final_input(project_root)
    df = normalize_input(raw_df)
    guides = build_guides(df)
    guides_df = guides_to_dataframe(guides)
    summary_df = build_target_summary(df)

    df.to_csv(out_dir / "fixed_final_input_copy.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "target_summary.csv", index=False, encoding="utf-8-sig")
    guides_df.to_csv(out_dir / "target_shell_background_guides.csv", index=False, encoding="utf-8-sig")

    guide_map = {g.wise_name: g for g in guides}
    for _, row in df.iterrows():
        guide = guide_map[str(row["wise_name"])]
        save_per_target_summary(row, guide, out_dir)
        save_plot_for_target(row, guide, out_dir)

    write_manifest(project_root, out_dir, df, guides_df)
    write_summary_csv(out_dir, df, guides_df)

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - skeleton pipeline")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print(f"Input file  : {project_root / INPUT_FILE}")
    print(f"Output dir  : {out_dir}")
    print("")
    print("[OK] Created:")
    print(out_dir / "fixed_final_input_copy.csv")
    print(out_dir / "target_summary.csv")
    print(out_dir / "target_shell_background_guides.csv")
    print(out_dir / "run_manifest.txt")
    print(out_dir / "run_summary.csv")
    print(out_dir / "targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
