#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
Standard pipeline (circular schematic version) for WISE H II first-pass targets

Author: Kwon Dominicus

Purpose
-------
This script is the standard-stage pipeline for:

    src/Validation of Structural Contrast Baseline/standard/

It reads:

    data/derived/Validation of Structural Contrast Baseline/input/wise_hii_catalog/wise_hii_final_input.csv

and writes to:

    results/Validation of Structural Contrast Baseline/output/standard/YYYYMMDD_HHMMSS/

Revision note
-------------
This version adds a circular standard-baseline schematic with:
- central schematic circle (target center marker)
- inner baseline zone
- shell structure zone
- background baseline zone
- cross axes
- title and legend in Korean

The central circle is schematic, not to physical scale.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(r"C:\Users\mincu\Desktop\topological_gravity_project")

INPUT_FILE = Path("data") / "derived" / "Validation of Structural Contrast Baseline" / "input" / "wise_hii_catalog" / "wise_hii_final_input.csv"
RESULTS_BASE = Path("results") / "Validation of Structural Contrast Baseline" / "output" / "standard"


@dataclass
class TargetZones:
    wise_name: str
    radius_arcsec: float
    inner_inner_arcsec: float
    inner_outer_arcsec: float
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


def build_standard_zones(df: pd.DataFrame) -> List[TargetZones]:
    zones: List[TargetZones] = []
    for _, row in df.iterrows():
        r = float(row["radius_arcsec_num"])
        zones.append(
            TargetZones(
                wise_name=str(row["wise_name"]),
                radius_arcsec=r,
                inner_inner_arcsec=0.0,
                inner_outer_arcsec=round(max(0.50 * r, 1.0), 3),
                shell_inner_arcsec=round(0.60 * r, 3),
                shell_outer_arcsec=round(1.00 * r, 3),
                bg_inner_arcsec=round(1.20 * r, 3),
                bg_outer_arcsec=round(1.80 * r, 3),
            )
        )
    return zones


def zones_to_dataframe(zones: List[TargetZones]) -> pd.DataFrame:
    return pd.DataFrame([z.__dict__ for z in zones])


def build_baseline_comparison_table(df: pd.DataFrame, zones_df: pd.DataFrame) -> pd.DataFrame:
    zones_tmp = zones_df.rename(columns={"radius_arcsec": "zone_radius_arcsec"}).copy()
    merged = df.merge(zones_tmp, on="wise_name", how="left")
    merged["baseline_geometry_status"] = "defined"
    merged["baseline_ready_for_profile_extraction"] = "yes"

    keep_cols = [
        "wise_name",
        "manual_final_rank",
        "manual_reason",
        "manual_shell_type",
        "manual_background_ring_defined",
        "catalog_class",
        "shortlist_source",
        "ra",
        "dec",
        "radius_arcsec",
        "zone_radius_arcsec",
        "inner_inner_arcsec",
        "inner_outer_arcsec",
        "shell_inner_arcsec",
        "shell_outer_arcsec",
        "bg_inner_arcsec",
        "bg_outer_arcsec",
        "baseline_geometry_status",
        "baseline_ready_for_profile_extraction",
        "baseline_first_pass_score",
    ]
    for c in keep_cols:
        if c not in merged.columns:
            merged[c] = ""
    return merged[keep_cols].copy()


def save_linear_plot_for_target(target: pd.Series, zone: TargetZones, out_dir: Path) -> None:
    name = sanitize_wise_name(target["wise_name"])
    target_dir = out_dir / "targets" / name
    target_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvspan(zone.inner_inner_arcsec, zone.inner_outer_arcsec, alpha=0.18, label="inner region")
    ax.axvspan(zone.shell_inner_arcsec, zone.shell_outer_arcsec, alpha=0.25, label="shell region")
    ax.axvspan(zone.bg_inner_arcsec, zone.bg_outer_arcsec, alpha=0.18, label="background region")
    ax.axvline(zone.radius_arcsec, linestyle="--", label="catalog radius")
    ax.set_xlabel("Radius (arcsec)")
    ax.set_ylabel("Baseline zone level")
    ax.set_title(f"{target['wise_name']} standard baseline zones")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.legend()
    fig.tight_layout()
    fig.savefig(target_dir / f"{name}_standard_baseline_zones_linear.png", dpi=160)
    plt.close(fig)


def save_circular_plot_for_target(target: pd.Series, zone: TargetZones, out_dir: Path) -> None:
    name = sanitize_wise_name(target["wise_name"])
    target_dir = out_dir / "targets" / name
    target_dir.mkdir(parents=True, exist_ok=True)

    rmax = zone.bg_outer_arcsec
    center_circle_r = max(0.03 * rmax, 3.0)  # schematic center marker, not physical scale

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.08, 0.08, 0.72, 0.82])
    legend_ax = fig.add_axes([0.81, 0.72, 0.16, 0.2])
    title_ax = fig.add_axes([0.18, 0.92, 0.42, 0.06])

    # Title box
    title_ax.set_axis_off()
    title_ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, linewidth=1.2))
    title_ax.text(0.5, 0.5, f"{target['wise_name']} 표준 기준 구역 도식",
                  ha="center", va="center", fontsize=12)

    # Legend box
    legend_ax.set_axis_off()
    legend_ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, linewidth=1.2))
    legend_lines = [
        "범례",
        "중심 원: 중심 기준점",
        "내부 기준 구역: 0 ~ 0.5R",
        "쉘 구조 구역: 0.6R ~ 1.0R",
        "배경 기준 구역: 1.2R ~ 1.8R",
        "점선 반경: 카탈로그 기준 반지름",
    ]
    y = 0.84
    for i, txt in enumerate(legend_lines):
        fs = 11 if i == 0 else 9
        legend_ax.text(0.5 if i == 0 else 0.05, y, txt,
                       ha="center" if i == 0 else "left",
                       va="center", fontsize=fs)
        y -= 0.15 if i == 0 else 0.13

    # Main circular schematic
    ax.set_aspect("equal")
    ax.set_xlim(-rmax * 1.05, rmax * 1.05)
    ax.set_ylim(-rmax * 1.05, rmax * 1.05)

    # Cross axes
    ax.plot([-rmax, rmax], [0, 0], linewidth=0.8)
    ax.plot([0, 0], [-rmax, rmax], linewidth=0.8)

    # Zone circles
    for radius in [zone.inner_outer_arcsec, zone.shell_outer_arcsec, zone.bg_outer_arcsec]:
        ax.add_patch(Circle((0, 0), radius, fill=False, linewidth=0.8))

    # Catalog radius dashed circle
    ax.add_patch(Circle((0, 0), zone.radius_arcsec, fill=False, linestyle="--", linewidth=1.4))

    # Schematic central circle
    ax.add_patch(Circle((0, 0), center_circle_r, fill=False, linewidth=0.9))

    # Labels inside
    ax.text(0, zone.inner_outer_arcsec * 0.55, "내부", ha="center", va="center", fontsize=11)
    ax.text(0, (zone.shell_inner_arcsec + zone.shell_outer_arcsec) / 2, "쉘", ha="center", va="center", fontsize=11)
    ax.text(0, (zone.bg_inner_arcsec + zone.bg_outer_arcsec) / 2, "배경", ha="center", va="center", fontsize=11)

    # Outer annotations
    ax.text(zone.inner_outer_arcsec + 5, 5, "0.5R", fontsize=9)
    ax.text(zone.shell_outer_arcsec + 5, 5, "1.0R", fontsize=9)
    ax.text(zone.bg_outer_arcsec + 5, 5, "1.8R", fontsize=9)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(target_dir / f"{name}_standard_baseline_zones_circular.png", dpi=160)
    plt.close(fig)


def save_per_target_summary(target: pd.Series, zone: TargetZones, out_dir: Path) -> None:
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
        f"ra: {target.get('ra', '')}",
        f"dec: {target.get('dec', '')}",
        f"radius_arcsec: {target.get('radius_arcsec', '')}",
        "",
        "standard baseline zones",
        f"inner_inner_arcsec: {zone.inner_inner_arcsec}",
        f"inner_outer_arcsec: {zone.inner_outer_arcsec}",
        f"shell_inner_arcsec: {zone.shell_inner_arcsec}",
        f"shell_outer_arcsec: {zone.shell_outer_arcsec}",
        f"bg_inner_arcsec: {zone.bg_inner_arcsec}",
        f"bg_outer_arcsec: {zone.bg_outer_arcsec}",
        "",
        "central schematic circle",
        "meaning: target center / center-object schematic marker",
        "note: not to physical scale",
    ]
    (target_dir / f"{name}_standard_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(project_root: Path, out_dir: Path, df: pd.DataFrame) -> None:
    lines = [
        "Validation of Structural Contrast Baseline",
        "Standard pipeline manifest",
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
        "This is the standard baseline structure stage.",
        "The output defines inner/shell/background comparison zones for later profile extraction and topological comparison.",
        "Circular plots include a schematic central circle for the target center.",
    ]
    (out_dir / "run_manifest.txt").write_text("\n".join(lines), encoding="utf-8")


def write_summary_csv(out_dir: Path, df: pd.DataFrame, zones_df: pd.DataFrame) -> None:
    rows = [
        {"metric": "target_count", "value": len(df)},
        {"metric": "radius_min_arcsec", "value": float(df["radius_arcsec_num"].min())},
        {"metric": "radius_median_arcsec", "value": float(df["radius_arcsec_num"].median())},
        {"metric": "radius_max_arcsec", "value": float(df["radius_arcsec_num"].max())},
        {"metric": "inner_outer_max_arcsec", "value": float(zones_df["inner_outer_arcsec"].max())},
        {"metric": "shell_outer_max_arcsec", "value": float(zones_df["shell_outer_arcsec"].max())},
        {"metric": "bg_outer_max_arcsec", "value": float(zones_df["bg_outer_arcsec"].max())},
    ]
    pd.DataFrame(rows).to_csv(out_dir / "run_summary.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()

    out_dir = create_timestamped_output_dir(project_root)

    raw_df = load_final_input(project_root)
    df = normalize_input(raw_df)
    zones = build_standard_zones(df)
    zones_df = zones_to_dataframe(zones)
    comparison_df = build_baseline_comparison_table(df, zones_df)

    df.to_csv(out_dir / "fixed_final_input_copy.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(out_dir / "standard_baseline_comparison_table.csv", index=False, encoding="utf-8-sig")
    zones_df.to_csv(out_dir / "target_standard_zones.csv", index=False, encoding="utf-8-sig")

    zone_map = {z.wise_name: z for z in zones}
    for _, row in df.iterrows():
        zone = zone_map[str(row["wise_name"])]
        save_per_target_summary(row, zone, out_dir)
        save_linear_plot_for_target(row, zone, out_dir)
        save_circular_plot_for_target(row, zone, out_dir)

    write_manifest(project_root, out_dir, df)
    write_summary_csv(out_dir, df, zones_df)

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - standard pipeline (circular)")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print(f"Input file  : {project_root / INPUT_FILE}")
    print(f"Output dir  : {out_dir}")
    print("")
    print("[OK] Created:")
    print(out_dir / "fixed_final_input_copy.csv")
    print(out_dir / "standard_baseline_comparison_table.csv")
    print(out_dir / "target_standard_zones.csv")
    print(out_dir / "run_manifest.txt")
    print(out_dir / "run_summary.csv")
    print(out_dir / "targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
