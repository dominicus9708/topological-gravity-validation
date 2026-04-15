#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
Standard pipeline for common5 final input (v2)

Author: Kwon Dominicus

Placement
---------
src/Validation of Structural Contrast Baseline/standard/

Input
-----
data/derived/Validation of Structural Contrast Baseline/input/standard/wise_hii_common5/wise_hii_common5_standard_final_input.csv

Output
------
results/Validation of Structural Contrast Baseline/output/standard/YYYYMMDD_HHMMSS/

Notes
-----
- This is an execution pipeline, not a derived-processing script.
- Final input is treated as fixed official input.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import pandas as pd

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\mincu\Desktop\topological_gravity_project")

INPUT_FILE = (
    Path("data")
    / "derived"
    / "Validation of Structural Contrast Baseline"
    / "input"
    / "standard"
    / "wise_hii_common5"
    / "wise_hii_common5_standard_final_input.csv"
)

RESULTS_BASE = (
    Path("results")
    / "Validation of Structural Contrast Baseline"
    / "output"
    / "standard"
)


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
        "catalog_class",
        "input_group",
        "priority_rank",
        "glon",
        "glat",
        "ra",
        "dec",
        "radius_arcsec",
        "hii_region_name",
        "membership",
        "baseline_ready_flag",
    ]
    ensure_required_columns(df, required)

    df["radius_arcsec_num"] = pd.to_numeric(df["radius_arcsec"], errors="coerce")
    if df["radius_arcsec_num"].isna().any():
        bad = df.loc[df["radius_arcsec_num"].isna(), "wise_name"].tolist()
        raise ValueError("Invalid radius_arcsec values for: " + ", ".join(map(str, bad)))

    df["priority_rank_num"] = pd.to_numeric(df["priority_rank"], errors="coerce")
    df["wise_name"] = df["wise_name"].astype(str).str.strip()

    if df["wise_name"].duplicated().any():
        dup = df.loc[df["wise_name"].duplicated(), "wise_name"].tolist()
        raise ValueError("Duplicate wise_name entries found: " + ", ".join(dup))

    not_ready = df.loc[df["baseline_ready_flag"].astype(str).str.lower().isin(["false", "0", "no"]), "wise_name"].tolist()
    if not_ready:
        raise ValueError("Rows not baseline_ready_flag=True/usable: " + ", ".join(not_ready))

    return df.sort_values(
        by=["priority_rank_num", "radius_arcsec_num", "wise_name"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)


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
        "catalog_class",
        "input_group",
        "priority_rank",
        "hii_region_name",
        "membership",
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
        "score_total",
        "region_bucket",
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
    ax.axvspan(zone.inner_inner_arcsec, zone.inner_outer_arcsec, alpha=0.35, label="inner region")
    ax.axvspan(zone.shell_inner_arcsec, zone.shell_outer_arcsec, alpha=0.35, label="shell region")
    ax.axvspan(zone.bg_inner_arcsec, zone.bg_outer_arcsec, alpha=0.35, label="background region")
    ax.axvline(zone.radius_arcsec, linestyle="-", label="catalog radius")
    ax.set_xlabel("Radius (arcsec)")
    ax.set_ylabel("Baseline zone level")
    ax.set_title(f"{target['wise_name']} standard baseline zones")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.legend()
    fig.tight_layout()
    fig.savefig(target_dir / f"{name}_standard_baseline_zones_linear.png", dpi=160)
    plt.close(fig)


def wrap_note_by_periods(text: str, width: int = 95) -> str:
    parts = [p.strip() for p in text.split(".") if p.strip()]
    lines: List[str] = []
    for part in parts:
        sentence = part + "."
        wrapped = textwrap.wrap(sentence, width=width)
        lines.extend(wrapped if wrapped else [sentence])
    return "\n".join(lines)


def make_radius_label(value: str, arcsec: float) -> str:
    return value + "\n(" + f"{arcsec:.1f}" + '")'


def save_circular_plot_for_target(target: pd.Series, zone: TargetZones, out_dir: Path) -> None:
    name = sanitize_wise_name(target["wise_name"])
    target_dir = out_dir / "targets" / name
    target_dir.mkdir(parents=True, exist_ok=True)

    rmax = zone.bg_outer_arcsec
    center_circle_r = max(0.03 * rmax, 3.0)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.05, 0.14, 0.74, 0.76])

    fig.text(
        0.5,
        0.95,
        f"{target['wise_name']} Standard Baseline Zone Schematic",
        ha="center",
        va="center",
        fontsize=18,
    )

    note_text = (
        f"Note. R is the catalog reference radius. For this target, R = {zone.radius_arcsec:.1f} arcsec. "
        f"The small central circle is a schematic center marker. It is not to physical scale."
    )
    fig.text(
        0.12,
        0.05,
        wrap_note_by_periods(note_text, width=95),
        ha="left",
        va="bottom",
        fontsize=10,
    )

    legend_lines = [
        ("Legend", 12, "center"),
        ("Yellow center circle: center marker", 9.7, "left"),
        (f"Inner baseline zone: 0 to 0.5R ({zone.inner_outer_arcsec:.1f}\")", 9.2, "left"),
        (
            f"Shell structure zone: 0.6R to 1.0R ({zone.shell_inner_arcsec:.1f}\" to {zone.shell_outer_arcsec:.1f}\")",
            9.2,
            "left",
        ),
        (
            f"Background baseline zone: 1.2R to 1.8R ({zone.bg_inner_arcsec:.1f}\" to {zone.bg_outer_arcsec:.1f}\")",
            9.2,
            "left",
        ),
        (f"Reference radius: R = {zone.radius_arcsec:.1f} arcsec", 9.2, "left"),
    ]

    max_chars = max(len(txt) for txt, _, _ in legend_lines)
    legend_width = min(max(0.26, 0.11 + max_chars * 0.010), 0.415)
    legend_height = (len(legend_lines) - 1) * 0.0025 + 0.12
    legend_right = 0.97
    legend_left = legend_right - legend_width
    legend_bottom = 0.05
    legend_ax = fig.add_axes([legend_left, legend_bottom, legend_width, legend_height])

    legend_ax.set_axis_off()
    legend_ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, linewidth=1.2))
    y = 0.88
    for i, (txt, fs, align) in enumerate(legend_lines):
        legend_ax.text(
            0.5 if align == "center" else 0.05,
            y,
            txt,
            ha=align,
            va="center",
            fontsize=fs,
        )
        y -= 0.13 if i == 0 else 0.125

    ax.set_aspect("equal")
    ax.set_xlim(-rmax * 1.06, rmax * 1.06)
    ax.set_ylim(-rmax * 1.06, rmax * 1.06)

    ax.plot([-rmax, rmax], [0, 0], linewidth=1.0, color="#2C7FB8")
    ax.plot([0, 0], [-rmax, rmax], linewidth=1.0, color="#D95F0E")

    ax.add_patch(Circle((0, 0), zone.bg_outer_arcsec, facecolor="#C6DBEF", edgecolor="none", alpha=0.85))
    ax.add_patch(Circle((0, 0), zone.bg_inner_arcsec, facecolor="white", edgecolor="none"))

    ax.add_patch(Circle((0, 0), zone.shell_outer_arcsec, facecolor="#9ECAE1", edgecolor="none", alpha=0.95))
    ax.add_patch(Circle((0, 0), zone.shell_inner_arcsec, facecolor="white", edgecolor="none"))

    ax.add_patch(Circle((0, 0), zone.inner_outer_arcsec, facecolor="#FFF7BC", edgecolor="none", alpha=0.98))

    for rad, lw in [
        (zone.inner_outer_arcsec, 1.1),
        (zone.shell_outer_arcsec, 1.4),
        (zone.bg_outer_arcsec, 1.1),
    ]:
        ax.add_patch(Circle((0, 0), rad, fill=False, linestyle="-", linewidth=lw, edgecolor="black"))

    for rad in [zone.shell_inner_arcsec, zone.bg_inner_arcsec]:
        ax.add_patch(Circle((0, 0), rad, fill=False, linestyle="--", linewidth=1.2, edgecolor="black"))

    ax.add_patch(Circle((0, 0), center_circle_r, facecolor="#FFD92F", edgecolor="black", linewidth=1.0))

    ax.text(0, zone.inner_outer_arcsec * 0.56, "Inner", ha="center", va="center", fontsize=16)
    ax.text(0, (zone.shell_inner_arcsec + zone.shell_outer_arcsec) / 2, "Shell", ha="center", va="center", fontsize=16)
    ax.text(0, (zone.bg_inner_arcsec + zone.bg_outer_arcsec) / 2, "Background", ha="center", va="center", fontsize=16)

    label_y_offset = 10
    ax.text(zone.inner_outer_arcsec, label_y_offset, make_radius_label("0.5R", zone.inner_outer_arcsec), fontsize=10.5, ha="center", va="bottom")
    ax.text(zone.shell_outer_arcsec, label_y_offset, make_radius_label("1.0R", zone.shell_outer_arcsec), fontsize=10.5, ha="center", va="bottom")
    ax.text(zone.bg_outer_arcsec, label_y_offset, make_radius_label("1.8R", zone.bg_outer_arcsec), fontsize=10.5, ha="center", va="bottom")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(target_dir / f"{name}_standard_baseline_zones_circular.png", dpi=180)
    plt.close(fig)


def save_per_target_summary(target: pd.Series, zone: TargetZones, out_dir: Path) -> None:
    name = sanitize_wise_name(target["wise_name"])
    target_dir = out_dir / "targets" / name
    target_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"wise_name: {target['wise_name']}",
        f"priority_rank: {target.get('priority_rank', '')}",
        f"catalog_class: {target.get('catalog_class', '')}",
        f"input_group: {target.get('input_group', '')}",
        f"hii_region_name: {target.get('hii_region_name', '')}",
        f"membership: {target.get('membership', '')}",
        f"ra: {target.get('ra', '')}",
        f"dec: {target.get('dec', '')}",
        f"radius_arcsec: {target.get('radius_arcsec', '')}",
        f"fits_source_service: {target.get('fits_source_service', '')}",
        f"fits_band: {target.get('fits_band', '')}",
        f"fits_url: {target.get('fits_url', '')}",
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
        "Standard pipeline manifest (common5 v2)",
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
        lines.append(f"- {row['wise_name']} | rank={row.get('priority_rank', '')} | radius_arcsec={row.get('radius_arcsec', '')}")
    lines += [
        "",
        "meaning",
        "-" * 20,
        "This is the standard baseline structure stage for common5 final input.",
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
    print("Validation of Structural Contrast Baseline - standard pipeline (common5 v2)")
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
