#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_PATCH_CSV = Path(
    r"data/derived/Echo of Light/input/T Pyxidis/t_pyx_standard_echo_patches_original_paper.csv"
)
DEFAULT_INTERPRETATION_CSV = Path(
    r"data/derived/Echo of Light/input/T Pyxidis/t_pyx_standard_interpretation_table.csv"
)
DEFAULT_TIMELINE_CSV = Path(
    r"data/derived/Echo of Light/input/T Pyxidis/t_pyx_standard_observation_timeline_table.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"results/Echo of Light/output/T Pyxidis/standard/integrated_standard"
)

ANGLE_DEG_MAP = {
    "north": 90,
    "north #2": 75,
    "south": 270,
    "south #2": 255,
    "northwest": 135,
    "west": 180,
}

FAMILY_MAP = {
    "North family": ["N1", "N2"],
    "South family": ["S1", "S2"],
    "West family": ["NW", "W"],
}

# display-only overlay values
WD_MASS_SOL = 0.70
DONOR_MASS_SOL = 0.14
DISPLAY_SEPARATION_ARCSEC = 0.10
DISTANCE_KPC = 4.8


def ensure_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def make_output_dir(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def arcsec_to_pc(theta_arcsec: float, distance_kpc: float) -> float:
    return distance_kpc * 1000.0 * theta_arcsec / 206265.0


def arcsec_to_ly(theta_arcsec: float, distance_kpc: float) -> float:
    return arcsec_to_pc(theta_arcsec, distance_kpc) * 3.26156


def arcsec_to_au(theta_arcsec: float, distance_kpc: float) -> float:
    return arcsec_to_pc(theta_arcsec, distance_kpc) * 206265.0


def prepare_patch_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = to_numeric(out, ["delay_time_days", "theta_arcsec", "z_1e17_cm"])
    out["angle_deg"] = out["location_label"].map(ANGLE_DEG_MAP)
    out["angle_rad"] = np.deg2rad(out["angle_deg"])
    out["theta_pc_at_4p8kpc"] = out["theta_arcsec"].apply(lambda x: arcsec_to_pc(float(x), DISTANCE_KPC))
    out["theta_ly_at_4p8kpc"] = out["theta_arcsec"].apply(lambda x: arcsec_to_ly(float(x), DISTANCE_KPC))
    out["theta_au_at_4p8kpc"] = out["theta_arcsec"].apply(lambda x: arcsec_to_au(float(x), DISTANCE_KPC))
    return out


def binary_overlay_positions():
    # display-only: the two markers are intentionally exaggerated but made to overlap
    # around the true center of mass, with both marker centers pulled toward 0,0.
    half_sep = DISPLAY_SEPARATION_ARCSEC / 2.0
    return {
        "wd_x": -half_sep,
        "donor_x": half_sep,
        "wd_y": 0.0,
        "donor_y": 0.0,
        "com_x": 0.0,
        "com_y": 0.0,
    }


def add_arcsec_distance_legend(fig):
    lines = []
    for r in [1, 2, 3, 4, 5]:
        au = arcsec_to_au(r, DISTANCE_KPC)
        ly = arcsec_to_ly(r, DISTANCE_KPC)
        lines.append(f'{r}" = {au:,.0f} AU = {ly:.3f} ly')
    text = "Angular scale\n" + "\n".join(lines)
    fig.text(
        0.77, 0.12, text,
        ha="left", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9)
    )


def add_binary_name_legend(fig):
    text = (
        "Central binary legend\n"
        "Left marker: White dwarf\n"
        "Right marker: Donor companion\n"
        "Center marker: COM at exact graph center\n"
        "The two stars are drawn heavily compressed and\n"
        "partly overlapping around the COM to show both:\n"
        "(1) the binary nature, and\n"
        "(2) the fact that the true physical separation is\n"
        "far too small to resolve on the echo-patch scale."
    )
    fig.text(
        0.77, 0.74, text,
        ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9)
    )


def save_circular_layout(df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    ax.scatter(df["angle_rad"], df["theta_arcsec"], s=95)
    for _, row in df.iterrows():
        ax.text(
            row["angle_rad"],
            row["theta_arcsec"] + 0.18,
            f"{row['echo_patch']}\n{int(round(row['delay_time_days']))} d",
            ha="center",
            va="center",
            fontsize=9,
        )

    for _, patches in FAMILY_MAP.items():
        sub = df[df["echo_patch"].isin(patches)].copy().sort_values("delay_time_days")
        if len(sub) >= 2:
            ax.annotate(
                "",
                xy=(float(sub.iloc[1]["angle_rad"]), float(sub.iloc[1]["theta_arcsec"])),
                xytext=(float(sub.iloc[0]["angle_rad"]), float(sub.iloc[0]["theta_arcsec"])),
                arrowprops=dict(arrowstyle="->", lw=1.4),
            )

    ring_vals = [1, 2, 3, 4, 5]
    ax.set_rticks(ring_vals)
    ax.set_yticklabels([f'{r}"' for r in ring_vals])

    # central binary overlay directly on graph center
    inset = fig.add_axes([0.41, 0.41, 0.18, 0.18])
    inset.set_aspect("equal")
    pos = binary_overlay_positions()
    inset.scatter([pos["wd_x"]], [pos["wd_y"]], s=150, marker="o", alpha=0.95)
    inset.scatter([pos["donor_x"]], [pos["donor_y"]], s=150, marker="o", alpha=0.95)
    inset.scatter([pos["com_x"]], [pos["com_y"]], s=65, marker="+", zorder=5)
    inset.set_xlim(-0.20, 0.20)
    inset.set_ylim(-0.12, 0.12)
    inset.axis("off")

    add_binary_name_legend(fig)
    add_arcsec_distance_legend(fig)

    ax.set_title("T Pyxidis standard: integrated circular echo-patch layout")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_delay_theta_by_family(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    family_colors = {
        "North family": "tab:blue",
        "South family": "tab:orange",
        "West family": "tab:green",
    }
    for family_name, patches in FAMILY_MAP.items():
        sub = df[df["echo_patch"].isin(patches)].copy().sort_values("delay_time_days")
        color = family_colors[family_name]
        ax.scatter(sub["delay_time_days"], sub["theta_arcsec"], s=75, label=family_name, color=color)
        if len(sub) >= 2:
            ax.annotate(
                "",
                xy=(float(sub.iloc[1]["delay_time_days"]), float(sub.iloc[1]["theta_arcsec"])),
                xytext=(float(sub.iloc[0]["delay_time_days"]), float(sub.iloc[0]["theta_arcsec"])),
                arrowprops=dict(arrowstyle="->", lw=1.3, color=color),
            )
        for _, row in sub.iterrows():
            ax.text(row["delay_time_days"], row["theta_arcsec"], f" {row['echo_patch']}", fontsize=9)
    ax.set_title("T Pyxidis standard: delay vs angular distance by patch family")
    ax.set_xlabel("Delay time (days)")
    ax.set_ylabel("Angular distance θ (arcsec)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_integrated_patch_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "echo_patch",
        "location_label",
        "delay_time_days",
        "theta_arcsec",
        "theta_au_at_4p8kpc",
        "theta_ly_at_4p8kpc",
        "z_1e17_cm",
    ]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()
    out["system_context"] = "recurrent nova binary: white dwarf + donor companion"
    out["medium_context"] = "pre-existing / previously ejected surrounding echo medium"
    out["display_note"] = "binary overlay in circular plot is illustrative, not to physical scale"
    return out


def build_summary_dict(
    patch_csv: Path,
    interpretation_csv: Path,
    timeline_csv: Path,
    out_dir: Path,
    patch_df: pd.DataFrame,
    interp_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
) -> dict:
    ring_scale = []
    for r in [1, 2, 3, 4, 5]:
        ring_scale.append({
            "arcsec": r,
            "au_at_4p8kpc": round(arcsec_to_au(r, DISTANCE_KPC), 2),
            "ly_at_4p8kpc": round(arcsec_to_ly(r, DISTANCE_KPC), 6),
        })

    return {
        "pipeline_name": "T Pyxidis Standard Integrated Pipeline",
        "created_at": datetime.now().isoformat(),
        "input_files": {
            "patch_csv": str(patch_csv),
            "interpretation_csv": str(interpretation_csv),
            "timeline_csv": str(timeline_csv),
        },
        "results_dir": str(out_dir),
        "row_counts": {
            "patch_rows": int(len(patch_df)),
            "interpretation_rows": int(len(interp_df)),
            "timeline_rows": int(len(timeline_df)),
        },
        "system_context": {
            "classification": "recurrent nova close binary",
            "primary": "white dwarf",
            "secondary": "donor companion star",
            "primary_mass_msun": WD_MASS_SOL,
            "secondary_mass_msun": DONOR_MASS_SOL,
            "distance_kpc_for_ring_conversion": DISTANCE_KPC,
        },
        "ring_scale_reference": ring_scale,
        "notes": [
            "Circular plot keeps arcsec labels on the original radial rings.",
            "Arcsec-to-distance conversion is moved into a separate lower-right legend box using AU and light-year.",
            "Binary names are moved into a separate upper-right legend box.",
            "COM marker is plotted directly at the graph center without text label.",
            "The white dwarf and donor companion are drawn as a deliberately compressed, partly overlapping pair centered on the COM.",
            "Standard outputs remain observational / geometric references rather than topological reinterpretation terms.",
        ],
    }


def write_readme(out_dir: Path) -> None:
    text = """README_results.txt
==================

This results folder was generated by:
    src/echo of light/T Pyxidis/standard/t_pyx_standard_integrated_pipeline.py

What changed in this revision
-----------------------------
- COM marker remains at the exact graph center, but COM text is removed
- white dwarf and donor companion are drawn partly overlapping around the COM
- radial ring labels keep only the original arcsec marks
- AU and light-year conversions are moved to a separate lower-right legend box
- central binary names are moved to a separate upper-right legend box
"""
    (out_dir / "README_results.txt").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="T Pyxidis standard integrated pipeline")
    parser.add_argument("--patch-csv", default=str(DEFAULT_PATCH_CSV))
    parser.add_argument("--interpretation-csv", default=str(DEFAULT_INTERPRETATION_CSV))
    parser.add_argument("--timeline-csv", default=str(DEFAULT_TIMELINE_CSV))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    patch_csv = ensure_file(Path(args.patch_csv))
    interpretation_csv = ensure_file(Path(args.interpretation_csv))
    timeline_csv = ensure_file(Path(args.timeline_csv))

    patch_df = prepare_patch_table(load_csv(patch_csv))
    interp_df = load_csv(interpretation_csv)
    timeline_df = load_csv(timeline_csv)

    out_dir = make_output_dir(Path(args.output_root))
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    save_circular_layout(patch_df, plots_dir / "t_pyx_standard_integrated_circular_layout.png")
    save_delay_theta_by_family(patch_df, plots_dir / "t_pyx_standard_integrated_delay_theta_family.png")

    integrated_patch = build_integrated_patch_table(patch_df)
    write_csv(integrated_patch, tables_dir / "t_pyx_standard_integrated_patch_table.csv")
    write_csv(interp_df, tables_dir / "t_pyx_standard_interpretation_table.csv")
    write_csv(timeline_df, tables_dir / "t_pyx_standard_observation_timeline_table.csv")

    summary = build_summary_dict(
        patch_csv=patch_csv,
        interpretation_csv=interpretation_csv,
        timeline_csv=timeline_csv,
        out_dir=out_dir,
        patch_df=patch_df,
        interp_df=interp_df,
        timeline_df=timeline_df,
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_readme(out_dir)
    print(f"[DONE] Results written to: {out_dir}")


if __name__ == "__main__":
    main()
