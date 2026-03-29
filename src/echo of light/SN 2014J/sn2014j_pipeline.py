#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SN 2014J pipeline
-----------------
Input folder:
    data/derived/Echo of Light/input/SN 2014J

Output root:
    results/Echo of Light/output/SN 2014J/YYYYMMDD_HHMMSS

This pipeline reads:
- SN2014J_model_input_radius_v2.csv
- SN2014J_Yang2017_Table4_arc_geometry.csv
- SN2014J_Yang2017_Table3_photometry.csv
- SN2014J_Graur2018_Table1_late_photometry.csv
- SN2014J_direct_and_late_extension_reference.csv

And writes:
- copies of all inputs
- standard plots for direct geometry and late photometry
- a comparison plot where direct-table geometry and late-time photometry
  are shown as separate series with different labels/colors
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT_DIR = Path("data/derived/Echo of Light/input/SN 2014J")
DEFAULT_OUTPUT_ROOT = Path("results/Echo of Light/output/SN 2014J")


def ensure_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def make_output_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = root / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_direct_radius_plot(df: pd.DataFrame, out_path: Path) -> None:
    p = df.copy()
    p["time_value"] = pd.to_numeric(p["time_value"], errors="coerce")
    p["radius_arcsec"] = pd.to_numeric(p["radius_arcsec"], errors="coerce")
    p["radius_arcsec_err"] = pd.to_numeric(p["radius_arcsec_err"], errors="coerce")
    p = p.dropna(subset=["time_value", "radius_arcsec"]).sort_values("time_value")

    plt.figure(figsize=(8, 5))
    for label, g in p.groupby("component_label", dropna=False):
        lab = str(label) if pd.notna(label) else "unlabeled_component"
        if "radius_arcsec_err" in g.columns and g["radius_arcsec_err"].notna().any():
            plt.errorbar(g["time_value"], g["radius_arcsec"], yerr=g["radius_arcsec_err"], marker="o", label=lab)
        else:
            plt.plot(g["time_value"], g["radius_arcsec"], marker="o", label=lab)
    plt.xlabel("time_value")
    plt.ylabel("radius_arcsec")
    plt.title("SN 2014J Direct Geometry Radius vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_arc_geometry_context_plot(df: pd.DataFrame, out_path: Path) -> None:
    p = df.copy()
    p["time_value"] = pd.to_numeric(p["time_value"], errors="coerce")
    p["foreground_distance_pc"] = pd.to_numeric(p["foreground_distance_pc"], errors="coerce")
    p["projected_radius_pc"] = pd.to_numeric(p["projected_radius_pc"], errors="coerce")
    p = p.dropna(subset=["time_value"]).sort_values("time_value")

    plt.figure(figsize=(8, 5))
    if p["foreground_distance_pc"].notna().any():
        plt.plot(p["time_value"], p["foreground_distance_pc"], marker="o", label="foreground_distance_pc")
    if p["projected_radius_pc"].notna().any():
        plt.plot(p["time_value"], p["projected_radius_pc"], marker="o", label="projected_radius_pc")
    plt.xlabel("time_value")
    plt.ylabel("pc")
    plt.title("SN 2014J Arc Geometry Context vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_late_photometry_plot(df: pd.DataFrame, out_path: Path) -> None:
    p = df.copy()
    p["phase_days"] = pd.to_numeric(p["phase_days"], errors="coerce")
    p["magnitude"] = pd.to_numeric(p["magnitude"], errors="coerce")
    p["magnitude_err"] = pd.to_numeric(p["magnitude_err"], errors="coerce")
    p["is_upper_limit"] = p["is_upper_limit"].astype(str).str.lower().isin(["true", "1", "yes"])
    p = p.dropna(subset=["phase_days", "magnitude"]).sort_values("phase_days")

    plt.figure(figsize=(8, 5))
    for flt, g in p.groupby("filter", dropna=False):
        det = g[~g["is_upper_limit"]].copy()
        upl = g[g["is_upper_limit"]].copy()
        if not det.empty:
            if det["magnitude_err"].notna().any():
                plt.errorbar(det["phase_days"], det["magnitude"], yerr=det["magnitude_err"], marker="o", label=f"{flt} detected")
            else:
                plt.plot(det["phase_days"], det["magnitude"], marker="o", label=f"{flt} detected")
        if not upl.empty:
            plt.scatter(upl["phase_days"], upl["magnitude"], marker="v", label=f"{flt} upper_limit")
    plt.gca().invert_yaxis()
    plt.xlabel("phase_days")
    plt.ylabel("Vega magnitude")
    plt.title("SN 2014J Late-time Photometry vs Phase")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_reference_dual_origin_plot(df: pd.DataFrame, out_path: Path) -> None:
    p = df.copy()
    p["time_value"] = pd.to_numeric(p["time_value"], errors="coerce")
    p["value_num"] = pd.to_numeric(p["value"], errors="coerce")
    p = p.dropna(subset=["time_value", "value_num"]).sort_values("time_value")

    geom = p[p["data_origin"].astype(str) == "direct_table_geometry"].copy()
    late = p[p["data_origin"].astype(str) == "late_time_photometry"].copy()

    plt.figure(figsize=(8, 5))
    if not geom.empty:
        plt.plot(
            geom["time_value"],
            geom["value_num"],
            marker="o",
            linestyle="-",
            label="direct_table_geometry",
        )
    if not late.empty:
        plt.plot(
            late["time_value"],
            late["value_num"],
            marker="s",
            linestyle="--",
            label="late_time_photometry",
        )
    plt.gca().invert_yaxis() if (late["unit"].astype(str).eq("Vega_mag").any() if not late.empty else False) else None
    plt.xlabel("time_value")
    plt.ylabel("mixed value scale")
    plt.title("SN 2014J Direct vs Late Extension Reference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SN 2014J pipeline")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-input-csv", default="SN2014J_model_input_radius_v2.csv")
    parser.add_argument("--yang-table4-csv", default="SN2014J_Yang2017_Table4_arc_geometry.csv")
    parser.add_argument("--yang-table3-csv", default="SN2014J_Yang2017_Table3_photometry.csv")
    parser.add_argument("--graur-late-csv", default="SN2014J_Graur2018_Table1_late_photometry.csv")
    parser.add_argument("--reference-csv", default="SN2014J_direct_and_late_extension_reference.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)

    model_input_df = load_csv(ensure_file(input_dir / args.model_input_csv))
    yang_t4_df = load_csv(ensure_file(input_dir / args.yang_table4_csv))
    yang_t3_df = load_csv(ensure_file(input_dir / args.yang_table3_csv))
    graur_df = load_csv(ensure_file(input_dir / args.graur_late_csv))
    reference_df = load_csv(ensure_file(input_dir / args.reference_csv))

    out_dir = make_output_dir(output_root)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    write_csv(model_input_df, out_dir / "model_input_radius_copy.csv")
    write_csv(yang_t4_df, out_dir / "yang_table4_arc_geometry_copy.csv")
    write_csv(yang_t3_df, out_dir / "yang_table3_photometry_copy.csv")
    write_csv(graur_df, out_dir / "graur_late_photometry_copy.csv")
    write_csv(reference_df, out_dir / "direct_and_late_reference_copy.csv")

    save_direct_radius_plot(model_input_df, plots_dir / "sn2014j_direct_geometry_radius_vs_time.png")
    save_arc_geometry_context_plot(model_input_df, plots_dir / "sn2014j_arc_geometry_context_vs_time.png")
    save_late_photometry_plot(graur_df, plots_dir / "sn2014j_late_photometry_vs_phase.png")
    save_reference_dual_origin_plot(reference_df, plots_dir / "sn2014j_direct_vs_late_reference.png")

    summary = {
        "pipeline_name": "SN 2014J pipeline",
        "created_at": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "results_dir": str(out_dir),
        "row_counts": {
            "model_input_radius_v2": int(len(model_input_df)),
            "yang_table4_arc_geometry": int(len(yang_t4_df)),
            "yang_table3_photometry": int(len(yang_t3_df)),
            "graur_late_photometry": int(len(graur_df)),
            "direct_and_late_reference": int(len(reference_df)),
        },
        "notes": [
            "This pipeline reads the SN 2014J input folder directly.",
            "Direct-table geometry and late-time photometry are kept as separate data classes.",
            "The comparison plot overlays different origins with different markers/line styles."
        ],
        "generated_plots": [
            "plots/sn2014j_direct_geometry_radius_vs_time.png",
            "plots/sn2014j_arc_geometry_context_vs_time.png",
            "plots/sn2014j_late_photometry_vs_phase.png",
            "plots/sn2014j_direct_vs_late_reference.png",
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    (out_dir / "README_results.txt").write_text(
        "SN 2014J pipeline results folder.\n"
        "Uses direct-table geometry and late-time photometry from the input/SN 2014J folder.\n",
        encoding="utf-8",
    )

    print(f"[DONE] Results written to: {out_dir}")


if __name__ == "__main__":
    main()
