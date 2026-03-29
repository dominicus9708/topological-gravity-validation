#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

def save_radius_plot(df: pd.DataFrame, out_path: Path) -> None:
    p = df.copy()
    p["time_value"] = pd.to_numeric(p["time_value"], errors="coerce")
    p["radius_arcsec"] = pd.to_numeric(p["radius_arcsec"], errors="coerce")
    p["radius_arcsec_err"] = pd.to_numeric(p["radius_arcsec_err"], errors="coerce")
    p = p.dropna(subset=["time_value", "radius_arcsec"]).sort_values("time_value")

    plt.figure(figsize=(8, 5))
    for label, g in p.groupby("component_label", dropna=False):
        lab = str(label) if pd.notna(label) else "unlabeled_component"
        plt.errorbar(g["time_value"], g["radius_arcsec"], yerr=g["radius_arcsec_err"], marker="o", label=lab)
    plt.xlabel("time_value")
    plt.ylabel("radius_arcsec")
    plt.title("SN 2014J Direct-Table Radius vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_geometry_context_plot(df: pd.DataFrame, out_path: Path) -> None:
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

def main() -> None:
    parser = argparse.ArgumentParser(description="SN 2014J standard pipeline v2")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-input-csv", default="SN2014J_model_input_radius_v2.csv")
    parser.add_argument("--arc-geometry-csv", default="SN2014J_Yang2017_Table4_arc_geometry.csv")
    parser.add_argument("--photometry-csv", default="SN2014J_Yang2017_Table3_photometry.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)

    model_input_df = load_csv(ensure_file(input_dir / args.model_input_csv))
    arc_df = load_csv(ensure_file(input_dir / args.arc_geometry_csv))
    phot_df = load_csv(ensure_file(input_dir / args.photometry_csv))

    out_dir = make_output_dir(output_root)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    write_csv(model_input_df, out_dir / "model_input_radius_copy.csv")
    write_csv(arc_df, out_dir / "arc_geometry_copy.csv")
    write_csv(phot_df, out_dir / "photometry_copy.csv")

    save_radius_plot(model_input_df, plots_dir / "sn2014j_direct_table_radius_vs_time.png")
    save_geometry_context_plot(model_input_df, plots_dir / "sn2014j_arc_geometry_context_vs_time.png")

    summary = {
        "pipeline_name": "SN 2014J standard pipeline v2",
        "created_at": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "results_dir": str(out_dir),
        "row_counts": {
            "model_input_radius_v2": int(len(model_input_df)),
            "arc_geometry_table4": int(len(arc_df)),
            "photometry_table3": int(len(phot_df)),
        },
        "notes": [
            "This version is based on direct values extracted from Yang et al. 2017 Table 3 and Table 4.",
            "The radius time series uses direct arc geometry table values rather than sparse preview-only estimates."
        ],
        "generated_plots": [
            "plots/sn2014j_direct_table_radius_vs_time.png",
            "plots/sn2014j_arc_geometry_context_vs_time.png",
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    (out_dir / "README_results.txt").write_text(
        "SN 2014J standard pipeline v2 results.\n"
        "Based on direct Table 3 and Table 4 values from Yang et al. 2017.\n",
        encoding="utf-8",
    )

    print(f"[DONE] Results written to: {out_dir}")

if __name__ == "__main__":
    main()
