#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V838 Mon Light Echo Pipeline with Graphs
---------------------------------------
Purpose:
- Read the actual V838 Mon CSV inputs already prepared in the project
- Save results under output/V838Mon/YYYYMMDD_HHMMSS
- Reproduce standard-reference data views before any topological-gravity correction term

Default input folder:
    data/derived/Echo of Light/input

Expected default input files:
    - v838mon_measurements_verified.csv
    - v838mon_model_input_all.csv
    - v838mon_model_input_tylenda.csv
    - v838mon_model_input_crause.csv

Default output root:
    output/V838Mon
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path("data/derived/Echo of Light/input")
DEFAULT_OUTPUT_ROOT = Path("output/V838Mon")


def ensure_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def make_output_dir(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def to_numeric(df: pd.DataFrame, columns) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def save_radius_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plot_df = to_numeric(df, ["time_value", "radius_arcsec"]).dropna(subset=["time_value", "radius_arcsec"]).sort_values("time_value")
    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["time_value"], plot_df["radius_arcsec"], marker="o")
    plt.xlabel("time_value")
    plt.ylabel("radius_arcsec")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_tylenda_center_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plot_df = to_numeric(df, ["time_value", "x_center_arcsec", "y_center_arcsec"]).dropna(subset=["time_value"]).sort_values("time_value")
    plt.figure(figsize=(8, 5))
    if "x_center_arcsec" in plot_df.columns and plot_df["x_center_arcsec"].notna().any():
        plt.plot(plot_df["time_value"], plot_df["x_center_arcsec"], marker="o", label="x_center_arcsec")
    if "y_center_arcsec" in plot_df.columns and plot_df["y_center_arcsec"].notna().any():
        plt.plot(plot_df["time_value"], plot_df["y_center_arcsec"], marker="o", label="y_center_arcsec")
    plt.xlabel("time_value")
    plt.ylabel("center_arcsec")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_crause_offset_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plot_df = to_numeric(df, ["time_value", "ra_offset_arcsec", "dec_offset_arcsec"]).dropna(subset=["time_value"]).sort_values("time_value")
    plt.figure(figsize=(8, 5))
    if "ra_offset_arcsec" in plot_df.columns and plot_df["ra_offset_arcsec"].notna().any():
        plt.plot(plot_df["time_value"], plot_df["ra_offset_arcsec"], marker="o", label="ra_offset_arcsec")
    if "dec_offset_arcsec" in plot_df.columns and plot_df["dec_offset_arcsec"].notna().any():
        plt.plot(plot_df["time_value"], plot_df["dec_offset_arcsec"], marker="o", label="dec_offset_arcsec")
    plt.xlabel("time_value")
    plt.ylabel("offset_arcsec")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_summary(verified_df: pd.DataFrame, all_df: pd.DataFrame, tylenda_df: pd.DataFrame, crause_df: pd.DataFrame,
                  input_dir: Path, out_dir: Path, filenames: Dict[str, str]) -> Dict:
    return {
        "pipeline_name": "V838 Mon Light Echo Pipeline with Graphs",
        "created_at": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "results_dir": str(out_dir),
        "input_files": filenames,
        "row_counts": {
            "measurements_verified": int(len(verified_df)),
            "model_input_all": int(len(all_df)),
            "model_input_tylenda": int(len(tylenda_df)),
            "model_input_crause": int(len(crause_df)),
        },
        "dataset_counts_in_verified": (
            verified_df["dataset"].astype(str).value_counts().to_dict()
            if "dataset" in verified_df.columns else {}
        ),
        "generated_plots": [
            "plots/crause_radius_vs_time.png",
            "plots/tylenda_radius_vs_time.png",
            "plots/tylenda_center_shift.png",
            "plots/crause_center_offset.png",
        ],
        "notes": [
            "This version is a standard-reference graph reproduction step.",
            "It is intended to visualize the Tylenda and Crause inputs before any topological-gravity correction term is added.",
            "Tylenda and Crause time conventions remain intentionally separated.",
        ],
    }


def write_readme(out_dir: Path) -> None:
    text = """README_results.txt
==================

This results folder was generated by:
    src/echo of light/v838mon/v838mon_echo_pipeline_graphs.py

Purpose
-------
This version reproduces standard-reference plots from the already prepared V838 Mon input tables.

Files written here
------------------
- verified_copy.csv
- model_input_all_copy.csv
- model_input_tylenda_copy.csv
- model_input_crause_copy.csv
- verified_tylenda_only.csv
- verified_crause_only.csv
- dataset_row_counts.csv
- summary.json
- README_results.txt

Plots written here
------------------
- plots/crause_radius_vs_time.png
- plots/tylenda_radius_vs_time.png
- plots/tylenda_center_shift.png
- plots/crause_center_offset.png

Interpretation status
---------------------
This is still a standard-reference reproduction step.
It is not yet the topological-gravity correction step.

Important caution
-----------------
Do not merge Tylenda and Crause time systems unless you explicitly convert them.
"""
    (out_dir / "README_results.txt").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="V838 Mon pipeline with graphs")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--verified-csv", default="v838mon_measurements_verified.csv")
    parser.add_argument("--all-csv", default="v838mon_model_input_all.csv")
    parser.add_argument("--tylenda-csv", default="v838mon_model_input_tylenda.csv")
    parser.add_argument("--crause-csv", default="v838mon_model_input_crause.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)

    verified_path = ensure_file(input_dir / args.verified_csv)
    all_path = ensure_file(input_dir / args.all_csv)
    tylenda_path = ensure_file(input_dir / args.tylenda_csv)
    crause_path = ensure_file(input_dir / args.crause_csv)

    verified_df = load_csv(verified_path)
    all_df = load_csv(all_path)
    tylenda_df = load_csv(tylenda_path)
    crause_df = load_csv(crause_path)

    out_dir = make_output_dir(output_root)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    write_csv(verified_df, out_dir / "verified_copy.csv")
    write_csv(all_df, out_dir / "model_input_all_copy.csv")
    write_csv(tylenda_df, out_dir / "model_input_tylenda_copy.csv")
    write_csv(crause_df, out_dir / "model_input_crause_copy.csv")

    if "dataset" in verified_df.columns:
        verified_tylenda = verified_df[verified_df["dataset"].astype(str).str.startswith("Tylenda", na=False)].copy()
        verified_crause = verified_df[verified_df["dataset"].astype(str).str.startswith("Crause", na=False)].copy()
    else:
        verified_tylenda = pd.DataFrame()
        verified_crause = pd.DataFrame()

    write_csv(verified_tylenda, out_dir / "verified_tylenda_only.csv")
    write_csv(verified_crause, out_dir / "verified_crause_only.csv")

    counts_df = pd.DataFrame([
        {"file": "verified", "rows": len(verified_df)},
        {"file": "model_input_all", "rows": len(all_df)},
        {"file": "model_input_tylenda", "rows": len(tylenda_df)},
        {"file": "model_input_crause", "rows": len(crause_df)},
        {"file": "verified_tylenda_only", "rows": len(verified_tylenda)},
        {"file": "verified_crause_only", "rows": len(verified_crause)},
    ])
    write_csv(counts_df, out_dir / "dataset_row_counts.csv")

    # Save plots
    save_radius_plot(crause_df, plots_dir / "crause_radius_vs_time.png", "Crause Radius vs Time")
    save_radius_plot(tylenda_df, plots_dir / "tylenda_radius_vs_time.png", "Tylenda Radius vs Time")
    save_tylenda_center_plot(tylenda_df, plots_dir / "tylenda_center_shift.png", "Tylenda Center Shift")
    save_crause_offset_plot(crause_df, plots_dir / "crause_center_offset.png", "Crause Center Offset")

    summary = build_summary(
        verified_df, all_df, tylenda_df, crause_df, input_dir, out_dir,
        {
            "verified_csv": args.verified_csv,
            "all_csv": args.all_csv,
            "tylenda_csv": args.tylenda_csv,
            "crause_csv": args.crause_csv,
        }
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_readme(out_dir)
    print(f"[DONE] Results written to: {out_dir}")


if __name__ == "__main__":
    main()
