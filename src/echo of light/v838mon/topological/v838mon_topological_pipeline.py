#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V838 Mon Topological Pipeline (trial version)
---------------------------------------------
Purpose
-------
This script is the first topological-side trial pipeline for V838 Mon.

It does NOT claim to prove a final topological-gravity explanation.
Instead, it provides a controlled framework to:

1. load the already prepared V838 Mon input tables,
2. preserve standard-reference observables,
3. apply a configurable topological correction term using proxy columns if available,
4. save trial comparison CSV files and plots.

Default input folder
--------------------
data/derived/Echo of Light/input

Expected default files
----------------------
- v838mon_measurements_verified.csv
- v838mon_model_input_all.csv
- v838mon_model_input_tylenda.csv
- v838mon_model_input_crause.csv

Default output root
-------------------
results/Echo of Light/output/V838Mon/topological

Default script location
-----------------------
src/echo of light/v838mon/topological/v838mon_topological_pipeline.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT_DIR = Path("data/derived/Echo of Light/input")
DEFAULT_OUTPUT_ROOT = Path("results/Echo of Light/output/V838Mon/topological")


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
    out_dir = root / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def ensure_proxy_column(df: pd.DataFrame, col: str, default_value: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns:
        out[col] = default_value
    out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default_value)
    return out


def apply_radius_topological_trial(
    df: pd.DataFrame,
    beta_sigma: float,
    beta_dsigma: float,
) -> pd.DataFrame:
    out = df.copy()
    out = ensure_proxy_column(out, "sigma_proxy", 0.0)
    out = ensure_proxy_column(out, "dsigma_dt_proxy", 0.0)
    out = to_numeric(out, ["radius_arcsec"])

    out["topological_radius_adjustment"] = (
        beta_sigma * out["sigma_proxy"] +
        beta_dsigma * out["dsigma_dt_proxy"]
    )
    out["radius_arcsec_topological_trial"] = out["radius_arcsec"] + out["topological_radius_adjustment"]
    return out


def apply_tylenda_center_trial(
    df: pd.DataFrame,
    gamma_sigma: float,
    gamma_dsigma: float,
) -> pd.DataFrame:
    out = df.copy()
    out = ensure_proxy_column(out, "sigma_proxy", 0.0)
    out = ensure_proxy_column(out, "dsigma_dt_proxy", 0.0)
    out = to_numeric(out, ["x_center_arcsec", "y_center_arcsec"])

    shift = gamma_sigma * out["sigma_proxy"] + gamma_dsigma * out["dsigma_dt_proxy"]
    out["topological_center_adjustment"] = shift
    if "x_center_arcsec" in out.columns:
        out["x_center_arcsec_topological_trial"] = out["x_center_arcsec"] + shift
    if "y_center_arcsec" in out.columns:
        out["y_center_arcsec_topological_trial"] = out["y_center_arcsec"] + shift
    return out


def apply_crause_offset_trial(
    df: pd.DataFrame,
    gamma_sigma: float,
    gamma_dsigma: float,
) -> pd.DataFrame:
    out = df.copy()
    out = ensure_proxy_column(out, "sigma_proxy", 0.0)
    out = ensure_proxy_column(out, "dsigma_dt_proxy", 0.0)
    out = to_numeric(out, ["ra_offset_arcsec", "dec_offset_arcsec"])

    shift = gamma_sigma * out["sigma_proxy"] + gamma_dsigma * out["dsigma_dt_proxy"]
    out["topological_center_adjustment"] = shift
    if "ra_offset_arcsec" in out.columns:
        out["ra_offset_arcsec_topological_trial"] = out["ra_offset_arcsec"] + shift
    if "dec_offset_arcsec" in out.columns:
        out["dec_offset_arcsec_topological_trial"] = out["dec_offset_arcsec"] + shift
    return out


def save_radius_trial_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plot_df = to_numeric(df, ["time_value", "radius_arcsec", "radius_arcsec_topological_trial", "topological_radius_adjustment"])
    plot_df = plot_df.dropna(subset=["time_value", "radius_arcsec"]).sort_values("time_value")

    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["time_value"], plot_df["radius_arcsec"], marker="o", label="observed_radius_arcsec")
    if "radius_arcsec_topological_trial" in plot_df.columns and plot_df["radius_arcsec_topological_trial"].notna().any():
        plt.plot(plot_df["time_value"], plot_df["radius_arcsec_topological_trial"], marker="o", label="topological_trial_radius_arcsec")
    plt.xlabel("time_value")
    plt.ylabel("radius_arcsec")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_radius_adjustment_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plot_df = to_numeric(df, ["time_value", "topological_radius_adjustment"]).dropna(subset=["time_value"]).sort_values("time_value")

    plt.figure(figsize=(8, 5))
    if "topological_radius_adjustment" in plot_df.columns and plot_df["topological_radius_adjustment"].notna().any():
        plt.plot(plot_df["time_value"], plot_df["topological_radius_adjustment"], marker="o", label="topological_radius_adjustment")
    plt.xlabel("time_value")
    plt.ylabel("adjustment")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_tylenda_center_trial_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plot_df = to_numeric(
        df,
        [
            "time_value",
            "x_center_arcsec",
            "y_center_arcsec",
            "x_center_arcsec_topological_trial",
            "y_center_arcsec_topological_trial",
        ],
    ).dropna(subset=["time_value"]).sort_values("time_value")

    plt.figure(figsize=(8, 5))
    if "x_center_arcsec" in plot_df.columns and plot_df["x_center_arcsec"].notna().any():
        plt.plot(plot_df["time_value"], plot_df["x_center_arcsec"], marker="o", label="x_center_arcsec")
    if "x_center_arcsec_topological_trial" in plot_df.columns and plot_df["x_center_arcsec_topological_trial"].notna().any():
        plt.plot(plot_df["time_value"], plot_df["x_center_arcsec_topological_trial"], marker="o", label="x_center_arcsec_topological_trial")
    if "y_center_arcsec" in plot_df.columns and plot_df["y_center_arcsec"].notna().any():
        plt.plot(plot_df["time_value"], plot_df["y_center_arcsec"], marker="o", label="y_center_arcsec")
    if "y_center_arcsec_topological_trial" in plot_df.columns and plot_df["y_center_arcsec_topological_trial"].notna().any():
        plt.plot(plot_df["time_value"], plot_df["y_center_arcsec_topological_trial"], marker="o", label="y_center_arcsec_topological_trial")
    plt.xlabel("time_value")
    plt.ylabel("center_arcsec")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_crause_offset_trial_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    plot_df = to_numeric(
        df,
        [
            "time_value",
            "ra_offset_arcsec",
            "dec_offset_arcsec",
            "ra_offset_arcsec_topological_trial",
            "dec_offset_arcsec_topological_trial",
        ],
    ).dropna(subset=["time_value"]).sort_values("time_value")

    plt.figure(figsize=(8, 5))
    if "ra_offset_arcsec" in plot_df.columns and plot_df["ra_offset_arcsec"].notna().any():
        plt.plot(plot_df["time_value"], plot_df["ra_offset_arcsec"], marker="o", label="ra_offset_arcsec")
    if "ra_offset_arcsec_topological_trial" in plot_df.columns and plot_df["ra_offset_arcsec_topological_trial"].notna().any():
        plt.plot(plot_df["time_value"], plot_df["ra_offset_arcsec_topological_trial"], marker="o", label="ra_offset_arcsec_topological_trial")
    if "dec_offset_arcsec" in plot_df.columns and plot_df["dec_offset_arcsec"].notna().any():
        plt.plot(plot_df["time_value"], plot_df["dec_offset_arcsec"], marker="o", label="dec_offset_arcsec")
    if "dec_offset_arcsec_topological_trial" in plot_df.columns and plot_df["dec_offset_arcsec_topological_trial"].notna().any():
        plt.plot(plot_df["time_value"], plot_df["dec_offset_arcsec_topological_trial"], marker="o", label="dec_offset_arcsec_topological_trial")
    plt.xlabel("time_value")
    plt.ylabel("offset_arcsec")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def detect_proxy_status(df: pd.DataFrame) -> Dict[str, bool]:
    return {
        "has_sigma_proxy": "sigma_proxy" in df.columns,
        "has_dsigma_dt_proxy": "dsigma_dt_proxy" in df.columns,
    }


def write_readme(out_dir: Path) -> None:
    text = """README_results.txt
==================

This results folder was generated by:
    src/echo of light/v838mon/topological/v838mon_topological_pipeline.py

Purpose
-------
This is the first topological-side trial version for V838 Mon.

Interpretation status
---------------------
This version does not claim a final topological-gravity derivation.
It applies configurable trial correction terms using sigma_proxy and dsigma_dt_proxy
if those columns exist. If they do not exist, they default to zero.

Files written here
------------------
- verified_copy.csv
- model_input_all_copy.csv
- model_input_tylenda_copy.csv
- model_input_crause_copy.csv
- tylenda_topological_trial.csv
- crause_topological_trial.csv
- dataset_row_counts.csv
- summary.json
- README_results.txt

Plots written here
------------------
- plots/tylenda_radius_topological_trial.png
- plots/crause_radius_topological_trial.png
- plots/tylenda_center_topological_trial.png
- plots/crause_offset_topological_trial.png
- plots/tylenda_radius_adjustment.png
- plots/crause_radius_adjustment.png

Important caution
-----------------
This is a trial comparison framework, not a claimed final proof.
Tylenda and Crause time systems remain separated.
"""
    (out_dir / "README_results.txt").write_text(text, encoding="utf-8")


def build_summary(
    verified_df: pd.DataFrame,
    all_df: pd.DataFrame,
    tylenda_df: pd.DataFrame,
    crause_df: pd.DataFrame,
    tylenda_trial_df: pd.DataFrame,
    crause_trial_df: pd.DataFrame,
    input_dir: Path,
    out_dir: Path,
    filenames: Dict[str, str],
    params: Dict[str, float],
) -> Dict:
    return {
        "pipeline_name": "V838 Mon Topological Pipeline (trial version)",
        "created_at": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "results_dir": str(out_dir),
        "input_files": filenames,
        "row_counts": {
            "measurements_verified": int(len(verified_df)),
            "model_input_all": int(len(all_df)),
            "model_input_tylenda": int(len(tylenda_df)),
            "model_input_crause": int(len(crause_df)),
            "tylenda_topological_trial": int(len(tylenda_trial_df)),
            "crause_topological_trial": int(len(crause_trial_df)),
        },
        "parameters": params,
        "proxy_detection": {
            "tylenda": detect_proxy_status(tylenda_df),
            "crause": detect_proxy_status(crause_df),
        },
        "generated_plots": [
            "plots/tylenda_radius_topological_trial.png",
            "plots/crause_radius_topological_trial.png",
            "plots/tylenda_center_topological_trial.png",
            "plots/crause_offset_topological_trial.png",
            "plots/tylenda_radius_adjustment.png",
            "plots/crause_radius_adjustment.png",
        ],
        "notes": [
            "This is the first topological-side trial layer built on top of the already prepared V838 Mon input tables.",
            "If sigma_proxy or dsigma_dt_proxy are absent, the pipeline uses zero-valued defaults.",
            "This means the first run can serve as a structural framework check even before proxy columns are added.",
            "A later version can replace the proxy columns with observationally motivated structural quantities.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V838 Mon topological trial pipeline")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--verified-csv", default="v838mon_measurements_verified.csv")
    parser.add_argument("--all-csv", default="v838mon_model_input_all.csv")
    parser.add_argument("--tylenda-csv", default="v838mon_model_input_tylenda.csv")
    parser.add_argument("--crause-csv", default="v838mon_model_input_crause.csv")

    parser.add_argument("--beta-sigma", type=float, default=1.0)
    parser.add_argument("--beta-dsigma", type=float, default=0.0)
    parser.add_argument("--gamma-sigma", type=float, default=0.25)
    parser.add_argument("--gamma-dsigma", type=float, default=0.0)

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

    tylenda_trial_df = apply_radius_topological_trial(
        tylenda_df,
        beta_sigma=args.beta_sigma,
        beta_dsigma=args.beta_dsigma,
    )
    tylenda_trial_df = apply_tylenda_center_trial(
        tylenda_trial_df,
        gamma_sigma=args.gamma_sigma,
        gamma_dsigma=args.gamma_dsigma,
    )

    crause_trial_df = apply_radius_topological_trial(
        crause_df,
        beta_sigma=args.beta_sigma,
        beta_dsigma=args.beta_dsigma,
    )
    crause_trial_df = apply_crause_offset_trial(
        crause_trial_df,
        gamma_sigma=args.gamma_sigma,
        gamma_dsigma=args.gamma_dsigma,
    )

    write_csv(tylenda_trial_df, out_dir / "tylenda_topological_trial.csv")
    write_csv(crause_trial_df, out_dir / "crause_topological_trial.csv")

    counts_df = pd.DataFrame([
        {"file": "verified", "rows": len(verified_df)},
        {"file": "model_input_all", "rows": len(all_df)},
        {"file": "model_input_tylenda", "rows": len(tylenda_df)},
        {"file": "model_input_crause", "rows": len(crause_df)},
        {"file": "tylenda_topological_trial", "rows": len(tylenda_trial_df)},
        {"file": "crause_topological_trial", "rows": len(crause_trial_df)},
    ])
    write_csv(counts_df, out_dir / "dataset_row_counts.csv")

    save_radius_trial_plot(
        tylenda_trial_df,
        plots_dir / "tylenda_radius_topological_trial.png",
        "Tylenda Radius: Observed vs Topological Trial",
    )
    save_radius_trial_plot(
        crause_trial_df,
        plots_dir / "crause_radius_topological_trial.png",
        "Crause Radius: Observed vs Topological Trial",
    )
    save_tylenda_center_trial_plot(
        tylenda_trial_df,
        plots_dir / "tylenda_center_topological_trial.png",
        "Tylenda Center: Observed vs Topological Trial",
    )
    save_crause_offset_trial_plot(
        crause_trial_df,
        plots_dir / "crause_offset_topological_trial.png",
        "Crause Offset: Observed vs Topological Trial",
    )
    save_radius_adjustment_plot(
        tylenda_trial_df,
        plots_dir / "tylenda_radius_adjustment.png",
        "Tylenda Topological Radius Adjustment",
    )
    save_radius_adjustment_plot(
        crause_trial_df,
        plots_dir / "crause_radius_adjustment.png",
        "Crause Topological Radius Adjustment",
    )

    summary = build_summary(
        verified_df=verified_df,
        all_df=all_df,
        tylenda_df=tylenda_df,
        crause_df=crause_df,
        tylenda_trial_df=tylenda_trial_df,
        crause_trial_df=crause_trial_df,
        input_dir=input_dir,
        out_dir=out_dir,
        filenames={
            "verified_csv": args.verified_csv,
            "all_csv": args.all_csv,
            "tylenda_csv": args.tylenda_csv,
            "crause_csv": args.crause_csv,
        },
        params={
            "beta_sigma": args.beta_sigma,
            "beta_dsigma": args.beta_dsigma,
            "gamma_sigma": args.gamma_sigma,
            "gamma_dsigma": args.gamma_dsigma,
        },
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_readme(out_dir)
    print(f"[DONE] Results written to: {out_dir}")


if __name__ == "__main__":
    main()
