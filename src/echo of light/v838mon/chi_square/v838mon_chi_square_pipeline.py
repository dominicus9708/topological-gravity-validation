#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("data/derived/Echo of Light/input")
DEFAULT_RESULTS_ROOT = Path("results/Echo of Light/output/V838Mon/topological/chi_square")
DEFAULT_MASS_RESULTS_ROOT = Path("results/Echo of Light/output/V838Mon/topological/mass")


def ensure_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def latest_timestamp_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Results root does not exist: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No timestamp result folders found under: {root}")
    return sorted(candidates)[-1]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def make_output_dir(root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_base_input(input_dir: Path, filename: str) -> pd.DataFrame:
    df = pd.read_csv(ensure_file(input_dir / filename), encoding="utf-8-sig")
    for col in ["time_value", "radius_arcsec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_mass_trial(mass_results_root: Path, filename: str, explicit_results_dir: str | None) -> Tuple[pd.DataFrame, Path]:
    result_dir = Path(explicit_results_dir) if explicit_results_dir else latest_timestamp_dir(mass_results_root)
    df = pd.read_csv(ensure_file(result_dir / filename), encoding="utf-8-sig")
    for col in ["time_value", "radius_arcsec", "sigma_mass_volume", "Dw_trial_mass_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, result_dir


def build_standard_radius(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("time_value").reset_index(drop=True)
    out["radius_standard"] = out["radius_arcsec"]
    return out


def build_topological_radius(df: pd.DataFrame, beta_radius: float) -> pd.DataFrame:
    out = df.copy().sort_values("time_value").reset_index(drop=True)
    if "sigma_mass_volume" not in out.columns:
        raise KeyError("sigma_mass_volume column not found in mass trial results.")
    out["radius_standard"] = out["radius_arcsec"]
    out["radius_topological"] = out["radius_arcsec"] * (1.0 + beta_radius * out["sigma_mass_volume"])
    return out


def attach_sigma_obs(df: pd.DataFrame, relative_error: float, floor_arcsec: float) -> pd.DataFrame:
    out = df.copy()
    out["sigma_radius_obs"] = np.maximum(relative_error * out["radius_arcsec"], floor_arcsec)
    return out


def compute_chi_square(df: pd.DataFrame, model_col: str) -> Dict[str, float]:
    valid = df[["radius_arcsec", "sigma_radius_obs", model_col]].dropna().copy()
    if valid.empty:
        return {
            "n_points": 0,
            "chi2": float("nan"),
            "dof": float("nan"),
            "reduced_chi2": float("nan"),
            "rmse": float("nan"),
        }

    resid = valid["radius_arcsec"] - valid[model_col]
    chi2 = np.sum((resid / valid["sigma_radius_obs"]) ** 2)
    n_points = len(valid)
    # heuristic 1-parameter penalty for topological model; 0 for standard baseline copy
    n_params = 0 if model_col == "radius_standard" else 1
    dof = max(n_points - n_params, 1)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return {
        "n_points": int(n_points),
        "chi2": float(chi2),
        "dof": int(dof),
        "reduced_chi2": float(chi2 / dof),
        "rmse": rmse,
    }


def build_metrics_table(dataset_name: str, standard_metrics: Dict[str, float], topo_metrics: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "model": "standard_baseline",
                **standard_metrics,
            },
            {
                "dataset": dataset_name,
                "model": "topological_mass_volume_trial",
                **topo_metrics,
            },
        ]
    )


def save_plot(df: pd.DataFrame, dataset_name: str, out_path: Path) -> None:
    p = df.dropna(subset=["time_value", "radius_arcsec"]).sort_values("time_value")
    plt.figure(figsize=(8, 5))
    plt.plot(p["time_value"], p["radius_arcsec"], marker="o", label="observed_radius_arcsec")
    if "radius_standard" in p.columns:
        plt.plot(p["time_value"], p["radius_standard"], marker="o", label="radius_standard")
    if "radius_topological" in p.columns:
        plt.plot(p["time_value"], p["radius_topological"], marker="o", label="radius_topological")
    plt.xlabel("time_value")
    plt.ylabel("radius_arcsec")
    plt.title(f"{dataset_name} Radius: Observed vs Standard vs Topological")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_residual_plot(df: pd.DataFrame, dataset_name: str, out_path: Path) -> None:
    p = df.dropna(subset=["time_value", "radius_arcsec"]).sort_values("time_value").copy()
    p["resid_standard"] = p["radius_arcsec"] - p["radius_standard"]
    p["resid_topological"] = p["radius_arcsec"] - p["radius_topological"]
    plt.figure(figsize=(8, 5))
    plt.plot(p["time_value"], p["resid_standard"], marker="o", label="obs - standard")
    plt.plot(p["time_value"], p["resid_topological"], marker="o", label="obs - topological")
    plt.xlabel("time_value")
    plt.ylabel("radius residual (arcsec)")
    plt.title(f"{dataset_name} Radius Residuals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="V838 Mon chi-square evaluation pipeline")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--mass-results-root", default=str(DEFAULT_MASS_RESULTS_ROOT))
    parser.add_argument("--mass-results-dir", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_RESULTS_ROOT))

    parser.add_argument("--tylenda-input-csv", default="v838mon_model_input_tylenda.csv")
    parser.add_argument("--crause-input-csv", default="v838mon_model_input_crause.csv")
    parser.add_argument("--tylenda-mass-csv", default="v838mon_tylenda_mass_volume_trial.csv")
    parser.add_argument("--crause-mass-csv", default="v838mon_crause_mass_volume_trial.csv")

    parser.add_argument("--radius-relative-error", type=float, default=0.03)
    parser.add_argument("--radius-error-floor", type=float, default=0.3)
    parser.add_argument("--beta-radius", type=float, default=0.05)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    mass_results_root = Path(args.mass_results_root)
    output_root = Path(args.output_root)

    # observed inputs
    tylenda_obs = load_base_input(input_dir, args.tylenda_input_csv)
    crause_obs = load_base_input(input_dir, args.crause_input_csv)

    # mass/topological trial results
    tylenda_mass, used_mass_dir = load_mass_trial(mass_results_root, args.tylenda_mass_csv, args.mass_results_dir)
    crause_mass, used_mass_dir_2 = load_mass_trial(mass_results_root, args.crause_mass_csv, args.mass_results_dir)

    # standard baseline = copy of observed radius
    tylenda_std = build_standard_radius(tylenda_obs)
    crause_std = build_standard_radius(crause_obs)

    # topological radius from sigma_mass_volume
    tylenda_topo = build_topological_radius(tylenda_mass, args.beta_radius)
    crause_topo = build_topological_radius(crause_mass, args.beta_radius)

    # observed sigma
    tylenda_eval = attach_sigma_obs(tylenda_topo, args.radius_relative_error, args.radius_error_floor)
    crause_eval = attach_sigma_obs(crause_topo, args.radius_relative_error, args.radius_error_floor)

    # standard evaluation uses identical observed radius baseline
    tylenda_std_eval = attach_sigma_obs(tylenda_std, args.radius_relative_error, args.radius_error_floor)
    crause_std_eval = attach_sigma_obs(crause_std, args.radius_relative_error, args.radius_error_floor)

    tylenda_standard_metrics = compute_chi_square(tylenda_std_eval, "radius_standard")
    crause_standard_metrics = compute_chi_square(crause_std_eval, "radius_standard")
    tylenda_topo_metrics = compute_chi_square(tylenda_eval, "radius_topological")
    crause_topo_metrics = compute_chi_square(crause_eval, "radius_topological")

    output_dir = make_output_dir(output_root)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save per-dataset tables
    write_csv(tylenda_eval, output_dir / "v838mon_tylenda_chi_square_input.csv")
    write_csv(crause_eval, output_dir / "v838mon_crause_chi_square_input.csv")

    metrics_df = pd.concat(
        [
            build_metrics_table("Tylenda", tylenda_standard_metrics, tylenda_topo_metrics),
            build_metrics_table("Crause", crause_standard_metrics, crause_topo_metrics),
        ],
        ignore_index=True,
    )
    write_csv(metrics_df, output_dir / "v838mon_chi_square_summary.csv")

    # aggregate
    std_total_chi2 = tylenda_standard_metrics["chi2"] + crause_standard_metrics["chi2"]
    topo_total_chi2 = tylenda_topo_metrics["chi2"] + crause_topo_metrics["chi2"]
    std_total_dof = tylenda_standard_metrics["dof"] + crause_standard_metrics["dof"]
    topo_total_dof = tylenda_topo_metrics["dof"] + crause_topo_metrics["dof"]

    aggregate = {
        "used_mass_results_dir": str(used_mass_dir),
        "used_mass_results_dir_crause": str(used_mass_dir_2),
        "parameters": {
            "radius_relative_error": args.radius_relative_error,
            "radius_error_floor": args.radius_error_floor,
            "beta_radius": args.beta_radius,
        },
        "aggregate_metrics": {
            "standard_total_chi2": float(std_total_chi2),
            "standard_total_dof": int(std_total_dof),
            "standard_reduced_chi2": float(std_total_chi2 / max(std_total_dof, 1)),
            "topological_total_chi2": float(topo_total_chi2),
            "topological_total_dof": int(topo_total_dof),
            "topological_reduced_chi2": float(topo_total_chi2 / max(topo_total_dof, 1)),
            "delta_chi2_topological_minus_standard": float(topo_total_chi2 - std_total_chi2),
        },
        "notes": [
            "Tylenda and Crause are evaluated separately and only aggregated at the final metric stage.",
            "Standard baseline is a trivial observational baseline copy, so its chi-square is expected to be zero.",
            "The meaningful comparison is whether the topological radius trial remains within acceptable residual size under the assumed radius uncertainty.",
            "Observed radius uncertainty is modeled as max(relative_error * radius, error_floor).",
        ],
    }
    with (output_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    save_plot(tylenda_eval, "Tylenda", plots_dir / "tylenda_radius_model_comparison.png")
    save_plot(crause_eval, "Crause", plots_dir / "crause_radius_model_comparison.png")
    save_residual_plot(tylenda_eval, "Tylenda", plots_dir / "tylenda_radius_residuals.png")
    save_residual_plot(crause_eval, "Crause", plots_dir / "crause_radius_residuals.png")

    readme = f"""README_results.txt
==================
This results folder was generated by:
    src/echo of light/v838mon/chi_square/v838mon_chi_square_pipeline.py

Purpose
-------
This pipeline evaluates radius-based chi-square metrics for V838 Mon.

Important interpretation
------------------------
- Tylenda and Crause time systems remain separated.
- The direct observational baseline reproduces the observed radius exactly, so its chi-square is zero by construction.
- The informative quantity is the topological trial residual size relative to the assumed observational uncertainty.
- This is therefore a consistency-oriented chi-square layer, not a proof that the model is superior to the standard interpretation.

Used mass results directory
---------------------------
{used_mass_dir}

Main outputs
------------
- v838mon_tylenda_chi_square_input.csv
- v838mon_crause_chi_square_input.csv
- v838mon_chi_square_summary.csv
- aggregate_metrics.json
- plots/*.png
"""
    (output_dir / "README_results.txt").write_text(readme, encoding="utf-8")

    print(f"[DONE] Results written to: {output_dir}")


if __name__ == "__main__":
    main()
