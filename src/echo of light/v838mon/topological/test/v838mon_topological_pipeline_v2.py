#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V838 Mon Topological Pipeline v2
--------------------------------
Purpose
-------
A trial topological pipeline that computes sigma_proxy automatically from
observed center-offset magnitude when no explicit proxy columns are present.

Key idea
--------
- Crause: use sqrt(ra_offset_arcsec^2 + dec_offset_arcsec^2)
- Tylenda: use sqrt(x_center_arcsec^2 + y_center_arcsec^2)

Then normalize sigma_proxy to [0, 1] within each dataset and estimate
dsigma_dt_proxy by finite differences over time_value.

This is NOT a final derivation of topological gravity.
It is a pragmatic proxy-based test layer.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
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


def normalize_series_01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    smin = valid.min()
    smax = valid.max()
    if np.isclose(smax, smin):
        out = pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
        out.loc[valid.index] = 1.0
        return out
    return (s - smin) / (smax - smin)


def finite_diff(time_s: pd.Series, value_s: pd.Series) -> pd.Series:
    t = pd.to_numeric(time_s, errors="coerce")
    v = pd.to_numeric(value_s, errors="coerce")
    out = pd.Series(np.zeros(len(v)), index=v.index, dtype=float)
    valid = (~t.isna()) & (~v.isna())
    if valid.sum() < 2:
        return out
    idx = t[valid].sort_values().index
    tv = t.loc[idx].to_numpy(dtype=float)
    vv = v.loc[idx].to_numpy(dtype=float)
    dvdt = np.zeros_like(vv)
    for i in range(len(vv)):
        if i == 0:
            dt = tv[i + 1] - tv[i]
            dvdt[i] = 0.0 if np.isclose(dt, 0.0) else (vv[i + 1] - vv[i]) / dt
        elif i == len(vv) - 1:
            dt = tv[i] - tv[i - 1]
            dvdt[i] = 0.0 if np.isclose(dt, 0.0) else (vv[i] - vv[i - 1]) / dt
        else:
            dt = tv[i + 1] - tv[i - 1]
            dvdt[i] = 0.0 if np.isclose(dt, 0.0) else (vv[i + 1] - vv[i - 1]) / dt
    out.loc[idx] = dvdt
    return out


def compute_sigma_proxy(df: pd.DataFrame, dataset_kind: str) -> pd.DataFrame:
    out = df.copy()
    if "sigma_proxy" in out.columns and pd.to_numeric(out["sigma_proxy"], errors="coerce").notna().any():
        out["sigma_proxy_source"] = "input_column"
        if "dsigma_dt_proxy" not in out.columns or not pd.to_numeric(out["dsigma_dt_proxy"], errors="coerce").notna().any():
            out["dsigma_dt_proxy"] = finite_diff(out["time_value"], out["sigma_proxy"])
        return out

    out = to_numeric(out, ["time_value", "ra_offset_arcsec", "dec_offset_arcsec", "x_center_arcsec", "y_center_arcsec"])

    if dataset_kind.lower() == "crause":
        mag = np.sqrt(
            out["ra_offset_arcsec"].fillna(0.0) ** 2 +
            out["dec_offset_arcsec"].fillna(0.0) ** 2
        )
        out["sigma_proxy_raw"] = mag
        out["sigma_proxy"] = normalize_series_01(out["sigma_proxy_raw"])
        out["sigma_proxy_source"] = "offset_magnitude_normalized"
    else:
        mag = np.sqrt(
            out["x_center_arcsec"].fillna(0.0) ** 2 +
            out["y_center_arcsec"].fillna(0.0) ** 2
        )
        out["sigma_proxy_raw"] = mag
        out["sigma_proxy"] = normalize_series_01(out["sigma_proxy_raw"])
        out["sigma_proxy_source"] = "center_shift_magnitude_normalized"

    out["dsigma_dt_proxy"] = finite_diff(out["time_value"], out["sigma_proxy"])
    return out


def apply_radius_trial(df: pd.DataFrame, beta_sigma: float, beta_dsigma: float) -> pd.DataFrame:
    out = df.copy()
    out = to_numeric(out, ["radius_arcsec", "sigma_proxy", "dsigma_dt_proxy"])
    out["topological_radius_adjustment"] = beta_sigma * out["sigma_proxy"] + beta_dsigma * out["dsigma_dt_proxy"]
    out["radius_arcsec_topological_trial"] = out["radius_arcsec"] + out["topological_radius_adjustment"]
    return out


def apply_tylenda_center_trial(df: pd.DataFrame, gamma_sigma: float, gamma_dsigma: float) -> pd.DataFrame:
    out = df.copy()
    out = to_numeric(out, ["x_center_arcsec", "y_center_arcsec", "sigma_proxy", "dsigma_dt_proxy"])
    shift = gamma_sigma * out["sigma_proxy"] + gamma_dsigma * out["dsigma_dt_proxy"]
    out["topological_center_adjustment"] = shift
    out["x_center_arcsec_topological_trial"] = out["x_center_arcsec"] + shift
    out["y_center_arcsec_topological_trial"] = out["y_center_arcsec"] + shift
    return out


def apply_crause_offset_trial(df: pd.DataFrame, gamma_sigma: float, gamma_dsigma: float) -> pd.DataFrame:
    out = df.copy()
    out = to_numeric(out, ["ra_offset_arcsec", "dec_offset_arcsec", "sigma_proxy", "dsigma_dt_proxy"])
    shift = gamma_sigma * out["sigma_proxy"] + gamma_dsigma * out["dsigma_dt_proxy"]
    out["topological_center_adjustment"] = shift
    out["ra_offset_arcsec_topological_trial"] = out["ra_offset_arcsec"] + shift
    out["dec_offset_arcsec_topological_trial"] = out["dec_offset_arcsec"] + shift
    return out


def save_plot_radius(df: pd.DataFrame, out_path: Path, title: str) -> None:
    p = to_numeric(df, ["time_value", "radius_arcsec", "radius_arcsec_topological_trial"]).dropna(subset=["time_value"]).sort_values("time_value")
    plt.figure(figsize=(8, 5))
    plt.plot(p["time_value"], p["radius_arcsec"], marker="o", label="observed_radius_arcsec")
    plt.plot(p["time_value"], p["radius_arcsec_topological_trial"], marker="o", label="topological_trial_radius_arcsec")
    plt.xlabel("time_value")
    plt.ylabel("radius_arcsec")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_plot_adjustment(df: pd.DataFrame, out_path: Path, title: str) -> None:
    p = to_numeric(df, ["time_value", "topological_radius_adjustment"]).dropna(subset=["time_value"]).sort_values("time_value")
    plt.figure(figsize=(8, 5))
    plt.plot(p["time_value"], p["topological_radius_adjustment"], marker="o", label="topological_radius_adjustment")
    plt.xlabel("time_value")
    plt.ylabel("adjustment")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_plot_sigma(df: pd.DataFrame, out_path: Path, title: str) -> None:
    p = to_numeric(df, ["time_value", "sigma_proxy", "dsigma_dt_proxy"]).dropna(subset=["time_value"]).sort_values("time_value")
    plt.figure(figsize=(8, 5))
    plt.plot(p["time_value"], p["sigma_proxy"], marker="o", label="sigma_proxy")
    plt.plot(p["time_value"], p["dsigma_dt_proxy"], marker="o", label="dsigma_dt_proxy")
    plt.xlabel("time_value")
    plt.ylabel("proxy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_plot_tylenda_center(df: pd.DataFrame, out_path: Path, title: str) -> None:
    p = to_numeric(df, ["time_value", "x_center_arcsec", "y_center_arcsec", "x_center_arcsec_topological_trial", "y_center_arcsec_topological_trial"]).dropna(subset=["time_value"]).sort_values("time_value")
    plt.figure(figsize=(8, 5))
    plt.plot(p["time_value"], p["x_center_arcsec"], marker="o", label="x_center_arcsec")
    plt.plot(p["time_value"], p["x_center_arcsec_topological_trial"], marker="o", label="x_center_arcsec_topological_trial")
    plt.plot(p["time_value"], p["y_center_arcsec"], marker="o", label="y_center_arcsec")
    plt.plot(p["time_value"], p["y_center_arcsec_topological_trial"], marker="o", label="y_center_arcsec_topological_trial")
    plt.xlabel("time_value")
    plt.ylabel("center_arcsec")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_plot_crause_offset(df: pd.DataFrame, out_path: Path, title: str) -> None:
    p = to_numeric(df, ["time_value", "ra_offset_arcsec", "dec_offset_arcsec", "ra_offset_arcsec_topological_trial", "dec_offset_arcsec_topological_trial"]).dropna(subset=["time_value"]).sort_values("time_value")
    plt.figure(figsize=(8, 5))
    plt.plot(p["time_value"], p["ra_offset_arcsec"], marker="o", label="ra_offset_arcsec")
    plt.plot(p["time_value"], p["ra_offset_arcsec_topological_trial"], marker="o", label="ra_offset_arcsec_topological_trial")
    plt.plot(p["time_value"], p["dec_offset_arcsec"], marker="o", label="dec_offset_arcsec")
    plt.plot(p["time_value"], p["dec_offset_arcsec_topological_trial"], marker="o", label="dec_offset_arcsec_topological_trial")
    plt.xlabel("time_value")
    plt.ylabel("offset_arcsec")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_readme(out_dir: Path) -> None:
    text = """README_results.txt
==================

This results folder was generated by:
    src/echo of light/v838mon/topological/v838mon_topological_pipeline_v2.py

Purpose
-------
This is the second topological-side trial version for V838 Mon.
Unlike v1, it automatically computes sigma_proxy when no explicit proxy columns exist.

Proxy definitions
-----------------
- Crause:
  sigma_proxy_raw = sqrt(ra_offset_arcsec^2 + dec_offset_arcsec^2)
- Tylenda:
  sigma_proxy_raw = sqrt(x_center_arcsec^2 + y_center_arcsec^2)

Then sigma_proxy is normalized to [0, 1] within each dataset and dsigma_dt_proxy
is estimated from finite differences over time_value.

Interpretation status
---------------------
This is still a proxy-based trial framework, not a final derivation.

Important caution
-----------------
Tylenda and Crause time systems remain separated.
"""
    (out_dir / "README_results.txt").write_text(text, encoding="utf-8")


def build_summary(verified_df, all_df, tylenda_df, crause_df, tylenda_trial_df, crause_trial_df, input_dir, out_dir, filenames, params):
    return {
        "pipeline_name": "V838 Mon Topological Pipeline v2",
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
        "sigma_proxy_source": {
            "tylenda": sorted(set(tylenda_trial_df.get("sigma_proxy_source", pd.Series(dtype=str)).astype(str).tolist())),
            "crause": sorted(set(crause_trial_df.get("sigma_proxy_source", pd.Series(dtype=str)).astype(str).tolist())),
        },
        "generated_plots": [
            "plots/tylenda_radius_topological_trial.png",
            "plots/crause_radius_topological_trial.png",
            "plots/tylenda_center_topological_trial.png",
            "plots/crause_offset_topological_trial.png",
            "plots/tylenda_radius_adjustment.png",
            "plots/crause_radius_adjustment.png",
            "plots/tylenda_sigma_proxy.png",
            "plots/crause_sigma_proxy.png",
        ],
        "notes": [
            "v2 computes sigma_proxy automatically from observed center-offset magnitude when no input proxy columns exist.",
            "This is still a heuristic trial layer, not a final proof.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V838 Mon topological trial pipeline v2")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--verified-csv", default="v838mon_measurements_verified.csv")
    parser.add_argument("--all-csv", default="v838mon_model_input_all.csv")
    parser.add_argument("--tylenda-csv", default="v838mon_model_input_tylenda.csv")
    parser.add_argument("--crause-csv", default="v838mon_model_input_crause.csv")
    parser.add_argument("--beta-sigma", type=float, default=1.0)
    parser.add_argument("--beta-dsigma", type=float, default=0.25)
    parser.add_argument("--gamma-sigma", type=float, default=0.25)
    parser.add_argument("--gamma-dsigma", type=float, default=0.10)
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

    tylenda_proxy_df = compute_sigma_proxy(tylenda_df, "tylenda")
    crause_proxy_df = compute_sigma_proxy(crause_df, "crause")

    tylenda_trial_df = apply_radius_trial(tylenda_proxy_df, args.beta_sigma, args.beta_dsigma)
    tylenda_trial_df = apply_tylenda_center_trial(tylenda_trial_df, args.gamma_sigma, args.gamma_dsigma)

    crause_trial_df = apply_radius_trial(crause_proxy_df, args.beta_sigma, args.beta_dsigma)
    crause_trial_df = apply_crause_offset_trial(crause_trial_df, args.gamma_sigma, args.gamma_dsigma)

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

    save_plot_radius(tylenda_trial_df, plots_dir / "tylenda_radius_topological_trial.png", "Tylenda Radius: Observed vs Topological Trial")
    save_plot_radius(crause_trial_df, plots_dir / "crause_radius_topological_trial.png", "Crause Radius: Observed vs Topological Trial")
    save_plot_tylenda_center(tylenda_trial_df, plots_dir / "tylenda_center_topological_trial.png", "Tylenda Center: Observed vs Topological Trial")
    save_plot_crause_offset(crause_trial_df, plots_dir / "crause_offset_topological_trial.png", "Crause Offset: Observed vs Topological Trial")
    save_plot_adjustment(tylenda_trial_df, plots_dir / "tylenda_radius_adjustment.png", "Tylenda Topological Radius Adjustment")
    save_plot_adjustment(crause_trial_df, plots_dir / "crause_radius_adjustment.png", "Crause Topological Radius Adjustment")
    save_plot_sigma(tylenda_trial_df, plots_dir / "tylenda_sigma_proxy.png", "Tylenda Sigma Proxy")
    save_plot_sigma(crause_trial_df, plots_dir / "crause_sigma_proxy.png", "Crause Sigma Proxy")

    summary = build_summary(
        verified_df, all_df, tylenda_df, crause_df, tylenda_trial_df, crause_trial_df,
        input_dir, out_dir,
        {
            "verified_csv": args.verified_csv,
            "all_csv": args.all_csv,
            "tylenda_csv": args.tylenda_csv,
            "crause_csv": args.crause_csv,
        },
        {
            "beta_sigma": args.beta_sigma,
            "beta_dsigma": args.beta_dsigma,
            "gamma_sigma": args.gamma_sigma,
            "gamma_dsigma": args.gamma_dsigma,
        }
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_readme(out_dir)
    print(f"[DONE] Results written to: {out_dir}")


if __name__ == "__main__":
    main()
