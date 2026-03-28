#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V838 Mon Mass Structural Proxy Pipeline
--------------------------------------
Purpose
-------
This pipeline combines:
1. auxiliary mass/ejecta context from
   data/derived/Echo of Light/comparison/v838mon_paper_comparison_auxiliary.csv
2. light-echo radius inputs from
   data/derived/Echo of Light/input/v838mon_model_input_crause.csv
   data/derived/Echo of Light/input/v838mon_model_input_tylenda.csv

It then builds a heuristic mass-informed structural proxy and a trial effective
dimension time series. This is a consistency-oriented experiment, not a strict
physical derivation.

Default script location
-----------------------
src/echo of light/v838mon/mass/v838mon_mass_structural_proxy_pipeline.py

Default output root
-------------------
results/Echo of Light/output/V838Mon/topological/mass
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_COMPARISON_DIR = Path("data/derived/Echo of Light/comparison")
DEFAULT_INPUT_DIR = Path("data/derived/Echo of Light/input")
DEFAULT_OUTPUT_ROOT = Path("results/Echo of Light/output/V838Mon/topological/mass")


# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------
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


def to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# ---------------------------------------------------------------------
# Date / phase parsing
# ---------------------------------------------------------------------
def parse_observation_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["observation_date_dt"] = pd.to_datetime(out["observation_date"], errors="coerce")
    return out


def infer_phase_date(text: str) -> Optional[pd.Timestamp]:
    s = str(text).strip()

    # explicit YYYY-MM-DD
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        return pd.to_datetime(m.group(1), errors="coerce")

    # YYYY-MM to midpoint of month
    m = re.search(r"(\d{4})-(\d{2})", s)
    if m:
        y = int(m.group(1))
        mo = int(m.group(2))
        return pd.Timestamp(year=y, month=mo, day=15)

    # patterns used in the auxiliary table
    if "2002-01 to 2002-03" in s or "2002 january" in s.lower():
        return pd.Timestamp("2002-02-05")
    if "week before 2002-03 maximum" in s.lower():
        return pd.Timestamp("2002-03-01")
    if "2002-04 to 2002-05" in s:
        return pd.Timestamp("2002-04-20")
    if "jd 535-555 in 2002" in s.lower():
        return pd.Timestamp("2002-04-25")
    if "2002 event summarized later" in s.lower():
        return pd.Timestamp("2002-06-01")
    if "2002-05 to 2004-12" in s:
        return pd.Timestamp("2002-05-20")
    if "pre-2002" in s.lower():
        return pd.Timestamp("2001-12-31")

    return None


# ---------------------------------------------------------------------
# Auxiliary mass proxy construction
# ---------------------------------------------------------------------
def midpoint_or_upper(row: pd.Series) -> float:
    vmin = pd.to_numeric(row.get("value_min", np.nan), errors="coerce")
    vmax = pd.to_numeric(row.get("value_max", np.nan), errors="coerce")
    value_type = str(row.get("value_type", "")).strip().lower()
    unit = str(row.get("unit", "")).strip()

    if unit == "M_sun":
        if value_type == "upper_limit":
            if pd.notna(vmax):
                return float(vmax)
            if pd.notna(vmin):
                return float(vmin)
            return 0.0
        if pd.notna(vmin) and pd.notna(vmax):
            return float((vmin + vmax) / 2.0)
        if pd.notna(vmin):
            return float(vmin)
        if pd.notna(vmax):
            return float(vmax)
        return 0.0

    # non-mass items converted to small event strengths
    if unit == "flag":
        return 0.02
    if unit == "count":
        return 0.01 * float(vmax if pd.notna(vmax) else vmin if pd.notna(vmin) else 0.0)
    if unit == "km_s^-1":
        # very weak contextual contribution
        base = float(vmax if pd.notna(vmax) else vmin if pd.notna(vmin) else 0.0)
        return 0.001 * math.log1p(max(base, 0.0))

    return 0.0


def build_auxiliary_phase_table(aux_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in aux_df.iterrows():
        phase_date = infer_phase_date(row.get("approx_date_or_period", ""))
        mass_equivalent_proxy = midpoint_or_upper(row)
        rows.append(
            {
                "aux_id": row.get("aux_id", ""),
                "comparison_group": row.get("comparison_group", ""),
                "time_phase": row.get("time_phase", ""),
                "approx_date_or_period": row.get("approx_date_or_period", ""),
                "phase_date": phase_date,
                "parameter_name": row.get("parameter_name", ""),
                "value_min": row.get("value_min", ""),
                "value_max": row.get("value_max", ""),
                "unit": row.get("unit", ""),
                "value_type": row.get("value_type", ""),
                "paper_source": row.get("paper_source", ""),
                "comparison_role": row.get("comparison_role", ""),
                "can_join_to_geometry": row.get("can_join_to_geometry", ""),
                "mass_equivalent_proxy": mass_equivalent_proxy,
                "notes": row.get("notes", ""),
            }
        )
    out = pd.DataFrame(rows)
    out["phase_date"] = pd.to_datetime(out["phase_date"], errors="coerce")
    return out.sort_values(["phase_date", "aux_id"], na_position="last").reset_index(drop=True)


def cumulative_mass_proxy_for_date(phase_df: pd.DataFrame, obs_date: pd.Timestamp) -> float:
    if pd.isna(obs_date):
        return 0.0
    eligible = phase_df[phase_df["phase_date"].notna() & (phase_df["phase_date"] <= obs_date)]
    if eligible.empty:
        return 0.0
    return float(eligible["mass_equivalent_proxy"].sum())


def normalize_series_01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    smin = valid.min()
    smax = valid.max()
    if np.isclose(smin, smax):
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


# ---------------------------------------------------------------------
# Radius -> echo volume proxy
# ---------------------------------------------------------------------
def build_dataset_mass_volume_trial(
    df: pd.DataFrame,
    aux_phase_df: pd.DataFrame,
    d_bg: float,
    lambda_sigma: float,
    radius_power: float = 3.0,
) -> pd.DataFrame:
    out = df.copy()
    out = parse_observation_dates(out)
    out = to_numeric(out, ["time_value", "radius_arcsec"])

    # cumulative mass proxy by date
    out["mass_proxy_raw"] = out["observation_date_dt"].apply(lambda d: cumulative_mass_proxy_for_date(aux_phase_df, d))

    # echo geometric volume proxy from observed radius
    out["echo_volume_proxy_raw"] = np.where(
        out["radius_arcsec"].notna(),
        np.power(np.maximum(out["radius_arcsec"], 0.0), radius_power),
        np.nan,
    )

    # avoid zero division
    out["echo_volume_proxy_safe"] = out["echo_volume_proxy_raw"].replace(0, np.nan)

    # mass/volume structural proxy
    out["mass_volume_proxy_raw"] = out["mass_proxy_raw"] / out["echo_volume_proxy_safe"]
    out["mass_volume_proxy_raw"] = out["mass_volume_proxy_raw"].replace([np.inf, -np.inf], np.nan)

    # log compression helps with rough bounds
    out["mass_volume_proxy_log"] = np.log1p(out["mass_volume_proxy_raw"].fillna(0.0))

    # normalized sigma
    out["sigma_mass_volume"] = normalize_series_01(out["mass_volume_proxy_log"])
    out["dsigma_mass_volume_dt"] = finite_diff(out["time_value"], out["sigma_mass_volume"])

    # trial effective dimension (heuristic)
    out["Dw_trial_mass_volume"] = d_bg - lambda_sigma * out["sigma_mass_volume"]

    return out


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def _prep_plot(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return to_numeric(df, cols).dropna(subset=["time_value"]).sort_values("time_value")


def save_single_plot(df: pd.DataFrame, x: str, y_cols: List[Tuple[str, str]], out_path: Path, title: str, ylabel: str) -> None:
    p = _prep_plot(df, [x] + [c for c, _ in y_cols])
    plt.figure(figsize=(8, 5))
    for col, label in y_cols:
        if col in p.columns and p[col].notna().any():
            plt.plot(p[x], p[col], marker="o", label=label)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Summary / README
# ---------------------------------------------------------------------
def write_readme(out_dir: Path) -> None:
    text = """README_results.txt
==================

This results folder was generated by:
    src/echo of light/v838mon/mass/v838mon_mass_structural_proxy_pipeline.py

Purpose
-------
This is a mass-informed structural proxy experiment for V838 Mon.

Inputs used
-----------
1. comparison auxiliary table:
   data/derived/Echo of Light/comparison/v838mon_paper_comparison_auxiliary.csv

2. light echo radius tables:
   data/derived/Echo of Light/input/v838mon_model_input_crause.csv
   data/derived/Echo of Light/input/v838mon_model_input_tylenda.csv

Important interpretation note
-----------------------------
- mass_proxy_raw is a heuristic cumulative event-strength proxy derived from literature values
- echo_volume_proxy_raw is built from observed light-echo radius^3
- this is an echo-geometric volume proxy, NOT a direct ejecta volume measurement
- Dw_trial_mass_volume is a trial effective dimension index, NOT a strict derivation of D_w

This pipeline is intended only as a consistency-oriented test layer.
"""
    (out_dir / "README_results.txt").write_text(text, encoding="utf-8")


def build_summary(
    aux_df: pd.DataFrame,
    aux_phase_df: pd.DataFrame,
    tylenda_trial_df: pd.DataFrame,
    crause_trial_df: pd.DataFrame,
    comparison_dir: Path,
    input_dir: Path,
    out_dir: Path,
    params: Dict[str, float],
) -> Dict[str, object]:
    return {
        "pipeline_name": "V838 Mon Mass Structural Proxy Pipeline",
        "created_at": datetime.now().isoformat(),
        "comparison_dir": str(comparison_dir),
        "input_dir": str(input_dir),
        "results_dir": str(out_dir),
        "row_counts": {
            "auxiliary_input": int(len(aux_df)),
            "auxiliary_phase_table": int(len(aux_phase_df)),
            "tylenda_trial_rows": int(len(tylenda_trial_df)),
            "crause_trial_rows": int(len(crause_trial_df)),
        },
        "parameters": params,
        "notes": [
            "This pipeline combines auxiliary mass/ejecta context with light-echo radius tables.",
            "echo_volume_proxy_raw is derived from observed radius^3 and is only a geometric proxy.",
            "Dw_trial_mass_volume is a heuristic trial index for consistency-oriented comparison.",
        ],
        "generated_plots": [
            "plots/tylenda_mass_proxy_vs_time.png",
            "plots/crause_mass_proxy_vs_time.png",
            "plots/tylenda_echo_volume_proxy_vs_time.png",
            "plots/crause_echo_volume_proxy_vs_time.png",
            "plots/tylenda_sigma_mass_volume_vs_time.png",
            "plots/crause_sigma_mass_volume_vs_time.png",
            "plots/tylenda_dw_trial_mass_volume_vs_time.png",
            "plots/crause_dw_trial_mass_volume_vs_time.png",
        ],
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="V838 Mon mass structural proxy pipeline")
    parser.add_argument("--comparison-dir", default=str(DEFAULT_COMPARISON_DIR))
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))

    parser.add_argument("--aux-csv", default="v838mon_paper_comparison_auxiliary.csv")
    parser.add_argument("--tylenda-csv", default="v838mon_model_input_tylenda.csv")
    parser.add_argument("--crause-csv", default="v838mon_model_input_crause.csv")

    parser.add_argument("--d-bg", type=float, default=3.0)
    parser.add_argument("--lambda-sigma", type=float, default=0.8)
    parser.add_argument("--radius-power", type=float, default=3.0)

    args = parser.parse_args()

    comparison_dir = Path(args.comparison_dir)
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)

    aux_path = ensure_file(comparison_dir / args.aux_csv)
    tylenda_path = ensure_file(input_dir / args.tylenda_csv)
    crause_path = ensure_file(input_dir / args.crause_csv)

    aux_df = load_csv(aux_path)
    tylenda_df = load_csv(tylenda_path)
    crause_df = load_csv(crause_path)

    aux_phase_df = build_auxiliary_phase_table(aux_df)

    tylenda_trial_df = build_dataset_mass_volume_trial(
        tylenda_df,
        aux_phase_df=aux_phase_df,
        d_bg=args.d_bg,
        lambda_sigma=args.lambda_sigma,
        radius_power=args.radius_power,
    )
    crause_trial_df = build_dataset_mass_volume_trial(
        crause_df,
        aux_phase_df=aux_phase_df,
        d_bg=args.d_bg,
        lambda_sigma=args.lambda_sigma,
        radius_power=args.radius_power,
    )

    out_dir = make_output_dir(output_root)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save tables
    write_csv(aux_phase_df, out_dir / "v838mon_mass_phase_table.csv")
    write_csv(tylenda_trial_df, out_dir / "v838mon_tylenda_mass_volume_trial.csv")
    write_csv(crause_trial_df, out_dir / "v838mon_crause_mass_volume_trial.csv")

    merged_proxy = pd.concat(
        [
            tylenda_trial_df.assign(dataset_source="Tylenda"),
            crause_trial_df.assign(dataset_source="Crause"),
        ],
        ignore_index=True,
    )
    write_csv(merged_proxy, out_dir / "v838mon_mass_volume_trial_merged.csv")

    # Plots
    save_single_plot(
        tylenda_trial_df,
        "time_value",
        [("mass_proxy_raw", "mass_proxy_raw")],
        plots_dir / "tylenda_mass_proxy_vs_time.png",
        "Tylenda Mass Proxy vs Time",
        "mass_proxy_raw",
    )
    save_single_plot(
        crause_trial_df,
        "time_value",
        [("mass_proxy_raw", "mass_proxy_raw")],
        plots_dir / "crause_mass_proxy_vs_time.png",
        "Crause Mass Proxy vs Time",
        "mass_proxy_raw",
    )
    save_single_plot(
        tylenda_trial_df,
        "time_value",
        [("echo_volume_proxy_raw", "echo_volume_proxy_raw")],
        plots_dir / "tylenda_echo_volume_proxy_vs_time.png",
        "Tylenda Echo Volume Proxy vs Time",
        "echo_volume_proxy_raw",
    )
    save_single_plot(
        crause_trial_df,
        "time_value",
        [("echo_volume_proxy_raw", "echo_volume_proxy_raw")],
        plots_dir / "crause_echo_volume_proxy_vs_time.png",
        "Crause Echo Volume Proxy vs Time",
        "echo_volume_proxy_raw",
    )
    save_single_plot(
        tylenda_trial_df,
        "time_value",
        [("sigma_mass_volume", "sigma_mass_volume"), ("dsigma_mass_volume_dt", "dsigma_mass_volume_dt")],
        plots_dir / "tylenda_sigma_mass_volume_vs_time.png",
        "Tylenda Sigma Mass-Volume Proxy vs Time",
        "proxy",
    )
    save_single_plot(
        crause_trial_df,
        "time_value",
        [("sigma_mass_volume", "sigma_mass_volume"), ("dsigma_mass_volume_dt", "dsigma_mass_volume_dt")],
        plots_dir / "crause_sigma_mass_volume_vs_time.png",
        "Crause Sigma Mass-Volume Proxy vs Time",
        "proxy",
    )
    save_single_plot(
        tylenda_trial_df,
        "time_value",
        [("Dw_trial_mass_volume", "Dw_trial_mass_volume")],
        plots_dir / "tylenda_dw_trial_mass_volume_vs_time.png",
        "Tylenda Trial Effective Dimension vs Time",
        "Dw_trial_mass_volume",
    )
    save_single_plot(
        crause_trial_df,
        "time_value",
        [("Dw_trial_mass_volume", "Dw_trial_mass_volume")],
        plots_dir / "crause_dw_trial_mass_volume_vs_time.png",
        "Crause Trial Effective Dimension vs Time",
        "Dw_trial_mass_volume",
    )

    summary = build_summary(
        aux_df=aux_df,
        aux_phase_df=aux_phase_df,
        tylenda_trial_df=tylenda_trial_df,
        crause_trial_df=crause_trial_df,
        comparison_dir=comparison_dir,
        input_dir=input_dir,
        out_dir=out_dir,
        params={
            "d_bg": args.d_bg,
            "lambda_sigma": args.lambda_sigma,
            "radius_power": args.radius_power,
        },
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_readme(out_dir)
    print(f"[DONE] Results written to: {out_dir}")


if __name__ == "__main__":
    main()
