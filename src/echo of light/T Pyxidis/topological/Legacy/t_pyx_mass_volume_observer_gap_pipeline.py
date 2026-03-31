#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T Pyxidis Mass-Volume + Observer-Gap Pipeline
--------------------------------------------
Purpose
-------
This pipeline reconstructs the T Pyxidis topological trial in a way that is
more consistent with the V838 Mon mass-structural-proxy style, while still
respecting the patch-based nature of the T Pyxidis light-echo data.

It combines:

1. patch geometry observables
   - delay_time_days
   - theta_arcsec
   - z_1e17_cm

2. a heuristic local mass-volume structural proxy
   - mass proxy inferred from earlier / more directly disturbed echo patches
   - geometric volume proxy inferred from theta^3

3. a theory-aligned observer-gap term
   - inspired by Paper 1: difference between internal disturbed structure
     and external-observer describability

This is a consistency-oriented experiment, not a strict physical derivation.

Default script location
-----------------------
src/echo of light/T Pyxidis/topological/t_pyx_mass_volume_observer_gap_pipeline.py

Default output root
-------------------
results/Echo of Light/output/T Pyxidis/topological/mass_volume_observer_gap
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = Path(
    r"data/derived/Echo of Light/input/T Pyxidis/t_pyx_standard_echo_patches_original_paper.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"results/Echo of Light/output/T Pyxidis/topological/mass_volume_observer_gap"
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


def validate_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


# ---------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Patch mass / volume proxy
# ---------------------------------------------------------------------
def build_patch_trial_table(
    df: pd.DataFrame,
    d_bg: float,
    lambda_sigma: float,
    radius_power: float,
    w_mass_volume: float,
    w_local_burst: float,
    w_observer_gap: float,
    local_strength: float,
) -> pd.DataFrame:
    out = df.copy()
    out = to_numeric(out, ["delay_time_days", "theta_arcsec", "z_1e17_cm"])

    # ----- Base observables -----
    out["theta_norm"] = normalize_series_01(out["theta_arcsec"])
    out["delay_norm"] = normalize_series_01(out["delay_time_days"])
    out["z_abs_1e17_cm"] = out["z_1e17_cm"].abs()
    out["z_abs_norm"] = normalize_series_01(out["z_abs_1e17_cm"])

    # ----- Local burst term -----
    # Earlier / more directly disturbed patches are assigned larger local degradation.
    out["sigma_local_burst"] = local_strength * (1.0 - out["delay_norm"])

    # ----- Observer-gap term -----
    # External observer mismatch between projected angular structure and line-of-sight displacement.
    out["sigma_observer_gap"] = 0.50 * out["z_abs_norm"] + 0.50 * (out["delay_norm"] - out["theta_norm"]).abs()

    # ----- Mass proxy -----
    # Heuristic local mass-context proxy:
    # earlier / more directly affected patches receive larger mass-context contribution.
    # This mirrors the V838 Mon logic of combining structural disturbance with a mass-like proxy.
    out["mass_proxy_raw"] = 1.0 + 4.0 * out["sigma_local_burst"]

    # ----- Echo geometric volume proxy -----
    # Patch-based analogue of V838Mon radius^3 volume proxy.
    out["echo_volume_proxy_raw"] = np.power(np.maximum(out["theta_arcsec"], 0.0), radius_power)
    out["echo_volume_proxy_safe"] = out["echo_volume_proxy_raw"].replace(0, np.nan)

    # ----- Mass / volume structural proxy -----
    out["mass_volume_proxy_raw"] = out["mass_proxy_raw"] / out["echo_volume_proxy_safe"]
    out["mass_volume_proxy_raw"] = out["mass_volume_proxy_raw"].replace([np.inf, -np.inf], np.nan)
    out["mass_volume_proxy_log"] = np.log1p(out["mass_volume_proxy_raw"].fillna(0.0))
    out["sigma_mass_volume"] = normalize_series_01(out["mass_volume_proxy_log"])

    # ----- Total structural deviation -----
    weight_sum = w_mass_volume + w_local_burst + w_observer_gap
    if np.isclose(weight_sum, 0.0):
        raise ValueError("At least one structural weight must be non-zero.")

    out["sigma_total"] = (
        w_mass_volume * out["sigma_mass_volume"]
        + w_local_burst * out["sigma_local_burst"]
        + w_observer_gap * out["sigma_observer_gap"]
    ) / weight_sum

    out["Dw_trial_mass_volume_observer_gap"] = d_bg - lambda_sigma * out["sigma_total"]

    # ----- Delay driver -----
    # Still a comparison-layer proxy, but now includes mass-volume structure as requested.
    out["delay_driver"] = (
        w_mass_volume * out["sigma_mass_volume"]
        + w_local_burst * out["sigma_local_burst"]
        + w_observer_gap * out["sigma_observer_gap"]
    ) / weight_sum

    dmin = out["delay_time_days"].min()
    dmax = out["delay_time_days"].max()
    out["delay_topological_proxy_days"] = dmin + out["delay_driver"] * (dmax - dmin)

    # Visual placement
    out["angle_deg"] = out["location_label"].map(ANGLE_DEG_MAP)
    out["angle_rad"] = np.deg2rad(out["angle_deg"])

    return out


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def _prep_plot(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    p = to_numeric(df, cols)
    return p.copy()


def save_circular_layout(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    sc = ax.scatter(
        df["angle_rad"],
        df["theta_arcsec"],
        c=df["Dw_trial_mass_volume_observer_gap"],
        s=95,
    )
    for _, row in df.iterrows():
        ax.text(
            row["angle_rad"],
            row["theta_arcsec"] + 0.18,
            f"{row['echo_patch']}\nDw={row['Dw_trial_mass_volume_observer_gap']:.2f}",
            ha="center",
            va="center",
            fontsize=9,
        )

    for _, patches in FAMILY_MAP.items():
        sub = df[df["echo_patch"].isin(patches)].sort_values("delay_time_days")
        ax.plot(sub["angle_rad"], sub["theta_arcsec"], linewidth=1.2)

    plt.colorbar(sc, pad=0.1, label="Dw_trial_mass_volume_observer_gap")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_family_delay_theta_comparison(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for family_name, patches in FAMILY_MAP.items():
        sub = df[df["echo_patch"].isin(patches)].sort_values("delay_time_days")
        ax.plot(sub["delay_time_days"], sub["theta_arcsec"], marker="o", label=f"{family_name} standard")
        ax.plot(
            sub["delay_topological_proxy_days"],
            sub["theta_arcsec"],
            marker="o",
            linestyle="--",
            label=f"{family_name} topological",
        )
    for _, row in df.iterrows():
        ax.text(row["delay_time_days"], row["theta_arcsec"], f" {row['echo_patch']}", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Delay time (days)")
    ax.set_ylabel("Angular distance θ (arcsec)")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_components_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    ax.plot(x, df["sigma_mass_volume"], marker="o", label="sigma_mass_volume")
    ax.plot(x, df["sigma_local_burst"], marker="o", label="sigma_local_burst")
    ax.plot(x, df["sigma_observer_gap"], marker="o", label="sigma_observer_gap")
    ax.plot(x, df["sigma_total"], marker="o", label="sigma_total")
    ax.set_xticks(x)
    ax.set_xticklabels(df["echo_patch"])
    ax.set_title(title)
    ax.set_xlabel("Echo patch")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_delay_driver_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    ax.plot(x, df["delay_norm"], marker="o", label="standard_delay_norm")
    ax.plot(x, df["delay_driver"], marker="o", label="delay_driver")
    ax.set_xticks(x)
    ax.set_xticklabels(df["echo_patch"])
    ax.set_title(title)
    ax.set_xlabel("Echo patch")
    ax.set_ylabel("Normalized value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ---------------------------------------------------------------------
# Summary / README
# ---------------------------------------------------------------------
def write_readme(out_dir: Path) -> None:
    text = """README_results.txt
==================

This results folder was generated by:
    src/echo of light/T Pyxidis/topological/t_pyx_mass_volume_observer_gap_pipeline.py

Purpose
-------
This is a mass-volume + observer-gap structural proxy experiment for T Pyxidis.

Inputs used
-----------
1. patch geometry input:
   data/derived/Echo of Light/input/T Pyxidis/t_pyx_standard_echo_patches_original_paper.csv

Important interpretation note
-----------------------------
- mass_proxy_raw is a heuristic local mass-context proxy derived from early disturbed patches
- echo_volume_proxy_raw is derived from theta^3 and is only a geometric patch-volume proxy
- sigma_mass_volume is a structural density-style proxy, not a direct ejecta-density measurement
- Dw_trial_mass_volume_observer_gap is a trial effective dimension index, not a strict derivation of D_w

This pipeline is intended only as a consistency-oriented test layer.
"""
    (out_dir / "README_results.txt").write_text(text, encoding="utf-8")


def build_summary(
    patch_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    input_csv: Path,
    out_dir: Path,
    params: Dict[str, float],
) -> Dict[str, object]:
    return {
        "pipeline_name": "T Pyxidis Mass-Volume + Observer-Gap Pipeline",
        "created_at": datetime.now().isoformat(),
        "input_csv": str(input_csv),
        "results_dir": str(out_dir),
        "row_counts": {
            "patch_input_rows": int(len(patch_df)),
            "trial_rows": int(len(trial_df)),
        },
        "parameters": params,
        "notes": [
            "This pipeline keeps the T Pyx patch-based interface while borrowing the V838Mon-style mass-volume proxy logic.",
            "mass_proxy_raw is a heuristic local mass-context term, not a direct measured ejecta mass.",
            "echo_volume_proxy_raw is a patch geometric proxy built from theta^3.",
            "delay_driver remains a comparison-layer proxy, but now includes mass-volume structure.",
        ],
        "generated_files": [
            "t_pyx_mass_volume_observer_gap_patch_comparison.csv",
            "plots/t_pyx_mass_volume_observer_gap_circular_layout.png",
            "plots/t_pyx_mass_volume_observer_gap_delay_theta_comparison.png",
            "plots/t_pyx_mass_volume_observer_gap_components.png",
            "plots/t_pyx_mass_volume_observer_gap_delay_driver.png",
        ],
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="T Pyxidis mass-volume + observer-gap pipeline")
    parser.add_argument("--patch-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))

    parser.add_argument("--d-bg", type=float, default=3.0)
    parser.add_argument("--lambda-sigma", type=float, default=0.8)
    parser.add_argument("--radius-power", type=float, default=3.0)

    parser.add_argument("--w-mass-volume", type=float, default=0.35)
    parser.add_argument("--w-local-burst", type=float, default=0.35)
    parser.add_argument("--w-observer-gap", type=float, default=0.30)

    parser.add_argument("--local-strength", type=float, default=1.0)

    args = parser.parse_args()

    patch_csv = ensure_file(Path(args.patch_csv))
    output_root = Path(args.output_root)

    patch_df = load_csv(patch_csv)
    validate_columns(
        patch_df,
        ["echo_patch", "location_label", "delay_time_days", "theta_arcsec", "z_1e17_cm"],
        "patch CSV",
    )

    trial_df = build_patch_trial_table(
        patch_df,
        d_bg=args.d_bg,
        lambda_sigma=args.lambda_sigma,
        radius_power=args.radius_power,
        w_mass_volume=args.w_mass_volume,
        w_local_burst=args.w_local_burst,
        w_observer_gap=args.w_observer_gap,
        local_strength=args.local_strength,
    )

    out_dir = make_output_dir(output_root)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save tables
    write_csv(trial_df, out_dir / "t_pyx_mass_volume_observer_gap_patch_comparison.csv")

    # Plots
    save_circular_layout(
        trial_df,
        plots_dir / "t_pyx_mass_volume_observer_gap_circular_layout.png",
        "T Pyxidis: circular patch layout (mass-volume + observer-gap model)",
    )
    save_family_delay_theta_comparison(
        trial_df,
        plots_dir / "t_pyx_mass_volume_observer_gap_delay_theta_comparison.png",
        "T Pyxidis: delay vs angular distance, standard vs topological (mass-volume + observer-gap)",
    )
    save_components_plot(
        trial_df,
        plots_dir / "t_pyx_mass_volume_observer_gap_components.png",
        "T Pyxidis: mass-volume vs local-burst vs observer-gap structural terms",
    )
    save_delay_driver_plot(
        trial_df,
        plots_dir / "t_pyx_mass_volume_observer_gap_delay_driver.png",
        "T Pyxidis: standard delay norm vs topological delay driver (mass-volume + observer-gap)",
    )

    summary = build_summary(
        patch_df=patch_df,
        trial_df=trial_df,
        input_csv=patch_csv,
        out_dir=out_dir,
        params={
            "d_bg": args.d_bg,
            "lambda_sigma": args.lambda_sigma,
            "radius_power": args.radius_power,
            "w_mass_volume": args.w_mass_volume,
            "w_local_burst": args.w_local_burst,
            "w_observer_gap": args.w_observer_gap,
            "local_strength": args.local_strength,
        },
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_readme(out_dir)
    print(f"[DONE] Results written to: {out_dir}")


if __name__ == "__main__":
    main()
