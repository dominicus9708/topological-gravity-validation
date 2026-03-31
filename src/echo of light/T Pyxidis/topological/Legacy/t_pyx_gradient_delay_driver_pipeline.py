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

DEFAULT_INPUT_CSV = Path(
    r"data/derived/Echo of Light/input/T Pyxidis/t_pyx_standard_echo_patches_original_paper.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"results/Echo of Light/output/T Pyxidis/topological/gradient_delay_driver"
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


def add_family_gradient(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    out = df.copy()
    out[out_col] = np.nan
    for _, patches in FAMILY_MAP.items():
        sub = out[out["echo_patch"].isin(patches)].copy().sort_values("delay_time_days")
        if len(sub) == 0:
            continue
        vals = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(dtype=float)
        grad = np.zeros(len(sub), dtype=float)
        if len(sub) == 1:
            grad[0] = 0.0
        else:
            grad[0] = vals[1] - vals[0]
            for i in range(1, len(sub)):
                grad[i] = vals[i] - vals[i - 1]
        out.loc[sub.index, out_col] = grad
    out[out_col] = out[out_col].fillna(0.0)
    return out


def build_patch_trial_table(
    df: pd.DataFrame,
    d_bg: float,
    lambda_sigma: float,
    radius_power: float,
    w_mass_volume: float,
    w_local_burst: float,
    w_observer_gap: float,
    local_strength: float,
    driver_geom_weight: float,
    driver_grad_mv_weight: float,
    driver_grad_burst_weight: float,
    driver_gap_weight: float,
) -> pd.DataFrame:
    out = df.copy()
    out = to_numeric(out, ["delay_time_days", "theta_arcsec", "z_1e17_cm"])

    # ----- Structural observables -----
    out["theta_norm"] = normalize_series_01(out["theta_arcsec"])
    out["delay_norm"] = normalize_series_01(out["delay_time_days"])
    out["z_abs_1e17_cm"] = out["z_1e17_cm"].abs()
    out["z_abs_norm"] = normalize_series_01(out["z_abs_1e17_cm"])

    # ----- Structural interpretation layer -----
    out["sigma_local_burst"] = local_strength * (1.0 - out["delay_norm"])
    out["sigma_observer_gap"] = 0.50 * out["z_abs_norm"] + 0.50 * (out["delay_norm"] - out["theta_norm"]).abs()

    out["mass_proxy_raw"] = 1.0 + 4.0 * out["sigma_local_burst"]
    out["echo_volume_proxy_raw"] = np.power(np.maximum(out["theta_arcsec"], 0.0), radius_power)
    out["echo_volume_proxy_safe"] = out["echo_volume_proxy_raw"].replace(0, np.nan)
    out["mass_volume_proxy_raw"] = out["mass_proxy_raw"] / out["echo_volume_proxy_safe"]
    out["mass_volume_proxy_raw"] = out["mass_volume_proxy_raw"].replace([np.inf, -np.inf], np.nan)
    out["mass_volume_proxy_log"] = np.log1p(out["mass_volume_proxy_raw"].fillna(0.0))
    out["sigma_mass_volume"] = normalize_series_01(out["mass_volume_proxy_log"])

    weight_sum = w_mass_volume + w_local_burst + w_observer_gap
    if np.isclose(weight_sum, 0.0):
        raise ValueError("At least one structural weight must be non-zero.")

    out["sigma_total"] = (
        w_mass_volume * out["sigma_mass_volume"]
        + w_local_burst * out["sigma_local_burst"]
        + w_observer_gap * out["sigma_observer_gap"]
    ) / weight_sum
    out["Dw_trial_gradient_driver"] = d_bg - lambda_sigma * out["sigma_total"]

    # ----- Delay-driver layer (separated) -----
    # geometry propagation skeleton
    out["driver_geom"] = 0.55 * out["theta_norm"] + 0.45 * out["z_abs_norm"]

    # family-internal gradient terms
    out = add_family_gradient(out, "sigma_mass_volume", "grad_sigma_mass_volume")
    out = add_family_gradient(out, "sigma_local_burst", "grad_sigma_local_burst")

    out["driver_grad_mass_volume"] = normalize_series_01(out["grad_sigma_mass_volume"].abs())
    out["driver_grad_local_burst"] = normalize_series_01((-out["grad_sigma_local_burst"]).clip(lower=0.0))

    # observer-gap correction
    out["driver_gap"] = out["sigma_observer_gap"]

    driver_w_sum = driver_geom_weight + driver_grad_mv_weight + driver_grad_burst_weight + driver_gap_weight
    if np.isclose(driver_w_sum, 0.0):
        raise ValueError("At least one delay-driver weight must be non-zero.")

    out["delay_driver"] = (
        driver_geom_weight * out["driver_geom"]
        + driver_grad_mv_weight * out["driver_grad_mass_volume"]
        + driver_grad_burst_weight * out["driver_grad_local_burst"]
        + driver_gap_weight * out["driver_gap"]
    ) / driver_w_sum

    dmin = out["delay_time_days"].min()
    dmax = out["delay_time_days"].max()
    out["delay_topological_proxy_days"] = dmin + out["delay_driver"] * (dmax - dmin)

    out["angle_deg"] = out["location_label"].map(ANGLE_DEG_MAP)
    out["angle_rad"] = np.deg2rad(out["angle_deg"])
    return out


def draw_arrow_on_polar(ax, theta1, r1, theta2, r2):
    ax.annotate("", xy=(theta2, r2), xytext=(theta1, r1), arrowprops=dict(arrowstyle="->", lw=1.4))


def save_circular_layout(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    sc = ax.scatter(df["angle_rad"], df["theta_arcsec"], c=df["Dw_trial_gradient_driver"], s=95)
    for _, row in df.iterrows():
        ax.text(
            row["angle_rad"],
            row["theta_arcsec"] + 0.18,
            f"{row['echo_patch']}\nDw={row['Dw_trial_gradient_driver']:.2f}",
            ha="center",
            va="center",
            fontsize=9,
        )

    # arrows only for genuine within-family progression
    for _, patches in FAMILY_MAP.items():
        sub = df[df["echo_patch"].isin(patches)].copy().sort_values("delay_time_days")
        if len(sub) >= 2:
            draw_arrow_on_polar(
                ax,
                float(sub.iloc[0]["angle_rad"]),
                float(sub.iloc[0]["theta_arcsec"]),
                float(sub.iloc[1]["angle_rad"]),
                float(sub.iloc[1]["theta_arcsec"]),
            )

    plt.colorbar(sc, pad=0.1, label="Dw_trial_gradient_driver")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_family_delay_theta_points(df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for family_name, patches in FAMILY_MAP.items():
        sub = df[df["echo_patch"].isin(patches)].copy().sort_values("delay_time_days")

        ax.scatter(sub["delay_time_days"], sub["theta_arcsec"], s=70, label=f"{family_name} standard")
        ax.scatter(sub["delay_topological_proxy_days"], sub["theta_arcsec"], s=70, marker="x", label=f"{family_name} topological")

        if len(sub) >= 2:
            # standard arrow
            ax.annotate(
                "",
                xy=(sub.iloc[1]["delay_time_days"], sub.iloc[1]["theta_arcsec"]),
                xytext=(sub.iloc[0]["delay_time_days"], sub.iloc[0]["theta_arcsec"]),
                arrowprops=dict(arrowstyle="->", lw=1.2),
            )
            # topological arrow
            ax.annotate(
                "",
                xy=(sub.iloc[1]["delay_topological_proxy_days"], sub.iloc[1]["theta_arcsec"]),
                xytext=(sub.iloc[0]["delay_topological_proxy_days"], sub.iloc[0]["theta_arcsec"]),
                arrowprops=dict(arrowstyle="->", lw=1.2, linestyle="--"),
            )

        for _, row in sub.iterrows():
            ax.text(row["delay_time_days"], row["theta_arcsec"], f" {row['echo_patch']}", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("Delay time (days)")
    ax.set_ylabel("Angular distance θ (arcsec)")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_components_bars(df: pd.DataFrame, out_path: Path, title: str) -> None:
    order = ["N1", "S1", "N2", "S2", "NW", "W"]
    p = df.set_index("echo_patch").loc[order].reset_index()
    x = np.arange(len(p))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - 1.5 * width, p["sigma_mass_volume"], width=width, label="sigma_mass_volume")
    ax.bar(x - 0.5 * width, p["sigma_local_burst"], width=width, label="sigma_local_burst")
    ax.bar(x + 0.5 * width, p["sigma_observer_gap"], width=width, label="sigma_observer_gap")
    ax.bar(x + 1.5 * width, p["sigma_total"], width=width, label="sigma_total")
    ax.set_xticks(x)
    ax.set_xticklabels(p["echo_patch"])
    ax.set_title(title)
    ax.set_xlabel("Echo patch")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_delay_driver_points(df: pd.DataFrame, out_path: Path, title: str) -> None:
    order = ["N1", "S1", "N2", "S2", "NW", "W"]
    p = df.set_index("echo_patch").loc[order].reset_index()
    x = np.arange(len(p))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, p["delay_norm"], s=75, label="standard_delay_norm")
    ax.scatter(x, p["delay_driver"], s=75, marker="x", label="delay_driver")

    for i, row in p.iterrows():
        ax.text(i, row["delay_norm"], f" {row['echo_patch']}", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(p["echo_patch"])
    ax.set_title(title)
    ax.set_xlabel("Echo patch")
    ax.set_ylabel("Normalized value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_summary(patch_df: pd.DataFrame, trial_df: pd.DataFrame, input_csv: Path, out_dir: Path, params: dict) -> dict:
    return {
        "pipeline_name": "T Pyxidis Gradient Delay Driver Pipeline",
        "created_at": datetime.now().isoformat(),
        "input_csv": str(input_csv),
        "results_dir": str(out_dir),
        "row_counts": {
            "patch_input_rows": int(len(patch_df)),
            "trial_rows": int(len(trial_df)),
        },
        "parameters": params,
        "notes": [
            "delay_driver is separated from sigma_total.",
            "delay_driver uses geometry skeleton + family gradients + observer-gap correction.",
            "lines/arrows are used only for genuine change indicators within each family.",
            "patch-comparison component charts are shown as bars rather than continuous lines.",
        ],
    }


def write_readme(out_dir: Path) -> None:
    text = """README_results.txt
==================

This results folder was generated by:
    src/echo of light/T Pyxidis/topological/t_pyx_gradient_delay_driver_pipeline.py

Purpose
-------
This version revises the delay-driver construction to better reflect:
- geometry propagation skeleton
- family-internal gradient / recovery terms
- observer-gap correction

Graph policy
------------
- lines / arrows only for genuine change indicators
- circular layout uses arrows to indicate within-family direction
- comparison profiles use points or bars rather than continuous lines

Important interpretation note
-----------------------------
- sigma_total remains the structural interpretation layer
- delay_driver is now separated from sigma_total
- delay_topological_proxy_days is still a comparison-layer proxy, not a strict light-echo derivation
"""
    (out_dir / "README_results.txt").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="T Pyxidis gradient delay driver pipeline")
    parser.add_argument("--patch-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--d-bg", type=float, default=3.0)
    parser.add_argument("--lambda-sigma", type=float, default=0.8)
    parser.add_argument("--radius-power", type=float, default=3.0)
    parser.add_argument("--w-mass-volume", type=float, default=0.35)
    parser.add_argument("--w-local-burst", type=float, default=0.35)
    parser.add_argument("--w-observer-gap", type=float, default=0.30)
    parser.add_argument("--local-strength", type=float, default=1.0)
    parser.add_argument("--driver-geom-weight", type=float, default=0.40)
    parser.add_argument("--driver-grad-mv-weight", type=float, default=0.25)
    parser.add_argument("--driver-grad-burst-weight", type=float, default=0.20)
    parser.add_argument("--driver-gap-weight", type=float, default=0.15)
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
        driver_geom_weight=args.driver_geom_weight,
        driver_grad_mv_weight=args.driver_grad_mv_weight,
        driver_grad_burst_weight=args.driver_grad_burst_weight,
        driver_gap_weight=args.driver_gap_weight,
    )

    out_dir = make_output_dir(output_root)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    write_csv(trial_df, out_dir / "t_pyx_gradient_delay_driver_patch_comparison.csv")

    save_circular_layout(
        trial_df,
        plots_dir / "t_pyx_gradient_delay_driver_circular_layout.png",
        "T Pyxidis: circular patch layout (gradient delay-driver model)",
    )
    save_family_delay_theta_points(
        trial_df,
        plots_dir / "t_pyx_gradient_delay_driver_family_delay_theta.png",
        "T Pyxidis: family delay-theta comparison (gradient delay-driver)",
    )
    save_components_bars(
        trial_df,
        plots_dir / "t_pyx_gradient_delay_driver_components.png",
        "T Pyxidis: structural components (bars)",
    )
    save_delay_driver_points(
        trial_df,
        plots_dir / "t_pyx_gradient_delay_driver_profile.png",
        "T Pyxidis: standard delay norm vs gradient delay driver (points)",
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
            "driver_geom_weight": args.driver_geom_weight,
            "driver_grad_mv_weight": args.driver_grad_mv_weight,
            "driver_grad_burst_weight": args.driver_grad_burst_weight,
            "driver_gap_weight": args.driver_gap_weight,
        },
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_readme(out_dir)
    print(f"[DONE] Results written to: {out_dir}")


if __name__ == "__main__":
    main()