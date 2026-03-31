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
DEFAULT_OUTPUT_ROOT = Path(
    r"results/Echo of Light/output/T Pyxidis/topological/echo_medium_centered_integrated"
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

FAMILY_COLORS = {
    "North family": "tab:blue",
    "South family": "tab:orange",
    "West family": "tab:green",
}

WD_MASS_SOL = 0.70
DONOR_MASS_SOL = 0.14
DISPLAY_SEPARATION_ARCSEC = 0.07
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


def arcsec_to_pc(theta_arcsec: float, distance_kpc: float) -> float:
    return distance_kpc * 1000.0 * theta_arcsec / 206265.0


def arcsec_to_ly(theta_arcsec: float, distance_kpc: float) -> float:
    return arcsec_to_pc(theta_arcsec, distance_kpc) * 3.26156


def arcsec_to_au(theta_arcsec: float, distance_kpc: float) -> float:
    return arcsec_to_pc(theta_arcsec, distance_kpc) * 206265.0


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


def prepare_patch_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = to_numeric(out, ["delay_time_days", "theta_arcsec", "z_1e17_cm"])
    out["angle_deg"] = out["location_label"].map(ANGLE_DEG_MAP)
    out["angle_rad"] = np.deg2rad(out["angle_deg"])
    out["theta_au_at_4p8kpc"] = out["theta_arcsec"].apply(lambda x: arcsec_to_au(float(x), DISTANCE_KPC))
    out["theta_ly_at_4p8kpc"] = out["theta_arcsec"].apply(lambda x: arcsec_to_ly(float(x), DISTANCE_KPC))
    return out


def build_topological_table(
    df: pd.DataFrame,
    d_bg: float,
    lambda_sigma: float,
    radius_power: float,
    w_medium_structure: float,
    w_observer_gap: float,
    w_gradient_driver: float,
    w_gap_driver: float,
    w_delay_structure: float,
) -> pd.DataFrame:
    out = prepare_patch_table(df)

    out["theta_norm"] = normalize_series_01(out["theta_arcsec"])
    out["delay_norm"] = normalize_series_01(out["delay_time_days"])
    out["z_abs_1e17_cm"] = out["z_1e17_cm"].abs()
    out["z_abs_norm"] = normalize_series_01(out["z_abs_1e17_cm"])

    out["echo_medium_mass_proxy_raw"] = (
        1.0
        + 2.0 * out["delay_norm"]
        + 1.0 * out["z_abs_norm"]
        + 0.5 * out["theta_norm"]
    )
    out["echo_medium_volume_proxy_raw"] = np.power(np.maximum(out["theta_arcsec"], 0.0), radius_power)
    out["echo_medium_volume_proxy_safe"] = out["echo_medium_volume_proxy_raw"].replace(0, np.nan)
    out["echo_medium_density_proxy_raw"] = (
        out["echo_medium_mass_proxy_raw"] / out["echo_medium_volume_proxy_safe"]
    )
    out["echo_medium_density_proxy_raw"] = out["echo_medium_density_proxy_raw"].replace([np.inf, -np.inf], np.nan)
    out["echo_medium_density_proxy_log"] = np.log1p(out["echo_medium_density_proxy_raw"].fillna(0.0))

    out["sigma_echo_medium_structure"] = normalize_series_01(out["echo_medium_density_proxy_log"])
    out["D_echo_medium"] = d_bg - lambda_sigma * out["sigma_echo_medium_structure"]

    out["gap_term"] = (d_bg - out["D_echo_medium"]).abs()
    out["gap_term_norm"] = normalize_series_01(out["gap_term"])

    struct_weight_sum = w_medium_structure + w_observer_gap
    out["sigma_total"] = (
        w_medium_structure * out["sigma_echo_medium_structure"]
        + w_observer_gap * out["gap_term_norm"]
    ) / struct_weight_sum
    out["Dw_trial_echo_medium"] = d_bg - lambda_sigma * out["sigma_total"]

    out = add_family_gradient(out, "D_echo_medium", "grad_D_echo_medium")
    out["medium_gradient_term"] = normalize_series_01(out["grad_D_echo_medium"].abs())

    driver_weight_sum = w_gradient_driver + w_gap_driver + w_delay_structure
    out["distortion_driver"] = normalize_series_01(
        (
            w_gradient_driver * out["medium_gradient_term"]
            + w_gap_driver * out["gap_term_norm"]
            + w_delay_structure * out["sigma_echo_medium_structure"]
        ) / driver_weight_sum
    )

    dmin = out["delay_time_days"].min()
    dmax = out["delay_time_days"].max()
    out["delay_topological_proxy_days"] = dmin + out["distortion_driver"] * (dmax - dmin)
    return out


def binary_overlay_positions():
    half_sep = DISPLAY_SEPARATION_ARCSEC / 2.0
    return {
        "wd_x": -half_sep,
        "donor_x": half_sep,
        "wd_y": 0.0,
        "donor_y": 0.0,
        "radius": 0.045,
    }


def add_binary_name_legend(fig):
    text = (
        "Central binary legend\n"
        "Blue circle: White dwarf\n"
        "Orange circle: Donor companion\n"
        "Center marker: COM\n"
        "Compressed, partly overlapping display\n"
        "centered on the true graph center\n"
        "(illustrative; not to physical scale)"
    )
    fig.text(
        0.975, 0.74, text,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95)
    )


def add_arcsec_distance_legend(fig):
    lines = []
    for r in [1, 2, 3, 4, 5]:
        au = arcsec_to_au(r, DISTANCE_KPC)
        ly = arcsec_to_ly(r, DISTANCE_KPC)
        lines.append(f'{r}" = {au:,.0f} AU = {ly:.3f} ly')
    text = "Angular scale\n" + "\n".join(lines)
    fig.text(
        0.975, 0.12, text,
        ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95)
    )


def save_topological_circular_layout(df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    sc = ax.scatter(df["angle_rad"], df["theta_arcsec"], c=df["Dw_trial_echo_medium"], s=95)
    for _, row in df.iterrows():
        ax.text(
            row["angle_rad"],
            row["theta_arcsec"] + 0.18,
            f"{row['echo_patch']}\nDw={row['Dw_trial_echo_medium']:.2f}",
            ha="center",
            va="center",
            fontsize=9,
        )

    for family_name, patches in FAMILY_MAP.items():
        sub = df[df["echo_patch"].isin(patches)].copy().sort_values("delay_time_days")
        if len(sub) >= 2:
            color = FAMILY_COLORS[family_name]
            ax.annotate(
                "",
                xy=(float(sub.iloc[1]["angle_rad"]), float(sub.iloc[1]["theta_arcsec"])),
                xytext=(float(sub.iloc[0]["angle_rad"]), float(sub.iloc[0]["theta_arcsec"])),
                arrowprops=dict(arrowstyle="->", lw=1.4, color=color),
            )

    ax.set_rticks([1, 2, 3, 4, 5])
    ax.set_yticklabels([f'{r}"' for r in [1, 2, 3, 4, 5]])

    inset = ax.inset_axes([0.448, 0.448, 0.104, 0.104], transform=ax.transAxes)
    inset.set_aspect("equal")
    pos = binary_overlay_positions()

    from matplotlib.patches import Circle
    wd_circle = Circle((pos["wd_x"], pos["wd_y"]), pos["radius"], fill=True, facecolor="tab:blue", edgecolor="tab:blue", linewidth=1.2, alpha=0.95)
    donor_circle = Circle((pos["donor_x"], pos["donor_y"]), pos["radius"], fill=True, facecolor="tab:orange", edgecolor="tab:orange", linewidth=1.2, alpha=0.95)
    inset.add_patch(wd_circle)
    inset.add_patch(donor_circle)
    inset.scatter([0.0], [0.0], s=70, marker="+", color="black", zorder=6)
    inset.set_xlim(-0.15, 0.15)
    inset.set_ylim(-0.10, 0.10)
    inset.axis("off")

    fig.subplots_adjust(right=0.78)
    add_binary_name_legend(fig)
    add_arcsec_distance_legend(fig)

    plt.colorbar(sc, pad=0.08, label="Dw_trial_echo_medium")
    ax.set_title("T Pyxidis topological: integrated circular echo-patch layout")
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_family_delay_theta_comparison(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for family_name, patches in FAMILY_MAP.items():
        color = FAMILY_COLORS[family_name]
        sub = df[df["echo_patch"].isin(patches)].copy().sort_values("delay_time_days")

        ax.scatter(sub["delay_time_days"], sub["theta_arcsec"], s=75, label=f"{family_name} standard", color=color)
        ax.scatter(sub["delay_topological_proxy_days"], sub["theta_arcsec"], s=75, marker="x", label=f"{family_name} topological", color=color)

        if len(sub) >= 2:
            ax.annotate(
                "",
                xy=(float(sub.iloc[1]["delay_time_days"]), float(sub.iloc[1]["theta_arcsec"])),
                xytext=(float(sub.iloc[0]["delay_time_days"]), float(sub.iloc[0]["theta_arcsec"])),
                arrowprops=dict(arrowstyle="->", lw=1.2, color=color),
            )
            ax.annotate(
                "",
                xy=(float(sub.iloc[1]["delay_topological_proxy_days"]), float(sub.iloc[1]["theta_arcsec"])),
                xytext=(float(sub.iloc[0]["delay_topological_proxy_days"]), float(sub.iloc[0]["theta_arcsec"])),
                arrowprops=dict(arrowstyle="->", lw=1.2, color=color, linestyle="--"),
            )

        for _, row in sub.iterrows():
            ax.text(row["delay_time_days"], row["theta_arcsec"], f" {row['echo_patch']}", fontsize=9)

    ax.set_title("T Pyxidis topological: delay vs angular distance by patch family")
    ax.set_xlabel("Delay time (days)")
    ax.set_ylabel("Angular distance θ (arcsec)")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_component_bars(df: pd.DataFrame, out_path: Path) -> None:
    order = ["N1", "S1", "N2", "S2", "NW", "W"]
    p = df.set_index("echo_patch").loc[order].reset_index()
    x = np.arange(len(p))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, p["sigma_echo_medium_structure"], width=width, label="sigma_echo_medium_structure")
    ax.bar(x, p["gap_term_norm"], width=width, label="gap_term_norm")
    ax.bar(x + width, p["medium_gradient_term"], width=width, label="medium_gradient_term")
    ax.set_xticks(x)
    ax.set_xticklabels(p["echo_patch"])
    ax.set_title("T Pyxidis topological: structural components")
    ax.set_xlabel("Echo patch")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_driver_points(df: pd.DataFrame, out_path: Path) -> None:
    order = ["N1", "S1", "N2", "S2", "NW", "W"]
    p = df.set_index("echo_patch").loc[order].reset_index()
    x = np.arange(len(p))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, p["delay_norm"], s=75, label="standard_delay_norm")
    ax.scatter(x, p["distortion_driver"], s=75, marker="x", label="distortion_driver")
    for i, row in p.iterrows():
        ax.text(i, row["delay_norm"], f" {row['echo_patch']}", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(p["echo_patch"])
    ax.set_title("T Pyxidis topological: standard delay norm vs distortion driver")
    ax.set_xlabel("Echo patch")
    ax.set_ylabel("Normalized value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_summary(patch_csv: Path, out_dir: Path, df: pd.DataFrame, params: dict) -> dict:
    ring_scale = []
    for r in [1, 2, 3, 4, 5]:
        ring_scale.append({
            "arcsec": r,
            "au_at_4p8kpc": round(arcsec_to_au(r, DISTANCE_KPC), 2),
            "ly_at_4p8kpc": round(arcsec_to_ly(r, DISTANCE_KPC), 6),
        })

    return {
        "pipeline_name": "T Pyxidis Topological Echo-Medium-Centered Integrated Pipeline",
        "created_at": datetime.now().isoformat(),
        "input_patch_csv": str(patch_csv),
        "results_dir": str(out_dir),
        "row_count": int(len(df)),
        "system_context": {
            "classification": "recurrent nova close binary",
            "primary": "white dwarf",
            "secondary": "donor companion star",
            "distance_kpc_for_ring_conversion": DISTANCE_KPC,
        },
        "ring_scale_reference": ring_scale,
        "model_context": {
            "centering": "echo-medium-centered",
            "removed_terms": [
                "sigma_local_burst",
                "driver_grad_burst_term",
                "geometry skeleton term"
            ],
            "active_terms": [
                "sigma_echo_medium_structure",
                "gap_term_norm",
                "medium_gradient_term",
                "distortion_driver"
            ],
        },
        "parameters": params,
        "notes": [
            "Topological circular layout now follows the same display convention as the integrated standard pipeline.",
            "The central binary overlay is illustrative and not to physical scale.",
            "COM marker is placed at the exact graph center.",
            "The structural task now centers on refining echo_medium proxies rather than restoring burst-centered terms."
        ],
    }


def write_readme(out_dir: Path) -> None:
    text = """README_results.txt
==================

This results folder was generated by:
    src/echo of light/T Pyxidis/topological/t_pyx_topological_integrated_pipeline.py

What this integrated topological pipeline does
----------------------------------------------
- reads the standard patch table
- builds an echo-medium-centered topological interpretation
- creates a standard-matched circular plot
- creates a family delay-theta comparison plot
- creates component bars and driver comparison plot
- writes integrated CSV outputs
- writes one summary.json

Model policy
------------
- burst-centered terms remain removed
- geometry skeleton term remains removed
- distortion_driver uses echo-medium structure + observer gap + medium gradient
"""
    (out_dir / "README_results.txt").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="T Pyxidis topological integrated pipeline")
    parser.add_argument("--patch-csv", default=str(DEFAULT_PATCH_CSV))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--d-bg", type=float, default=3.0)
    parser.add_argument("--lambda-sigma", type=float, default=0.8)
    parser.add_argument("--radius-power", type=float, default=3.0)
    parser.add_argument("--w-medium-structure", type=float, default=0.60)
    parser.add_argument("--w-observer-gap", type=float, default=0.40)
    parser.add_argument("--w-gradient-driver", type=float, default=0.40)
    parser.add_argument("--w-gap-driver", type=float, default=0.35)
    parser.add_argument("--w-delay-structure", type=float, default=0.25)
    args = parser.parse_args()

    patch_csv = ensure_file(Path(args.patch_csv))
    patch_df = load_csv(patch_csv)

    out_dir = make_output_dir(Path(args.output_root))
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    topo_df = build_topological_table(
        patch_df,
        d_bg=args.d_bg,
        lambda_sigma=args.lambda_sigma,
        radius_power=args.radius_power,
        w_medium_structure=args.w_medium_structure,
        w_observer_gap=args.w_observer_gap,
        w_gradient_driver=args.w_gradient_driver,
        w_gap_driver=args.w_gap_driver,
        w_delay_structure=args.w_delay_structure,
    )

    write_csv(topo_df, tables_dir / "t_pyx_topological_integrated_patch_table.csv")

    save_topological_circular_layout(topo_df, plots_dir / "t_pyx_topological_integrated_circular_layout.png")
    save_family_delay_theta_comparison(topo_df, plots_dir / "t_pyx_topological_integrated_delay_theta_family.png")
    save_component_bars(topo_df, plots_dir / "t_pyx_topological_integrated_components.png")
    save_driver_points(topo_df, plots_dir / "t_pyx_topological_integrated_driver_points.png")

    params = {
        "d_bg": args.d_bg,
        "lambda_sigma": args.lambda_sigma,
        "radius_power": args.radius_power,
        "w_medium_structure": args.w_medium_structure,
        "w_observer_gap": args.w_observer_gap,
        "w_gradient_driver": args.w_gradient_driver,
        "w_gap_driver": args.w_gap_driver,
        "w_delay_structure": args.w_delay_structure,
    }
    summary = build_summary(patch_csv, out_dir, topo_df, params)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_readme(out_dir)
    print(f"[DONE] Results written to: {out_dir}")


if __name__ == "__main__":
    main()
