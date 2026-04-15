#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
Topological pipeline for common5 final input (v4)

Author: Kwon Dominicus

Placement
---------
src/Validation of Structural Contrast Baseline/topological/

Input
-----
data/derived/Validation of Structural Contrast Baseline/input/topological/wise_hii_common5/wise_hii_common5_topological_final_input.csv

Output
------
results/Validation of Structural Contrast Baseline/output/topological/YYYYMMDD_HHMMSS/

Notes
-----
- This is an execution pipeline, not a derived-processing script.
- Final input is treated as fixed official input.
- This version reads mass/proxy information directly from the common5 topological final input.
- Added:
  1) standard-style circular zone schematic in topological output
  2) Dw_obs zone bar plot
  3) previous alpha_obs and L(<r) plots retained
  4) mass-volume bridge plot retained
"""

from __future__ import annotations

import math
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\mincu\Desktop\topological_gravity_project")

INPUT_FILE = (
    Path("data")
    / "derived"
    / "Validation of Structural Contrast Baseline"
    / "input"
    / "topological"
    / "wise_hii_common5"
    / "wise_hii_common5_topological_final_input.csv"
)

PROFILE_DIR = (
    Path("data")
    / "derived"
    / "Validation of Structural Contrast Baseline"
    / "input"
    / "wise_hii_catalog"
    / "radial_profiles"
)

RESULTS_BASE = (
    Path("results")
    / "Validation of Structural Contrast Baseline"
    / "output"
    / "topological"
)

EPS = 1e-9


@dataclass
class TargetZones:
    wise_name: str
    radius_arcsec: float
    inner_inner_arcsec: float
    inner_outer_arcsec: float
    shell_inner_arcsec: float
    shell_outer_arcsec: float
    bg_inner_arcsec: float
    bg_outer_arcsec: float


def ensure_required_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: " + ", ".join(missing))


def sanitize_wise_name(name: str) -> str:
    text = str(name).strip()
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def create_timestamped_output_dir(project_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = project_root / RESULTS_BASE / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_final_input(project_root: Path) -> pd.DataFrame:
    path = project_root / INPUT_FILE
    if not path.exists():
        raise FileNotFoundError(f"Final input file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = [
        "wise_name",
        "priority_rank",
        "catalog_class",
        "ra",
        "dec",
        "radius_arcsec",
        "hii_region_name",
        "membership",
        "fits_downloadable",
    ]
    ensure_required_columns(df, required, "Topological final input")

    df["radius_arcsec_num"] = pd.to_numeric(df["radius_arcsec"], errors="coerce")
    if df["radius_arcsec_num"].isna().any():
        bad = df.loc[df["radius_arcsec_num"].isna(), "wise_name"].tolist()
        raise ValueError("Invalid radius_arcsec values for: " + ", ".join(map(str, bad)))

    df["priority_rank_num"] = pd.to_numeric(df["priority_rank"], errors="coerce")
    df["wise_name"] = df["wise_name"].astype(str).str.strip()

    if df["wise_name"].duplicated().any():
        dup = df.loc[df["wise_name"].duplicated(), "wise_name"].tolist()
        raise ValueError("Duplicate wise_name entries found: " + ", ".join(dup))

    return df.sort_values(
        by=["priority_rank_num", "radius_arcsec_num", "wise_name"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_standard_zones(df: pd.DataFrame) -> List[TargetZones]:
    zones: List[TargetZones] = []
    for _, row in df.iterrows():
        r = float(row["radius_arcsec_num"])
        zones.append(
            TargetZones(
                wise_name=str(row["wise_name"]),
                radius_arcsec=r,
                inner_inner_arcsec=0.0,
                inner_outer_arcsec=round(max(0.50 * r, 1.0), 3),
                shell_inner_arcsec=round(0.60 * r, 3),
                shell_outer_arcsec=round(1.00 * r, 3),
                bg_inner_arcsec=round(1.20 * r, 3),
                bg_outer_arcsec=round(1.80 * r, 3),
            )
        )
    return zones


def find_profile_file(project_root: Path, wise_name: str) -> Optional[Path]:
    base = project_root / PROFILE_DIR
    candidates = [
        base / f"{wise_name}_radial_profile.csv",
        base / f"{wise_name}.csv",
        base / f"{sanitize_wise_name(wise_name)}_radial_profile.csv",
        base / f"{sanitize_wise_name(wise_name)}.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_profile_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    radius_candidates = ["radius_arcsec", "r_arcsec", "r", "radius"]
    intensity_candidates = ["intensity", "value", "flux", "surface_brightness", "I"]

    radius_col = next((c for c in radius_candidates if c in df.columns), None)
    intensity_col = next((c for c in intensity_candidates if c in df.columns), None)

    if radius_col is None or intensity_col is None:
        raise ValueError(
            f"Profile CSV missing required columns. Found columns={list(df.columns)}. "
            "Need one radius column from [radius_arcsec, r_arcsec, r, radius] and "
            "one intensity column from [intensity, value, flux, surface_brightness, I]."
        )

    out = pd.DataFrame(
        {
            "radius_arcsec": pd.to_numeric(df[radius_col], errors="coerce"),
            "intensity": pd.to_numeric(df[intensity_col], errors="coerce"),
        }
    ).dropna()

    if out.empty:
        raise ValueError(f"Profile CSV has no usable rows: {path}")

    out = out.sort_values("radius_arcsec").reset_index(drop=True)
    out = out[out["radius_arcsec"] > 0].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"Profile CSV has no strictly positive radius rows: {path}")
    return out


def build_observational_structure_profile(profile: pd.DataFrame) -> pd.DataFrame:
    df = profile.copy()

    r = df["radius_arcsec"].to_numpy(dtype=float)
    I = df["intensity"].to_numpy(dtype=float)

    kernel = I * r
    cumulative = np.zeros_like(r, dtype=float)
    if len(r) > 0:
        cumulative[0] = max(kernel[0] * r[0], EPS)
        for i in range(1, len(r)):
            dr = max(r[i] - r[i - 1], EPS)
            cumulative[i] = cumulative[i - 1] + 0.5 * (kernel[i] + kernel[i - 1]) * dr

    cumulative = np.maximum(cumulative, EPS)

    log_r = np.log(np.maximum(r, EPS))
    log_L = np.log(cumulative)

    if len(df) >= 3:
        alpha_obs = np.gradient(log_L, log_r)
    elif len(df) == 2:
        slope = (log_L[1] - log_L[0]) / max(log_r[1] - log_r[0], EPS)
        alpha_obs = np.array([slope, slope], dtype=float)
    else:
        alpha_obs = np.array([np.nan], dtype=float)

    df["kernel_Ir"] = kernel
    df["L_cumulative"] = cumulative
    df["alpha_obs"] = alpha_obs
    return df


def weighted_zone_mean(profile_struct: pd.DataFrame, value_col: str, r_in: float, r_out: float) -> Optional[float]:
    subset = profile_struct[
        (profile_struct["radius_arcsec"] >= r_in) & (profile_struct["radius_arcsec"] <= r_out)
    ].copy()
    subset = subset[np.isfinite(subset[value_col])]
    if subset.empty:
        return None

    weights = subset["radius_arcsec"].clip(lower=1e-6)
    denom = float(weights.sum())
    if denom <= 0:
        return None
    return float((subset[value_col] * weights).sum() / denom)


def sphere_shell_volume(r_in: float, r_out: float) -> float:
    return (4.0 / 3.0) * math.pi * max(r_out**3 - r_in**3, 0.0)


def derive_mass_proxy(row: pd.Series) -> tuple[Optional[float], str]:
    direct_mass = pd.to_numeric(pd.Series([row.get("mass_value_msun", np.nan)]), errors="coerce").iloc[0]
    if pd.notna(direct_mass) and float(direct_mass) > 0:
        return float(direct_mass), "direct_mass_value_msun"

    # Robust handling:
    # - some rows may store true log10(N_Ly) values, typically around 47 to 51
    # - some rows may store linear N_Ly values directly, e.g. 1.7e48
    log_nly = pd.to_numeric(pd.Series([row.get("log_nly", np.nan)]), errors="coerce").iloc[0]
    if pd.notna(log_nly):
        val = float(log_nly)
        if 30.0 <= val <= 60.0:
            return float(10 ** (val - 49.0)), "radio_proxy_from_log_Nly_log10_input"
        if val > 1.0e6:
            return float(val / 1.0e49), "radio_proxy_from_log_Nly_linear_input"
        if val > 0:
            return float(val), "radio_proxy_from_log_Nly_small_positive_input"

    proxy_value = pd.to_numeric(pd.Series([row.get("proxy_value", np.nan)]), errors="coerce").iloc[0]
    proxy_kind = str(row.get("proxy_kind", "")).strip().lower()
    if pd.notna(proxy_value) and proxy_kind in {"radio_continuum", "log_nly"}:
        val = float(proxy_value)
        if proxy_kind == "log_nly" and 30.0 <= val <= 60.0:
            return float(10 ** (val - 49.0)), "proxy_value_from_log_nly_log10_input"
        if proxy_kind == "log_nly" and val > 1.0e6:
            return float(val / 1.0e49), "proxy_value_from_log_nly_linear_input"
        return float(val), f"proxy_value_from_{proxy_kind}"

    spectral_type = str(row.get("spectral_type", "")).strip()
    if spectral_type:
        return 1.0, "spectral_type_presence_proxy"

    radio_proxy = str(row.get("radio_proxy_available", "")).strip().lower()
    if radio_proxy in {"yes", "true", "1"}:
        return 1.0, "radio_proxy_presence"

    return None, "no_usable_mass_proxy"


def derive_physical_radius_scale_pc(row: pd.Series) -> tuple[Optional[float], str]:
    dist_kpc = pd.to_numeric(pd.Series([row.get("dist_kpc", np.nan)]), errors="coerce").iloc[0]
    radius_arcsec = pd.to_numeric(pd.Series([row.get("radius_arcsec", np.nan)]), errors="coerce").iloc[0]

    if pd.notna(dist_kpc) and dist_kpc > 0 and pd.notna(radius_arcsec) and radius_arcsec > 0:
        theta_rad = math.radians(float(radius_arcsec) / 3600.0)
        radius_pc = float(dist_kpc) * 1000.0 * theta_rad
        return radius_pc, "derived_from_dist_kpc_and_radius_arcsec"

    return None, "no_physical_radius_scale"


def convert_zone_to_pc(zone: TargetZones, radius_scale_pc: float) -> dict:
    arc_to_pc = radius_scale_pc / max(zone.radius_arcsec, EPS)
    return {
        "inner_inner_pc": zone.inner_inner_arcsec * arc_to_pc,
        "inner_outer_pc": zone.inner_outer_arcsec * arc_to_pc,
        "shell_inner_pc": zone.shell_inner_arcsec * arc_to_pc,
        "shell_outer_pc": zone.shell_outer_arcsec * arc_to_pc,
        "bg_inner_pc": zone.bg_inner_arcsec * arc_to_pc,
        "bg_outer_pc": zone.bg_outer_arcsec * arc_to_pc,
        "arcsec_to_pc_scale": arc_to_pc,
    }


def classify_v3(status: str, sigma_mass_volume: Optional[float]) -> str:
    if status != "mass_volume_ready":
        return status
    if sigma_mass_volume is None:
        return "mass_volume_ready_but_sigma_missing"
    if sigma_mass_volume > 0.15:
        return "shell_mass_volume_enhanced"
    if sigma_mass_volume > 0.03:
        return "weak_shell_mass_volume_excess"
    if sigma_mass_volume < -0.15:
        return "shell_mass_volume_suppressed"
    return "mass_volume_near_balanced"


def save_alpha_profile_plot(wise_name: str, profile_struct: pd.DataFrame, zone: TargetZones, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(profile_struct["radius_arcsec"], profile_struct["alpha_obs"], label="alpha_obs(r)")

    for x in [zone.inner_outer_arcsec, zone.shell_outer_arcsec, zone.bg_outer_arcsec]:
        ax.axvline(x, linestyle="-", linewidth=1.1)
    for x in [zone.shell_inner_arcsec, zone.bg_inner_arcsec]:
        ax.axvline(x, linestyle="--", linewidth=1.1)

    ax.set_xlabel("Radius (arcsec)")
    ax.set_ylabel("alpha_obs")
    ax.set_title(f"{wise_name} observational scaling profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{sanitize_wise_name(wise_name)}_alpha_obs_profile.png", dpi=180)
    plt.close(fig)


def save_cumulative_profile_plot(wise_name: str, profile_struct: pd.DataFrame, zone: TargetZones, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(profile_struct["radius_arcsec"], profile_struct["L_cumulative"], label="L(<r)")

    for x in [zone.inner_outer_arcsec, zone.shell_outer_arcsec, zone.bg_outer_arcsec]:
        ax.axvline(x, linestyle="-", linewidth=1.1)
    for x in [zone.shell_inner_arcsec, zone.bg_inner_arcsec]:
        ax.axvline(x, linestyle="--", linewidth=1.1)

    ax.set_xlabel("Radius (arcsec)")
    ax.set_ylabel("L(<r)")
    ax.set_title(f"{wise_name} cumulative structural proxy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{sanitize_wise_name(wise_name)}_L_cumulative_profile.png", dpi=180)
    plt.close(fig)


def save_dw_obs_zone_bar_plot(
    wise_name: str,
    D_inner: Optional[float],
    D_shell: Optional[float],
    D_bg: Optional[float],
    out_dir: Path,
) -> None:
    labels = ["Inner", "Shell", "Background"]
    values = [
        np.nan if D_inner is None else D_inner,
        np.nan if D_shell is None else D_shell,
        np.nan if D_bg is None else D_bg,
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values)
    ax.set_ylabel("D_w_obs")
    ax.set_title(f"{wise_name} observational structural dimension by zone")
    fig.tight_layout()
    fig.savefig(out_dir / f"{sanitize_wise_name(wise_name)}_Dw_obs_zone_bar.png", dpi=180)
    plt.close(fig)


def save_mass_bridge_plot(
    wise_name: str,
    D_shell: Optional[float],
    D_bg: Optional[float],
    shell_mv_density: Optional[float],
    bg_mv_density: Optional[float],
    out_dir: Path,
) -> None:
    labels = ["Dw_shell", "Dw_bg", "MVdens_shell", "MVdens_bg"]
    values = [
        np.nan if D_shell is None else D_shell,
        np.nan if D_bg is None else D_bg,
        np.nan if shell_mv_density is None else shell_mv_density,
        np.nan if bg_mv_density is None else bg_mv_density,
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_title(f"{wise_name} mass-volume bridge summary")
    fig.tight_layout()
    fig.savefig(out_dir / f"{sanitize_wise_name(wise_name)}_mass_bridge_profile.png", dpi=180)
    plt.close(fig)


def wrap_note_by_periods(text: str, width: int = 95) -> str:
    parts = [p.strip() for p in text.split(".") if p.strip()]
    lines: List[str] = []
    for part in parts:
        sentence = part + "."
        wrapped = textwrap.wrap(sentence, width=width)
        lines.extend(wrapped if wrapped else [sentence])
    return "\n".join(lines)


def make_radius_label(value: str, arcsec: float) -> str:
    return value + "\n(" + f"{arcsec:.1f}" + '")'


def save_circular_plot_for_target(target: pd.Series, zone: TargetZones, out_dir: Path) -> None:
    rmax = zone.bg_outer_arcsec
    center_circle_r = max(0.03 * rmax, 3.0)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.05, 0.14, 0.74, 0.76])

    fig.text(
        0.5,
        0.95,
        f"{target['wise_name']} Topological Zone Schematic",
        ha="center",
        va="center",
        fontsize=18,
    )

    note_text = (
        f"Note. R is the catalog reference radius. For this target, R = {zone.radius_arcsec:.1f} arcsec. "
        f"The small central circle is a schematic center marker. It is not to physical scale. "
        f"Topological output uses the same baseline zone geometry for fair comparison."
    )
    fig.text(
        0.12,
        0.05,
        wrap_note_by_periods(note_text, width=95),
        ha="left",
        va="bottom",
        fontsize=10,
    )

    legend_lines = [
        ("Legend", 12, "center"),
        ("Yellow center circle: center marker", 9.7, "left"),
        (f"Inner baseline zone: 0 to 0.5R ({zone.inner_outer_arcsec:.1f}\")", 9.2, "left"),
        (
            f"Shell structure zone: 0.6R to 1.0R ({zone.shell_inner_arcsec:.1f}\" to {zone.shell_outer_arcsec:.1f}\")",
            9.2,
            "left",
        ),
        (
            f"Background baseline zone: 1.2R to 1.8R ({zone.bg_inner_arcsec:.1f}\" to {zone.bg_outer_arcsec:.1f}\")",
            9.2,
            "left",
        ),
        (f"Reference radius: R = {zone.radius_arcsec:.1f} arcsec", 9.2, "left"),
    ]

    max_chars = max(len(txt) for txt, _, _ in legend_lines)
    legend_width = min(max(0.26, 0.11 + max_chars * 0.010), 0.415)
    legend_height = (len(legend_lines) - 1) * 0.0025 + 0.12
    legend_right = 0.97
    legend_left = legend_right - legend_width
    legend_bottom = 0.05
    legend_ax = fig.add_axes([legend_left, legend_bottom, legend_width, legend_height])

    legend_ax.set_axis_off()
    legend_ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, linewidth=1.2))
    y = 0.88
    for i, (txt, fs, align) in enumerate(legend_lines):
        legend_ax.text(
            0.5 if align == "center" else 0.05,
            y,
            txt,
            ha=align,
            va="center",
            fontsize=fs,
        )
        y -= 0.13 if i == 0 else 0.125

    ax.set_aspect("equal")
    ax.set_xlim(-rmax * 1.06, rmax * 1.06)
    ax.set_ylim(-rmax * 1.06, rmax * 1.06)

    ax.plot([-rmax, rmax], [0, 0], linewidth=1.0, color="#2C7FB8")
    ax.plot([0, 0], [-rmax, rmax], linewidth=1.0, color="#D95F0E")

    ax.add_patch(Circle((0, 0), zone.bg_outer_arcsec, facecolor="#C6DBEF", edgecolor="none", alpha=0.85))
    ax.add_patch(Circle((0, 0), zone.bg_inner_arcsec, facecolor="white", edgecolor="none"))

    ax.add_patch(Circle((0, 0), zone.shell_outer_arcsec, facecolor="#9ECAE1", edgecolor="none", alpha=0.95))
    ax.add_patch(Circle((0, 0), zone.shell_inner_arcsec, facecolor="white", edgecolor="none"))

    ax.add_patch(Circle((0, 0), zone.inner_outer_arcsec, facecolor="#FFF7BC", edgecolor="none", alpha=0.98))

    for rad, lw in [
        (zone.inner_outer_arcsec, 1.1),
        (zone.shell_outer_arcsec, 1.4),
        (zone.bg_outer_arcsec, 1.1),
    ]:
        ax.add_patch(Circle((0, 0), rad, fill=False, linestyle="-", linewidth=lw, edgecolor="black"))

    for rad in [zone.shell_inner_arcsec, zone.bg_inner_arcsec]:
        ax.add_patch(Circle((0, 0), rad, fill=False, linestyle="--", linewidth=1.2, edgecolor="black"))

    ax.add_patch(Circle((0, 0), center_circle_r, facecolor="#FFD92F", edgecolor="black", linewidth=1.0))

    ax.text(0, zone.inner_outer_arcsec * 0.56, "Inner", ha="center", va="center", fontsize=16)
    ax.text(0, (zone.shell_inner_arcsec + zone.shell_outer_arcsec) / 2, "Shell", ha="center", va="center", fontsize=16)
    ax.text(0, (zone.bg_inner_arcsec + zone.bg_outer_arcsec) / 2, "Background", ha="center", va="center", fontsize=16)

    label_y_offset = 10
    ax.text(zone.inner_outer_arcsec, label_y_offset, make_radius_label("0.5R", zone.inner_outer_arcsec), fontsize=10.5, ha="center", va="bottom")
    ax.text(zone.shell_outer_arcsec, label_y_offset, make_radius_label("1.0R", zone.shell_outer_arcsec), fontsize=10.5, ha="center", va="bottom")
    ax.text(zone.bg_outer_arcsec, label_y_offset, make_radius_label("1.8R", zone.bg_outer_arcsec), fontsize=10.5, ha="center", va="bottom")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(out_dir / f"{sanitize_wise_name(target['wise_name'])}_topological_zones_circular.png", dpi=180)
    plt.close(fig)


def save_summary_txt(
    wise_name: str,
    profile_path: Optional[Path],
    input_row: pd.Series,
    D_inner: Optional[float],
    D_shell: Optional[float],
    D_bg: Optional[float],
    mass_proxy: Optional[float],
    mass_proxy_kind: str,
    radius_scale_pc: Optional[float],
    radius_scale_kind: str,
    shell_mv_density: Optional[float],
    bg_mv_density: Optional[float],
    sigma_mass_volume: Optional[float],
    status: str,
    flag: str,
    out_dir: Path,
) -> None:
    lines = [
        f"wise_name: {wise_name}",
        f"profile_file: {profile_path if profile_path else 'not_found'}",
        "",
        "observational structural dimensions",
        f"Dw_obs_inner: {'' if D_inner is None else D_inner}",
        f"Dw_obs_shell: {'' if D_shell is None else D_shell}",
        f"Dw_obs_background: {'' if D_bg is None else D_bg}",
        "",
        "mass/proxy bridge from final input",
        f"mass_source_key: {input_row.get('mass_source_key', '')}",
        f"mass_value_type: {input_row.get('mass_value_type', '')}",
        f"proxy_kind: {input_row.get('proxy_kind', '')}",
        f"log_nly: {input_row.get('log_nly', '')}",
        f"spectral_type: {input_row.get('spectral_type', '')}",
        f"radio_proxy_available: {input_row.get('radio_proxy_available', '')}",
        "",
        f"mass_proxy: {'' if mass_proxy is None else mass_proxy}",
        f"mass_proxy_kind: {mass_proxy_kind}",
        f"radius_scale_pc: {'' if radius_scale_pc is None else radius_scale_pc}",
        f"radius_scale_kind: {radius_scale_kind}",
        f"shell_mass_volume_density: {'' if shell_mv_density is None else shell_mv_density}",
        f"background_mass_volume_density: {'' if bg_mv_density is None else bg_mv_density}",
        f"sigma_mass_volume: {'' if sigma_mass_volume is None else sigma_mass_volume}",
        "",
        f"bridge_status: {status}",
        f"interpretation_flag: {flag}",
    ]
    (out_dir / f"{sanitize_wise_name(wise_name)}_topological_common5_v4_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def compute_target_result(row: pd.Series, zone: TargetZones, project_root: Path, target_dir: Path) -> dict:
    wise_name = str(row["wise_name"])
    profile_path = find_profile_file(project_root, wise_name)

    D_inner = None
    D_shell = None
    D_bg = None
    mass_proxy = None
    mass_proxy_kind = "not_evaluated"
    radius_scale_pc = None
    radius_scale_kind = "not_evaluated"
    shell_mv_density = None
    bg_mv_density = None
    sigma_mass_volume = None
    status = "missing_observational_profile"

    save_circular_plot_for_target(row, zone, target_dir)

    if profile_path is not None:
        profile = load_profile_csv(profile_path)
        profile_struct = build_observational_structure_profile(profile)

        D_inner = weighted_zone_mean(profile_struct, "alpha_obs", zone.inner_inner_arcsec, zone.inner_outer_arcsec)
        D_shell = weighted_zone_mean(profile_struct, "alpha_obs", zone.shell_inner_arcsec, zone.shell_outer_arcsec)
        D_bg = weighted_zone_mean(profile_struct, "alpha_obs", zone.bg_inner_arcsec, zone.bg_outer_arcsec)

        mass_proxy, mass_proxy_kind = derive_mass_proxy(row)
        radius_scale_pc, radius_scale_kind = derive_physical_radius_scale_pc(row)

        if (D_shell is not None and D_bg is not None and
            mass_proxy is not None and radius_scale_pc is not None and radius_scale_pc > 0):
            zone_pc = convert_zone_to_pc(zone, radius_scale_pc)
            shell_vol = sphere_shell_volume(zone_pc["shell_inner_pc"], zone_pc["shell_outer_pc"])
            bg_vol = sphere_shell_volume(zone_pc["bg_inner_pc"], zone_pc["bg_outer_pc"])

            if shell_vol > 0 and bg_vol > 0:
                shell_mv_density = mass_proxy * max(D_shell, 0.0) / shell_vol
                bg_mv_density = mass_proxy * max(D_bg, 0.0) / bg_vol
                sigma_mass_volume = (shell_mv_density - bg_mv_density) / (bg_mv_density + EPS)
                status = "mass_volume_ready"
            else:
                status = "missing_zone_volume"
        else:
            reasons = []
            if D_shell is None or D_bg is None:
                reasons.append("missing_Dw_obs")
            if mass_proxy is None:
                reasons.append(mass_proxy_kind)
            if radius_scale_pc is None:
                reasons.append(radius_scale_kind)
            status = "observational_only__" + "__".join(reasons) if reasons else "observational_only"

        save_alpha_profile_plot(wise_name, profile_struct, zone, target_dir)
        save_cumulative_profile_plot(wise_name, profile_struct, zone, target_dir)
        save_dw_obs_zone_bar_plot(wise_name, D_inner, D_shell, D_bg, target_dir)

        profile_struct.to_csv(
            target_dir / f"{sanitize_wise_name(wise_name)}_processed_structure_profile_common5_v4.csv",
            index=False,
            encoding="utf-8-sig",
        )

    flag = classify_v3(status, sigma_mass_volume)
    save_mass_bridge_plot(wise_name, D_shell, D_bg, shell_mv_density, bg_mv_density, target_dir)
    save_summary_txt(
        wise_name=wise_name,
        profile_path=profile_path,
        input_row=row,
        D_inner=D_inner,
        D_shell=D_shell,
        D_bg=D_bg,
        mass_proxy=mass_proxy,
        mass_proxy_kind=mass_proxy_kind,
        radius_scale_pc=radius_scale_pc,
        radius_scale_kind=radius_scale_kind,
        shell_mv_density=shell_mv_density,
        bg_mv_density=bg_mv_density,
        sigma_mass_volume=sigma_mass_volume,
        status=status,
        flag=flag,
        out_dir=target_dir,
    )

    return {
        "wise_name": wise_name,
        "radius_arcsec": zone.radius_arcsec,
        "Dw_obs_inner": D_inner,
        "Dw_obs_shell": D_shell,
        "Dw_obs_background": D_bg,
        "mass_proxy": mass_proxy,
        "mass_proxy_kind": mass_proxy_kind,
        "radius_scale_pc": radius_scale_pc,
        "radius_scale_kind": radius_scale_kind,
        "shell_mass_volume_density": shell_mv_density,
        "background_mass_volume_density": bg_mv_density,
        "sigma_mass_volume": sigma_mass_volume,
        "bridge_status": status,
        "interpretation_flag": flag,
        "profile_file": str(profile_path) if profile_path else "",
    }


def write_manifest(project_root: Path, out_dir: Path, df: pd.DataFrame) -> None:
    lines = [
        "Validation of Structural Contrast Baseline",
        "Topological pipeline manifest (common5 v4)",
        "=" * 60,
        "",
        f"project_root: {project_root}",
        f"input_file: {project_root / INPUT_FILE}",
        f"profile_dir: {project_root / PROFILE_DIR}",
        f"output_dir: {out_dir}",
        "",
        f"target_count: {len(df)}",
        "",
        "meaning",
        "-" * 20,
        "This common5 v3 pipeline reads fixed final input and radial profiles.",
        "Mass/proxy bridge is read directly from the topological final input file.",
        "It preserves the observational structure chain and adds conservative mass-volume interpretation.",
        "Added graphs: standard-style circular zone schematic and Dw_obs zone bar.",
    ]
    (out_dir / "run_manifest.txt").write_text("\n".join(lines), encoding="utf-8")


def write_run_summary(out_dir: Path, summary_df: pd.DataFrame) -> None:
    rows = [
        {"metric": "target_count", "value": len(summary_df)},
        {"metric": "targets_with_profile", "value": int(summary_df["profile_file"].astype(str).ne("").sum())},
        {"metric": "targets_mass_volume_ready", "value": int((summary_df["bridge_status"] == "mass_volume_ready").sum())},
        {"metric": "targets_observational_only", "value": int(summary_df["bridge_status"].astype(str).str.startswith("observational_only").sum())},
        {"metric": "shell_mass_volume_enhanced_count", "value": int((summary_df["interpretation_flag"] == "shell_mass_volume_enhanced").sum())},
        {"metric": "weak_shell_mass_volume_excess_count", "value": int((summary_df["interpretation_flag"] == "weak_shell_mass_volume_excess").sum())},
        {"metric": "shell_mass_volume_suppressed_count", "value": int((summary_df["interpretation_flag"] == "shell_mass_volume_suppressed").sum())},
        {"metric": "mass_volume_near_balanced_count", "value": int((summary_df["interpretation_flag"] == "mass_volume_near_balanced").sum())},
    ]
    pd.DataFrame(rows).to_csv(out_dir / "run_summary.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()

    out_dir = create_timestamped_output_dir(project_root)

    raw_df = load_final_input(project_root)
    df = normalize_input(raw_df)
    zones = build_standard_zones(df)
    zone_map = {z.wise_name: z for z in zones}

    results = []
    targets_root = out_dir / "targets"
    targets_root.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        wise_name = str(row["wise_name"])
        zone = zone_map[wise_name]
        target_dir = targets_root / sanitize_wise_name(wise_name)
        target_dir.mkdir(parents=True, exist_ok=True)
        results.append(compute_target_result(row, zone, project_root, target_dir))

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(out_dir / "topological_summary_common5_v4.csv", index=False, encoding="utf-8-sig")

    write_manifest(project_root, out_dir, df)
    write_run_summary(out_dir, summary_df)

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - topological pipeline (common5 v4)")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print(f"Input file  : {project_root / INPUT_FILE}")
    print(f"Profile dir : {project_root / PROFILE_DIR}")
    print(f"Output dir  : {out_dir}")
    print("")
    print("[OK] Created:")
    print(out_dir / "topological_summary_common5_v4.csv")
    print(out_dir / "run_manifest.txt")
    print(out_dir / "run_summary.csv")
    print(out_dir / "targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
