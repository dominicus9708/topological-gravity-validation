#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
Topological pipeline v2

Author: Kwon Dominicus

V2 concept
----------
This version replaces the v1 direct zone-mean contrast with a closer proxy to
the paper-series structure:

    I(r) -> L(<r) -> alpha_obs(r) -> D_w_obs(zone) -> sigma_obs

Definitions used in this practical observational proxy:
    L(<r) = integral_0^r I(rho) * rho d(rho)
    alpha_obs(r) ~ d ln L(<r) / d ln r
    D_w_obs(zone) = weighted mean of alpha_obs(r) over the zone
    sigma_obs_shell = D_w_obs(shell) - D_w_obs(background)

This is still a first-pass observational proxy, not a final physical mass-volume
derivation.

Expected input
--------------
    data/derived/Validation of Structural Contrast Baseline/input/wise_hii_catalog/wise_hii_final_input.csv
    data/derived/Validation of Structural Contrast Baseline/input/wise_hii_catalog/radial_profiles/<wise_name>_radial_profile.csv

Recommended placement
---------------------
    src/Validation of Structural Contrast Baseline/topological/
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\mincu\Desktop\topological_gravity_project")

INPUT_FILE = (
    Path("data")
    / "derived"
    / "Validation of Structural Contrast Baseline"
    / "input"
    / "wise_hii_catalog"
    / "wise_hii_final_input.csv"
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
    / "topological_v2"
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


def ensure_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Final input is missing required columns: " + ", ".join(missing))


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
    return pd.read_csv(path, dtype=str, low_memory=False)


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = [
        "wise_name",
        "manual_final_rank",
        "manual_reason",
        "manual_shell_type",
        "manual_background_ring_defined",
        "catalog_class",
        "ra",
        "dec",
        "radius_arcsec",
        "target_status",
        "input_ready",
    ]
    ensure_required_columns(df, required)

    df["radius_arcsec_num"] = pd.to_numeric(df["radius_arcsec"], errors="coerce")
    if df["radius_arcsec_num"].isna().any():
        bad = df.loc[df["radius_arcsec_num"].isna(), "wise_name"].tolist()
        raise ValueError("Invalid radius_arcsec values for: " + ", ".join(map(str, bad)))

    df["manual_final_rank_num"] = pd.to_numeric(df["manual_final_rank"], errors="coerce")
    df["wise_name"] = df["wise_name"].astype(str).str.strip()

    if df["wise_name"].duplicated().any():
        dup = df.loc[df["wise_name"].duplicated(), "wise_name"].tolist()
        raise ValueError("Duplicate wise_name entries found: " + ", ".join(dup))

    not_ready = df.loc[df["input_ready"].astype(str).str.lower().ne("yes"), "wise_name"].tolist()
    if not_ready:
        raise ValueError("Rows not marked input_ready=yes: " + ", ".join(not_ready))

    not_fixed = df.loc[df["target_status"].astype(str).ne("fixed_final_input"), "wise_name"].tolist()
    if not_fixed:
        raise ValueError("Rows not fixed_final_input: " + ", ".join(not_fixed))

    return df.sort_values(
        by=["manual_final_rank_num", "radius_arcsec_num", "wise_name"],
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

    # cumulative structure proxy L(<r) = integral I(r) * r dr
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


def classify_sigma_obs(sigma_obs_shell: Optional[float]) -> str:
    if sigma_obs_shell is None:
        return "missing_observational_profile"
    if sigma_obs_shell > 0.15:
        return "shell_structurally_enhanced"
    if sigma_obs_shell > 0.03:
        return "weak_shell_structural_excess"
    if sigma_obs_shell < -0.15:
        return "shell_structurally_suppressed"
    return "near_balanced"


def save_dw_zone_bar(
    wise_name: str,
    D_inner: Optional[float],
    D_shell: Optional[float],
    D_bg: Optional[float],
    out_dir: Path,
) -> None:
    values = [
        float("nan") if D_inner is None else D_inner,
        float("nan") if D_shell is None else D_shell,
        float("nan") if D_bg is None else D_bg,
    ]
    labels = ["Inner", "Shell", "Background"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylabel("D_w_obs")
    ax.set_title(f"{wise_name} observational structural dimension by zone")
    fig.tight_layout()
    fig.savefig(out_dir / f"{sanitize_wise_name(wise_name)}_Dw_obs_zone_bar.png", dpi=180)
    plt.close(fig)


def save_alpha_profile_plot(
    wise_name: str,
    profile_struct: pd.DataFrame,
    zone: TargetZones,
    out_dir: Path,
) -> None:
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


def save_cumulative_profile_plot(
    wise_name: str,
    profile_struct: pd.DataFrame,
    zone: TargetZones,
    out_dir: Path,
) -> None:
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


def save_summary_txt(
    wise_name: str,
    zone: TargetZones,
    profile_path: Optional[Path],
    D_inner: Optional[float],
    D_shell: Optional[float],
    D_bg: Optional[float],
    sigma_obs_shell: Optional[float],
    flag: str,
    out_dir: Path,
) -> None:
    lines = [
        f"wise_name: {wise_name}",
        f"radius_arcsec: {zone.radius_arcsec:.3f}",
        f"profile_file: {profile_path if profile_path else 'not_found'}",
        "",
        "zones",
        f"inner: {zone.inner_inner_arcsec:.3f} to {zone.inner_outer_arcsec:.3f}",
        f"shell: {zone.shell_inner_arcsec:.3f} to {zone.shell_outer_arcsec:.3f}",
        f"background: {zone.bg_inner_arcsec:.3f} to {zone.bg_outer_arcsec:.3f}",
        "",
        "observational structural dimension",
        f"Dw_obs_inner: {'' if D_inner is None else D_inner}",
        f"Dw_obs_shell: {'' if D_shell is None else D_shell}",
        f"Dw_obs_background: {'' if D_bg is None else D_bg}",
        f"sigma_obs_shell: {'' if sigma_obs_shell is None else sigma_obs_shell}",
        "",
        f"interpretation_flag: {flag}",
    ]
    (out_dir / f"{sanitize_wise_name(wise_name)}_topological_v2_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def compute_target_result(
    row: pd.Series,
    zone: TargetZones,
    project_root: Path,
    target_dir: Path,
) -> dict:
    wise_name = str(row["wise_name"])
    profile_path = find_profile_file(project_root, wise_name)

    D_inner = None
    D_shell = None
    D_bg = None
    sigma_obs_shell = None

    if profile_path is not None:
        profile = load_profile_csv(profile_path)
        profile_struct = build_observational_structure_profile(profile)

        D_inner = weighted_zone_mean(profile_struct, "alpha_obs", zone.inner_inner_arcsec, zone.inner_outer_arcsec)
        D_shell = weighted_zone_mean(profile_struct, "alpha_obs", zone.shell_inner_arcsec, zone.shell_outer_arcsec)
        D_bg = weighted_zone_mean(profile_struct, "alpha_obs", zone.bg_inner_arcsec, zone.bg_outer_arcsec)

        if D_shell is not None and D_bg is not None:
            sigma_obs_shell = D_shell - D_bg

        save_alpha_profile_plot(wise_name, profile_struct, zone, target_dir)
        save_cumulative_profile_plot(wise_name, profile_struct, zone, target_dir)

        # Save expanded processed structure profile for audit
        profile_struct.to_csv(
            target_dir / f"{sanitize_wise_name(wise_name)}_processed_structure_profile.csv",
            index=False,
            encoding="utf-8-sig",
        )

    flag = classify_sigma_obs(sigma_obs_shell)
    save_dw_zone_bar(wise_name, D_inner, D_shell, D_bg, target_dir)
    save_summary_txt(
        wise_name=wise_name,
        zone=zone,
        profile_path=profile_path,
        D_inner=D_inner,
        D_shell=D_shell,
        D_bg=D_bg,
        sigma_obs_shell=sigma_obs_shell,
        flag=flag,
        out_dir=target_dir,
    )

    return {
        "wise_name": wise_name,
        "radius_arcsec": zone.radius_arcsec,
        "Dw_obs_inner": D_inner,
        "Dw_obs_shell": D_shell,
        "Dw_obs_background": D_bg,
        "sigma_obs_shell": sigma_obs_shell,
        "interpretation_flag": flag,
        "profile_file": str(profile_path) if profile_path else "",
    }


def write_manifest(project_root: Path, out_dir: Path, df: pd.DataFrame) -> None:
    lines = [
        "Validation of Structural Contrast Baseline",
        "Topological pipeline v2 manifest",
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
        "This v2 pipeline reads the final input directly from the input layer.",
        "It re-creates the standard inner/shell/background zones.",
        "It transforms I(r) into L(<r), alpha_obs(r), and D_w_obs(zone).",
        "Final shell contrast is sigma_obs_shell = D_w_obs(shell) - D_w_obs(background).",
    ]
    (out_dir / "run_manifest.txt").write_text("\n".join(lines), encoding="utf-8")


def write_run_summary(out_dir: Path, summary_df: pd.DataFrame) -> None:
    rows = [
        {"metric": "target_count", "value": len(summary_df)},
        {"metric": "targets_with_profile", "value": int(summary_df["profile_file"].astype(str).ne("").sum())},
        {"metric": "targets_with_sigma_obs", "value": int(summary_df["sigma_obs_shell"].notna().sum())},
        {"metric": "shell_structurally_enhanced_count", "value": int((summary_df["interpretation_flag"] == "shell_structurally_enhanced").sum())},
        {"metric": "weak_shell_structural_excess_count", "value": int((summary_df["interpretation_flag"] == "weak_shell_structural_excess").sum())},
        {"metric": "shell_structurally_suppressed_count", "value": int((summary_df["interpretation_flag"] == "shell_structurally_suppressed").sum())},
        {"metric": "near_balanced_count", "value": int((summary_df["interpretation_flag"] == "near_balanced").sum())},
        {"metric": "missing_profile_count", "value": int((summary_df["interpretation_flag"] == "missing_observational_profile").sum())},
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
    summary_df.to_csv(out_dir / "topological_summary_v2.csv", index=False, encoding="utf-8-sig")

    write_manifest(project_root, out_dir, df)
    write_run_summary(out_dir, summary_df)

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - topological pipeline v2")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print(f"Input file  : {project_root / INPUT_FILE}")
    print(f"Profile dir : {project_root / PROFILE_DIR}")
    print(f"Output dir  : {out_dir}")
    print("")
    print("[OK] Created:")
    print(out_dir / "topological_summary_v2.csv")
    print(out_dir / "run_manifest.txt")
    print(out_dir / "run_summary.csv")
    print(out_dir / "targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
