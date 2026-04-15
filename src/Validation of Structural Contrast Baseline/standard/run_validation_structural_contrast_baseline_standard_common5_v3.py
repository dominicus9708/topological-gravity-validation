#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
Standard pipeline for common5 final input (v3)

Author: Kwon Dominicus

Placement
---------
src/Validation of Structural Contrast Baseline/standard/

Input
-----
data/derived/Validation of Structural Contrast Baseline/input/standard/wise_hii_common5/wise_hii_common5_standard_final_input.csv

Output
------
results/Validation of Structural Contrast Baseline/output/standard/YYYYMMDD_HHMMSS/

Notes
-----
- This is an execution pipeline, not a derived-processing script.
- Final input is treated as fixed official input.
- v3 adds graph consistency with the topological pipeline:
  1) observational intensity profile with common zone boundaries
  2) alpha_obs(r) profile
  3) cumulative structural proxy L(<r)
  4) Dw_obs zone bar plot
- The old circular zone schematic is intentionally omitted in v3.
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
    / "standard"
    / "wise_hii_common5"
    / "wise_hii_common5_standard_final_input.csv"
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
    / "standard"
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
        "catalog_class",
        "input_group",
        "priority_rank",
        "glon",
        "glat",
        "ra",
        "dec",
        "radius_arcsec",
        "hii_region_name",
        "membership",
        "baseline_ready_flag",
    ]
    ensure_required_columns(df, required)

    df["radius_arcsec_num"] = pd.to_numeric(df["radius_arcsec"], errors="coerce")
    if df["radius_arcsec_num"].isna().any():
        bad = df.loc[df["radius_arcsec_num"].isna(), "wise_name"].tolist()
        raise ValueError("Invalid radius_arcsec values for: " + ", ".join(map(str, bad)))

    df["priority_rank_num"] = pd.to_numeric(df["priority_rank"], errors="coerce")
    df["wise_name"] = df["wise_name"].astype(str).str.strip()

    if df["wise_name"].duplicated().any():
        dup = df.loc[df["wise_name"].duplicated(), "wise_name"].tolist()
        raise ValueError("Duplicate wise_name entries found: " + ", ".join(dup))

    not_ready = df.loc[
        df["baseline_ready_flag"].astype(str).str.lower().isin(["false", "0", "no"]),
        "wise_name",
    ].tolist()
    if not_ready:
        raise ValueError("Rows not baseline_ready_flag=True/usable: " + ", ".join(not_ready))

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


def zones_to_dataframe(zones: List[TargetZones]) -> pd.DataFrame:
    return pd.DataFrame([z.__dict__ for z in zones])


def build_baseline_comparison_table(df: pd.DataFrame, zones_df: pd.DataFrame) -> pd.DataFrame:
    zones_tmp = zones_df.rename(columns={"radius_arcsec": "zone_radius_arcsec"}).copy()
    merged = df.merge(zones_tmp, on="wise_name", how="left")
    merged["baseline_geometry_status"] = "defined"
    merged["baseline_ready_for_profile_extraction"] = "yes"

    keep_cols = [
        "wise_name",
        "catalog_class",
        "input_group",
        "priority_rank",
        "hii_region_name",
        "membership",
        "ra",
        "dec",
        "radius_arcsec",
        "zone_radius_arcsec",
        "inner_inner_arcsec",
        "inner_outer_arcsec",
        "shell_inner_arcsec",
        "shell_outer_arcsec",
        "bg_inner_arcsec",
        "bg_outer_arcsec",
        "baseline_geometry_status",
        "baseline_ready_for_profile_extraction",
        "score_total",
        "region_bucket",
    ]
    for c in keep_cols:
        if c not in merged.columns:
            merged[c] = ""
    return merged[keep_cols].copy()


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


def save_intensity_profile_plot(wise_name: str, profile: pd.DataFrame, zone: TargetZones, target_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(profile["radius_arcsec"], profile["intensity"], label="intensity(r)")

    for x in [zone.inner_outer_arcsec, zone.shell_outer_arcsec, zone.bg_outer_arcsec]:
        ax.axvline(x, linestyle="-", linewidth=1.1)
    for x in [zone.shell_inner_arcsec, zone.bg_inner_arcsec]:
        ax.axvline(x, linestyle="--", linewidth=1.1)

    ax.set_xlabel("Radius (arcsec)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"{wise_name} observational intensity profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(target_dir / f"{sanitize_wise_name(wise_name)}_intensity_profile.png", dpi=180)
    plt.close(fig)


def save_alpha_profile_plot(wise_name: str, profile_struct: pd.DataFrame, zone: TargetZones, target_dir: Path) -> None:
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
    fig.savefig(target_dir / f"{sanitize_wise_name(wise_name)}_alpha_obs_profile.png", dpi=180)
    plt.close(fig)


def save_cumulative_profile_plot(wise_name: str, profile_struct: pd.DataFrame, zone: TargetZones, target_dir: Path) -> None:
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
    fig.savefig(target_dir / f"{sanitize_wise_name(wise_name)}_L_cumulative_profile.png", dpi=180)
    plt.close(fig)


def save_dw_obs_zone_bar_plot(
    wise_name: str,
    D_inner: Optional[float],
    D_shell: Optional[float],
    D_bg: Optional[float],
    target_dir: Path,
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
    fig.savefig(target_dir / f"{sanitize_wise_name(wise_name)}_Dw_obs_zone_bar.png", dpi=180)
    plt.close(fig)


def save_per_target_summary(
    target: pd.Series,
    zone: TargetZones,
    out_dir: Path,
    profile_path: Optional[Path],
    D_inner: Optional[float],
    D_shell: Optional[float],
    D_bg: Optional[float],
) -> None:
    name = sanitize_wise_name(target["wise_name"])
    target_dir = out_dir / "targets" / name
    target_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"wise_name: {target['wise_name']}",
        f"priority_rank: {target.get('priority_rank', '')}",
        f"catalog_class: {target.get('catalog_class', '')}",
        f"input_group: {target.get('input_group', '')}",
        f"hii_region_name: {target.get('hii_region_name', '')}",
        f"membership: {target.get('membership', '')}",
        f"ra: {target.get('ra', '')}",
        f"dec: {target.get('dec', '')}",
        f"radius_arcsec: {target.get('radius_arcsec', '')}",
        f"profile_file: {profile_path if profile_path else 'not_found'}",
        "",
        "standard baseline zones",
        f"inner_inner_arcsec: {zone.inner_inner_arcsec}",
        f"inner_outer_arcsec: {zone.inner_outer_arcsec}",
        f"shell_inner_arcsec: {zone.shell_inner_arcsec}",
        f"shell_outer_arcsec: {zone.shell_outer_arcsec}",
        f"bg_inner_arcsec: {zone.bg_inner_arcsec}",
        f"bg_outer_arcsec: {zone.bg_outer_arcsec}",
        "",
        "observational structural dimensions",
        f"Dw_obs_inner: {'' if D_inner is None else D_inner}",
        f"Dw_obs_shell: {'' if D_shell is None else D_shell}",
        f"Dw_obs_background: {'' if D_bg is None else D_bg}",
    ]
    (target_dir / f"{name}_standard_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(project_root: Path, out_dir: Path, df: pd.DataFrame) -> None:
    lines = [
        "Validation of Structural Contrast Baseline",
        "Standard pipeline manifest (common5 v3)",
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
        "This is the standard baseline structure stage for common5 final input.",
        "v3 adds profile-based graphs to align graph style with the topological pipeline.",
        "The circular schematic was intentionally omitted in v3.",
    ]
    (out_dir / "run_manifest.txt").write_text("\n".join(lines), encoding="utf-8")


def write_summary_csv(out_dir: Path, summary_rows: list[dict]) -> None:
    pd.DataFrame(summary_rows).to_csv(out_dir / "standard_summary_common5_v3.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()

    out_dir = create_timestamped_output_dir(project_root)

    raw_df = load_final_input(project_root)
    df = normalize_input(raw_df)
    zones = build_standard_zones(df)
    zones_df = zones_to_dataframe(zones)
    comparison_df = build_baseline_comparison_table(df, zones_df)

    df.to_csv(out_dir / "fixed_final_input_copy.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(out_dir / "standard_baseline_comparison_table.csv", index=False, encoding="utf-8-sig")
    zones_df.to_csv(out_dir / "target_standard_zones.csv", index=False, encoding="utf-8-sig")

    zone_map = {z.wise_name: z for z in zones}
    summary_rows: list[dict] = []

    for _, row in df.iterrows():
        wise_name = str(row["wise_name"])
        name = sanitize_wise_name(wise_name)
        target_dir = out_dir / "targets" / name
        target_dir.mkdir(parents=True, exist_ok=True)
        zone = zone_map[wise_name]

        profile_path = find_profile_file(project_root, wise_name)
        D_inner = None
        D_shell = None
        D_bg = None

        if profile_path is not None:
            profile = load_profile_csv(profile_path)
            profile_struct = build_observational_structure_profile(profile)

            D_inner = weighted_zone_mean(profile_struct, "alpha_obs", zone.inner_inner_arcsec, zone.inner_outer_arcsec)
            D_shell = weighted_zone_mean(profile_struct, "alpha_obs", zone.shell_inner_arcsec, zone.shell_outer_arcsec)
            D_bg = weighted_zone_mean(profile_struct, "alpha_obs", zone.bg_inner_arcsec, zone.bg_outer_arcsec)

            save_intensity_profile_plot(wise_name, profile, zone, target_dir)
            save_alpha_profile_plot(wise_name, profile_struct, zone, target_dir)
            save_cumulative_profile_plot(wise_name, profile_struct, zone, target_dir)
            save_dw_obs_zone_bar_plot(wise_name, D_inner, D_shell, D_bg, target_dir)

            profile_struct.to_csv(
                target_dir / f"{name}_processed_structure_profile_standard_v3.csv",
                index=False,
                encoding="utf-8-sig",
            )

        save_per_target_summary(row, zone, out_dir, profile_path, D_inner, D_shell, D_bg)

        summary_rows.append(
            {
                "wise_name": wise_name,
                "radius_arcsec": zone.radius_arcsec,
                "Dw_obs_inner": D_inner,
                "Dw_obs_shell": D_shell,
                "Dw_obs_background": D_bg,
                "profile_file": str(profile_path) if profile_path else "",
                "profile_status": "ok" if profile_path else "missing_profile",
            }
        )

    write_manifest(project_root, out_dir, df)
    write_summary_csv(out_dir, summary_rows)

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - standard pipeline (common5 v3)")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print(f"Input file  : {project_root / INPUT_FILE}")
    print(f"Profile dir : {project_root / PROFILE_DIR}")
    print(f"Output dir  : {out_dir}")
    print("")
    print("[OK] Created:")
    print(out_dir / "fixed_final_input_copy.csv")
    print(out_dir / "standard_baseline_comparison_table.csv")
    print(out_dir / "target_standard_zones.csv")
    print(out_dir / "standard_summary_common5_v3.csv")
    print(out_dir / "run_manifest.txt")
    print(out_dir / "targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
