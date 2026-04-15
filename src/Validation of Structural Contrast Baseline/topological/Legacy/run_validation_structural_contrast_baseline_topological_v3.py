#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
Topological pipeline v3 (mass-bridge aware)

Author: Kwon Dominicus

V3 concept
----------
This version extends the V2 observational-structure pipeline by reading the
mass-bridge layer:

    data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/wise_hii_mass_bridge.csv

It keeps the V2 structural backbone:

    I(r) -> L(<r) -> alpha_obs(r) -> D_w_obs(zone)

and then applies a conservative mass-volume bridge only when a usable bridge
input exists.

Current bridge policy
---------------------
1) If direct mass_value_msun is provided, use it as the primary mass bridge.
2) Else if log_Nly_s_minus_1 is available, use a dimensionless ionizing proxy
   scale:
       mass_proxy_scale = 10^(log_Nly - 49)
   This is NOT claimed to be a literal stellar mass in solar units.
   It is only a conservative bridge weight relative to a 10^49 s^-1 reference.
3) If no usable bridge exists, keep the target in observational-only status.

Zone volume policy
------------------
If effective_radius_pc is available in the bridge file, zone radii are converted
to physical radii using that effective radius as the physical counterpart of the
catalog angular radius. Then approximate spherical shell volumes are used.

If physical radius information is unavailable, the target remains in
observational-only status.

Outputs
-------
results/Validation of Structural Contrast Baseline/output/topological/YYYYMMDD_HHMMSS/
    topological_summary_v3.csv
    run_summary.csv
    run_manifest.txt
    targets/<wise_name>/
        *_mass_bridge_profile.png
        *_topological_v3_summary.txt
"""

from __future__ import annotations

import math
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

MASS_BRIDGE_FILE = (
    Path("data")
    / "raw"
    / "Validation of Structural Contrast Baseline"
    / "wise_hii_catalog"
    / "wise_hii_mass_bridge.csv"
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
    return pd.read_csv(path, dtype=str, low_memory=False)


def load_mass_bridge(project_root: Path) -> pd.DataFrame:
    path = project_root / MASS_BRIDGE_FILE
    if not path.exists():
        raise FileNotFoundError(f"Mass bridge file not found: {path}")
    return pd.read_csv(path, low_memory=False)


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
    ensure_required_columns(df, required, "Final input")

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


def normalize_mass_bridge(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ["wise_name", "mass_bridge_status", "preferred_mass_type"]
    ensure_required_columns(df, required, "Mass bridge")

    df["wise_name"] = df["wise_name"].astype(str).str.strip()

    numeric_cols = [
        "mass_value_msun",
        "mass_value_min_msun",
        "mass_value_max_msun",
        "distance_kpc",
        "galactocentric_distance_kpc",
        "effective_radius_pc",
        "flux_1p4GHz_Jy",
        "log_Nly_s_minus_1",
        "stromgren_radius_pc",
        "dynamical_age_Myr",
    ]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["bridge_ready_for_v3", "mass_input_mode_for_v3", "mass_source_note", "source_bundle"]:
        if c not in df.columns:
            df[c] = ""

    df = df.drop_duplicates(subset=["wise_name"]).reset_index(drop=True)
    return df


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


def get_mass_bridge_row(mass_df: pd.DataFrame, wise_name: str) -> Optional[pd.Series]:
    row = mass_df.loc[mass_df["wise_name"] == wise_name]
    if row.empty:
        return None
    return row.iloc[0]


def derive_mass_proxy(row: Optional[pd.Series]) -> tuple[Optional[float], str]:
    if row is None:
        return None, "missing_mass_bridge_row"

    direct_mass = row.get("mass_value_msun", np.nan)
    if pd.notna(direct_mass) and float(direct_mass) > 0:
        return float(direct_mass), "direct_mass_value_msun"

    log_nly = row.get("log_Nly_s_minus_1", np.nan)
    if pd.notna(log_nly):
        # Conservative dimensionless proxy relative to 1e49 s^-1
        return float(10 ** (float(log_nly) - 49.0)), "radio_proxy_from_log_Nly"

    return None, "no_usable_mass_proxy"


def derive_physical_radius_scale_pc(row: Optional[pd.Series]) -> tuple[Optional[float], str]:
    if row is None:
        return None, "missing_mass_bridge_row"

    eff_r = row.get("effective_radius_pc", np.nan)
    if pd.notna(eff_r) and float(eff_r) > 0:
        return float(eff_r), "effective_radius_pc"

    return None, "no_physical_radius_scale"


def convert_zone_to_pc(zone: TargetZones, radius_scale_pc: float) -> dict:
    # radius_scale_pc is interpreted as the physical counterpart of the catalog radius R
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


def save_summary_txt(
    wise_name: str,
    profile_path: Optional[Path],
    mass_bridge_row: Optional[pd.Series],
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
        "mass bridge",
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
    if mass_bridge_row is not None:
        lines += [
            "",
            "mass bridge source context",
            f"mass_bridge_status: {mass_bridge_row.get('mass_bridge_status', '')}",
            f"preferred_mass_type: {mass_bridge_row.get('preferred_mass_type', '')}",
            f"bridge_ready_for_v3: {mass_bridge_row.get('bridge_ready_for_v3', '')}",
            f"mass_input_mode_for_v3: {mass_bridge_row.get('mass_input_mode_for_v3', '')}",
            f"source_bundle: {mass_bridge_row.get('source_bundle', '')}",
        ]
    (out_dir / f"{sanitize_wise_name(wise_name)}_topological_v3_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def compute_target_result(
    row: pd.Series,
    zone: TargetZones,
    mass_df: pd.DataFrame,
    project_root: Path,
    target_dir: Path,
) -> dict:
    wise_name = str(row["wise_name"])
    profile_path = find_profile_file(project_root, wise_name)
    bridge_row = get_mass_bridge_row(mass_df, wise_name)

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

    if profile_path is not None:
        profile = load_profile_csv(profile_path)
        profile_struct = build_observational_structure_profile(profile)

        D_inner = weighted_zone_mean(profile_struct, "alpha_obs", zone.inner_inner_arcsec, zone.inner_outer_arcsec)
        D_shell = weighted_zone_mean(profile_struct, "alpha_obs", zone.shell_inner_arcsec, zone.shell_outer_arcsec)
        D_bg = weighted_zone_mean(profile_struct, "alpha_obs", zone.bg_inner_arcsec, zone.bg_outer_arcsec)

        mass_proxy, mass_proxy_kind = derive_mass_proxy(bridge_row)
        radius_scale_pc, radius_scale_kind = derive_physical_radius_scale_pc(bridge_row)

        if (D_shell is not None and D_bg is not None and
            mass_proxy is not None and radius_scale_pc is not None and radius_scale_pc > 0):
            zone_pc = convert_zone_to_pc(zone, radius_scale_pc)
            shell_vol = sphere_shell_volume(zone_pc["shell_inner_pc"], zone_pc["shell_outer_pc"])
            bg_vol = sphere_shell_volume(zone_pc["bg_inner_pc"], zone_pc["bg_outer_pc"])

            if shell_vol > 0 and bg_vol > 0:
                # Conservative bridge:
                # use the same proxy mass scale on both zones, but let structural dimension modulate
                # an effective mass-volume density proxy.
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

        profile_struct.to_csv(
            target_dir / f"{sanitize_wise_name(wise_name)}_processed_structure_profile_v3.csv",
            index=False,
            encoding="utf-8-sig",
        )

    flag = classify_v3(status, sigma_mass_volume)
    save_mass_bridge_plot(wise_name, D_shell, D_bg, shell_mv_density, bg_mv_density, target_dir)
    save_summary_txt(
        wise_name=wise_name,
        profile_path=profile_path,
        mass_bridge_row=bridge_row,
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

    result = {
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
    return result


def write_manifest(project_root: Path, out_dir: Path, df: pd.DataFrame) -> None:
    lines = [
        "Validation of Structural Contrast Baseline",
        "Topological pipeline v3 manifest",
        "=" * 60,
        "",
        f"project_root: {project_root}",
        f"input_file: {project_root / INPUT_FILE}",
        f"profile_dir: {project_root / PROFILE_DIR}",
        f"mass_bridge_file: {project_root / MASS_BRIDGE_FILE}",
        f"output_dir: {out_dir}",
        "",
        f"target_count: {len(df)}",
        "",
        "meaning",
        "-" * 20,
        "This v3 pipeline reads the final input, radial profiles, and mass-bridge layer.",
        "It preserves the V2 observational structure chain.",
        "Mass-volume bridge is applied only when a usable mass proxy and physical radius scale exist.",
        "radio proxy from log_Nly is treated as a conservative relative mass bridge, not a literal stellar mass.",
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
    mass_df = normalize_mass_bridge(load_mass_bridge(project_root))
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
        results.append(compute_target_result(row, zone, mass_df, project_root, target_dir))

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(out_dir / "topological_summary_v3.csv", index=False, encoding="utf-8-sig")

    write_manifest(project_root, out_dir, df)
    write_run_summary(out_dir, summary_df)

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - topological pipeline v3")
    print("=" * 72)
    print(f"Project root    : {project_root}")
    print(f"Input file      : {project_root / INPUT_FILE}")
    print(f"Profile dir     : {project_root / PROFILE_DIR}")
    print(f"Mass bridge file: {project_root / MASS_BRIDGE_FILE}")
    print(f"Output dir      : {out_dir}")
    print("")
    print("[OK] Created:")
    print(out_dir / "topological_summary_v3.csv")
    print(out_dir / "run_manifest.txt")
    print(out_dir / "run_summary.csv")
    print(out_dir / "targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
