#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
Topological pipeline v1

Author: Kwon Dominicus
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
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

    return out.sort_values("radius_arcsec").reset_index(drop=True)


def area_weighted_mean_from_profile(profile: pd.DataFrame, r_in: float, r_out: float) -> Optional[float]:
    subset = profile[(profile["radius_arcsec"] >= r_in) & (profile["radius_arcsec"] <= r_out)].copy()
    if subset.empty:
        return None

    weights = subset["radius_arcsec"].clip(lower=1e-6)
    denom = float(weights.sum())
    if denom <= 0:
        return None
    return float((subset["intensity"] * weights).sum() / denom)


def classify_sigma(sigma_topo: Optional[float], shell_minus_bg: Optional[float]) -> str:
    if sigma_topo is None:
        return "missing_observational_profile"
    if shell_minus_bg is None:
        return "undetermined"
    if sigma_topo > 0.15 and shell_minus_bg > 0:
        return "shell_enhanced"
    if sigma_topo > 0.02 and shell_minus_bg > 0:
        return "weak_shell_contrast"
    if sigma_topo < -0.02:
        return "background_dominated"
    return "near_balanced"


def save_zone_means_bar(
    wise_name: str,
    I_inner: Optional[float],
    I_shell: Optional[float],
    I_bg: Optional[float],
    out_dir: Path,
) -> None:
    values = [
        float("nan") if I_inner is None else I_inner,
        float("nan") if I_shell is None else I_shell,
        float("nan") if I_bg is None else I_bg,
    ]
    labels = ["Inner", "Shell", "Background"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_ylabel("Mean observational intensity")
    ax.set_title(f"{wise_name} zone means")
    fig.tight_layout()
    fig.savefig(out_dir / f"{sanitize_wise_name(wise_name)}_zone_means_bar.png", dpi=180)
    plt.close(fig)


def save_radial_profile_plot(
    wise_name: str,
    profile: pd.DataFrame,
    zone: TargetZones,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(profile["radius_arcsec"], profile["intensity"])

    for x in [zone.inner_outer_arcsec, zone.shell_outer_arcsec, zone.bg_outer_arcsec]:
        ax.axvline(x, linestyle="-", linewidth=1.1)
    for x in [zone.shell_inner_arcsec, zone.bg_inner_arcsec]:
        ax.axvline(x, linestyle="--", linewidth=1.1)

    ax.set_xlabel("Radius (arcsec)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"{wise_name} radial profile")
    fig.tight_layout()
    fig.savefig(out_dir / f"{sanitize_wise_name(wise_name)}_radial_profile.png", dpi=180)
    plt.close(fig)


def save_topological_summary_txt(
    wise_name: str,
    zone: TargetZones,
    profile_path: Optional[Path],
    I_inner: Optional[float],
    I_shell: Optional[float],
    I_bg: Optional[float],
    I_ref: Optional[float],
    sigma_topo: Optional[float],
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
        "means",
        f"I_inner: {'' if I_inner is None else I_inner}",
        f"I_shell: {'' if I_shell is None else I_shell}",
        f"I_bg: {'' if I_bg is None else I_bg}",
        f"I_ref: {'' if I_ref is None else I_ref}",
        f"sigma_topo: {'' if sigma_topo is None else sigma_topo}",
        "",
        f"interpretation_flag: {flag}",
    ]
    (out_dir / f"{sanitize_wise_name(wise_name)}_topological_summary.txt").write_text(
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

    I_inner = None
    I_shell = None
    I_bg = None
    I_ref = None
    sigma_topo = None

    if profile_path is not None:
        profile = load_profile_csv(profile_path)
        I_inner = area_weighted_mean_from_profile(profile, zone.inner_inner_arcsec, zone.inner_outer_arcsec)
        I_shell = area_weighted_mean_from_profile(profile, zone.shell_inner_arcsec, zone.shell_outer_arcsec)
        I_bg = area_weighted_mean_from_profile(profile, zone.bg_inner_arcsec, zone.bg_outer_arcsec)

        if I_inner is not None and I_bg is not None:
            I_ref = 0.5 * (I_inner + I_bg)

        if I_shell is not None and I_ref is not None:
            sigma_topo = (I_shell - I_ref) / (I_ref + EPS)

        save_radial_profile_plot(wise_name, profile, zone, target_dir)

    shell_minus_bg = None if (I_shell is None or I_bg is None) else (I_shell - I_bg)
    shell_minus_inner = None if (I_shell is None or I_inner is None) else (I_shell - I_inner)
    flag = classify_sigma(sigma_topo, shell_minus_bg)

    save_zone_means_bar(wise_name, I_inner, I_shell, I_bg, target_dir)
    save_topological_summary_txt(
        wise_name=wise_name,
        zone=zone,
        profile_path=profile_path,
        I_inner=I_inner,
        I_shell=I_shell,
        I_bg=I_bg,
        I_ref=I_ref,
        sigma_topo=sigma_topo,
        flag=flag,
        out_dir=target_dir,
    )

    return {
        "wise_name": wise_name,
        "radius_arcsec": zone.radius_arcsec,
        "I_inner": I_inner,
        "I_shell": I_shell,
        "I_bg": I_bg,
        "I_ref": I_ref,
        "shell_minus_bg": shell_minus_bg,
        "shell_minus_inner": shell_minus_inner,
        "sigma_topo": sigma_topo,
        "interpretation_flag": flag,
        "profile_file": str(profile_path) if profile_path else "",
    }


def write_manifest(project_root: Path, out_dir: Path, df: pd.DataFrame) -> None:
    lines = [
        "Validation of Structural Contrast Baseline",
        "Topological pipeline manifest",
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
        "This pipeline reads the final input directly from the input layer.",
        "It re-creates the standard inner/shell/background zones and computes sigma_topo from observational zone means when a radial profile CSV is available.",
    ]
    (out_dir / "run_manifest.txt").write_text("\n".join(lines), encoding="utf-8")


def write_run_summary(out_dir: Path, summary_df: pd.DataFrame) -> None:
    rows = [
        {"metric": "target_count", "value": len(summary_df)},
        {"metric": "targets_with_profile", "value": int(summary_df["profile_file"].astype(str).ne("").sum())},
        {"metric": "targets_with_sigma", "value": int(summary_df["sigma_topo"].notna().sum())},
        {"metric": "shell_enhanced_count", "value": int((summary_df["interpretation_flag"] == "shell_enhanced").sum())},
        {"metric": "weak_shell_contrast_count", "value": int((summary_df["interpretation_flag"] == "weak_shell_contrast").sum())},
        {"metric": "background_dominated_count", "value": int((summary_df["interpretation_flag"] == "background_dominated").sum())},
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
    summary_df.to_csv(out_dir / "topological_summary.csv", index=False, encoding="utf-8-sig")

    write_manifest(project_root, out_dir, df)
    write_run_summary(out_dir, summary_df)

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - topological pipeline v1")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print(f"Input file  : {project_root / INPUT_FILE}")
    print(f"Profile dir : {project_root / PROFILE_DIR}")
    print(f"Output dir  : {out_dir}")
    print("")
    print("[OK] Created:")
    print(out_dir / "topological_summary.csv")
    print(out_dir / "run_manifest.txt")
    print(out_dir / "run_summary.csv")
    print(out_dir / "targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
