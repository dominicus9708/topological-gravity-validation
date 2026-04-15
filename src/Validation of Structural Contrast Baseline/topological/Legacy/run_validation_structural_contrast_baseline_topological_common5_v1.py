#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
Topological pipeline for common5 final input (proxy-bridge aware)

Author: Kwon Dominicus
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

RESULTS_BASE = (
    Path("results")
    / "Validation of Structural Contrast Baseline"
    / "output"
    / "topological"
    / "wise_hii_common5"
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
        raise ValueError("Topological final input is missing required columns: " + ", ".join(missing))

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
        raise FileNotFoundError(f"Topological final input file not found: {path}")
    return pd.read_csv(path, low_memory=False)

def truthy(v) -> bool:
    if pd.isna(v):
        return False
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1","true","t","yes","y","verified","downloaded","ok","ready","api_xml_ready"}

def has_text(v) -> bool:
    return pd.notna(v) and str(v).strip() != ""

def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = [
        "wise_name", "catalog_class", "input_group", "priority_rank",
        "ra", "dec", "radius_arcsec", "fits_source_service", "fits_url",
        "mass_source_key", "mass_value_type", "proxy_kind"
    ]
    ensure_required_columns(df, required)

    df["wise_name"] = df["wise_name"].astype(str).str.strip()
    df["radius_arcsec_num"] = pd.to_numeric(df["radius_arcsec"], errors="coerce")
    df["priority_rank_num"] = pd.to_numeric(df["priority_rank"], errors="coerce")
    df["score_total_num"] = pd.to_numeric(df.get("score_total", pd.Series([None]*len(df))), errors="coerce")
    if df["wise_name"].duplicated().any():
        dup = df.loc[df["wise_name"].duplicated(), "wise_name"].tolist()
        raise ValueError("Duplicate wise_name entries found: " + ", ".join(dup))
    bad_radius = df.loc[df["radius_arcsec_num"].isna(), "wise_name"].tolist()
    if bad_radius:
        raise ValueError("Invalid radius_arcsec values for: " + ", ".join(map(str, bad_radius)))
    return df.sort_values(
        by=["priority_rank_num", "score_total_num", "radius_arcsec_num", "wise_name"],
        ascending=[True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)

def build_standard_zones(df: pd.DataFrame) -> List[TargetZones]:
    zones = []
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

def sphere_shell_volume(r_in: float, r_out: float) -> float:
    return (4.0 / 3.0) * math.pi * max(r_out**3 - r_in**3, 0.0)

def estimate_physical_radius_pc(row: pd.Series) -> tuple[Optional[float], str]:
    dist_kpc = pd.to_numeric(row.get("dist_kpc"), errors="coerce")
    radius_arcsec = pd.to_numeric(row.get("radius_arcsec"), errors="coerce")
    if pd.notna(dist_kpc) and pd.notna(radius_arcsec) and dist_kpc > 0 and radius_arcsec > 0:
        # small-angle physical radius in pc
        radius_pc = float(dist_kpc) * 1000.0 * (float(radius_arcsec) / 206265.0)
        return radius_pc, "small_angle_from_dist_kpc"
    return None, "missing_dist_kpc"

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

def derive_mass_proxy(row: pd.Series) -> tuple[Optional[float], str]:
    direct_mass = pd.to_numeric(row.get("mass_value_msun"), errors="coerce")
    if pd.notna(direct_mass) and direct_mass > 0:
        return float(direct_mass), "direct_mass_value_msun"

    log_nly = pd.to_numeric(row.get("log_nly"), errors="coerce")
    if pd.notna(log_nly):
        return float(10 ** (float(log_nly) - 49.0)), "radio_proxy_from_log_nly"

    spectral_type = str(row.get("spectral_type", "")).strip()
    if spectral_type:
        return 1.0, "spectral_type_proxy_unit_weight"

    radio_ok = str(row.get("radio_proxy_available", "")).strip().lower()
    if radio_ok in {"yes","true","1"}:
        proxy_val = pd.to_numeric(row.get("proxy_value"), errors="coerce")
        if pd.notna(proxy_val) and proxy_val > 0:
            return float(proxy_val), "radio_proxy_numeric"
        return 1.0, "radio_proxy_unit_weight"

    ion = str(row.get("ionizing_source_reference", "")).strip()
    if ion:
        return 1.0, "ionizing_source_unit_weight"

    return None, "no_usable_mass_proxy"

def classify_topological_status(row: pd.Series, mass_proxy: Optional[float], radius_scale_pc: Optional[float]) -> str:
    if not truthy(row.get("fits_downloadable")) and not truthy(row.get("fits_local_verified")) and not truthy(row.get("fits_image_plane_verified")):
        return "missing_fits_ready"
    if mass_proxy is None:
        return "observational_only_no_mass_proxy"
    if radius_scale_pc is None:
        return "observational_only_no_physical_radius"
    return "mass_volume_ready"

def save_mass_bridge_plot(wise_name: str, shell_mv_density: Optional[float], bg_mv_density: Optional[float], sigma_mass_volume: Optional[float], out_dir: Path) -> None:
    labels = ["MVdens_shell", "MVdens_bg", "sigma_mass_volume"]
    values = [
        np.nan if shell_mv_density is None else shell_mv_density,
        np.nan if bg_mv_density is None else bg_mv_density,
        np.nan if sigma_mass_volume is None else sigma_mass_volume,
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_title(f"{wise_name} common5 mass-volume bridge summary")
    fig.tight_layout()
    fig.savefig(out_dir / f"{sanitize_wise_name(wise_name)}_mass_bridge_profile.png", dpi=180)
    plt.close(fig)

def save_summary_txt(row: pd.Series, zone: TargetZones, mass_proxy: Optional[float], mass_proxy_kind: str,
                     radius_scale_pc: Optional[float], radius_scale_kind: str, shell_mv_density: Optional[float],
                     bg_mv_density: Optional[float], sigma_mass_volume: Optional[float], status: str,
                     out_dir: Path) -> None:
    wise_name = str(row["wise_name"])
    lines = [
        f"wise_name: {wise_name}",
        f"catalog_class: {row.get('catalog_class', '')}",
        f"input_group: {row.get('input_group', '')}",
        f"priority_rank: {row.get('priority_rank', '')}",
        f"hii_region_name: {row.get('hii_region_name', '')}",
        f"membership: {row.get('membership', '')}",
        f"region_bucket: {row.get('region_bucket', '')}",
        f"score_total: {row.get('score_total', '')}",
        "",
        "topological bridge input",
        f"mass_source_key: {row.get('mass_source_key', '')}",
        f"mass_value_type: {row.get('mass_value_type', '')}",
        f"proxy_kind: {row.get('proxy_kind', '')}",
        f"proxy_value: {row.get('proxy_value', '')}",
        f"proxy_value_unit: {row.get('proxy_value_unit', '')}",
        f"log_nly: {row.get('log_nly', '')}",
        f"spectral_type: {row.get('spectral_type', '')}",
        f"radio_proxy_available: {row.get('radio_proxy_available', '')}",
        f"ionizing_source_reference: {row.get('ionizing_source_reference', '')}",
        "",
        f"mass_proxy: {'' if mass_proxy is None else mass_proxy}",
        f"mass_proxy_kind: {mass_proxy_kind}",
        f"radius_scale_pc: {'' if radius_scale_pc is None else radius_scale_pc}",
        f"radius_scale_kind: {radius_scale_kind}",
        f"shell_mass_volume_density: {'' if shell_mv_density is None else shell_mv_density}",
        f"background_mass_volume_density: {'' if bg_mv_density is None else bg_mv_density}",
        f"sigma_mass_volume: {'' if sigma_mass_volume is None else sigma_mass_volume}",
        f"bridge_status: {status}",
    ]
    (out_dir / f"{sanitize_wise_name(wise_name)}_topological_common5_summary.txt").write_text("\n".join(lines), encoding="utf-8")

def compute_target_result(row: pd.Series, zone: TargetZones, target_dir: Path) -> dict:
    mass_proxy, mass_proxy_kind = derive_mass_proxy(row)
    radius_scale_pc, radius_scale_kind = estimate_physical_radius_pc(row)
    status = classify_topological_status(row, mass_proxy, radius_scale_pc)

    shell_mv_density = None
    bg_mv_density = None
    sigma_mass_volume = None

    if status == "mass_volume_ready":
        zone_pc = convert_zone_to_pc(zone, radius_scale_pc)
        shell_vol = sphere_shell_volume(zone_pc["shell_inner_pc"], zone_pc["shell_outer_pc"])
        bg_vol = sphere_shell_volume(zone_pc["bg_inner_pc"], zone_pc["bg_outer_pc"])
        if shell_vol > 0 and bg_vol > 0:
            # conservative structural weighting from available proxy kind
            shell_weight = 1.05
            bg_weight = 0.95
            pk = str(row.get("proxy_kind", "")).strip().lower()
            if pk == "log_nly":
                shell_weight = 1.15
                bg_weight = 0.90
            elif pk == "spectral_type":
                shell_weight = 1.10
                bg_weight = 0.92
            elif pk == "radio_continuum":
                shell_weight = 1.08
                bg_weight = 0.94
            shell_mv_density = mass_proxy * shell_weight / shell_vol
            bg_mv_density = mass_proxy * bg_weight / bg_vol
            sigma_mass_volume = (shell_mv_density - bg_mv_density) / (bg_mv_density + EPS)

    save_mass_bridge_plot(str(row["wise_name"]), shell_mv_density, bg_mv_density, sigma_mass_volume, target_dir)
    save_summary_txt(row, zone, mass_proxy, mass_proxy_kind, radius_scale_pc, radius_scale_kind,
                     shell_mv_density, bg_mv_density, sigma_mass_volume, status, target_dir)

    return {
        "wise_name": str(row["wise_name"]),
        "catalog_class": row.get("catalog_class", ""),
        "priority_rank": row.get("priority_rank", ""),
        "hii_region_name": row.get("hii_region_name", ""),
        "membership": row.get("membership", ""),
        "radius_arcsec": zone.radius_arcsec,
        "mass_source_key": row.get("mass_source_key", ""),
        "mass_value_type": row.get("mass_value_type", ""),
        "proxy_kind": row.get("proxy_kind", ""),
        "log_nly": row.get("log_nly", ""),
        "spectral_type": row.get("spectral_type", ""),
        "radio_proxy_available": row.get("radio_proxy_available", ""),
        "fits_downloadable": row.get("fits_downloadable", ""),
        "fits_local_verified": row.get("fits_local_verified", ""),
        "mass_proxy": mass_proxy,
        "mass_proxy_kind": mass_proxy_kind,
        "radius_scale_pc": radius_scale_pc,
        "radius_scale_kind": radius_scale_kind,
        "shell_mass_volume_density": shell_mv_density,
        "background_mass_volume_density": bg_mv_density,
        "sigma_mass_volume": sigma_mass_volume,
        "bridge_status": status,
    }

def write_manifest(project_root: Path, out_dir: Path, df: pd.DataFrame) -> None:
    lines = [
        "Validation of Structural Contrast Baseline",
        "Topological common5 pipeline manifest",
        "=" * 60,
        "",
        f"project_root: {project_root}",
        f"input_file: {project_root / INPUT_FILE}",
        f"output_dir: {out_dir}",
        "",
        f"target_count: {len(df)}",
        "",
        "meaning",
        "-" * 20,
        "This common5 topological pipeline reads the integrated final topological input directly.",
        "It does not use the legacy separate mass_bridge.csv file.",
        "It derives a conservative mass-volume bridge from direct mass / log_nly / spectral type / radio proxy fields already present in the final input.",
    ]
    (out_dir / "run_manifest.txt").write_text("\n".join(lines), encoding="utf-8")

def write_run_summary(out_dir: Path, summary_df: pd.DataFrame) -> None:
    rows = [
        {"metric": "target_count", "value": len(summary_df)},
        {"metric": "mass_volume_ready_count", "value": int((summary_df["bridge_status"] == "mass_volume_ready").sum())},
        {"metric": "observational_only_no_mass_proxy_count", "value": int((summary_df["bridge_status"] == "observational_only_no_mass_proxy").sum())},
        {"metric": "observational_only_no_physical_radius_count", "value": int((summary_df["bridge_status"] == "observational_only_no_physical_radius").sum())},
    ]
    pd.DataFrame(rows).to_csv(out_dir / "run_summary.csv", index=False, encoding="utf-8-sig")

def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()

    out_dir = create_timestamped_output_dir(project_root)
    df = normalize_input(load_final_input(project_root))
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
        results.append(compute_target_result(row, zone, target_dir))

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(out_dir / "topological_summary_common5.csv", index=False, encoding="utf-8-sig")

    write_manifest(project_root, out_dir, df)
    write_run_summary(out_dir, summary_df)

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - topological pipeline common5")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print(f"Input file  : {project_root / INPUT_FILE}")
    print(f"Output dir  : {out_dir}")
    print("[OK] Created common5 topological outputs")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
