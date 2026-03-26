from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

G_KPC_KMS2_PER_MSUN = 4.30091e-6
DEFAULT_EPS = 1e-12


# -----------------------------------------------------------------------------
# V7 concept
#
# Keep the V6 structure-state pipeline, but simplify the coupling law.
#
# 1) Compute local scaling exponent alpha_i from shell structure.
# 2) Build effective structural dimension D_i from alpha_i.
# 3) Build two structural deviations:
#       sigma0_i = D_i - <D>_galaxy
#       sigma1_i = D_i - D_{i-1}
# 4) Define an effective structural mismatch:
#       DeltaD_eff_i = w0 * |sigma0_i| + w1 * |sigma1_i|
# 5) Define coupling with an exponential coherence law:
#       beta_i = beta0 * exp(-DeltaD_eff_i / b)
#    where b is the coherence / consistency allowance scale.
# 6) Keep the bounded independent translation term at acceleration level.
# -----------------------------------------------------------------------------


@dataclass
class ModelConfig:
    # Structural dimension mapping
    d_bg: float = 3.0
    c_alpha: float = 0.35
    alpha_clip: float = 3.0

    # Thickness / segmentation
    flare_alpha: float = 0.0
    segment_count: int = 5
    min_points_per_segment: int = 4
    h_fallback_kpc: float = 0.35
    h_ref_kpc: float = 0.35

    # Coherence coupling
    beta0: float = 2.5
    b: float = 0.30
    w_sigma0: float = 1.0
    w_sigma1: float = 1.0

    # Structure visibility / radial activation
    omega_density_power: float = 1.0
    omega_thickness_power: float = 1.0
    omega_radial_power: float = 0.5
    radial_activation_power: float = 1.0

    # Translation term
    translation_scale: float = 0.18
    translation_cap_factor: float = 0.85
    rho0_msun_per_kpc3: float = 1.0e8
    min_g_desc: float = 1e-10


@dataclass
class GalaxyThickness:
    galaxy: str
    h0_kpc: float
    flare_alpha: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Topostructure galaxy rotation pipeline v7 with exponential coherence coupling beta."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing per-galaxy enriched CSV files.")
    parser.add_argument(
        "--output-root",
        default=r"data\derived",
        help="Root directory under which crossmatched/<timestamp_tag>/ will be created. Default: data\\derived",
    )
    parser.add_argument("--thickness-csv", default=None, help="Optional metadata CSV with columns galaxy,h0_kpc[,flare_alpha].")
    parser.add_argument("--glob", default="*.csv", help="Input file glob pattern. Default: *.csv")
    parser.add_argument("--config-json", default=None, help="Optional JSON file overriding ModelConfig fields.")
    parser.add_argument("--segment-count", type=int, default=None, help="Override radial segment count.")
    parser.add_argument("--galaxy-filter", nargs="*", default=None, help="Optional list of galaxy names to process.")
    parser.add_argument("--tag", default="topostructure_v7", help="Optional tag to append after timestamp folder name.")
    parser.add_argument("--dpi", type=int, default=160, help="PNG DPI for diagnostic plots. Default: 160")
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "")


def timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_config(config_json: Optional[str], segment_count_override: Optional[int]) -> ModelConfig:
    config = ModelConfig()
    if config_json:
        with open(config_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for key, value in payload.items():
            if not hasattr(config, key):
                raise ValueError(f"Unknown config field: {key}")
            setattr(config, key, value)
    if segment_count_override is not None:
        config.segment_count = segment_count_override
    return config


def load_thickness_map(thickness_csv: Optional[str], config: ModelConfig) -> Dict[str, GalaxyThickness]:
    if not thickness_csv:
        return {}
    path = Path(thickness_csv)
    if not path.exists():
        raise FileNotFoundError(f"Thickness CSV not found: {thickness_csv}")
    df = pd.read_csv(path)
    required = {"galaxy", "h0_kpc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thickness CSV missing required columns: {sorted(missing)}")
    out: Dict[str, GalaxyThickness] = {}
    for _, row in df.iterrows():
        flare_alpha = float(row["flare_alpha"]) if "flare_alpha" in df.columns and pd.notna(row["flare_alpha"]) else config.flare_alpha
        out[normalize_name(row["galaxy"])] = GalaxyThickness(galaxy=str(row["galaxy"]), h0_kpc=float(row["h0_kpc"]), flare_alpha=flare_alpha)
    return out


def infer_galaxy_name(file_path: Path, df: pd.DataFrame) -> str:
    for column in ("galaxy", "galaxy_name", "name"):
        if column in df.columns and df[column].notna().any():
            return str(df[column].dropna().iloc[0]).strip()
    return file_path.stem.replace("_structure_enriched", "")


def require_numeric_column(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            return col
    raise ValueError(f"Required column for {label} not found. Tried: {list(candidates)}")


def first_present_numeric(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            return col
    return None


def prepare_dataframe(file_path: Path) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(file_path)
    galaxy = infer_galaxy_name(file_path, df)

    r_col = require_numeric_column(df, ["r_kpc", "r", "radius", "rad_kpc"], "radius")
    v_obs_col = require_numeric_column(df, ["v_obs_kmps", "v_obs", "vobs", "v_circ_obs", "v_rot"], "observed rotation")
    v_err_col = first_present_numeric(df, ["v_err_kmps", "v_err", "verr", "v_obs_err", "ev_obs"])

    v_bar_col = first_present_numeric(df, ["v_bar_kmps", "v_bar", "v_baryon", "v_baryonic", "v_bary", "v_tot_bar"])
    if v_bar_col is None:
        v_gas_col = first_present_numeric(df, ["v_gas_kmps", "v_gas", "vgas"])
        v_disk_col = first_present_numeric(df, ["v_disk_kmps", "v_disk", "vdisk", "v_stellar_disk"])
        v_bul_col = first_present_numeric(df, ["v_bul_kmps", "v_bul", "v_bulge", "vbul"])
        missing_components = [
            name for name, col in [("v_gas", v_gas_col), ("v_disk", v_disk_col), ("v_bulge", v_bul_col)] if col is None
        ]
        if missing_components:
            raise ValueError("Could not determine baryonic rotation column. Missing components: " + ", ".join(missing_components))
        df["v_bar_kmps"] = np.sqrt(
            np.clip(df[v_gas_col].to_numpy(dtype=float) ** 2, 0.0, None)
            + np.clip(df[v_disk_col].to_numpy(dtype=float) ** 2, 0.0, None)
            + np.clip(df[v_bul_col].to_numpy(dtype=float) ** 2, 0.0, None)
        )
        v_bar_col = "v_bar_kmps"

    keep_cols = [r_col, v_obs_col, v_bar_col]
    if v_err_col is not None:
        keep_cols.append(v_err_col)
    extra_cols = [c for c in df.columns if c not in keep_cols]
    df = df[keep_cols + extra_cols].copy()
    rename_map = {r_col: "r_kpc", v_obs_col: "v_obs_kms", v_bar_col: "v_bar_internal_kms"}
    if v_err_col is not None:
        rename_map[v_err_col] = "v_err_kms"
    df = df.rename(columns=rename_map)
    if "v_err_kms" not in df.columns:
        df["v_err_kms"] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["r_kpc", "v_obs_kms", "v_bar_internal_kms"]).copy()
    df = df[df["r_kpc"] > 0].copy()
    df = df.sort_values("r_kpc").reset_index(drop=True)
    if len(df) < 4:
        raise ValueError("Not enough valid radial points after cleaning.")
    return df, galaxy


def compute_enclosed_mass_proxy_msun(r_kpc: np.ndarray, v_bar_kms: np.ndarray) -> np.ndarray:
    return np.clip(r_kpc * np.square(v_bar_kms) / G_KPC_KMS2_PER_MSUN, 0.0, None)


def choose_segment_edges(r_kpc: np.ndarray, segment_count: int, min_points_per_segment: int) -> np.ndarray:
    n = len(r_kpc)
    limit = max(1, n // max(min_points_per_segment, 1))
    segment_count = max(1, min(segment_count, limit))
    if segment_count == 1:
        return np.array([r_kpc[0], r_kpc[-1]], dtype=float)
    edges = np.quantile(r_kpc, np.linspace(0.0, 1.0, segment_count + 1))
    edges[0] = r_kpc[0]
    edges[-1] = r_kpc[-1]
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.array([r_kpc[0], r_kpc[-1]], dtype=float)
    return edges


def radial_thickness_kpc(r_mid_kpc: float, thickness: Optional[GalaxyThickness], config: ModelConfig) -> float:
    h0 = thickness.h0_kpc if thickness is not None else config.h_fallback_kpc
    flare_alpha = thickness.flare_alpha if thickness is not None else config.flare_alpha
    return max(h0 * (1.0 + flare_alpha * max(r_mid_kpc, 0.0)), 1e-4)


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred)))) if len(y_true) else float("nan")


def safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")


def safe_reduced_chi2(y_true: np.ndarray, y_pred: np.ndarray, y_err: np.ndarray) -> float:
    mask = np.isfinite(y_err) & (y_err > 0)
    if not np.any(mask):
        return float("nan")
    resid = (y_true[mask] - y_pred[mask]) / y_err[mask]
    dof = max(len(resid) - 1, 1)
    return float(np.sum(np.square(resid)) / dof)


def build_segments_for_galaxy(galaxy: str, df: pd.DataFrame, thickness: Optional[GalaxyThickness], config: ModelConfig):
    r = df["r_kpc"].to_numpy(dtype=float)
    edges = choose_segment_edges(r, config.segment_count, config.min_points_per_segment)
    r_max = float(np.max(r)) if len(r) else 1.0
    segments: List[Dict[str, float]] = []
    point_rows: List[pd.DataFrame] = []

    for idx in range(len(edges) - 1):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        mask = (r >= left) & (r < right) if idx < len(edges) - 2 else (r >= left) & (r <= right)
        if not np.any(mask):
            continue
        local = df.loc[mask].copy()
        seg_r = local["r_kpc"].to_numpy(dtype=float)
        seg_v_bar = local["v_bar_internal_kms"].to_numpy(dtype=float)
        seg_v_obs = local["v_obs_kms"].to_numpy(dtype=float)
        seg_v_err = local["v_err_kms"].to_numpy(dtype=float)
        seg_m_enc = compute_enclosed_mass_proxy_msun(seg_r, seg_v_bar)

        r_left = float(seg_r.min())
        r_right = float(seg_r.max())
        r_mid = 0.5 * (r_left + r_right)
        dr = max(r_right - r_left, DEFAULT_EPS)
        h_i = radial_thickness_kpc(r_mid, thickness, config)
        volume = max(2.0 * math.pi * r_mid * dr * h_i, DEFAULT_EPS)
        m_inner = float(seg_m_enc[0])
        m_outer = float(seg_m_enc[-1])
        m_shell = max(m_outer - m_inner, DEFAULT_EPS)
        mu_shell = m_shell / volume

        segments.append(
            {
                "galaxy": galaxy,
                "segment_index": idx + 1,
                "r_left_kpc": r_left,
                "r_right_kpc": r_right,
                "r_mid_kpc": r_mid,
                "dr_kpc": dr,
                "h_i_kpc": h_i,
                "volume_kpc3": volume,
                "m_enc_inner_msun": m_inner,
                "m_enc_outer_msun": m_outer,
                "m_shell_msun": m_shell,
                "mu_shell_msun_per_kpc3": mu_shell,
                "v_bar_internal_segment_kms": float(np.nanmean(seg_v_bar)),
                "v_obs_segment_kms": float(np.nanmean(seg_v_obs)),
                "v_err_segment_kms": float(np.nanmean(seg_v_err)) if np.isfinite(seg_v_err).any() else np.nan,
                "n_points": int(mask.sum()),
                "r_max_gal_kpc": r_max,
            }
        )

    seg_table = pd.DataFrame(segments)
    if seg_table.empty:
        raise ValueError("No valid segments could be constructed.")

    # Local scaling exponent alpha_i from neighboring shell measure changes
    alpha_vals: List[float] = []
    mu = seg_table["mu_shell_msun_per_kpc3"].to_numpy(dtype=float)
    r_mid_arr = seg_table["r_mid_kpc"].to_numpy(dtype=float)
    n_seg = len(seg_table)
    for i in range(n_seg):
        if n_seg == 1:
            alpha_vals.append(0.0)
            continue
        if i == 0:
            j0, j1 = 0, 1
        else:
            j0, j1 = i - 1, i
        num = math.log(max(mu[j1], DEFAULT_EPS)) - math.log(max(mu[j0], DEFAULT_EPS))
        den = math.log(max(r_mid_arr[j1], DEFAULT_EPS)) - math.log(max(r_mid_arr[j0], DEFAULT_EPS))
        alpha_vals.append(num / den if abs(den) > DEFAULT_EPS else 0.0)
    seg_table["alpha_i"] = np.clip(alpha_vals, -config.alpha_clip, config.alpha_clip)
    seg_table["d_i"] = config.d_bg + config.c_alpha * seg_table["alpha_i"]

    total_shell_mass = float(seg_table["m_shell_msun"].sum())
    d_gal_mean = float(np.average(seg_table["d_i"], weights=seg_table["m_shell_msun"])) if total_shell_mass > 0 else float(seg_table["d_i"].mean())
    seg_table["d_gal_mean"] = d_gal_mean
    seg_table["sigma0_i"] = seg_table["d_i"] - d_gal_mean

    sigma1_vals: List[float] = []
    dvals = seg_table["d_i"].to_numpy(dtype=float)
    for i in range(n_seg):
        if n_seg == 1:
            sigma1_vals.append(0.0)
        elif i == 0:
            sigma1_vals.append(dvals[1] - dvals[0])
        else:
            sigma1_vals.append(dvals[i] - dvals[i - 1])
    seg_table["sigma1_i"] = sigma1_vals

    seg_table["delta_d_eff_i"] = config.w_sigma0 * np.abs(seg_table["sigma0_i"]) + config.w_sigma1 * np.abs(seg_table["sigma1_i"])
    seg_table["beta_i"] = config.beta0 * np.exp(-seg_table["delta_d_eff_i"] / max(config.b, DEFAULT_EPS))

    density_ratio = np.clip(seg_table["mu_shell_msun_per_kpc3"] / (seg_table["mu_shell_msun_per_kpc3"] + config.rho0_msun_per_kpc3), 0.0, 1.0)
    thickness_ratio = np.clip(config.h_ref_kpc / (seg_table["h_i_kpc"] + config.h_ref_kpc), 0.0, 1.0)
    radial_ratio = np.clip(seg_table["r_mid_kpc"] / np.clip(seg_table["r_max_gal_kpc"], DEFAULT_EPS, None), 0.0, 1.0)
    seg_table["density_ratio_i"] = density_ratio
    seg_table["thickness_visibility_i"] = thickness_ratio
    seg_table["radial_visibility_i"] = radial_ratio
    seg_table["omega_i"] = 1.0 + (
        np.power(np.clip(density_ratio, DEFAULT_EPS, None), config.omega_density_power)
        * np.power(np.clip(thickness_ratio, DEFAULT_EPS, None), config.omega_thickness_power)
        * np.power(np.clip(radial_ratio, DEFAULT_EPS, None), config.omega_radial_power)
    )
    seg_table["a_i"] = np.clip(radial_ratio, 0.0, 1.0) ** config.radial_activation_power

    g_bar_internal = np.square(seg_table["v_bar_internal_segment_kms"]) / np.clip(seg_table["r_mid_kpc"], DEFAULT_EPS, None)
    seg_table["g_bar_internal_km2s2_per_kpc"] = g_bar_internal
    density_boost = np.sqrt(np.clip(seg_table["mu_shell_msun_per_kpc3"] / config.rho0_msun_per_kpc3, DEFAULT_EPS, None))
    seg_table["g_scale_i"] = config.translation_scale * (g_bar_internal + density_boost)

    # Independent bounded translation term; nonnegative magnitude from mismatch.
    delta_mag = np.tanh(seg_table["delta_d_eff_i"] / max(config.b, DEFAULT_EPS))
    seg_table["delta_mag_i"] = delta_mag
    g_trans_raw = seg_table["beta_i"] * seg_table["omega_i"] * seg_table["a_i"] * seg_table["g_scale_i"] * delta_mag
    seg_table["g_trans_raw_km2s2_per_kpc"] = g_trans_raw
    g_cap = config.translation_cap_factor * g_bar_internal
    seg_table["g_cap_i"] = g_cap
    seg_table["g_trans_km2s2_per_kpc"] = g_cap * np.tanh(g_trans_raw / np.clip(g_cap, DEFAULT_EPS, None))
    seg_table["g_desc_km2s2_per_kpc"] = np.clip(g_bar_internal + seg_table["g_trans_km2s2_per_kpc"], config.min_g_desc, None)
    seg_table["v_desc_segment_kms"] = np.sqrt(np.clip(seg_table["g_desc_km2s2_per_kpc"] * seg_table["r_mid_kpc"], 0.0, None))

    for _, seg in seg_table.iterrows():
        mask = (df["r_kpc"] >= seg["r_left_kpc"]) & (df["r_kpc"] < seg["r_right_kpc"]) if int(seg["segment_index"]) < len(seg_table) else (df["r_kpc"] >= seg["r_left_kpc"]) & (df["r_kpc"] <= seg["r_right_kpc"])
        local = df.loc[mask].copy()
        if local.empty:
            continue
        local["galaxy"] = galaxy
        local["segment_index"] = int(seg["segment_index"])
        local["h_i_kpc"] = float(seg["h_i_kpc"])
        local["alpha_i"] = float(seg["alpha_i"])
        local["d_i"] = float(seg["d_i"])
        local["d_gal_mean"] = float(seg["d_gal_mean"])
        local["sigma0_i"] = float(seg["sigma0_i"])
        local["sigma1_i"] = float(seg["sigma1_i"])
        local["delta_d_eff_i"] = float(seg["delta_d_eff_i"])
        local["beta_i"] = float(seg["beta_i"])
        local["density_ratio_i"] = float(seg["density_ratio_i"])
        local["thickness_visibility_i"] = float(seg["thickness_visibility_i"])
        local["radial_visibility_i"] = float(seg["radial_visibility_i"])
        local["omega_i"] = float(seg["omega_i"])
        local["a_i"] = float(seg["a_i"])
        local["g_scale_i"] = float(seg["g_scale_i"])
        local["delta_mag_i"] = float(seg["delta_mag_i"])
        local["g_trans_raw_km2s2_per_kpc"] = float(seg["g_trans_raw_km2s2_per_kpc"])
        local["g_cap_i"] = float(seg["g_cap_i"])
        local["g_trans_km2s2_per_kpc"] = float(seg["g_trans_km2s2_per_kpc"])
        local["g_bar_internal_km2s2_per_kpc"] = np.square(local["v_bar_internal_kms"]) / np.clip(local["r_kpc"], DEFAULT_EPS, None)
        local["g_desc_km2s2_per_kpc"] = np.clip(local["g_bar_internal_km2s2_per_kpc"] + float(seg["g_trans_km2s2_per_kpc"]), config.min_g_desc, None)
        local["v_desc_kms"] = np.sqrt(np.clip(local["g_desc_km2s2_per_kpc"] * local["r_kpc"], 0.0, None))
        local["rotation_residual_desc_kms"] = local["v_obs_kms"] - local["v_desc_kms"]
        local["rotation_residual_bar_only_kms"] = local["v_obs_kms"] - local["v_bar_internal_kms"]
        point_rows.append(local)

    point_table = pd.concat(point_rows, ignore_index=True) if point_rows else pd.DataFrame()
    if point_table.empty:
        raise ValueError("No point-wise output rows were generated.")

    metrics = {
        "galaxy": galaxy,
        "n_points": int(len(point_table)),
        "n_segments": int(len(seg_table)),
        "d_gal_mean": d_gal_mean,
        "sigma0_abs_max": float(np.max(np.abs(seg_table["sigma0_i"]))),
        "sigma1_abs_max": float(np.max(np.abs(seg_table["sigma1_i"]))),
        "mean_beta_i": float(seg_table["beta_i"].mean()),
        "mean_delta_d_eff_i": float(seg_table["delta_d_eff_i"].mean()),
        "mean_g_trans": float(seg_table["g_trans_km2s2_per_kpc"].mean()),
        "rmse_bar_only_kms": safe_rmse(point_table["v_obs_kms"].to_numpy(), point_table["v_bar_internal_kms"].to_numpy()),
        "rmse_desc_kms": safe_rmse(point_table["v_obs_kms"].to_numpy(), point_table["v_desc_kms"].to_numpy()),
        "mae_bar_only_kms": safe_mae(point_table["v_obs_kms"].to_numpy(), point_table["v_bar_internal_kms"].to_numpy()),
        "mae_desc_kms": safe_mae(point_table["v_obs_kms"].to_numpy(), point_table["v_desc_kms"].to_numpy()),
        "reduced_chi2_bar_only": safe_reduced_chi2(point_table["v_obs_kms"].to_numpy(), point_table["v_bar_internal_kms"].to_numpy(), point_table["v_err_kms"].to_numpy()),
        "reduced_chi2_desc": safe_reduced_chi2(point_table["v_obs_kms"].to_numpy(), point_table["v_desc_kms"].to_numpy(), point_table["v_err_kms"].to_numpy()),
    }
    metrics["rmse_improvement_kms"] = metrics["rmse_bar_only_kms"] - metrics["rmse_desc_kms"]
    metrics["mae_improvement_kms"] = metrics["mae_bar_only_kms"] - metrics["mae_desc_kms"]
    return seg_table, point_table, metrics


def plot_one_galaxy(point_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    point_df = point_df.sort_values("r_kpc").reset_index(drop=True)
    galaxy = str(point_df["galaxy"].dropna().iloc[0]) if "galaxy" in point_df.columns and point_df["galaxy"].notna().any() else output_path.stem
    x = point_df["r_kpc"].to_numpy(dtype=float)
    v_obs = point_df["v_obs_kms"].to_numpy(dtype=float)
    v_bar = point_df["v_bar_internal_kms"].to_numpy(dtype=float)
    v_desc = point_df["v_desc_kms"].to_numpy(dtype=float)
    resid_bar = point_df["rotation_residual_bar_only_kms"].to_numpy(dtype=float)
    resid_desc = point_df["rotation_residual_desc_kms"].to_numpy(dtype=float)
    alpha_i = point_df["alpha_i"].to_numpy(dtype=float)
    sigma0 = point_df["sigma0_i"].to_numpy(dtype=float)
    sigma1 = point_df["sigma1_i"].to_numpy(dtype=float)
    beta = point_df["beta_i"].to_numpy(dtype=float)
    g_trans = point_df["g_trans_km2s2_per_kpc"].to_numpy(dtype=float)
    has_err = "v_err_kms" in point_df.columns and np.isfinite(point_df["v_err_kms"]).any()
    err = point_df["v_err_kms"].to_numpy(dtype=float) if has_err else None

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"{galaxy} Topostructure V7 Diagnostics", fontsize=14)

    ax = axes[0]
    if has_err:
        ax.errorbar(x, v_obs, yerr=err, fmt="o", markersize=4, linewidth=1, label="Observed")
    else:
        ax.plot(x, v_obs, marker="o", linewidth=1.2, label="Observed")
    ax.plot(x, v_bar, linewidth=1.6, label="Internal baryonic proxy")
    ax.plot(x, v_desc, linewidth=1.6, label="Externally describable model")
    ax.set_ylabel("Velocity (km/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.axhline(0.0, linewidth=1.0)
    ax.plot(x, resid_bar, marker="o", linewidth=1.2, label="Residual: obs - internal baryon")
    ax.plot(x, resid_desc, marker="o", linewidth=1.2, label="Residual: obs - describable model")
    ax.set_ylabel("Residual (km/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2]
    ax.plot(x, alpha_i, marker="o", linewidth=1.2, label="alpha_i")
    ax.plot(x, sigma0, marker="o", linewidth=1.2, label="sigma0_i")
    ax.plot(x, sigma1, marker="o", linewidth=1.2, label="sigma1_i")
    ax.plot(x, beta, marker="o", linewidth=1.2, label="beta_i")
    ax.plot(x, g_trans, marker="o", linewidth=1.2, label="g_trans")
    ax.set_ylabel("Structural diagnostics")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_xlabel("Radius (kpc)")

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def merge_outputs(pointwise_frames: List[pd.DataFrame], segment_frames: List[pd.DataFrame], summary_rows: List[Dict[str, float]], merged_dir: Path) -> None:
    merged_dir.mkdir(parents=True, exist_ok=True)
    if pointwise_frames:
        pd.concat(pointwise_frames, ignore_index=True).to_csv(merged_dir / "merged_pointwise_topostructure_v7.csv", index=False)
    if segment_frames:
        pd.concat(segment_frames, ignore_index=True).to_csv(merged_dir / "merged_segment_topostructure_v7.csv", index=False)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values(["rmse_desc_kms", "rmse_improvement_kms"], ascending=[True, False]).reset_index(drop=True)
        summary_df.to_csv(merged_dir / "merged_summary_topostructure_v7.csv", index=False)


def write_run_metadata(run_dir: Path, config: ModelConfig, args: argparse.Namespace, input_dir: Path) -> None:
    payload = {
        "config": asdict(config),
        "input_dir": str(input_dir),
        "output_root": str(args.output_root),
        "glob": args.glob,
        "thickness_csv": args.thickness_csv,
        "galaxy_filter": args.galaxy_filter,
        "tag": args.tag,
        "dpi": args.dpi,
        "model_note": "V7 uses exponential coherence coupling beta_i = beta0 * exp(-DeltaD_eff_i / b).",
    }
    (run_dir / "run_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config_json, args.segment_count)
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    run_name = timestamp_label() if not args.tag else f"{timestamp_label()}_{args.tag}"
    run_dir = output_root / "crossmatched" / run_name
    per_galaxy_dir = run_dir / "per_galaxy"
    merged_dir = run_dir / "merged"
    plots_dir = run_dir / "plots"
    per_galaxy_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    thickness_map = load_thickness_map(args.thickness_csv, config)
    galaxy_filter = {normalize_name(x) for x in args.galaxy_filter} if args.galaxy_filter else None
    files = sorted(input_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched pattern {args.glob!r} in {input_dir}")

    summary_rows: List[Dict[str, float]] = []
    failure_rows: List[Dict[str, str]] = []
    pointwise_frames: List[pd.DataFrame] = []
    segment_frames: List[pd.DataFrame] = []

    for file_path in files:
        try:
            df, galaxy = prepare_dataframe(file_path)
            if galaxy_filter is not None and normalize_name(galaxy) not in galaxy_filter:
                continue
            thickness = thickness_map.get(normalize_name(galaxy))
            seg_table, point_table, metrics = build_segments_for_galaxy(galaxy, df, thickness, config)
            slug = galaxy.replace("/", "_").replace(" ", "_")
            seg_path = per_galaxy_dir / f"{slug}_segment_structure_v7.csv"
            point_path = per_galaxy_dir / f"{slug}_pointwise_topostructure_v7.csv"
            plot_path = plots_dir / f"{slug}_topostructure_v7_diagnostics.png"
            seg_table.to_csv(seg_path, index=False)
            point_table.to_csv(point_path, index=False)
            plot_one_galaxy(point_table, plot_path, args.dpi)
            summary_rows.append(metrics)
            pointwise_frames.append(point_table.assign(source_file=point_path.name))
            segment_frames.append(seg_table.assign(source_file=seg_path.name))
            print(f"[OK] {galaxy} -> segments={metrics['n_segments']} rmse_desc={metrics['rmse_desc_kms']:.3f}")
        except Exception as exc:
            failure_rows.append({"file": str(file_path), "error": str(exc)})
            print(f"[FAILED] {file_path.name} -> {exc}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values(["rmse_desc_kms", "rmse_improvement_kms"], ascending=[True, False]).reset_index(drop=True)
        summary_df.to_csv(run_dir / "topostructure_summary_v7.csv", index=False)
        aggregate = {
            "galaxies_processed": int(len(summary_df)),
            "mean_rmse_bar_only_kms": float(summary_df["rmse_bar_only_kms"].mean()),
            "mean_rmse_desc_kms": float(summary_df["rmse_desc_kms"].mean()),
            "mean_rmse_improvement_kms": float(summary_df["rmse_improvement_kms"].mean()),
            "median_rmse_improvement_kms": float(summary_df["rmse_improvement_kms"].median()),
            "mean_mae_bar_only_kms": float(summary_df["mae_bar_only_kms"].mean()),
            "mean_mae_desc_kms": float(summary_df["mae_desc_kms"].mean()),
            "mean_d_gal_mean": float(summary_df["d_gal_mean"].mean()),
            "mean_beta_i": float(summary_df["mean_beta_i"].mean()),
            "mean_delta_d_eff_i": float(summary_df["mean_delta_d_eff_i"].mean()),
            "mean_g_trans": float(summary_df["mean_g_trans"].mean()),
        }
        (run_dir / "aggregate_metrics_v7.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(run_dir / "topostructure_failures_v7.csv", index=False)

    merge_outputs(pointwise_frames, segment_frames, summary_rows, merged_dir)
    write_run_metadata(run_dir, config, args, input_dir)

    print("\nRun complete.")
    print(f"Run folder: {run_dir}")
    print(f"Per-galaxy folder: {per_galaxy_dir}")
    print(f"Merged folder: {merged_dir}")
    print(f"Plots folder: {plots_dir}")
    if summary_rows:
        print(f"Processed galaxies: {len(summary_rows)}")
    if failure_rows:
        print(f"Failures: {len(failure_rows)}")


if __name__ == "__main__":
    main()
