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

DEFAULT_EPS = 1e-12
R0_KPC = 0.1
G_KPC_KM2S2_PER_MSUN = 4.30091e-6


# -----------------------------------------------------------------------------
# V8.5 formula-revised concept
#
# Input structure and output layout are intentionally preserved.
#
# Updated physical interpretation:
#   1) Each radial segment estimates an effective structural dimension Dw_i
#      from the vertical-plane geometry (thickness, radius) and baryonic mass,
#      following a discrete weighted-effective-dimension idea inspired by the
#      second paper.
#   2) Structural deviation sigma_i is not one-sided anymore. It is the mean
#      of the adjacent inter-segment mismatches, so each segment carries the
#      averaged structural contrast around it.
#   3) Structural coupling is computed separately from baryonic Newtonian
#      gravity. The coupling source is driven by adjacent-segment mismatch and
#      structural relaxation/coherence, reflecting the dynamical viewpoint of
#      the third paper.
#   4) A bounded observer/describability factor is introduced in Gamma_topo to
#      encode the possible mismatch between the internally experienced baryonic
#      structure and the externally reconstructed global description.
#
# Resulting chain:
#   geometry + baryonic mass -> Dw_i
#   averaged neighbor mismatch -> sigma_i
#   dynamic structural coupling -> info_source_i, I_struct_i
#   observer-gap weighted galaxy factor -> gamma_topo
#   g_topo + Newtonian baryonic contribution -> rotation comparison
# -----------------------------------------------------------------------------


@dataclass
class ModelConfig:
    segment_count: int = 5
    min_points_per_segment: int = 4
    h_fallback_kpc: float = 0.35
    flare_alpha: float = 0.0
    dpi: int = 160

    # informational source construction
    source_mode: str = "dynamic_structural"   # kept for compatibility; formula is now fixed to dynamic_structural
    hybrid_weight_sigma: float = 1.0
    hybrid_weight_dw: float = 1.0
    attenuation_lambda: float = 0.0

    # finite structural-information propagation scale
    c_info: float = 1.0
    tau_struct: float = 2.0
    ell_min_factor: float = 1.0
    ell_max_fraction_of_rmax: float = 0.6

    # galaxy-derived Gamma_topo parameters
    gamma0: float = 1.0e4
    gamma_mode: str = "structural_product"   # structural_product, inverse_sigma, coherence_weighted

    # normalization references for gamma law
    dw_ref: float = 0.03
    sigma_ref: float = 0.01
    rscale_ref: float = 2.5
    beta_ref: float = 0.95

    # exponents for structural_product
    gamma_exp_dw: float = 1.0
    gamma_exp_sigma: float = 1.0
    gamma_exp_rscale: float = 1.0
    gamma_exp_beta: float = 0.0

    # clipping for stability
    gamma_min: float = 1.0e-6
    gamma_max: float = 1.0e6


@dataclass
class GalaxyThickness:
    galaxy: str
    h0_kpc: float
    flare_alpha: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Topostructure galaxy rotation pipeline v8.5 with galaxy-derived Gamma_topo."
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
    parser.add_argument("--tag", default="topostructure_v8_5", help="Optional tag to append after timestamp folder name.")
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


def build_pointwise_thickness(df: pd.DataFrame, thickness: Optional[GalaxyThickness], config: ModelConfig) -> np.ndarray:
    existing = first_present_numeric(df, ["h_kpc", "h", "thickness_kpc", "disk_thickness_kpc"])
    r = df["r_kpc"].to_numpy(dtype=float)
    if existing is not None:
        h = df[existing].to_numpy(dtype=float)
        h = np.where(np.isfinite(h) & (h > 0), h, np.nan)
        if np.isfinite(h).any():
            fallback = np.nanmedian(h[np.isfinite(h)])
        else:
            fallback = thickness.h0_kpc if thickness is not None else config.h_fallback_kpc
        return np.where(np.isfinite(h) & (h > 0), h, fallback)

    h0 = thickness.h0_kpc if thickness is not None else config.h_fallback_kpc
    flare_alpha = thickness.flare_alpha if thickness is not None else config.flare_alpha
    return np.maximum(h0 * (1.0 + flare_alpha * np.maximum(r, 0.0)), 1e-4)


def prepare_dataframe(file_path: Path, thickness: Optional[GalaxyThickness], config: ModelConfig) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(file_path)
    galaxy = infer_galaxy_name(file_path, df)

    r_col = require_numeric_column(df, ["r_kpc", "r", "radius", "rad_kpc"], "radius")
    v_obs_col = require_numeric_column(df, ["v_obs_kmps", "v_obs", "vobs", "v_circ_obs", "v_rot"], "observed rotation")
    v_err_col = first_present_numeric(df, ["v_err_kmps", "v_err", "verr", "v_obs_err", "ev_obs"])

    v_vis_col = first_present_numeric(df, ["v_vis_kmps", "v_vis", "v_visible_kmps", "v_bar_kmps", "v_bar", "v_baryon", "v_baryonic", "v_bary", "v_tot_bar"])
    if v_vis_col is None:
        v_gas_col = first_present_numeric(df, ["v_gas_kmps", "v_gas", "vgas"])
        v_disk_col = first_present_numeric(df, ["v_disk_kmps", "v_disk", "vdisk", "v_stellar_disk"])
        v_bul_col = first_present_numeric(df, ["v_bul_kmps", "v_bul", "v_bulge", "vbul"])
        missing_components = [
            name for name, col in [("v_gas", v_gas_col), ("v_disk", v_disk_col), ("v_bulge", v_bul_col)] if col is None
        ]
        if missing_components:
            raise ValueError("Could not determine visible rotation column. Missing components: " + ", ".join(missing_components))
        df["v_vis_kmps"] = np.sqrt(
            np.clip(df[v_gas_col].to_numpy(dtype=float) ** 2, 0.0, None)
            + np.clip(df[v_disk_col].to_numpy(dtype=float) ** 2, 0.0, None)
            + np.clip(df[v_bul_col].to_numpy(dtype=float) ** 2, 0.0, None)
        )
        v_vis_col = "v_vis_kmps"

    keep_cols = [r_col, v_obs_col, v_vis_col]
    if v_err_col is not None:
        keep_cols.append(v_err_col)
    extra_cols = [c for c in df.columns if c not in keep_cols]
    df = df[keep_cols + extra_cols].copy()
    rename_map = {r_col: "r_kpc", v_obs_col: "v_obs_kms", v_vis_col: "v_vis_kms"}
    if v_err_col is not None:
        rename_map[v_err_col] = "v_err_kms"
    df = df.rename(columns=rename_map)
    if "v_err_kms" not in df.columns:
        df["v_err_kms"] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["r_kpc", "v_obs_kms", "v_vis_kms"]).copy()
    df = df[df["r_kpc"] > R0_KPC + DEFAULT_EPS].copy()
    df = df.sort_values("r_kpc").reset_index(drop=True)
    if len(df) < 4:
        raise ValueError("Not enough valid radial points after cleaning.")

    df["h_kpc"] = build_pointwise_thickness(df, thickness, config)
    return df, galaxy


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


def cumulative_with_attenuation(
    source: np.ndarray,
    dr: np.ndarray,
    radius: np.ndarray,
    attenuation_lambda: float,
    ell_info_kpc: float,
) -> np.ndarray:
    if len(source) == 0:
        return np.array([], dtype=float)

    radius_arr = np.asarray(radius, dtype=float)
    source_arr = np.asarray(source, dtype=float)
    dr_arr = np.asarray(dr, dtype=float)
    ell_eff = max(float(ell_info_kpc), DEFAULT_EPS)
    base_lambda = max(float(attenuation_lambda), 0.0)

    out = np.zeros_like(source_arr, dtype=float)
    for i in range(len(source_arr)):
        dist = np.abs(radius_arr[i] - radius_arr)
        kernel = np.exp(-dist / ell_eff)
        if base_lambda > 0.0:
            kernel *= np.exp(-base_lambda * dist)
        out[i] = float(np.sum(np.clip(source_arr * dr_arr, 0.0, None) * kernel))
    return out


def estimate_segment_baryonic_mass_msun(r_mid_kpc: float, v_vis_kms: float) -> float:
    r_eff = max(float(r_mid_kpc), DEFAULT_EPS)
    v_eff = max(float(v_vis_kms), 0.0)
    return float((v_eff ** 2) * r_eff / max(G_KPC_KM2S2_PER_MSUN, DEFAULT_EPS))


def compute_weighted_effective_dimension(
    r_mid_kpc: float,
    h_mid_kpc: float,
    mass_bar_msun: float,
    r_ref_kpc: float,
    mass_ref_msun: float,
) -> Tuple[float, float, float, float, float]:
    r_eff = max(float(r_mid_kpc), DEFAULT_EPS)
    h_eff = max(float(h_mid_kpc), DEFAULT_EPS)
    mass_eff = max(float(mass_bar_msun), DEFAULT_EPS)
    r_ref = max(float(r_ref_kpc), DEFAULT_EPS)
    mass_ref = max(float(mass_ref_msun), DEFAULT_EPS)

    radial_log = math.log(1.0 + r_eff / r_ref)
    radial_log = max(radial_log, 1e-6)

    alpha_geom = 2.0 + math.log(1.0 + h_eff / r_eff) / radial_log
    alpha_mass = 2.0 + math.log(1.0 + mass_eff / mass_ref) / radial_log

    alpha_geom = float(np.clip(alpha_geom, 2.0, 3.5))
    alpha_mass = float(np.clip(alpha_mass, 2.0, 3.5))

    w_geom = h_eff / (h_eff + r_eff)
    w_mass = mass_eff / (mass_eff + mass_ref)
    weight_sum = max(w_geom + w_mass, DEFAULT_EPS)
    dw = (w_geom * alpha_geom + w_mass * alpha_mass) / weight_sum
    dw = float(np.clip(dw, 2.0, 3.5))
    return dw, alpha_geom, alpha_mass, w_geom, w_mass


def build_segments_base(df: pd.DataFrame, galaxy: str, config: ModelConfig) -> pd.DataFrame:
    r = df["r_kpc"].to_numpy(dtype=float)
    edges = choose_segment_edges(r, config.segment_count, config.min_points_per_segment)
    r_max = float(np.max(r))
    r_min = float(np.min(r))
    r_scale = math.log(max(r_max / max(r_min, DEFAULT_EPS), 1.0 + DEFAULT_EPS))
    sigma_scale = 1.0 / max(r_scale, DEFAULT_EPS)
    r_ref_kpc = max(r_min, 0.25)

    provisional_rows: List[Dict[str, float]] = []
    for idx in range(len(edges) - 1):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        mask = (r >= left) & (r < right) if idx < len(edges) - 2 else (r >= left) & (r <= right)
        if not np.any(mask):
            continue
        local = df.loc[mask].copy()
        seg_r = local["r_kpc"].to_numpy(dtype=float)
        seg_v_obs = local["v_obs_kms"].to_numpy(dtype=float)
        seg_v_vis = local["v_vis_kms"].to_numpy(dtype=float)
        seg_v_err = local["v_err_kms"].to_numpy(dtype=float)
        seg_h = local["h_kpc"].to_numpy(dtype=float)

        r_left = float(seg_r.min())
        r_right = float(seg_r.max())
        r_mid = 0.5 * (r_left + r_right)
        dr = max(r_right - r_left, DEFAULT_EPS)
        h_mid = float(np.nanmean(seg_h))
        v_vis_seg = float(np.nanmean(seg_v_vis))
        mass_bar_seg = estimate_segment_baryonic_mass_msun(r_mid, v_vis_seg)

        provisional_rows.append(
            {
                "galaxy": galaxy,
                "segment_index": idx + 1,
                "r_left_kpc": r_left,
                "r_right_kpc": r_right,
                "r_mid_kpc": r_mid,
                "dr_kpc": dr,
                "h_i_kpc": h_mid,
                "mass_bar_segment_msun": mass_bar_seg,
                "v_vis_segment_kms": v_vis_seg,
                "v_obs_segment_kms": float(np.nanmean(seg_v_obs)),
                "v_err_segment_kms": float(np.nanmean(seg_v_err)) if np.isfinite(seg_v_err).any() else np.nan,
                "n_points": int(mask.sum()),
                "r_max_gal_kpc": r_max,
                "r_min_gal_kpc": r_min,
                "r_scale": r_scale,
                "sigma_scale": sigma_scale,
                "r_ref_kpc": r_ref_kpc,
            }
        )

    seg_table = pd.DataFrame(provisional_rows)
    if seg_table.empty:
        raise ValueError("No valid segments could be constructed.")

    mass_ref = float(np.nanmedian(np.clip(seg_table["mass_bar_segment_msun"].to_numpy(dtype=float), DEFAULT_EPS, None)))
    mass_ref = max(mass_ref, 1e6)
    seg_table["mass_ref_gal_msun"] = mass_ref

    dw_vals: List[float] = []
    alpha_geom_vals: List[float] = []
    alpha_mass_vals: List[float] = []
    w_geom_vals: List[float] = []
    w_mass_vals: List[float] = []
    for _, row in seg_table.iterrows():
        dw, alpha_geom, alpha_mass, w_geom, w_mass = compute_weighted_effective_dimension(
            r_mid_kpc=float(row["r_mid_kpc"]),
            h_mid_kpc=float(row["h_i_kpc"]),
            mass_bar_msun=float(row["mass_bar_segment_msun"]),
            r_ref_kpc=r_ref_kpc,
            mass_ref_msun=mass_ref,
        )
        dw_vals.append(dw)
        alpha_geom_vals.append(alpha_geom)
        alpha_mass_vals.append(alpha_mass)
        w_geom_vals.append(w_geom)
        w_mass_vals.append(w_mass)

    seg_table["alpha_geom_i"] = alpha_geom_vals
    seg_table["alpha_mass_i"] = alpha_mass_vals
    seg_table["weight_geom_i"] = w_geom_vals
    seg_table["weight_mass_i"] = w_mass_vals
    seg_table["Dw_i"] = dw_vals

    dw_arr = seg_table["Dw_i"].to_numpy(dtype=float)
    inter_segment_diff = np.abs(np.diff(dw_arr)) if len(dw_arr) > 1 else np.array([], dtype=float)
    sigma_vals: List[float] = []
    left_diff_vals: List[float] = []
    right_diff_vals: List[float] = []
    for i in range(len(seg_table)):
        local_diffs = []
        left_diff = float("nan")
        right_diff = float("nan")
        if i > 0:
            left_diff = float(inter_segment_diff[i - 1])
            local_diffs.append(left_diff)
        if i < len(seg_table) - 1:
            right_diff = float(inter_segment_diff[i])
            local_diffs.append(right_diff)
        sigma_i = float(np.mean(local_diffs)) if local_diffs else 0.0
        sigma_vals.append(sigma_i)
        left_diff_vals.append(left_diff)
        right_diff_vals.append(right_diff)

    seg_table["sigma_left_i"] = left_diff_vals
    seg_table["sigma_right_i"] = right_diff_vals
    seg_table["sigma_i"] = sigma_vals

    alpha_star = float(np.average(seg_table["Dw_i"].to_numpy(dtype=float), weights=np.clip(seg_table["mass_bar_segment_msun"].to_numpy(dtype=float), DEFAULT_EPS, None)))
    seg_table["alpha_star_gal"] = alpha_star
    seg_table["alpha_relax_i"] = np.abs(seg_table["Dw_i"] - alpha_star)
    seg_table["neighbor_gradient_i"] = seg_table["sigma_i"] / np.clip(seg_table["dr_kpc"], DEFAULT_EPS, None)

    mass_norm = np.sqrt(np.clip(seg_table["mass_bar_segment_msun"] / max(mass_ref, DEFAULT_EPS), DEFAULT_EPS, None))
    seg_table["mass_norm_i"] = mass_norm

    kappa_dyn = 1.0 / max(r_scale, DEFAULT_EPS)
    seg_table["kappa_dyn_gal"] = kappa_dyn
    seg_table["beta_i"] = np.exp(-kappa_dyn * seg_table["alpha_relax_i"] / np.clip(seg_table["sigma_i"] + seg_table["sigma_scale"], DEFAULT_EPS, None))
    seg_table["beta_i"] = np.clip(seg_table["beta_i"], 0.0, 1.0)

    source_core = mass_norm * seg_table["sigma_i"]
    seg_table["source_core_i"] = source_core
    seg_table["info_source_i"] = seg_table["source_core_i"] * seg_table["beta_i"]

    radius_arr = seg_table["r_mid_kpc"].to_numpy(dtype=float)
    dr_arr = seg_table["dr_kpc"].to_numpy(dtype=float)
    info_source_arr = seg_table["info_source_i"].to_numpy(dtype=float)

    median_dr = float(np.nanmedian(np.clip(dr_arr, DEFAULT_EPS, None)))
    cinfo_tau_kpc = max(float(config.c_info) * float(config.tau_struct), DEFAULT_EPS)
    ell_floor = max(float(config.ell_min_factor) * median_dr, DEFAULT_EPS)
    ell_ceiling = max(float(config.ell_max_fraction_of_rmax) * r_max, ell_floor)
    ell_info_kpc = float(np.clip(cinfo_tau_kpc, ell_floor, ell_ceiling))
    seg_table["c_info_gal"] = float(config.c_info)
    seg_table["tau_struct_gal"] = float(config.tau_struct)
    seg_table["ell_info_kpc"] = ell_info_kpc

    seg_table["I_struct_i"] = cumulative_with_attenuation(
        info_source_arr,
        dr_arr,
        radius_arr,
        config.attenuation_lambda,
        ell_info_kpc,
    )
    return seg_table


def derive_gamma(seg_table: pd.DataFrame, config: ModelConfig) -> Tuple[float, Dict[str, float]]:
    mean_dw_pert = float(np.mean(np.abs(seg_table["Dw_i"].to_numpy(dtype=float) - 2.0)))
    mean_sigma = float(np.mean(np.abs(seg_table["sigma_i"].to_numpy(dtype=float))))
    mean_beta = float(np.mean(seg_table["beta_i"].to_numpy(dtype=float)))
    r_scale = float(seg_table["r_scale"].iloc[0])

    mass_weights = np.clip(seg_table["mass_bar_segment_msun"].to_numpy(dtype=float), DEFAULT_EPS, None)
    dw_internal = float(np.average(seg_table["Dw_i"].to_numpy(dtype=float), weights=mass_weights))
    dw_external = float(np.mean(seg_table["Dw_i"].to_numpy(dtype=float)))
    observer_gap = abs(dw_internal - dw_external)
    observer_factor = 1.0 + observer_gap / max(mean_dw_pert + mean_sigma, DEFAULT_EPS)
    observer_factor = float(np.clip(observer_factor, 1.0, 5.0))

    if config.gamma_mode == "structural_product":
        f_dw = ((mean_dw_pert + DEFAULT_EPS) / max(config.dw_ref, DEFAULT_EPS)) ** config.gamma_exp_dw
        f_sigma = ((mean_sigma + DEFAULT_EPS) / max(config.sigma_ref, DEFAULT_EPS)) ** config.gamma_exp_sigma
        f_r = ((r_scale + DEFAULT_EPS) / max(config.rscale_ref, DEFAULT_EPS)) ** config.gamma_exp_rscale
        f_beta = ((mean_beta + DEFAULT_EPS) / max(config.beta_ref, DEFAULT_EPS)) ** config.gamma_exp_beta
        gamma_raw = config.gamma0 * f_dw * f_sigma * f_r * f_beta * observer_factor
        factors = {
            "mean_dw_pert": mean_dw_pert,
            "mean_sigma": mean_sigma,
            "mean_beta": mean_beta,
            "r_scale": r_scale,
            "dw_internal": dw_internal,
            "dw_external": dw_external,
            "observer_gap": observer_gap,
            "observer_factor": observer_factor,
            "factor_dw": f_dw,
            "factor_sigma": f_sigma,
            "factor_rscale": f_r,
            "factor_beta": f_beta,
            "gamma_raw": gamma_raw,
        }
    elif config.gamma_mode == "inverse_sigma":
        gamma_raw = config.gamma0 * (max(config.sigma_ref, DEFAULT_EPS) / max(mean_sigma, DEFAULT_EPS)) * observer_factor
        factors = {
            "mean_dw_pert": mean_dw_pert,
            "mean_sigma": mean_sigma,
            "mean_beta": mean_beta,
            "r_scale": r_scale,
            "dw_internal": dw_internal,
            "dw_external": dw_external,
            "observer_gap": observer_gap,
            "observer_factor": observer_factor,
            "factor_inverse_sigma": gamma_raw / max(config.gamma0, DEFAULT_EPS),
            "gamma_raw": gamma_raw,
        }
    elif config.gamma_mode == "coherence_weighted":
        gamma_raw = config.gamma0 * (r_scale / max(config.rscale_ref, DEFAULT_EPS)) * (mean_beta / max(config.beta_ref, DEFAULT_EPS)) * observer_factor
        factors = {
            "mean_dw_pert": mean_dw_pert,
            "mean_sigma": mean_sigma,
            "mean_beta": mean_beta,
            "r_scale": r_scale,
            "dw_internal": dw_internal,
            "dw_external": dw_external,
            "observer_gap": observer_gap,
            "observer_factor": observer_factor,
            "factor_coherence": gamma_raw / max(config.gamma0, DEFAULT_EPS),
            "gamma_raw": gamma_raw,
        }
    else:
        raise ValueError(f"Unsupported gamma_mode: {config.gamma_mode}")

    gamma = float(np.clip(gamma_raw, config.gamma_min, config.gamma_max))
    factors["gamma_topo"] = gamma
    return gamma, factors


def evaluate_with_gamma(seg_table_base: pd.DataFrame, gamma: float, factors: Dict[str, float], df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    seg_table = seg_table_base.copy()
    seg_table["gamma_topo"] = gamma
    for key, value in factors.items():
        seg_table[key] = value
    seg_table["g_topo_km2s2_per_kpc"] = gamma * seg_table["I_struct_i"] / np.clip(np.square(seg_table["r_mid_kpc"]), DEFAULT_EPS, None)
    seg_table["v_struct_kms"] = np.sqrt(np.clip(seg_table["r_mid_kpc"] * seg_table["g_topo_km2s2_per_kpc"], 0.0, None))
    seg_table["v_topo_total_segment_kms"] = np.sqrt(np.clip(np.square(seg_table["v_vis_segment_kms"]) + np.square(seg_table["v_struct_kms"]), 0.0, None))

    point_rows: List[pd.DataFrame] = []
    for _, seg in seg_table.iterrows():
        mask = (df["r_kpc"] >= seg["r_left_kpc"]) & (df["r_kpc"] < seg["r_right_kpc"]) if int(seg["segment_index"]) < len(seg_table) else (df["r_kpc"] >= seg["r_left_kpc"]) & (df["r_kpc"] <= seg["r_right_kpc"])
        local = df.loc[mask].copy()
        if local.empty:
            continue
        local["galaxy"] = seg["galaxy"]
        local["segment_index"] = int(seg["segment_index"])
        for col in [
            "Dw_i",
            "sigma_i",
            "beta_i",
            "source_core_i",
            "info_source_i",
            "I_struct_i",
            "gamma_topo",
            "alpha_geom_i",
            "alpha_mass_i",
            "weight_geom_i",
            "weight_mass_i",
            "mass_bar_segment_msun",
            "mass_norm_i",
            "alpha_relax_i",
            "neighbor_gradient_i",
            "ell_info_kpc",
            "c_info_gal",
            "tau_struct_gal",
        ]:
            local[col] = float(seg[col])
        for col in factors.keys():
            local[col] = float(factors[col])
        local["g_topo_km2s2_per_kpc"] = np.clip(float(gamma) * float(seg["I_struct_i"]) / np.clip(np.square(local["r_kpc"]), DEFAULT_EPS, None), 0.0, None)
        local["v_struct_kms"] = np.sqrt(np.clip(local["r_kpc"] * local["g_topo_km2s2_per_kpc"], 0.0, None))
        local["v_topo_total_kms"] = np.sqrt(np.clip(np.square(local["v_vis_kms"]) + np.square(local["v_struct_kms"]), 0.0, None))
        local["rotation_residual_topo_kms"] = local["v_obs_kms"] - local["v_topo_total_kms"]
        local["rotation_residual_vis_only_kms"] = local["v_obs_kms"] - local["v_vis_kms"]
        point_rows.append(local)

    point_table = pd.concat(point_rows, ignore_index=True) if point_rows else pd.DataFrame()
    if point_table.empty:
        raise ValueError("No point-wise output rows were generated.")

    metrics = {
        "galaxy": str(seg_table["galaxy"].iloc[0]),
        "gamma_topo": gamma,
        **factors,
        "n_points": int(len(point_table)),
        "n_segments": int(len(seg_table)),
        "mean_Dw_i": float(seg_table["Dw_i"].mean()),
        "mean_sigma_i": float(seg_table["sigma_i"].mean()),
        "mean_beta_i": float(seg_table["beta_i"].mean()),
        "mean_info_source_i": float(seg_table["info_source_i"].mean()),
        "mean_v_struct_kms": float(seg_table["v_struct_kms"].mean()),
        "rmse_vis_only_kms": safe_rmse(point_table["v_obs_kms"].to_numpy(), point_table["v_vis_kms"].to_numpy()),
        "rmse_topo_kms": safe_rmse(point_table["v_obs_kms"].to_numpy(), point_table["v_topo_total_kms"].to_numpy()),
        "mae_vis_only_kms": safe_mae(point_table["v_obs_kms"].to_numpy(), point_table["v_vis_kms"].to_numpy()),
        "mae_topo_kms": safe_mae(point_table["v_obs_kms"].to_numpy(), point_table["v_topo_total_kms"].to_numpy()),
        "reduced_chi2_vis_only": safe_reduced_chi2(point_table["v_obs_kms"].to_numpy(), point_table["v_vis_kms"].to_numpy(), point_table["v_err_kms"].to_numpy()),
        "reduced_chi2_topo": safe_reduced_chi2(point_table["v_obs_kms"].to_numpy(), point_table["v_topo_total_kms"].to_numpy(), point_table["v_err_kms"].to_numpy()),
    }
    metrics["rmse_improvement_kms"] = metrics["rmse_vis_only_kms"] - metrics["rmse_topo_kms"]
    metrics["mae_improvement_kms"] = metrics["mae_vis_only_kms"] - metrics["mae_topo_kms"]
    return seg_table, point_table, metrics


def plot_one_galaxy(point_df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    point_df = point_df.sort_values("r_kpc").reset_index(drop=True)
    galaxy = str(point_df["galaxy"].dropna().iloc[0])
    gamma = float(point_df["gamma_topo"].dropna().iloc[0])
    x = point_df["r_kpc"].to_numpy(dtype=float)
    v_obs = point_df["v_obs_kms"].to_numpy(dtype=float)
    v_vis = point_df["v_vis_kms"].to_numpy(dtype=float)
    v_topo = point_df["v_topo_total_kms"].to_numpy(dtype=float)
    resid_vis = point_df["rotation_residual_vis_only_kms"].to_numpy(dtype=float)
    resid_topo = point_df["rotation_residual_topo_kms"].to_numpy(dtype=float)
    dw = point_df["Dw_i"].to_numpy(dtype=float)
    sigma = point_df["sigma_i"].to_numpy(dtype=float)
    beta = point_df["beta_i"].to_numpy(dtype=float)
    v_struct = point_df["v_struct_kms"].to_numpy(dtype=float)
    has_err = "v_err_kms" in point_df.columns and np.isfinite(point_df["v_err_kms"]).any()
    err = point_df["v_err_kms"].to_numpy(dtype=float) if has_err else None

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"{galaxy} Topostructure V8.5 Gamma={gamma:.3g}", fontsize=14)

    ax = axes[0]
    if has_err:
        ax.errorbar(x, v_obs, yerr=err, fmt="o", markersize=4, linewidth=1, label="Observed")
    else:
        ax.plot(x, v_obs, marker="o", linewidth=1.2, label="Observed")
    ax.plot(x, v_vis, linewidth=1.6, label="Visible-only")
    ax.plot(x, v_topo, linewidth=1.6, label="Visible + informational topology")
    ax.plot(x, v_struct, linewidth=1.2, label="Structural-only")
    ax.set_ylabel("Velocity (km/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.axhline(0.0, linewidth=1.0)
    ax.plot(x, resid_vis, marker="o", linewidth=1.2, label="Residual: obs - visible")
    ax.plot(x, resid_topo, marker="o", linewidth=1.2, label="Residual: obs - topo")
    ax.set_ylabel("Residual (km/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2]
    ax.plot(x, dw, marker="o", linewidth=1.2, label="Dw")
    ax.plot(x, sigma, marker="o", linewidth=1.2, label="sigma")
    ax.plot(x, beta, marker="o", linewidth=1.2, label="beta")
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
        pd.concat(pointwise_frames, ignore_index=True).to_csv(merged_dir / "merged_pointwise_topostructure_v8_5.csv", index=False)
    if segment_frames:
        pd.concat(segment_frames, ignore_index=True).to_csv(merged_dir / "merged_segment_topostructure_v8_5.csv", index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(merged_dir / "merged_summary_topostructure_v8_5.csv", index=False)


def write_run_metadata(run_dir: Path, config: ModelConfig, args: argparse.Namespace, input_dir: Path) -> None:
    payload = {
        "config": asdict(config),
        "input_dir": str(input_dir),
        "output_root": str(args.output_root),
        "glob": args.glob,
        "thickness_csv": args.thickness_csv,
        "galaxy_filter": args.galaxy_filter,
        "tag": args.tag,
        "model_note": "V8.5 formula-revised: input/output preserved, while Dw, sigma, dynamic structural coupling, and observer-gap-weighted Gamma_topo were updated.",
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
            raw_df = pd.read_csv(file_path)
            galaxy_name = infer_galaxy_name(file_path, raw_df)
            if galaxy_filter is not None and normalize_name(galaxy_name) not in galaxy_filter:
                continue
            thickness = thickness_map.get(normalize_name(galaxy_name))
            df, galaxy = prepare_dataframe(file_path, thickness, config)
            seg_base = build_segments_base(df, galaxy, config)
            gamma, factors = derive_gamma(seg_base, config)
            seg_eval, point_eval, metrics = evaluate_with_gamma(seg_base, gamma, factors, df)

            slug = galaxy.replace("/", "_").replace(" ", "_")
            seg_eval.to_csv(per_galaxy_dir / f"{slug}_segment_structure_v8_5.csv", index=False)
            point_eval.to_csv(per_galaxy_dir / f"{slug}_pointwise_topostructure_v8_5.csv", index=False)
            plot_one_galaxy(point_eval, plots_dir / f"{slug}_topostructure_v8_5.png", config.dpi)

            summary_rows.append(metrics)
            pointwise_frames.append(point_eval.assign(source_file=f"{slug}_pointwise_topostructure_v8_5.csv"))
            segment_frames.append(seg_eval.assign(source_file=f"{slug}_segment_structure_v8_5.csv"))
            print(f"[OK] {galaxy} -> gamma_topo={gamma:.3g} rmse_topo={metrics['rmse_topo_kms']:.3f} improvement={metrics['rmse_improvement_kms']:.3f}")
        except Exception as exc:
            failure_rows.append({"file": str(file_path), "error": str(exc)})
            print(f"[FAILED] {file_path.name} -> {exc}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(["rmse_improvement_kms", "rmse_topo_kms"], ascending=[False, True]).reset_index(drop=True)
        summary_df.to_csv(run_dir / "topostructure_summary_v8_5.csv", index=False)
        aggregate = {
            "galaxies_processed": int(len(summary_df)),
            "mean_gamma_topo": float(summary_df["gamma_topo"].mean()),
            "median_gamma_topo": float(summary_df["gamma_topo"].median()),
            "mean_rmse_improvement_kms": float(summary_df["rmse_improvement_kms"].mean()),
            "median_rmse_improvement_kms": float(summary_df["rmse_improvement_kms"].median()),
            "mean_v_struct_kms": float(summary_df["mean_v_struct_kms"].mean()),
        }
        (run_dir / "aggregate_metrics_v8_5.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(run_dir / "topostructure_failures_v8_5.csv", index=False)

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
