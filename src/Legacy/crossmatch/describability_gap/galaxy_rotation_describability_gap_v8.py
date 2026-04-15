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
G_KPC_KM2S2_PER_MSUN = 4.30091e-6


@dataclass
class ModelConfig:
    segment_count: int = 5
    min_points_per_segment: int = 4
    h_fallback_kpc: float = 0.35
    dpi: int = 160

    # effective structural dimension
    r_ref_kpc: float = 0.25
    rho_ref_floor_msun_per_kpc3: float = 1.0e6
    w_geom: float = 1.0
    w_dens: float = 1.0
    dw_min: float = 2.0
    dw_max: float = 3.5

    # smoothed derivative source
    dw_smoothing_window: int = 3

    # finite-speed propagation
    c_info: float = 1.0
    tau_struct: float = 2.0
    ell_decay_factor: float = 1.0
    ell_min_factor: float = 1.0
    ell_max_fraction_of_rmax: float = 0.6
    propagation_source_power: float = 1.0
    causal_taper_width_fraction: float = 0.15

    # propagation-to-gap
    delta_floor: float = 0.0
    delta_ceiling: float = 1.0
    delta_p0_scale: float = 1.0

    tag_name: str = "describability_gap_v8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finite-speed causal describability-gap pipeline for galaxy rotation data."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing per-galaxy enriched CSV files.")
    parser.add_argument(
        "--output-root",
        default=r"data\derived\crossmatched",
        help=r"Root directory under which <timestamp_tag>/ will be created. Default: data\derived\crossmatched",
    )
    parser.add_argument("--glob", default="*.csv", help="Input file glob pattern. Default: *.csv")
    parser.add_argument("--config-json", default=None, help="Optional JSON file overriding ModelConfig fields.")
    parser.add_argument("--segment-count", type=int, default=None, help="Override radial segment count.")
    parser.add_argument("--galaxy-filter", nargs="*", default=None, help="Optional list of galaxy names to process.")
    parser.add_argument("--tag", default="describability_gap_v8", help="Tag appended to timestamp folder name.")
    return parser.parse_args()


def timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "")


def load_config(config_json: Optional[str], segment_count_override: Optional[int], tag_override: Optional[str]) -> ModelConfig:
    config = ModelConfig()
    if config_json:
        payload = json.loads(Path(config_json).read_text(encoding="utf-8"))
        for key, value in payload.items():
            if not hasattr(config, key):
                raise ValueError(f"Unknown config field: {key}")
            setattr(config, key, value)
    if segment_count_override is not None:
        config.segment_count = segment_count_override
    if tag_override:
        config.tag_name = tag_override
    return config


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


def infer_galaxy_name(file_path: Path, df: pd.DataFrame) -> str:
    for column in ("galaxy", "galaxy_name", "name"):
        if column in df.columns and df[column].notna().any():
            return str(df[column].dropna().iloc[0]).strip()
    return file_path.stem.replace("_structure_enriched", "")


def build_pointwise_thickness(df: pd.DataFrame, config: ModelConfig) -> np.ndarray:
    if "structure_2014_z0" in df.columns:
        z0 = pd.to_numeric(df["structure_2014_z0"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(z0).any() and np.nanmedian(z0) > 0:
            fallback = float(np.nanmedian(z0[np.isfinite(z0) & (z0 > 0)]))
            return np.where(np.isfinite(z0) & (z0 > 0), z0, fallback)
    existing = first_present_numeric(df, ["h_kpc", "h", "thickness_kpc", "disk_thickness_kpc"])
    if existing is not None:
        h = df[existing].to_numpy(dtype=float)
        h = np.where(np.isfinite(h) & (h > 0), h, np.nan)
        fallback = float(np.nanmedian(h[np.isfinite(h)])) if np.isfinite(h).any() else config.h_fallback_kpc
        return np.where(np.isfinite(h) & (h > 0), h, fallback)
    if "structure_2014_h" in df.columns and "structure_2014_z0_over_h" in df.columns:
        h_rad = pd.to_numeric(df["structure_2014_h"], errors="coerce").to_numpy(dtype=float)
        ratio = pd.to_numeric(df["structure_2014_z0_over_h"], errors="coerce").to_numpy(dtype=float)
        z0 = h_rad * ratio
        if np.isfinite(z0).any() and np.nanmedian(z0) > 0:
            fallback = float(np.nanmedian(z0[np.isfinite(z0) & (z0 > 0)]))
            return np.where(np.isfinite(z0) & (z0 > 0), z0, fallback)
    return np.full(len(df), float(config.h_fallback_kpc), dtype=float)


def prepare_dataframe(file_path: Path, config: ModelConfig) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(file_path)
    galaxy = infer_galaxy_name(file_path, df)
    r_col = require_numeric_column(df, ["r_kpc", "r", "radius", "rad_kpc"], "radius")
    v_obs_col = require_numeric_column(df, ["v_obs_kmps", "v_obs", "vobs", "v_rot"], "observed rotation")
    v_err_col = first_present_numeric(df, ["v_err_kmps", "v_err", "verr", "v_obs_err"])

    v_vis_col = first_present_numeric(df, ["v_vis_kms", "v_vis_kmps", "v_vis", "v_visible_kmps", "v_bar_kmps", "v_bar", "v_baryon"])
    if v_vis_col is None:
        v_gas_col = first_present_numeric(df, ["v_gas_kmps", "v_gas", "vgas"])
        v_disk_col = first_present_numeric(df, ["v_disk_kmps", "v_disk", "vdisk"])
        v_bul_col = first_present_numeric(df, ["v_bul_kmps", "v_bul", "v_bulge", "vbul"])
        missing_components = [name for name, col in [("v_gas", v_gas_col), ("v_disk", v_disk_col), ("v_bul", v_bul_col)] if col is None]
        if missing_components:
            raise ValueError("Could not determine visible rotation column. Missing components: " + ", ".join(missing_components))
        df["v_vis_kms"] = np.sqrt(
            np.clip(df[v_gas_col].to_numpy(dtype=float) ** 2, 0.0, None)
            + np.clip(df[v_disk_col].to_numpy(dtype=float) ** 2, 0.0, None)
            + np.clip(df[v_bul_col].to_numpy(dtype=float) ** 2, 0.0, None)
        )
        v_vis_col = "v_vis_kms"

    out = pd.DataFrame(
        {
            "galaxy": galaxy,
            "r_kpc": pd.to_numeric(df[r_col], errors="coerce"),
            "v_obs_kms": pd.to_numeric(df[v_obs_col], errors="coerce"),
            "v_vis_kms": pd.to_numeric(df[v_vis_col], errors="coerce"),
            "v_err_kms": pd.to_numeric(df[v_err_col], errors="coerce") if v_err_col is not None else np.nan,
        }
    )
    out["h_kpc"] = build_pointwise_thickness(df, config)
    out["source_path"] = str(file_path)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["r_kpc", "v_obs_kms", "v_vis_kms"]).sort_values("r_kpc").reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid rows after cleaning.")
    return out, galaxy


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


def estimate_segment_baryonic_mass_msun(r_mid_kpc: float, v_vis_kms: float) -> float:
    r_eff = max(float(r_mid_kpc), DEFAULT_EPS)
    v_eff = max(float(v_vis_kms), 0.0)
    return float((v_eff ** 2) * r_eff / G_KPC_KM2S2_PER_MSUN)


def compute_effective_dimension(r_left_kpc: float, r_right_kpc: float, r_mid_kpc: float, h_mid_kpc: float, mass_bar_msun: float, config: ModelConfig, rho_ref_msun_per_kpc3: float) -> Dict[str, float]:
    wedge_area_eff = max(0.5 * (r_right_kpc ** 2 - r_left_kpc ** 2), DEFAULT_EPS)
    volume_eff = max(h_mid_kpc * wedge_area_eff, DEFAULT_EPS)
    rho_eff = max(mass_bar_msun, DEFAULT_EPS) / volume_eff
    radial_log = max(math.log(1.0 + max(r_mid_kpc, DEFAULT_EPS) / max(config.r_ref_kpc, DEFAULT_EPS)), 1e-6)
    area_scale = max(math.sqrt(wedge_area_eff), DEFAULT_EPS)

    alpha_geom = 2.0 + math.log(1.0 + max(h_mid_kpc, DEFAULT_EPS) / area_scale) / radial_log
    alpha_dens = 2.0 + math.log(1.0 + rho_eff / max(rho_ref_msun_per_kpc3, DEFAULT_EPS)) / radial_log
    alpha_geom = float(np.clip(alpha_geom, config.dw_min, config.dw_max))
    alpha_dens = float(np.clip(alpha_dens, config.dw_min, config.dw_max))
    Dw = (config.w_geom * alpha_geom + config.w_dens * alpha_dens) / max(config.w_geom + config.w_dens, DEFAULT_EPS)
    Dw = float(np.clip(Dw, config.dw_min, config.dw_max))
    return {
        "segment_wedge_volume_eff_kpc3": volume_eff,
        "segment_density_eff_msun_per_kpc3": rho_eff,
        "alpha_geom_i": alpha_geom,
        "alpha_dens_i": alpha_dens,
        "Dw_i": Dw,
    }


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    n = arr.size
    if n == 0:
        return arr
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = float(np.mean(arr[lo:hi]))
    return out


def positive_median(arr: np.ndarray, fallback: float = 1.0) -> float:
    arr = np.asarray(arr, dtype=float)
    valid = arr[np.isfinite(arr) & (arr > 0)]
    if valid.size == 0:
        return float(fallback)
    return float(np.nanmedian(valid))


def sat_unit(arr: np.ndarray, scale: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    scale = max(float(scale), DEFAULT_EPS)
    arr = np.clip(arr, 0.0, None)
    return arr / (arr + scale)


def interpolate_segment_values(point_r: np.ndarray, seg_r: np.ndarray, seg_vals: np.ndarray) -> np.ndarray:
    point_r = np.asarray(point_r, dtype=float)
    seg_r = np.asarray(seg_r, dtype=float)
    seg_vals = np.asarray(seg_vals, dtype=float)
    if seg_r.size == 1:
        return np.full_like(point_r, float(seg_vals[0]), dtype=float)
    order = np.argsort(seg_r)
    xr = seg_r[order]
    yr = seg_vals[order]
    return np.interp(point_r, xr, yr, left=yr[0], right=yr[-1])


def causal_gate(radial_gap: float, horizon: float, taper_width: float) -> float:
    if radial_gap < 0:
        return 0.0
    if radial_gap <= horizon:
        return 1.0
    if taper_width <= DEFAULT_EPS:
        return 0.0
    excess = radial_gap - horizon
    if excess >= taper_width:
        return 0.0
    # smooth taper to avoid hard discontinuity
    return 0.5 * (1.0 + math.cos(math.pi * excess / taper_width))


def build_segment_table(df: pd.DataFrame, galaxy: str, config: ModelConfig) -> pd.DataFrame:
    r = df["r_kpc"].to_numpy(dtype=float)
    edges = choose_segment_edges(r, config.segment_count, config.min_points_per_segment)
    rows: List[Dict[str, float]] = []
    for idx in range(len(edges) - 1):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        mask = (r >= left) & (r < right) if idx < len(edges) - 2 else (r >= left) & (r <= right)
        if not np.any(mask):
            continue
        local = df.loc[mask].copy()
        seg_r = local["r_kpc"].to_numpy(dtype=float)
        r_left = float(np.min(seg_r))
        r_right = float(np.max(seg_r))
        r_mid = 0.5 * (r_left + r_right)
        h_mid = float(np.nanmean(local["h_kpc"].to_numpy(dtype=float)))
        v_vis_seg = float(np.nanmean(local["v_vis_kms"].to_numpy(dtype=float)))
        rows.append({
            "galaxy": galaxy,
            "segment_index": idx + 1,
            "r_left_kpc": r_left,
            "r_right_kpc": r_right,
            "r_mid_kpc": r_mid,
            "dr_kpc": max(r_right - r_left, DEFAULT_EPS),
            "h_i_kpc": h_mid,
            "v_obs_segment_kms": float(np.nanmean(local["v_obs_kms"].to_numpy(dtype=float))),
            "v_vis_segment_kms": v_vis_seg,
            "v_err_segment_kms": float(np.nanmean(local["v_err_kms"].to_numpy(dtype=float))) if np.isfinite(local["v_err_kms"]).any() else np.nan,
            "mass_bar_segment_msun": estimate_segment_baryonic_mass_msun(r_mid, v_vis_seg),
            "n_points": int(len(local)),
        })

    seg = pd.DataFrame(rows)
    if seg.empty:
        raise ValueError("No valid segments could be constructed.")

    wedge_seed = np.clip(0.5 * (seg["r_right_kpc"].to_numpy() ** 2 - seg["r_left_kpc"].to_numpy() ** 2) * np.clip(seg["h_i_kpc"].to_numpy(), DEFAULT_EPS, None), DEFAULT_EPS, None)
    rho_seed = np.clip(seg["mass_bar_segment_msun"].to_numpy(), DEFAULT_EPS, None) / wedge_seed
    rho_ref = max(float(np.nanmedian(rho_seed)), config.rho_ref_floor_msun_per_kpc3)
    seg["density_ref_gal_msun_per_kpc3"] = rho_ref
    extra = [compute_effective_dimension(float(row.r_left_kpc), float(row.r_right_kpc), float(row.r_mid_kpc), float(row.h_i_kpc), float(row.mass_bar_segment_msun), config, rho_ref) for row in seg.itertuples()]
    seg = pd.concat([seg.reset_index(drop=True), pd.DataFrame(extra)], axis=1)

    r_mid = seg["r_mid_kpc"].to_numpy(dtype=float)
    Dw = seg["Dw_i"].to_numpy(dtype=float)
    Dw_smooth = rolling_mean(Dw, config.dw_smoothing_window)
    dDw_dr = np.gradient(Dw_smooth, r_mid)
    source_raw = np.abs(dDw_dr)
    source_term = np.power(sat_unit(source_raw, positive_median(source_raw, fallback=0.1)), config.propagation_source_power)

    seg["Dw_smooth_i"] = Dw_smooth
    seg["dDw_dr_i"] = dDw_dr
    seg["propagation_source_i"] = source_term

    r_max = float(np.nanmax(r_mid)) if len(r_mid) else 1.0
    dr = np.clip(seg["dr_kpc"].to_numpy(dtype=float), DEFAULT_EPS, None)
    dr_med = float(np.nanmedian(dr))
    horizon = max(config.c_info * config.tau_struct, config.ell_min_factor * dr_med)
    ell_decay = max(config.ell_decay_factor * horizon, config.ell_min_factor * dr_med)
    ell_decay = min(ell_decay, max(config.ell_max_fraction_of_rmax * r_max, config.ell_min_factor * dr_med + DEFAULT_EPS))
    taper_width = max(config.causal_taper_width_fraction * horizon, 0.5 * dr_med)

    seg["causal_horizon_kpc"] = horizon
    seg["ell_info_kpc"] = ell_decay
    seg["c_info_gal"] = config.c_info
    seg["tau_struct_gal"] = config.tau_struct

    P_raw = np.zeros(len(seg), dtype=float)
    for i in range(len(seg)):
        total = 0.0
        for j in range(i + 1):
            radial_gap = max(r_mid[i] - r_mid[j], 0.0)
            gate = causal_gate(radial_gap, horizon, taper_width)
            if gate <= 0.0:
                continue
            attenuation = math.exp(-radial_gap / max(ell_decay, DEFAULT_EPS))
            shell_weight = dr[j] / max(r_max, DEFAULT_EPS)
            total += source_term[j] * attenuation * gate * shell_weight
        P_raw[i] = total

    p0 = config.delta_p0_scale * positive_median(P_raw, fallback=0.1)
    Delta = config.delta_floor + (config.delta_ceiling - config.delta_floor) * (P_raw / (P_raw + max(p0, DEFAULT_EPS)))

    seg["propagation_term_i"] = P_raw
    seg["delta_struct_i"] = np.clip(Delta, config.delta_floor, config.delta_ceiling)
    seg["DeltaD_i"] = seg["delta_struct_i"]
    return seg


def evaluate_pointwise(df: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    point = df.copy().reset_index(drop=True)
    point_r = point["r_kpc"].to_numpy(dtype=float)
    seg_r = seg["r_mid_kpc"].to_numpy(dtype=float)

    interp_cols = [
        "Dw_i",
        "Dw_smooth_i",
        "dDw_dr_i",
        "propagation_source_i",
        "propagation_term_i",
        "delta_struct_i",
        "DeltaD_i",
        "causal_horizon_kpc",
        "ell_info_kpc",
        "segment_density_eff_msun_per_kpc3",
        "segment_wedge_volume_eff_kpc3",
    ]
    for col in interp_cols:
        point[col] = interpolate_segment_values(point_r, seg_r, seg[col].to_numpy(dtype=float))

    seg_index_interp = interpolate_segment_values(point_r, seg_r, seg["segment_index"].to_numpy(dtype=float))
    point["segment_index"] = np.clip(np.rint(seg_index_interp), int(seg["segment_index"].min()), int(seg["segment_index"].max())).astype(int)
    point["v_bar_kms"] = point["v_vis_kms"]
    point["delta_v_i"] = np.abs(point["v_obs_kms"] - point["v_bar_kms"]) / np.clip(np.abs(point["v_obs_kms"]), DEFAULT_EPS, None)
    point["C_i"] = point["DeltaD_i"] * point["delta_v_i"]
    return point


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 3:
        return float("nan")
    if np.std(a[mask]) <= DEFAULT_EPS or np.std(b[mask]) <= DEFAULT_EPS:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def summarize_galaxy(point_df: pd.DataFrame, seg_df: pd.DataFrame) -> Dict[str, float]:
    deltaD = point_df["DeltaD_i"].to_numpy(dtype=float)
    delta_v = point_df["delta_v_i"].to_numpy(dtype=float)
    C = point_df["C_i"].to_numpy(dtype=float)
    return {
        "galaxy": str(point_df["galaxy"].iloc[0]),
        "n_points": int(len(point_df)),
        "n_segments": int(len(seg_df)),
        "mean_Dw_i": float(seg_df["Dw_i"].mean()),
        "mean_Dw_smooth_i": float(seg_df["Dw_smooth_i"].mean()),
        "mean_abs_dDw_dr_i": float(np.mean(np.abs(seg_df["dDw_dr_i"].to_numpy(dtype=float)))),
        "mean_DeltaD_i": float(np.mean(deltaD)),
        "std_DeltaD_i": float(np.std(deltaD)),
        "mean_delta_v_i": float(np.mean(delta_v)),
        "M_gal": float(np.mean(C)),
        "R_gal": safe_corr(deltaD, delta_v),
        "propagation_term_mean": float(np.mean(point_df["propagation_term_i"].to_numpy(dtype=float))),
        "propagation_source_mean": float(np.mean(point_df["propagation_source_i"].to_numpy(dtype=float))),
        "causal_horizon_kpc": float(seg_df["causal_horizon_kpc"].iloc[0]),
        "ell_info_kpc": float(seg_df["ell_info_kpc"].iloc[0]),
        "density_ref_gal_msun_per_kpc3": float(seg_df["density_ref_gal_msun_per_kpc3"].iloc[0]),
    }


def plot_one_galaxy(point_df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    d = point_df.sort_values("r_kpc").reset_index(drop=True)
    galaxy = str(d["galaxy"].iloc[0])
    x = d["r_kpc"].to_numpy(dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(f"{galaxy} describability-gap diagnostics", fontsize=14)

    ax = axes[0]
    has_err = np.isfinite(d["v_err_kms"]).any()
    if has_err:
        ax.errorbar(x, d["v_obs_kms"].to_numpy(dtype=float), yerr=d["v_err_kms"].to_numpy(dtype=float), fmt="o", markersize=4, linewidth=1, label="Observed")
    else:
        ax.plot(x, d["v_obs_kms"].to_numpy(dtype=float), marker="o", linewidth=1.2, label="Observed")
    ax.plot(x, d["v_bar_kms"].to_numpy(dtype=float), linewidth=1.5, label="Baryonic Newton baseline")
    ax.set_ylabel("Velocity (km/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(x, d["Dw_i"].to_numpy(dtype=float), marker="o", linewidth=1.2, label="Dw")
    ax.plot(x, d["Dw_smooth_i"].to_numpy(dtype=float), linewidth=1.5, label="Smoothed Dw")
    ax.plot(x, d["dDw_dr_i"].to_numpy(dtype=float), linewidth=1.2, label="dDw/dr")
    ax.set_ylabel("Structural profile")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2]
    ax.plot(x, d["propagation_source_i"].to_numpy(dtype=float), marker="o", linewidth=1.2, label="propagation source")
    ax.plot(x, d["propagation_term_i"].to_numpy(dtype=float), marker="o", linewidth=1.2, label="causal cumulative propagation")
    ax.plot(x, d["DeltaD_i"].to_numpy(dtype=float), marker="o", linewidth=1.2, label="describability gap")
    ax.set_ylabel("Causal propagation gap")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[3]
    ax.plot(x, d["DeltaD_i"].to_numpy(dtype=float), marker="o", linewidth=1.2, label="Describability gap")
    ax.plot(x, d["delta_v_i"].to_numpy(dtype=float), marker="o", linewidth=1.2, label="Rotation mismatch")
    ax.plot(x, d["C_i"].to_numpy(dtype=float), marker="o", linewidth=1.2, label="Consistency index")
    ax.set_xlabel("Radius (kpc)")
    ax.set_ylabel("Normalized indicators")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def merge_outputs(point_frames: List[pd.DataFrame], seg_frames: List[pd.DataFrame], summary_rows: List[Dict[str, float]], merged_dir: Path) -> None:
    merged_dir.mkdir(parents=True, exist_ok=True)
    if point_frames:
        pd.concat(point_frames, ignore_index=True).to_csv(merged_dir / "merged_pointwise_describability_gap_v8.csv", index=False)
    if seg_frames:
        pd.concat(seg_frames, ignore_index=True).to_csv(merged_dir / "merged_segment_describability_gap_v8.csv", index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(merged_dir / "merged_summary_describability_gap_v8.csv", index=False)


def write_run_metadata(run_dir: Path, config: ModelConfig, args: argparse.Namespace, input_dir: Path) -> None:
    payload = {
        "config": asdict(config),
        "input_dir": str(input_dir),
        "output_root": str(args.output_root),
        "glob": args.glob,
        "galaxy_filter": args.galaxy_filter,
        "tag": args.tag,
        "model_note": (
            "v8 adds explicit finite-speed causality to the derivative-driven propagation model. Rearrangement influence propagates outward only within a causal horizon c_info * tau_struct, with smooth taper and exponential decay."
        ),
        "minimal_story": {
            "Dw_i": "effective structural dimension per segment",
            "Dw_smooth_i": "smoothed structural profile",
            "dDw_dr_i": "continuous radial structural change used as rearrangement source",
            "causal_horizon_kpc": "c_info * tau_struct (minimum-adjusted)",
            "P_i": "outward cumulative attenuated sum of source from inner shells inside causal horizon",
            "DeltaD_i": "saturating function of P_i only",
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config_json, args.segment_count, args.tag)
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_root.mkdir(parents=True, exist_ok=True)

    run_dir = output_root / f"{timestamp_label()}_{config.tag_name}"
    per_galaxy_dir = run_dir / "per_galaxy"
    merged_dir = run_dir / "merged"
    plots_dir = run_dir / "plots"
    for p in (per_galaxy_dir, merged_dir, plots_dir):
        p.mkdir(parents=True, exist_ok=True)

    point_frames: List[pd.DataFrame] = []
    seg_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, float]] = []
    failure_rows: List[Dict[str, str]] = []

    allowed = None if not args.galaxy_filter else {normalize_name(x) for x in args.galaxy_filter}
    for file_path in sorted(input_dir.glob(args.glob)):
        if not file_path.is_file() or file_path.name.lower().startswith("read me"):
            continue
        try:
            df, galaxy = prepare_dataframe(file_path, config)
            if allowed is not None and normalize_name(galaxy) not in allowed:
                continue
            seg = build_segment_table(df, galaxy, config)
            point = evaluate_pointwise(df, seg)
            summary = summarize_galaxy(point, seg)

            point.to_csv(per_galaxy_dir / f"{galaxy}_pointwise_describability_gap_v8.csv", index=False)
            seg.to_csv(per_galaxy_dir / f"{galaxy}_segment_describability_gap_v8.csv", index=False)
            plot_one_galaxy(point, plots_dir / f"{galaxy}_describability_gap_v8.png", config.dpi)

            point_frames.append(point)
            seg_frames.append(seg)
            summary_rows.append(summary)
            print(f"[OK] {file_path.name}")
        except Exception as exc:
            failure_rows.append({"file": file_path.name, "error": str(exc)})
            print(f"[FAILED] {file_path.name} -> {exc}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(run_dir / "describability_gap_summary_v8.csv", index=False)
    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(run_dir / "describability_gap_failures_v8.csv", index=False)
    merge_outputs(point_frames, seg_frames, summary_rows, merged_dir)
    write_run_metadata(run_dir, config, args, input_dir)

    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    aggregate = {
        "n_success": len(summary_rows),
        "n_failed": len(failure_rows),
        "mean_M_gal": float(summary_df["M_gal"].mean()) if not summary_df.empty else float("nan"),
        "mean_R_gal": float(summary_df["R_gal"].mean()) if not summary_df.empty else float("nan"),
        "mean_propagation_term_mean": float(summary_df["propagation_term_mean"].mean()) if not summary_df.empty else float("nan"),
        "mean_abs_dDw_dr_i": float(summary_df["mean_abs_dDw_dr_i"].mean()) if not summary_df.empty else float("nan"),
        "mean_causal_horizon_kpc": float(summary_df["causal_horizon_kpc"].mean()) if not summary_df.empty else float("nan"),
    }
    (run_dir / "aggregate_metrics_v8.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
