from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Topological-structure galaxy rotation pipeline
#
# Purpose
#   - Segment each galaxy radially.
#   - Build a 3D structure cell per segment using mass proxy + radial length + thickness.
#   - Estimate effective structural dimension D_i.
#   - Derive structural deviation sigma_i and coupling beta_i without double-counting.
#   - Build a topological effective rotation prediction for consistency-oriented testing.
#
# Important note
#   This script does NOT claim direct measurement of the rigorous D_w from the second paper.
#   It constructs an observational proxy for a segment-wise effective structural dimension.
# -----------------------------------------------------------------------------

G_KPC_KMS2_PER_MSUN = 4.30091e-6  # kpc (km/s)^2 / Msun
DEFAULT_EPS = 1e-12


@dataclass
class ModelConfig:
    d_bg: float = 3.0
    a: float = 0.12
    rho0_msun_per_kpc3: float = 1.0e8
    beta0: float = 0.85
    eta: float = 0.0
    psi_mode: str = "exp"  # one of: none, exp, power
    psi_r0_kpc: float = 5.0
    psi_m: float = 0.5
    min_inside_sqrt: float = 1e-6
    flare_alpha: float = 0.0
    segment_count: int = 5
    min_points_per_segment: int = 4
    h_fallback_kpc: float = 0.35
    beta_mode: str = "constant"  # one of: constant, sigma_modulated


@dataclass
class GalaxyThickness:
    galaxy: str
    h0_kpc: float
    flare_alpha: float = 0.0


@dataclass
class SegmentRecord:
    galaxy: str
    segment_index: int
    r_left_kpc: float
    r_right_kpc: float
    r_mid_kpc: float
    dr_kpc: float
    h_i_kpc: float
    volume_kpc3: float
    m_enc_inner_msun: float
    m_enc_outer_msun: float
    m_shell_msun: float
    rho_eff_msun_per_kpc3: float
    d_i: float
    d_gal_mean: float
    sigma_i: float
    sigma_norm: float
    beta_i: float
    psi_i: float
    g_bar_km2s2_per_kpc: float
    g_topo_km2s2_per_kpc: float
    g_eff_km2s2_per_kpc: float
    v_bar_segment_kms: float
    v_topo_segment_kms: float
    v_eff_segment_kms: float
    n_points: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment-based topological structure pipeline for galaxy rotation data."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing per-galaxy CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write segment results, per-point results, and summary tables.",
    )
    parser.add_argument(
        "--thickness-csv",
        default=None,
        help=(
            "Optional metadata CSV with columns: galaxy,h0_kpc[,flare_alpha]. "
            "If omitted, h_fallback_kpc is used for every galaxy."
        ),
    )
    parser.add_argument(
        "--glob",
        default="*.csv",
        help="Glob pattern for input files inside --input-dir. Default: *.csv",
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help="Optional JSON file overriding ModelConfig fields.",
    )
    parser.add_argument(
        "--segment-count",
        type=int,
        default=None,
        help="Override radial segment count.",
    )
    parser.add_argument(
        "--galaxy-filter",
        nargs="*",
        default=None,
        help="Optional list of galaxy names to process.",
    )
    return parser.parse_args()


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
    if config.segment_count < 1:
        raise ValueError("segment_count must be >= 1")
    return config


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "")


# -----------------------------------------------------------------------------
# Input preparation
# -----------------------------------------------------------------------------

def load_thickness_map(thickness_csv: Optional[str], config: ModelConfig) -> Dict[str, GalaxyThickness]:
    if not thickness_csv:
        return {}

    df = pd.read_csv(thickness_csv)
    required = {"galaxy", "h0_kpc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thickness CSV missing required columns: {sorted(missing)}")

    thickness_map: Dict[str, GalaxyThickness] = {}
    for _, row in df.iterrows():
        galaxy_key = normalize_name(row["galaxy"])
        h0_kpc = float(row["h0_kpc"])
        flare_alpha = float(row["flare_alpha"]) if "flare_alpha" in df.columns and pd.notna(row.get("flare_alpha")) else config.flare_alpha
        thickness_map[galaxy_key] = GalaxyThickness(galaxy=str(row["galaxy"]), h0_kpc=h0_kpc, flare_alpha=flare_alpha)
    return thickness_map


def infer_galaxy_name(file_path: Path, df: pd.DataFrame) -> str:
    for column in ("galaxy", "galaxy_name", "name"):
        if column in df.columns and df[column].notna().any():
            return str(df[column].dropna().iloc[0]).strip()
    stem = file_path.stem
    for suffix in ("_normalized", "-normalized", "_clean", "-clean"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def require_numeric_column(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            return col
    raise ValueError(f"Required column for {label} not found. Tried: {list(candidates)}")


def prepare_dataframe(file_path: Path) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(file_path)
    galaxy = infer_galaxy_name(file_path, df)

    r_col = require_numeric_column(df, ["r", "radius", "r_kpc", "rad_kpc"], "radius")
    v_obs_col = require_numeric_column(df, ["v_obs", "vobs", "v_circ_obs", "v_rot"], "observed rotation")

    v_bar_col: Optional[str] = None
    for candidate in ("v_bar", "v_baryon", "v_baryonic", "v_bary", "v_tot_bar"):
        if candidate in df.columns:
            df[candidate] = pd.to_numeric(df[candidate], errors="coerce")
            v_bar_col = candidate
            break

    if v_bar_col is None:
        component_candidates = {
            "v_gas": ["v_gas", "vgas"],
            "v_disk": ["v_disk", "vdisk", "v_stellar_disk"],
            "v_bulge": ["v_bul", "v_bulge", "vbul"],
        }
        component_cols: Dict[str, Optional[str]] = {key: None for key in component_candidates}
        for key, candidates in component_candidates.items():
            for candidate in candidates:
                if candidate in df.columns:
                    df[candidate] = pd.to_numeric(df[candidate], errors="coerce")
                    component_cols[key] = candidate
                    break
        if all(component_cols.values()):
            df["v_bar"] = np.sqrt(
                np.clip(df[component_cols["v_gas"]] ** 2, 0.0, None)
                + np.clip(df[component_cols["v_disk"]] ** 2, 0.0, None)
                + np.clip(df[component_cols["v_bulge"]] ** 2, 0.0, None)
            )
            v_bar_col = "v_bar"
        else:
            raise ValueError(
                "Could not determine baryonic rotation column. Provide v_bar or the full component set: v_gas, v_disk, v_bulge."
            )

    v_err_col: Optional[str] = None
    for candidate in ("v_err", "verr", "v_obs_err", "ev_obs"):
        if candidate in df.columns:
            df[candidate] = pd.to_numeric(df[candidate], errors="coerce")
            v_err_col = candidate
            break

    keep_cols = [r_col, v_obs_col, v_bar_col]
    if v_err_col:
        keep_cols.append(v_err_col)
    extra_cols = [c for c in df.columns if c not in keep_cols]
    df = df[keep_cols + extra_cols].copy()
    df = df.rename(columns={r_col: "r_kpc", v_obs_col: "v_obs_kms", v_bar_col: "v_bar_kms"})
    if v_err_col:
        df = df.rename(columns={v_err_col: "v_err_kms"})
    else:
        df["v_err_kms"] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["r_kpc", "v_obs_kms", "v_bar_kms"]).copy()
    df = df[df["r_kpc"] > 0].copy()
    df = df.sort_values("r_kpc").reset_index(drop=True)
    if len(df) < 4:
        raise ValueError("Not enough valid radial points after cleaning.")
    return df, galaxy


# -----------------------------------------------------------------------------
# Segment construction
# -----------------------------------------------------------------------------

def compute_enclosed_mass_proxy_msun(r_kpc: np.ndarray, v_bar_kms: np.ndarray) -> np.ndarray:
    return np.clip(r_kpc * np.square(v_bar_kms) / G_KPC_KMS2_PER_MSUN, 0.0, None)


def choose_segment_edges(r_kpc: np.ndarray, segment_count: int, min_points_per_segment: int) -> np.ndarray:
    n = len(r_kpc)
    segment_count = max(1, min(segment_count, max(1, n // max(1, min_points_per_segment))))
    if segment_count == 1:
        return np.array([r_kpc[0], r_kpc[-1]], dtype=float)

    quantiles = np.linspace(0.0, 1.0, segment_count + 1)
    edges = np.quantile(r_kpc, quantiles)
    edges[0] = r_kpc[0]
    edges[-1] = r_kpc[-1]
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.array([r_kpc[0], r_kpc[-1]], dtype=float)
    return edges


def radial_thickness_kpc(r_mid_kpc: float, thickness: GalaxyThickness, config: ModelConfig) -> float:
    flare_alpha = thickness.flare_alpha if thickness is not None else config.flare_alpha
    h0 = thickness.h0_kpc if thickness is not None else config.h_fallback_kpc
    h_i = h0 * (1.0 + flare_alpha * max(r_mid_kpc, 0.0))
    return max(h_i, 1e-4)


def psi_function(r_mid_kpc: float, config: ModelConfig) -> float:
    if config.psi_mode == "none":
        return 1.0
    if config.psi_mode == "exp":
        scale = max(config.psi_r0_kpc, DEFAULT_EPS)
        return 1.0 - math.exp(-r_mid_kpc / scale)
    if config.psi_mode == "power":
        scale = max(config.psi_r0_kpc, DEFAULT_EPS)
        return max(r_mid_kpc / scale, DEFAULT_EPS) ** config.psi_m
    raise ValueError(f"Unsupported psi_mode: {config.psi_mode}")


def compute_beta_i(beta0: float, eta: float, sigma_norm: float, beta_mode: str) -> float:
    if beta_mode == "constant":
        return beta0
    if beta_mode == "sigma_modulated":
        return beta0 * (1.0 + eta * sigma_norm)
    raise ValueError(f"Unsupported beta_mode: {beta_mode}")


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def safe_reduced_chi2(y_true: np.ndarray, y_pred: np.ndarray, y_err: np.ndarray, dof_floor: int = 1) -> float:
    mask = np.isfinite(y_err) & (y_err > 0)
    if not np.any(mask):
        return float("nan")
    resid = (y_true[mask] - y_pred[mask]) / y_err[mask]
    dof = max(len(resid) - dof_floor, 1)
    return float(np.sum(np.square(resid)) / dof)


def build_segments_for_galaxy(
    galaxy: str,
    df: pd.DataFrame,
    thickness: Optional[GalaxyThickness],
    config: ModelConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    r = df["r_kpc"].to_numpy(dtype=float)
    v_obs = df["v_obs_kms"].to_numpy(dtype=float)
    v_bar = df["v_bar_kms"].to_numpy(dtype=float)
    v_err = df["v_err_kms"].to_numpy(dtype=float)

    m_enc = compute_enclosed_mass_proxy_msun(r, v_bar)
    edges = choose_segment_edges(r, config.segment_count, config.min_points_per_segment)

    segments: List[Dict[str, float]] = []
    point_rows: List[pd.DataFrame] = []

    for idx in range(len(edges) - 1):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        if idx < len(edges) - 2:
            mask = (r >= left) & (r < right)
        else:
            mask = (r >= left) & (r <= right)
        if not np.any(mask):
            continue

        seg_df = df.loc[mask].copy()
        seg_r = seg_df["r_kpc"].to_numpy(dtype=float)
        seg_v_bar = seg_df["v_bar_kms"].to_numpy(dtype=float)
        seg_v_obs = seg_df["v_obs_kms"].to_numpy(dtype=float)
        seg_v_err = seg_df["v_err_kms"].to_numpy(dtype=float)
        seg_m_enc = compute_enclosed_mass_proxy_msun(seg_r, seg_v_bar)

        r_left = float(seg_r.min())
        r_right = float(seg_r.max())
        r_mid = 0.5 * (r_left + r_right)
        dr = max(r_right - r_left, DEFAULT_EPS)

        h_i = radial_thickness_kpc(r_mid, thickness, config)
        volume = max(2.0 * math.pi * r_mid * dr * h_i, DEFAULT_EPS)

        # Differential enclosed-mass proxy across the segment.
        m_inner = float(seg_m_enc[0])
        m_outer = float(seg_m_enc[-1])
        m_shell = max(m_outer - m_inner, DEFAULT_EPS)
        rho_eff = m_shell / volume

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
                "rho_eff_msun_per_kpc3": rho_eff,
                "v_bar_segment_kms": float(np.nanmean(seg_v_bar)),
                "v_obs_segment_kms": float(np.nanmean(seg_v_obs)),
                "v_err_segment_kms": float(np.nanmean(seg_v_err)) if np.isfinite(seg_v_err).any() else np.nan,
                "n_points": int(mask.sum()),
            }
        )

    seg_table = pd.DataFrame(segments)
    if seg_table.empty:
        raise ValueError("No valid segments could be constructed.")

    seg_table["d_i"] = config.d_bg + config.a * np.log1p(seg_table["rho_eff_msun_per_kpc3"] / max(config.rho0_msun_per_kpc3, DEFAULT_EPS))
    total_shell_mass = float(seg_table["m_shell_msun"].sum())
    if total_shell_mass <= 0:
        d_gal_mean = float(seg_table["d_i"].mean())
    else:
        d_gal_mean = float(np.average(seg_table["d_i"], weights=seg_table["m_shell_msun"]))
    seg_table["d_gal_mean"] = d_gal_mean
    seg_table["sigma_i"] = seg_table["d_i"] - d_gal_mean

    sigma_abs_max = float(np.max(np.abs(seg_table["sigma_i"]))) if len(seg_table) > 0 else 0.0
    sigma_scale = sigma_abs_max if sigma_abs_max > 0 else 1.0
    seg_table["sigma_norm"] = seg_table["sigma_i"] / sigma_scale
    seg_table["beta_i"] = [compute_beta_i(config.beta0, config.eta, s, config.beta_mode) for s in seg_table["sigma_norm"]]
    seg_table["psi_i"] = [psi_function(r_mid, config) for r_mid in seg_table["r_mid_kpc"]]

    seg_table["g_bar_km2s2_per_kpc"] = np.square(seg_table["v_bar_segment_kms"]) / np.clip(seg_table["r_mid_kpc"], DEFAULT_EPS, None)
    seg_table["g_topo_km2s2_per_kpc"] = (
        seg_table["beta_i"] * seg_table["sigma_i"] * seg_table["g_bar_km2s2_per_kpc"] * seg_table["psi_i"]
    )
    seg_table["g_eff_km2s2_per_kpc"] = seg_table["g_bar_km2s2_per_kpc"] + seg_table["g_topo_km2s2_per_kpc"]

    inside = 1.0 + seg_table["beta_i"] * seg_table["sigma_i"] * seg_table["psi_i"]
    inside = np.clip(inside, config.min_inside_sqrt, None)
    seg_table["v_eff_segment_kms"] = seg_table["v_bar_segment_kms"] * np.sqrt(inside)
    seg_table["v_topo_segment_kms"] = np.sqrt(np.clip(np.square(seg_table["v_eff_segment_kms"]) - np.square(seg_table["v_bar_segment_kms"]), 0.0, None))

    # Broadcast segment-wise quantities back to the original radial points.
    for _, seg in seg_table.iterrows():
        if int(seg["segment_index"]) < len(seg_table):
            mask = (df["r_kpc"] >= seg["r_left_kpc"]) & (df["r_kpc"] < seg["r_right_kpc"])
        else:
            mask = (df["r_kpc"] >= seg["r_left_kpc"]) & (df["r_kpc"] <= seg["r_right_kpc"])
        local = df.loc[mask].copy()
        if local.empty:
            continue
        local["galaxy"] = galaxy
        local["segment_index"] = int(seg["segment_index"])
        local["h_i_kpc"] = float(seg["h_i_kpc"])
        local["d_i"] = float(seg["d_i"])
        local["d_gal_mean"] = float(seg["d_gal_mean"])
        local["sigma_i"] = float(seg["sigma_i"])
        local["sigma_norm"] = float(seg["sigma_norm"])
        local["beta_i"] = float(seg["beta_i"])
        local["psi_i"] = float(seg["psi_i"])
        inside_local = max(1.0 + float(seg["beta_i"]) * float(seg["sigma_i"]) * float(seg["psi_i"]), config.min_inside_sqrt)
        local["v_eff_kms"] = local["v_bar_kms"] * math.sqrt(inside_local)
        local["g_bar_km2s2_per_kpc"] = np.square(local["v_bar_kms"]) / np.clip(local["r_kpc"], DEFAULT_EPS, None)
        local["g_eff_km2s2_per_kpc"] = np.square(local["v_eff_kms"]) / np.clip(local["r_kpc"], DEFAULT_EPS, None)
        local["rotation_residual_kms"] = local["v_obs_kms"] - local["v_eff_kms"]
        local["rotation_residual_bar_only_kms"] = local["v_obs_kms"] - local["v_bar_kms"]
        point_rows.append(local)

    point_table = pd.concat(point_rows, ignore_index=True) if point_rows else pd.DataFrame()
    if point_table.empty:
        raise ValueError("No point-wise output rows were generated.")

    metrics = {
        "galaxy": galaxy,
        "n_points": int(len(point_table)),
        "n_segments": int(len(seg_table)),
        "d_gal_mean": d_gal_mean,
        "sigma_abs_max": sigma_abs_max,
        "rmse_bar_only_kms": safe_rmse(point_table["v_obs_kms"].to_numpy(), point_table["v_bar_kms"].to_numpy()),
        "rmse_topo_kms": safe_rmse(point_table["v_obs_kms"].to_numpy(), point_table["v_eff_kms"].to_numpy()),
        "mae_bar_only_kms": safe_mae(point_table["v_obs_kms"].to_numpy(), point_table["v_bar_kms"].to_numpy()),
        "mae_topo_kms": safe_mae(point_table["v_obs_kms"].to_numpy(), point_table["v_eff_kms"].to_numpy()),
        "reduced_chi2_bar_only": safe_reduced_chi2(
            point_table["v_obs_kms"].to_numpy(), point_table["v_bar_kms"].to_numpy(), point_table["v_err_kms"].to_numpy()
        ),
        "reduced_chi2_topo": safe_reduced_chi2(
            point_table["v_obs_kms"].to_numpy(), point_table["v_eff_kms"].to_numpy(), point_table["v_err_kms"].to_numpy()
        ),
    }
    metrics["rmse_improvement_kms"] = metrics["rmse_bar_only_kms"] - metrics["rmse_topo_kms"]
    metrics["mae_improvement_kms"] = metrics["mae_bar_only_kms"] - metrics["mae_topo_kms"]
    return seg_table, point_table, metrics


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------

def write_run_metadata(output_dir: Path, config: ModelConfig, args: argparse.Namespace) -> None:
    payload = {
        "config": asdict(config),
        "input_dir": str(args.input_dir),
        "glob": args.glob,
        "thickness_csv": args.thickness_csv,
        "galaxy_filter": args.galaxy_filter,
    }
    (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config_json, args.segment_count)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thickness_map = load_thickness_map(args.thickness_csv, config)
    galaxy_filter = {normalize_name(x) for x in args.galaxy_filter} if args.galaxy_filter else None

    files = sorted(input_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched pattern {args.glob!r} in {input_dir}")

    summary_rows: List[Dict[str, float]] = []
    failure_rows: List[Dict[str, str]] = []

    for file_path in files:
        try:
            df, galaxy = prepare_dataframe(file_path)
            galaxy_key = normalize_name(galaxy)
            if galaxy_filter is not None and galaxy_key not in galaxy_filter:
                continue

            thickness = thickness_map.get(galaxy_key)
            seg_table, point_table, metrics = build_segments_for_galaxy(galaxy, df, thickness, config)

            galaxy_slug = galaxy.replace("/", "_").replace(" ", "_")
            seg_table.to_csv(output_dir / f"{galaxy_slug}_segment_structure.csv", index=False)
            point_table.to_csv(output_dir / f"{galaxy_slug}_pointwise_topostructure.csv", index=False)
            summary_rows.append(metrics)
            print(f"[OK] {galaxy} -> segments={metrics['n_segments']} rmse_topo={metrics['rmse_topo_kms']:.3f}")
        except Exception as exc:
            failure_rows.append({"file": str(file_path), "error": str(exc)})
            print(f"[FAILED] {file_path.name} -> {exc}")

    summary_df = pd.DataFrame(summary_rows)
    failure_df = pd.DataFrame(failure_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(["rmse_topo_kms", "rmse_improvement_kms"], ascending=[True, False]).reset_index(drop=True)
        summary_df.to_csv(output_dir / "topostructure_summary.csv", index=False)

        aggregate = {
            "galaxies_processed": int(len(summary_df)),
            "mean_rmse_bar_only_kms": float(summary_df["rmse_bar_only_kms"].mean()),
            "mean_rmse_topo_kms": float(summary_df["rmse_topo_kms"].mean()),
            "mean_rmse_improvement_kms": float(summary_df["rmse_improvement_kms"].mean()),
            "median_rmse_improvement_kms": float(summary_df["rmse_improvement_kms"].median()),
            "mean_mae_bar_only_kms": float(summary_df["mae_bar_only_kms"].mean()),
            "mean_mae_topo_kms": float(summary_df["mae_topo_kms"].mean()),
            "mean_d_gal_mean": float(summary_df["d_gal_mean"].mean()),
        }
        (output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    if not failure_df.empty:
        failure_df.to_csv(output_dir / "topostructure_failures.csv", index=False)

    write_run_metadata(output_dir, config, args)
    print("\nRun complete.")
    print(f"Output directory: {output_dir}")
    if not summary_df.empty:
        print(f"Processed galaxies: {len(summary_df)}")
    if not failure_df.empty:
        print(f"Failures: {len(failure_df)}")


if __name__ == "__main__":
    main()
