from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd

from galaxy_rotation_pipeline_preprocessing import load_processed_sparc_table
from galaxy_rotation_pipeline_sigma_model import compute_sigma_profile
from galaxy_rotation_pipeline_acceleration_model import compute_baryonic_acceleration
from galaxy_rotation_pipeline_rotation_curve_model import compute_rotation_velocity
from galaxy_rotation_pipeline_metrics import (
    rmse,
    mae,
    mean_absolute_fractional_error,
    median_absolute_fractional_error,
    chi_square,
    reduced_chi_square,
    valid_point_count,
)
from galaxy_rotation_pipeline_plotting import save_rotation_curve_plot, save_metric_histogram

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "sparc_normalized_beta_abar"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "galaxy_rotation" / "pipeline_independent_topogravity"


def _resolve_galaxy_name(df: pd.DataFrame, file_path: Path) -> str:
    if "galaxy" in df.columns and not df["galaxy"].isna().all():
        name = str(df["galaxy"].iloc[0]).strip()
        if name:
            return name
    return file_path.stem.replace("_normalized", "")


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_run_output_dirs() -> dict[str, Path]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_root = DEFAULT_OUTPUT_ROOT / timestamp
    return {
        "run_root": _ensure_directory(run_root),
        "summary_dir": _ensure_directory(run_root / "summaries"),
        "table_dir": _ensure_directory(run_root / "tables"),
        "figure_dir": _ensure_directory(run_root / "figures"),
        "log_dir": _ensure_directory(run_root / "logs"),
        "metadata_dir": _ensure_directory(run_root / "metadata"),
    }


def _compute_component_velocities(
    r_kpc: np.ndarray,
    a_bar: np.ndarray,
    a_topo: np.ndarray,
    a_total: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r_kpc, dtype=float)
    v_bar = np.sqrt(np.clip(r * np.clip(np.asarray(a_bar, dtype=float), 0.0, None), 0.0, None))
    v_topo = np.sqrt(np.clip(r * np.clip(np.asarray(a_topo, dtype=float), 0.0, None), 0.0, None))
    v_total = np.sqrt(np.clip(r * np.clip(np.asarray(a_total, dtype=float), 0.0, None), 0.0, None))
    return v_bar, v_topo, v_total


def _write_run_log(log_path: Path, lines: list[str]) -> None:
    with open(log_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def compute_independent_topological_acceleration(
    r_kpc: np.ndarray,
    sigma_profile: np.ndarray,
    g_scale: float,
    lambda_reorg: float = 0.0,
    l0_kpc: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r_kpc, dtype=float)
    sigma = np.asarray(sigma_profile, dtype=float)

    if r.ndim != 1 or sigma.ndim != 1:
        raise ValueError("r_kpc and sigma_profile must be 1D arrays")
    if r.shape != sigma.shape:
        raise ValueError("r_kpc and sigma_profile must have the same shape")
    if r.size == 0:
        raise ValueError("Empty profile")
    if np.any(~np.isfinite(r)) or np.any(~np.isfinite(sigma)):
        raise ValueError("Non-finite values found in r_kpc or sigma_profile")
    if np.any(np.diff(r) <= 0):
        raise ValueError("r_kpc must be strictly increasing")

    dsigma_dr = np.gradient(sigma, r)
    a_topo = float(g_scale) * (sigma + float(lambda_reorg) * float(l0_kpc) * dsigma_dr)
    a_topo = np.where(np.isfinite(a_topo), a_topo, 0.0)
    return a_topo, dsigma_dr


def run_rotation_pipeline(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    g_scale: float = 300.0,
    lambda_reorg: float = 0.0,
    l0_kpc: float = 1.0,
    sigma_proxy_mode: str = "log_ratio",
    sigma_damping_mode: str = "bar_soft",
    sigma_positive_only: bool = False,
    D_bg: float = 0.0,
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
    sigma_weight_mode: str = "fixed",
    sigma_weight_value: float = 0.5,
    sigma_weight_g0: float | None = 1.0,
    sigma_weight_sigma0: float | None = 1.0,
    sigma_weight_u: float = 1.0,
    sigma_weight_v: float = 1.0,
    sigma_weight_lambda: float = 1.0,
    sigma_weight_g_choice: str = "gbar_char_logmean",
    sigma_weight_sigma_choice: str = "sigma_char_rms",
    sigma_soft_power: float = 1.0,
    sigma_floor_fraction: float = 0.15,
    sigma_normalize: bool = True,
    sigma_norm_mode: str = "tanh",
    sigma_norm_strength: float = 1.0,
    sigma_norm_floor_fraction: float = 0.25,
    sigma_norm_choice: str = "sigma_char_rms",
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {data_dir}")

    csv_files = sorted(data_dir.glob("*_normalized.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No normalized CSV files found in: {data_dir}")

    dirs = create_run_output_dirs()
    run_root = dirs["run_root"]
    summary_dir = dirs["summary_dir"]
    table_dir = dirs["table_dir"]
    figure_dir = dirs["figure_dir"]
    log_dir = dirs["log_dir"]
    metadata_dir = dirs["metadata_dir"]

    print("[INFO] run_root =", run_root)
    print("[INFO] g_scale =", g_scale)
    print("[INFO] lambda_reorg =", lambda_reorg)
    print("[INFO] l0_kpc =", l0_kpc)

    results: list[dict] = []

    for file_path in csv_files:
        try:
            df = load_processed_sparc_table(file_path)
            galaxy_name = _resolve_galaxy_name(df, file_path)

            r_kpc = df["r_kpc"].to_numpy(dtype=float)
            v_obs_kmps = df["v_obs_kmps"].to_numpy(dtype=float)
            v_err_kmps = df["v_err_kmps"].to_numpy(dtype=float)

            sigma_profile = compute_sigma_profile(
                galaxy_data=df,
                D_bg=D_bg,
                positive_only=sigma_positive_only,
                proxy_mode=sigma_proxy_mode,
                damping_mode=sigma_damping_mode,
                upsilon_disk=upsilon_disk,
                upsilon_bul=upsilon_bul,
                sigma_weight_mode=sigma_weight_mode,
                sigma_weight_value=sigma_weight_value,
                sigma_weight_g0=sigma_weight_g0,
                sigma_weight_sigma0=sigma_weight_sigma0,
                sigma_weight_u=sigma_weight_u,
                sigma_weight_v=sigma_weight_v,
                sigma_weight_lambda=sigma_weight_lambda,
                sigma_weight_g_choice=sigma_weight_g_choice,
                sigma_weight_sigma_choice=sigma_weight_sigma_choice,
                sigma_soft_power=sigma_soft_power,
                sigma_floor_fraction=sigma_floor_fraction,
                sigma_normalize=sigma_normalize,
                sigma_norm_mode=sigma_norm_mode,
                sigma_norm_strength=sigma_norm_strength,
                sigma_norm_floor_fraction=sigma_norm_floor_fraction,
                sigma_norm_choice=sigma_norm_choice,
            )

            a_bar = compute_baryonic_acceleration(
                galaxy_data=df,
                upsilon_disk=upsilon_disk,
                upsilon_bul=upsilon_bul,
            )
            a_topo, dsigma_dr = compute_independent_topological_acceleration(
                r_kpc=r_kpc,
                sigma_profile=sigma_profile,
                g_scale=g_scale,
                lambda_reorg=lambda_reorg,
                l0_kpc=l0_kpc,
            )
            a_total = np.clip(a_bar + a_topo, 0.0, None)

            v_model_kmps = compute_rotation_velocity(r_kpc, a_total)
            v_bar_kmps, v_topo_kmps, _ = _compute_component_velocities(
                r_kpc=r_kpc,
                a_bar=a_bar,
                a_topo=a_topo,
                a_total=a_total,
            )

            rmse_val = rmse(v_obs_kmps, v_model_kmps)
            mae_val = mae(v_obs_kmps, v_model_kmps)
            mafe_val = mean_absolute_fractional_error(v_obs_kmps, v_model_kmps)
            mdafe_val = median_absolute_fractional_error(v_obs_kmps, v_model_kmps)
            chi2_val = chi_square(v_obs_kmps, v_model_kmps, v_err_kmps)
            red_chi2_val = reduced_chi_square(v_obs_kmps, v_model_kmps, v_err_kmps, n_params=2)
            n_valid = valid_point_count(v_obs_kmps, v_model_kmps, v_err_kmps)

            detail_df = pd.DataFrame(
                {
                    "galaxy": galaxy_name,
                    "r_kpc": r_kpc,
                    "v_obs_kmps": v_obs_kmps,
                    "v_err_kmps": v_err_kmps,
                    "sigma_profile": sigma_profile,
                    "dsigma_dr": dsigma_dr,
                    "a_bar_kmps2_per_kpc": a_bar,
                    "a_topo_kmps2_per_kpc": a_topo,
                    "a_total_kmps2_per_kpc": a_total,
                    "v_bar_kmps": v_bar_kmps,
                    "v_topo_kmps": v_topo_kmps,
                    "v_model_kmps": v_model_kmps,
                    "g_scale": float(g_scale),
                    "lambda_reorg": float(lambda_reorg),
                    "l0_kpc": float(l0_kpc),
                }
            )
            detail_path = table_dir / f"{galaxy_name}_rotation_table.csv"
            detail_df.to_csv(detail_path, index=False)

            plot_path = save_rotation_curve_plot(
                galaxy_name=galaxy_name,
                r_kpc=r_kpc,
                v_obs_kmps=v_obs_kmps,
                v_err_kmps=v_err_kmps,
                v_model_kmps=v_model_kmps,
                output_dir=figure_dir,
                v_bar_kmps=v_bar_kmps,
                v_struct_kmps=v_topo_kmps,
                beta=float(g_scale),
                rmse_value=rmse_val,
                chi2_value=chi2_val,
                reduced_chi2_value=red_chi2_val,
            )

            results.append(
                {
                    "galaxy": galaxy_name,
                    "source_file": str(file_path),
                    "g_scale": float(g_scale),
                    "lambda_reorg": float(lambda_reorg),
                    "l0_kpc": float(l0_kpc),
                    "sigma_proxy_mode": sigma_proxy_mode,
                    "sigma_damping_mode": sigma_damping_mode,
                    "sigma_positive_only": bool(sigma_positive_only),
                    "D_bg": float(D_bg),
                    "upsilon_disk": float(upsilon_disk),
                    "upsilon_bul": float(upsilon_bul),
                    "rmse_kmps": float(rmse_val) if np.isfinite(rmse_val) else np.nan,
                    "mae_kmps": float(mae_val) if np.isfinite(mae_val) else np.nan,
                    "mean_absolute_fractional_error": float(mafe_val) if np.isfinite(mafe_val) else np.nan,
                    "median_absolute_fractional_error": float(mdafe_val) if np.isfinite(mdafe_val) else np.nan,
                    "chi_square": float(chi2_val) if np.isfinite(chi2_val) else np.nan,
                    "reduced_chi_square": float(red_chi2_val) if np.isfinite(red_chi2_val) else np.nan,
                    "n_points_total": int(len(df)),
                    "n_points_valid": int(n_valid),
                    "detail_table": str(detail_path),
                    "plot_path": str(plot_path),
                    "status": "ok",
                    "error": None,
                }
            )
            print(f"[OK] {galaxy_name}")

        except Exception as exc:
            results.append(
                {
                    "galaxy": file_path.stem.replace("_normalized", ""),
                    "source_file": str(file_path),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[FAILED] {file_path.name} -> {exc}")

    df_results = pd.DataFrame(results)
    summary_path = summary_dir / "rotation_pipeline_summary_independent_topogravity.csv"
    df_results.to_csv(summary_path, index=False)

    ok_df = df_results[df_results["status"] == "ok"].copy()
    metadata = {
        "timestamp_root": str(run_root.name),
        "data_dir": str(data_dir),
        "output_root": str(run_root),
        "n_files_found": int(len(csv_files)),
        "n_ok": int((df_results["status"] == "ok").sum()),
        "n_failed": int((df_results["status"] == "failed").sum()),
        "parameters": {
            "g_scale": float(g_scale),
            "lambda_reorg": float(lambda_reorg),
            "l0_kpc": float(l0_kpc),
            "sigma_proxy_mode": sigma_proxy_mode,
            "sigma_damping_mode": sigma_damping_mode,
            "sigma_positive_only": bool(sigma_positive_only),
            "D_bg": float(D_bg),
            "upsilon_disk": float(upsilon_disk),
            "upsilon_bul": float(upsilon_bul),
            "sigma_weight_mode": sigma_weight_mode,
            "sigma_weight_value": float(sigma_weight_value),
            "sigma_weight_g0": sigma_weight_g0,
            "sigma_weight_sigma0": sigma_weight_sigma0,
            "sigma_weight_u": float(sigma_weight_u),
            "sigma_weight_v": float(sigma_weight_v),
            "sigma_weight_lambda": float(sigma_weight_lambda),
            "sigma_weight_g_choice": sigma_weight_g_choice,
            "sigma_weight_sigma_choice": sigma_weight_sigma_choice,
            "sigma_soft_power": float(sigma_soft_power),
            "sigma_floor_fraction": float(sigma_floor_fraction),
            "sigma_normalize": bool(sigma_normalize),
            "sigma_norm_mode": sigma_norm_mode,
            "sigma_norm_strength": float(sigma_norm_strength),
            "sigma_norm_floor_fraction": float(sigma_norm_floor_fraction),
            "sigma_norm_choice": sigma_norm_choice,
        },
        "summary_csv": str(summary_path),
    }
    metadata_path = metadata_dir / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log_lines = [
        f"run_root: {run_root}",
        f"data_dir: {data_dir}",
        f"g_scale: {g_scale}",
        f"lambda_reorg: {lambda_reorg}",
        f"l0_kpc: {l0_kpc}",
        f"summary_csv: {summary_path}",
        f"metadata_json: {metadata_path}",
    ]
    _write_run_log(log_dir / "run_log.txt", log_lines)

    if not ok_df.empty:
        save_metric_histogram(
            metric_values=ok_df["rmse_kmps"].to_numpy(dtype=float),
            metric_name="RMSE_kmps",
            output_dir=figure_dir,
            filename="rmse_distribution.png",
            title="RMSE Distribution",
        )
        save_metric_histogram(
            metric_values=ok_df["chi_square"].to_numpy(dtype=float),
            metric_name="ChiSquare",
            output_dir=figure_dir,
            filename="chi_square_distribution.png",
            title="Chi-Square Distribution",
        )

    print(f"[DONE] summary saved to: {summary_path}")
    return df_results


if __name__ == "__main__":
    summary = run_rotation_pipeline(
        data_dir=DEFAULT_DATA_DIR,
        g_scale=300.0,
        lambda_reorg=0.0,
        l0_kpc=1.0,
        sigma_proxy_mode="log_ratio",
        sigma_damping_mode="bar_soft",
        sigma_positive_only=False,
        D_bg=0.0,
        upsilon_disk=1.0,
        upsilon_bul=1.0,
        sigma_weight_mode="fixed",
        sigma_weight_value=0.5,
        sigma_weight_g0=1.0,
        sigma_weight_sigma0=1.0,
        sigma_weight_u=1.0,
        sigma_weight_v=1.0,
        sigma_weight_lambda=1.0,
        sigma_weight_g_choice="gbar_char_logmean",
        sigma_weight_sigma_choice="sigma_char_rms",
        sigma_soft_power=1.0,
        sigma_floor_fraction=0.15,
        sigma_normalize=True,
        sigma_norm_mode="tanh",
        sigma_norm_strength=1.0,
        sigma_norm_floor_fraction=0.25,
        sigma_norm_choice="sigma_char_rms",
    )
    print(summary.head())
    print(f"\nTotal files processed: {len(summary)}")
