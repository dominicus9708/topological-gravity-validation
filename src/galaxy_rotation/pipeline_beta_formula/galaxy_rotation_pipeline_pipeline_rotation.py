from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd

from galaxy_rotation_pipeline_preprocessing import load_processed_sparc_table
from galaxy_rotation_pipeline_sigma_model import compute_sigma_profile
from galaxy_rotation_pipeline_beta_model import (
    get_constant_beta,
    get_structural_beta,
)
from galaxy_rotation_pipeline_acceleration_model import compute_total_acceleration
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
from galaxy_rotation_pipeline_plotting import (
    save_rotation_curve_plot,
    save_beta_histogram,
    save_metric_histogram,
)


FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "sparc_normalized_beta_abar"

# 베타 아바 결과물과 분리
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "galaxy_rotation" / "pipeline_beta_formula"
)


def _resolve_galaxy_name(df: pd.DataFrame, file_path: Path) -> str:
    """
    galaxy 열이 있으면 우선 사용하고, 없으면 파일명 stem을 사용합니다.
    """
    if "galaxy" in df.columns and not df["galaxy"].isna().all():
        name = str(df["galaxy"].iloc[0]).strip()
        if name:
            return name
    return file_path.stem.replace("_normalized", "")


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_run_output_dirs(beta_mode: str) -> dict[str, Path]:
    """
    초 단위 timestamp를 사용하여 실행 폴더를 생성합니다.
    같은 분 내 재실행 시 덮어쓰기 위험을 줄입니다.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_root = DEFAULT_OUTPUT_ROOT / f"{timestamp}_{beta_mode}"

    dirs = {
        "run_root": _ensure_directory(run_root),
        "summary_dir": _ensure_directory(run_root / "summaries"),
        "table_dir": _ensure_directory(run_root / "tables"),
        "figure_dir": _ensure_directory(run_root / "figures"),
        "log_dir": _ensure_directory(run_root / "logs"),
        "metadata_dir": _ensure_directory(run_root / "metadata"),
    }
    return dirs


def _compute_component_velocities(
    r_kpc: np.ndarray,
    a_bar: np.ndarray,
    a_struct: np.ndarray,
    a_total: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    각 가속도 항을 속도 등가량으로 환산합니다.
    """
    r = np.asarray(r_kpc, dtype=float)
    a_bar = np.asarray(a_bar, dtype=float)
    a_struct = np.asarray(a_struct, dtype=float)
    a_total = np.asarray(a_total, dtype=float)

    v_bar = np.sqrt(np.clip(r * np.clip(a_bar, 0.0, None), 0.0, None))
    v_struct = np.sqrt(np.clip(r * np.clip(a_struct, 0.0, None), 0.0, None))
    v_total = np.sqrt(np.clip(r * np.clip(a_total, 0.0, None), 0.0, None))

    return v_bar, v_struct, v_total


def _write_run_log(log_path: Path, lines: list[str]) -> None:
    with open(log_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def run_rotation_pipeline(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    beta_mode: str = "structural",
    beta_value: float = 200.0,
    structural_mode: str = "sigma_over_r",
    sigma_proxy_mode: str = "log_ratio",
    sigma_damping_mode: str = "bar_over_obs",
    sigma_positive_only: bool = True,
    D_bg: float = 0.0,
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
    beta_min: float = 0.0,
    beta_max: float = 2000.0,
    beta_ridge: float = 1.0e-12,
    use_nonnegative_target: bool = False,
) -> pd.DataFrame:
    """
    은하 회전곡선 베타 포뮬러 파이프라인 실행.

    beta_mode:
    - "constant"   : beta = beta_value
    - "structural" : 관측과 구조 신호의 정합으로 beta 유도
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {data_dir}")

    csv_files = sorted(data_dir.glob("*_normalized.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No normalized CSV files found in: {data_dir}")

    dirs = create_run_output_dirs(beta_mode=beta_mode)
    run_root = dirs["run_root"]
    summary_dir = dirs["summary_dir"]
    table_dir = dirs["table_dir"]
    figure_dir = dirs["figure_dir"]
    log_dir = dirs["log_dir"]
    metadata_dir = dirs["metadata_dir"]

    print("[INFO] run_root =", run_root)
    print("[INFO] beta_mode =", beta_mode)

    results: list[dict] = []
    failed_count = 0

    for file_path in csv_files:
        try:
            df = load_processed_sparc_table(file_path)
            galaxy_name = _resolve_galaxy_name(df, file_path)

            r_kpc = df["r_kpc"].to_numpy(dtype=float)
            v_obs_kmps = df["v_obs_kmps"].to_numpy(dtype=float)
            v_err_kmps = df["v_err_kmps"].to_numpy(dtype=float)

            # 1) sigma profile
            sigma_profile = compute_sigma_profile(
                galaxy_data=df,
                D_bg=D_bg,
                positive_only=sigma_positive_only,
                proxy_mode=sigma_proxy_mode,
                damping_mode=sigma_damping_mode,
                upsilon_disk=upsilon_disk,
                upsilon_bul=upsilon_bul,
            )

            # 2) beta
            if beta_mode == "constant":
                beta = get_constant_beta(beta_value)
            elif beta_mode == "structural":
                beta = get_structural_beta(
                    galaxy_data=df,
                    sigma_profile=sigma_profile,
                    structural_mode=structural_mode,
                    upsilon_disk=upsilon_disk,
                    upsilon_bul=upsilon_bul,
                    beta_min=beta_min,
                    beta_max=beta_max,
                    ridge=beta_ridge,
                    use_nonnegative_target=use_nonnegative_target,
                )
            else:
                raise ValueError(f"Unknown beta_mode: {beta_mode}")

            # 3) accelerations
            a_total, a_bar, a_struct = compute_total_acceleration(
                galaxy_data=df,
                sigma_profile=sigma_profile,
                beta=beta,
                structural_mode=structural_mode,
                upsilon_disk=upsilon_disk,
                upsilon_bul=upsilon_bul,
            )

            # 4) velocities
            v_model_kmps = compute_rotation_velocity(r_kpc, a_total)
            v_bar_kmps, v_struct_kmps, _ = _compute_component_velocities(
                r_kpc=r_kpc,
                a_bar=a_bar,
                a_struct=a_struct,
                a_total=a_total,
            )

            # 5) metrics
            rmse_val = rmse(v_obs_kmps, v_model_kmps)
            mae_val = mae(v_obs_kmps, v_model_kmps)
            mafe_val = mean_absolute_fractional_error(v_obs_kmps, v_model_kmps)
            mdafe_val = median_absolute_fractional_error(v_obs_kmps, v_model_kmps)
            chi2_val = chi_square(v_obs_kmps, v_model_kmps, v_err_kmps)
            red_chi2_val = reduced_chi_square(
                v_obs_kmps,
                v_model_kmps,
                v_err_kmps,
                n_params=1,
            )
            n_valid = valid_point_count(v_obs_kmps, v_model_kmps, v_err_kmps)

            # 6) per-galaxy detailed table
            detail_df = pd.DataFrame(
                {
                    "galaxy": galaxy_name,
                    "r_kpc": r_kpc,
                    "v_obs_kmps": v_obs_kmps,
                    "v_err_kmps": v_err_kmps,
                    "sigma_profile": sigma_profile,
                    "a_bar_kmps2_per_kpc": a_bar,
                    "a_struct_kmps2_per_kpc": a_struct,
                    "a_total_kmps2_per_kpc": a_total,
                    "v_bar_kmps": v_bar_kmps,
                    "v_struct_kmps": v_struct_kmps,
                    "v_model_kmps": v_model_kmps,
                    "beta": float(beta),
                    "beta_mode": beta_mode,
                }
            )
            detail_path = table_dir / f"{galaxy_name}_rotation_table.csv"
            detail_df.to_csv(detail_path, index=False)

            # 7) plot
            plot_path = save_rotation_curve_plot(
                galaxy_name=galaxy_name,
                r_kpc=r_kpc,
                v_obs_kmps=v_obs_kmps,
                v_err_kmps=v_err_kmps,
                v_model_kmps=v_model_kmps,
                output_dir=figure_dir,
                v_bar_kmps=v_bar_kmps,
                v_struct_kmps=v_struct_kmps,
                beta=float(beta),
                rmse_value=rmse_val,
                chi2_value=chi2_val,
                reduced_chi2_value=red_chi2_val,
            )

            results.append(
                {
                    "galaxy": galaxy_name,
                    "source_file": str(file_path),
                    "beta_mode": beta_mode,
                    "beta": float(beta),
                    "structural_mode": structural_mode,
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
            failed_count += 1
            results.append(
                {
                    "galaxy": file_path.stem.replace("_normalized", ""),
                    "source_file": str(file_path),
                    "beta_mode": beta_mode,
                    "beta": float(beta_value) if beta_mode == "constant" else np.nan,
                    "structural_mode": structural_mode,
                    "sigma_proxy_mode": sigma_proxy_mode,
                    "sigma_damping_mode": sigma_damping_mode,
                    "sigma_positive_only": bool(sigma_positive_only),
                    "D_bg": float(D_bg),
                    "upsilon_disk": float(upsilon_disk),
                    "upsilon_bul": float(upsilon_bul),
                    "rmse_kmps": np.nan,
                    "mae_kmps": np.nan,
                    "mean_absolute_fractional_error": np.nan,
                    "median_absolute_fractional_error": np.nan,
                    "chi_square": np.nan,
                    "reduced_chi_square": np.nan,
                    "n_points_total": np.nan,
                    "n_points_valid": np.nan,
                    "detail_table": None,
                    "plot_path": None,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[FAILED] {file_path.name} -> {exc}")

    df_results = pd.DataFrame(results)

    # summary table
    summary_name = (
        "rotation_pipeline_summary_constant.csv"
        if beta_mode == "constant"
        else "rotation_pipeline_summary_structural.csv"
    )
    summary_path = summary_dir / summary_name
    df_results.to_csv(summary_path, index=False)

    # aggregate metadata
    ok_df = df_results[df_results["status"] == "ok"].copy()

    metadata = {
        "mode": beta_mode,
        "timestamp_root": str(run_root.name),
        "data_dir": str(data_dir),
        "output_root": str(run_root),
        "n_files_found": int(len(csv_files)),
        "n_ok": int((df_results["status"] == "ok").sum()),
        "n_failed": int((df_results["status"] == "failed").sum()),
        "parameters": {
            "beta_value": float(beta_value),
            "structural_mode": structural_mode,
            "sigma_proxy_mode": sigma_proxy_mode,
            "sigma_damping_mode": sigma_damping_mode,
            "sigma_positive_only": bool(sigma_positive_only),
            "D_bg": float(D_bg),
            "upsilon_disk": float(upsilon_disk),
            "upsilon_bul": float(upsilon_bul),
            "beta_min": float(beta_min),
            "beta_max": float(beta_max),
            "beta_ridge": float(beta_ridge),
            "use_nonnegative_target": bool(use_nonnegative_target),
        },
        "summary_csv": str(summary_path),
    }

    if not ok_df.empty:
        metadata["aggregate_metrics"] = {
            "mean_beta": float(ok_df["beta"].mean()) if "beta" in ok_df.columns else None,
            "median_beta": float(ok_df["beta"].median()) if "beta" in ok_df.columns else None,
            "mean_rmse_kmps": float(ok_df["rmse_kmps"].mean()),
            "median_rmse_kmps": float(ok_df["rmse_kmps"].median()),
            "mean_chi_square": float(ok_df["chi_square"].mean()),
            "median_chi_square": float(ok_df["chi_square"].median()),
            "mean_reduced_chi_square": float(ok_df["reduced_chi_square"].mean()),
            "median_reduced_chi_square": float(ok_df["reduced_chi_square"].median()),
        }

    metadata_path = metadata_dir / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # run log
    log_lines = [
        f"run_root: {run_root}",
        f"data_dir: {data_dir}",
        f"beta_mode: {beta_mode}",
        f"structural_mode: {structural_mode}",
        f"sigma_proxy_mode: {sigma_proxy_mode}",
        f"sigma_damping_mode: {sigma_damping_mode}",
        f"sigma_positive_only: {sigma_positive_only}",
        f"D_bg: {D_bg}",
        f"upsilon_disk: {upsilon_disk}",
        f"upsilon_bul: {upsilon_bul}",
        f"beta_min: {beta_min}",
        f"beta_max: {beta_max}",
        f"beta_ridge: {beta_ridge}",
        f"use_nonnegative_target: {use_nonnegative_target}",
        f"n_files_found: {len(csv_files)}",
        f"n_ok: {(df_results['status'] == 'ok').sum()}",
        f"n_failed: {failed_count}",
        f"summary_csv: {summary_path}",
        f"metadata_json: {metadata_path}",
    ]
    _write_run_log(log_dir / "run_log.txt", log_lines)

    # histograms
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
        save_metric_histogram(
            metric_values=ok_df["reduced_chi_square"].to_numpy(dtype=float),
            metric_name="ReducedChiSquare",
            output_dir=figure_dir,
            filename="reduced_chi_square_distribution.png",
            title="Reduced Chi-Square Distribution",
        )
        save_beta_histogram(
            beta_values=ok_df["beta"].to_numpy(dtype=float),
            output_dir=figure_dir,
            filename="beta_distribution.png",
            title=f"Derived Beta Distribution ({beta_mode})",
        )

    print(f"[DONE] summary saved to: {summary_path}")
    return df_results


if __name__ == "__main__":
    summary = run_rotation_pipeline(
        data_dir=DEFAULT_DATA_DIR,
        beta_mode="structural",
        beta_value=200.0,
        structural_mode="sigma_over_r",
        sigma_proxy_mode="log_ratio",
        sigma_damping_mode="bar_over_obs",
        sigma_positive_only=True,
        D_bg=0.0,
        upsilon_disk=1.0,
        upsilon_bul=1.0,
        beta_min=0.0,
        beta_max=2000.0,
        beta_ridge=1.0e-12,
        use_nonnegative_target=False,
    )

    print(summary.head())
    print(f"\nTotal files processed: {len(summary)}")