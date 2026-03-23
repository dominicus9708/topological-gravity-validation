from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from galaxy_rotation_pipeline_preprocessing import load_processed_sparc_table
from galaxy_rotation_pipeline_sigma_model import compute_sigma_profile
from galaxy_rotation_pipeline_beta_model import (
    get_constant_beta,
    get_structural_beta,
)
from galaxy_rotation_pipeline_acceleration_model import (
    compute_structural_acceleration,
    compute_total_acceleration,
)
from galaxy_rotation_pipeline_rotation_curve_model import compute_rotation_velocity
from galaxy_rotation_pipeline_metrics import (
    rmse,
    mean_absolute_fractional_error,
)
from galaxy_rotation_pipeline_plotting import save_rotation_curve_plot


FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "sparc_normalized"
DEFAULT_RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_RESULTS_PLOT_DIR = PROJECT_ROOT / "results" / "plots" / "individual"


def ensure_directories(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _resolve_galaxy_name(df: pd.DataFrame, file_path: Path) -> str:
    if "galaxy" in df.columns and not df["galaxy"].empty:
        value = str(df["galaxy"].iloc[0]).strip()
        if value:
            return value

    name = file_path.stem
    if name.endswith("_normalized"):
        name = name[: -len("_normalized")]
    return name


def _compute_baryonic_newtonian_acceleration(df: pd.DataFrame, r_kpc: np.ndarray) -> np.ndarray:
    required = ["v_gas_kmps", "v_disk_kmps", "v_bul_kmps"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing baryonic velocity columns: {missing}")

    v_gas = df["v_gas_kmps"].to_numpy(dtype=float)
    v_disk = df["v_disk_kmps"].to_numpy(dtype=float)
    v_bul = df["v_bul_kmps"].to_numpy(dtype=float)

    v_bar_sq = v_gas**2 + v_disk**2 + v_bul**2
    return v_bar_sq / r_kpc


def run_rotation_pipeline(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    results_table_dir: str | Path = DEFAULT_RESULTS_TABLE_DIR,
    results_plot_dir: str | Path = DEFAULT_RESULTS_PLOT_DIR,
    beta_mode: str = "constant",
    beta_value: float = 200.0,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    results_table_dir = Path(results_table_dir)
    results_plot_dir = Path(results_plot_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {data_dir}")

    csv_files = sorted(data_dir.glob("*_normalized.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No normalized CSV files found in: {data_dir}")

    ensure_directories(results_table_dir, results_plot_dir)

    results = []

    for file_path in csv_files:
        try:
            df = load_processed_sparc_table(file_path)
            df = df.sort_values("r_kpc").reset_index(drop=True)

            galaxy_name = _resolve_galaxy_name(df, file_path)

            r_kpc = df["r_kpc"].to_numpy(dtype=float)
            v_obs_kmps = df["v_obs_kmps"].to_numpy(dtype=float)
            v_err_kmps = df["v_err_kmps"].to_numpy(dtype=float)

            a_newton = _compute_baryonic_newtonian_acceleration(df, r_kpc)

            sigma_profile = compute_sigma_profile(r_kpc, a_newton)

            if beta_mode == "constant":
                beta = get_constant_beta(beta_value)
            elif beta_mode == "structural":
                beta = get_structural_beta(df)
            else:
                raise ValueError(f"Unknown beta_mode: {beta_mode}")

            a_struct = compute_structural_acceleration(r_kpc, sigma_profile, beta)
            a_total = compute_total_acceleration(a_newton, a_struct)
            v_model_kmps = compute_rotation_velocity(a_total, r_kpc)

            rmse_val = rmse(v_obs_kmps, v_model_kmps)
            mf_error = mean_absolute_fractional_error(v_obs_kmps, v_model_kmps)

            results.append(
                {
                    "galaxy": galaxy_name,
                    "source_file": str(file_path),
                    "beta_mode": beta_mode,
                    "beta": float(beta),
                    "a_newton_source": "computed_from_baryonic_velocity",
                    "rmse_kmps": float(rmse_val),
                    "mean_absolute_fractional_error": float(mf_error),
                    "n_points": int(len(df)),
                    "status": "ok",
                }
            )

            plot_path = results_plot_dir / f"{galaxy_name}_rotation_curve.png"
            save_rotation_curve_plot(
                r_kpc=r_kpc,
                v_obs_kmps=v_obs_kmps,
                v_err_kmps=v_err_kmps,
                v_model_kmps=v_model_kmps,
                galaxy_name=galaxy_name,
                output_path=plot_path,
            )

            print(f"[OK] {galaxy_name}")

        except Exception as exc:
            results.append(
                {
                    "galaxy": file_path.stem,
                    "source_file": str(file_path),
                    "beta_mode": beta_mode,
                    "beta": float(beta_value) if beta_mode == "constant" else None,
                    "a_newton_source": None,
                    "rmse_kmps": None,
                    "mean_absolute_fractional_error": None,
                    "n_points": None,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[FAILED] {file_path.name} -> {exc}")

    df_results = pd.DataFrame(results)

    summary_name = (
        "rotation_pipeline_summary.csv"
        if beta_mode == "constant"
        else f"rotation_pipeline_summary_{beta_mode}.csv"
    )
    table_path = results_table_dir / summary_name
    df_results.to_csv(table_path, index=False)

    return df_results


if __name__ == "__main__":
    summary = run_rotation_pipeline(
        data_dir=DEFAULT_DATA_DIR,
        results_table_dir=DEFAULT_RESULTS_TABLE_DIR,
        results_plot_dir=DEFAULT_RESULTS_PLOT_DIR,
        beta_mode="constant",
        beta_value=200.0,
    )

    print(summary.head())
    print(f"\nTotal files processed: {len(summary)}")