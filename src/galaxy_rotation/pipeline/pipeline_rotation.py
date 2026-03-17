from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from preprocessing import load_processed_sparc_table
from sigma_model import compute_sigma_profile
from beta_model import get_constant_beta, get_structural_beta
from acceleration_model import (
    compute_newtonian_acceleration,
    compute_structural_acceleration,
    compute_total_acceleration,
)
from rotation_curve_model import compute_rotation_velocity
from metrics import rmse, mean_absolute_fractional_error
from plotting import save_rotation_curve_plot


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class PipelineConfig:
    data_path: str = "data_processed/sparc/processed_galaxy_table.csv"
    results_table_dir: str = "results/tables"
    results_plot_dir: str = "results/plots/individual"
    beta_mode: str = "constant"   # "constant" or "structural"
    beta_constant_value: float = 200.0
    save_plots: bool = True
    save_summary_table: bool = True
    eps: float = 1e-12


# ---------------------------------------------------------------------
# Per-galaxy result container
# ---------------------------------------------------------------------

@dataclass
class GalaxyFitResult:
    galaxy: str
    beta_mode: str
    beta_value: float
    rmse: float
    mean_fractional_error: float
    n_points: int


# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------

def ensure_directories(config: PipelineConfig) -> None:
    Path(config.results_table_dir).mkdir(parents=True, exist_ok=True)
    Path(config.results_plot_dir).mkdir(parents=True, exist_ok=True)


def validate_required_columns(df: pd.DataFrame) -> None:
    required = {
        "galaxy",
        "r",
        "v_obs",
        "v_gas",
        "v_disk",
        "v_bulge",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def group_galaxies(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    grouped: Dict[str, pd.DataFrame] = {}
    for galaxy_name, gdf in df.groupby("galaxy"):
        grouped[str(galaxy_name)] = gdf.sort_values("r").reset_index(drop=True)
    return grouped


# ---------------------------------------------------------------------
# Physics/model pipeline per galaxy
# ---------------------------------------------------------------------

def run_single_galaxy(
    galaxy_df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[GalaxyFitResult, pd.DataFrame]:
    galaxy_name = str(galaxy_df["galaxy"].iloc[0])

    r = galaxy_df["r"].to_numpy(dtype=float)
    v_obs = galaxy_df["v_obs"].to_numpy(dtype=float)
    v_gas = galaxy_df["v_gas"].to_numpy(dtype=float)
    v_disk = galaxy_df["v_disk"].to_numpy(dtype=float)
    v_bulge = galaxy_df["v_bulge"].to_numpy(dtype=float)

    # 1) Newtonian contribution
    g_newton = compute_newtonian_acceleration(
        r=r,
        v_gas=v_gas,
        v_disk=v_disk,
        v_bulge=v_bulge,
        eps=config.eps,
    )

    # 2) Sigma profile
    sigma = compute_sigma_profile(
        r=r,
        g_newton=g_newton,
    )

    # 3) Beta assignment
    if config.beta_mode == "constant":
        beta = get_constant_beta(config.beta_constant_value)
    elif config.beta_mode == "structural":
        beta = get_structural_beta(
            galaxy_data=galaxy_df,
            sigma_profile=sigma,
        )
    else:
        raise ValueError(f"Unsupported beta_mode: {config.beta_mode}")

    # 4) Structural acceleration
    g_struct = compute_structural_acceleration(
        r=r,
        sigma=sigma,
        beta=beta,
        eps=config.eps,
    )

    # 5) Total acceleration
    g_total = compute_total_acceleration(
        g_newton=g_newton,
        g_struct=g_struct,
    )

    # 6) Rotation velocity reconstruction
    v_model = compute_rotation_velocity(
        r=r,
        g_total=g_total,
        eps=config.eps,
    )

    # 7) Metrics
    fit_rmse = rmse(v_obs, v_model)
    fit_mfe = mean_absolute_fractional_error(v_obs, v_model, eps=config.eps)

    # 8) Build result row
    result = GalaxyFitResult(
        galaxy=galaxy_name,
        beta_mode=config.beta_mode,
        beta_value=float(beta),
        rmse=float(fit_rmse),
        mean_fractional_error=float(fit_mfe),
        n_points=int(len(r)),
    )

    comparison_df = pd.DataFrame(
        {
            "galaxy": galaxy_name,
            "r": r,
            "v_obs": v_obs,
            "v_model": v_model,
            "sigma": sigma,
            "g_newton": g_newton,
            "g_struct": g_struct,
            "g_total": g_total,
        }
    )

    return result, comparison_df


# ---------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------

def run_rotation_pipeline(config: Optional[PipelineConfig] = None) -> pd.DataFrame:
    if config is None:
        config = PipelineConfig()

    ensure_directories(config)

    df = load_processed_sparc_table(config.data_path)
    validate_required_columns(df)

    galaxy_map = group_galaxies(df)

    summary_rows: List[dict] = []

    for galaxy_name, galaxy_df in galaxy_map.items():
        try:
            result, comparison_df = run_single_galaxy(galaxy_df, config)
            summary_rows.append(asdict(result))

            if config.save_plots:
                plot_path = Path(config.results_plot_dir) / f"{galaxy_name}.png"
                save_rotation_curve_plot(
                    df=comparison_df,
                    galaxy_name=galaxy_name,
                    output_path=str(plot_path),
                )

        except Exception as exc:
            summary_rows.append(
                {
                    "galaxy": galaxy_name,
                    "beta_mode": config.beta_mode,
                    "beta_value": np.nan,
                    "rmse": np.nan,
                    "mean_fractional_error": np.nan,
                    "n_points": int(len(galaxy_df)),
                    "error": str(exc),
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    if config.save_summary_table:
        output_name = f"rotation_fit_summary_{config.beta_mode}.csv"
        output_path = Path(config.results_table_dir) / output_name
        summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return summary_df


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    config = PipelineConfig(
        data_path="data_processed/sparc/processed_galaxy_table.csv",
        results_table_dir="results/tables",
        results_plot_dir="results/plots/individual",
        beta_mode="constant",          # "constant" or "structural"
        beta_constant_value=200.0,
        save_plots=True,
        save_summary_table=True,
    )

    summary = run_rotation_pipeline(config)
    print(summary.head())
    print()
    print("Completed galaxy rotation validation pipeline.")
