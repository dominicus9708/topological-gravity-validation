# src/run_rotation_batch_beta_formula.py

from __future__ import annotations

from pathlib import Path
import traceback
import pandas as pd
import numpy as np

from data_loader import load_rotation_curve_file
from normalize_sparc import (
    normalize_sparc_dataframe,
    validate_normalized_sparc,
    sort_by_radius,
)
from sigma_model import compute_model_velocity_curve
from residuals import build_residual_dataframe, summarize_fit_metrics
from plot_rotation_curve import plot_rotation_curve, plot_residuals


# ----- paths -----

RAW_DIR = Path("data/raw/sparc_csv")
PROCESSED_DIR = Path("data/processed/sparc_normalized_beta_formula")
RESULTS_TABLE_DIR = Path("results/tables_beta_formula")
RESULTS_SUMMARY_DIR = Path("results/summaries_beta_formula")
FIGURE_DIR = Path("figures/rotation_curves_beta_formula")


# ----- model configuration -----

USE_POSITIVE_SIGMA_ONLY = True
SIGMA_RS_KPC = 5.0
N_PARAMS = 1


def ensure_directories() -> None:
    for path in [
        PROCESSED_DIR,
        RESULTS_TABLE_DIR,
        RESULTS_SUMMARY_DIR,
        FIGURE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def infer_beta_from_dataframe(df_norm: pd.DataFrame) -> float:
    """
    Reference-only beta inference from observed acceleration scale.

    beta_g = median(a_obs)

    Units:
        (km/s)^2 / kpc
    """
    a_obs = np.asarray(df_norm["a_obs_kmps2_per_kpc"].values, dtype=float)

    # robust central scale
    beta_g = float(np.median(np.abs(a_obs)))

    # safety floor
    beta_g = max(beta_g, 1.0e-12)
    return beta_g


def run_single_galaxy(file_path: Path, n_params: int = N_PARAMS) -> dict:
    galaxy_name = file_path.stem

    # 1. raw load
    df_raw = load_rotation_curve_file(file_path)

    # 2. normalize
    df_norm = normalize_sparc_dataframe(df_raw, galaxy_name=galaxy_name)
    validate_normalized_sparc(df_norm)
    df_norm = sort_by_radius(df_norm)

    # 3. save normalized table
    norm_path = PROCESSED_DIR / f"{galaxy_name}_normalized.csv"
    df_norm.to_csv(norm_path, index=False)

    # 4. infer beta from observed data
    beta_g = infer_beta_from_dataframe(df_norm)

    # 5. model prediction
    v_model_kmps = compute_model_velocity_curve(
        df_norm,
        beta=beta_g,
        use_positive_only=USE_POSITIVE_SIGMA_ONLY,
        rs_kpc=SIGMA_RS_KPC,
    )

    if len(v_model_kmps) != len(df_norm):
        raise ValueError(
            f"Model length mismatch for {galaxy_name}: "
            f"{len(v_model_kmps)} != {len(df_norm)}"
        )

    # 6. residual table
    df_resid = build_residual_dataframe(df_norm, v_model_kmps)
    resid_path = RESULTS_TABLE_DIR / f"{galaxy_name}_residuals.csv"
    df_resid.to_csv(resid_path, index=False)

    # 7. summary metrics
    summary = summarize_fit_metrics(df_resid, n_params=n_params)
    summary["beta_formula"] = "median(a_obs)"
    summary["beta_value"] = beta_g
    summary["rs_kpc"] = SIGMA_RS_KPC

    # 8. plots
    plot_rotation_curve(
        df_resid,
        output_path=str(FIGURE_DIR / f"{galaxy_name}_rotation.png"),
        show=False,
    )
    plot_residuals(
        df_resid,
        output_path=str(FIGURE_DIR / f"{galaxy_name}_residuals.png"),
        show=False,
    )

    summary["status"] = "success"
    summary["source_file"] = str(file_path)
    return summary


def run_all_galaxies(n_params: int = N_PARAMS) -> pd.DataFrame:
    ensure_directories()

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DIR}")

    all_files = sorted(RAW_DIR.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    results = []

    for file_path in all_files:
        galaxy_name = file_path.stem
        try:
            summary = run_single_galaxy(file_path, n_params=n_params)
            results.append(summary)
            print(f"[OK][beta_formula] {galaxy_name}")
        except Exception as exc:
            results.append(
                {
                    "galaxy": galaxy_name,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "source_file": str(file_path),
                    "beta_formula": "median(a_obs)",
                    "beta_value": np.nan,
                    "rs_kpc": SIGMA_RS_KPC,
                }
            )
            print(f"[FAIL][beta_formula] {galaxy_name}: {exc}")

    df_summary = pd.DataFrame(results)
    df_summary.to_csv(
        RESULTS_SUMMARY_DIR / "rotation_batch_summary_beta_formula.csv",
        index=False,
    )

    return df_summary


if __name__ == "__main__":
    summary_df = run_all_galaxies(n_params=N_PARAMS)
    print(summary_df.head())
    print(f"\nTotal galaxies processed: {len(summary_df)}")