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


# ----------------------------
# paths
# ----------------------------

RAW_DIR = Path("data/raw/sparc_csv")

PROCESSED_DIR = Path("data/processed/sparc_normalized_beta_abar")
RESULTS_TABLE_DIR = Path("results/tables_beta_abar")
RESULTS_SUMMARY_DIR = Path("results/summaries_beta_abar")

FIGURE_DIR = Path("figures/rotation_curves_beta_abar")


# ----------------------------
# model configuration
# ----------------------------

USE_POSITIVE_SIGMA_ONLY = True
SIGMA_RS_KPC = 5.0
N_PARAMS = 1


# ----------------------------
# directory creation
# ----------------------------

def ensure_directories() -> None:
    for path in [
        PROCESSED_DIR,
        RESULTS_TABLE_DIR,
        RESULTS_SUMMARY_DIR,
        FIGURE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


# ----------------------------
# beta = median(a_bar)
# ----------------------------

def infer_beta_from_abar(df_norm: pd.DataFrame) -> float:

    r = df_norm["r_kpc"].values

    v_gas = df_norm["v_gas_kmps"].values
    v_disk = df_norm["v_disk_kmps"].values
    v_bul = df_norm["v_bul_kmps"].values

    v_bar2 = v_gas**2 + v_disk**2 + v_bul**2

    a_bar = v_bar2 / r

    beta_g = float(np.median(a_bar))

    # 안전장치
    beta_g = max(beta_g, 1e-12)

    return beta_g


# ----------------------------
# single galaxy
# ----------------------------

def run_single_galaxy(file_path: Path, n_params: int = N_PARAMS) -> dict:

    galaxy_name = file_path.stem

    df_raw = load_rotation_curve_file(file_path)

    df_norm = normalize_sparc_dataframe(df_raw, galaxy_name=galaxy_name)

    validate_normalized_sparc(df_norm)

    df_norm = sort_by_radius(df_norm)

    norm_path = PROCESSED_DIR / f"{galaxy_name}_normalized.csv"

    df_norm.to_csv(norm_path, index=False)

    # beta 계산
    beta_g = infer_beta_from_abar(df_norm)

    v_model_kmps = compute_model_velocity_curve(
        df_norm,
        beta=beta_g,
        use_positive_only=USE_POSITIVE_SIGMA_ONLY,
        rs_kpc=SIGMA_RS_KPC,
    )

    df_resid = build_residual_dataframe(df_norm, v_model_kmps)

    resid_path = RESULTS_TABLE_DIR / f"{galaxy_name}_residuals.csv"

    df_resid.to_csv(resid_path, index=False)

    summary = summarize_fit_metrics(df_resid, n_params=n_params)

    summary["beta_formula"] = "median(a_bar)"
    summary["beta_value"] = beta_g
    summary["rs_kpc"] = SIGMA_RS_KPC

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


# ----------------------------
# batch run
# ----------------------------

def run_all_galaxies(n_params: int = N_PARAMS) -> pd.DataFrame:

    ensure_directories()

    all_files = sorted(RAW_DIR.glob("*.csv"))

    results = []

    for file_path in all_files:

        galaxy_name = file_path.stem

        try:

            summary = run_single_galaxy(file_path, n_params=n_params)

            results.append(summary)

            print(f"[OK][beta_abar] {galaxy_name}")

        except Exception as exc:

            results.append(
                {
                    "galaxy": galaxy_name,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "source_file": str(file_path),
                    "beta_formula": "median(a_bar)",
                    "beta_value": np.nan,
                    "rs_kpc": SIGMA_RS_KPC,
                }
            )

            print(f"[FAIL][beta_abar] {galaxy_name}: {exc}")

    df_summary = pd.DataFrame(results)

    df_summary.to_csv(
        RESULTS_SUMMARY_DIR / "rotation_batch_summary_beta_abar.csv",
        index=False,
    )

    return df_summary


# ----------------------------
# main
# ----------------------------

if __name__ == "__main__":

    summary_df = run_all_galaxies(n_params=N_PARAMS)

    print(summary_df.head())

    print(f"\nTotal galaxies processed: {len(summary_df)}")