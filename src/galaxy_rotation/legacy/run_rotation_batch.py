# src/run_rotation_batch.py

from __future__ import annotations

from pathlib import Path
import traceback
import pandas as pd

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
PROCESSED_DIR = Path("data/processed/sparc_normalized")
RESULTS_TABLE_DIR = Path("results/tables")
RESULTS_SUMMARY_DIR = Path("results/summaries")
FIGURE_DIR = Path("figures/rotation_curves")


# ----- model configuration -----

SIGMA_BETA = 200.0
USE_POSITIVE_SIGMA_ONLY = True
SIGMA_RS_KPC = 2.0
N_PARAMS = 1


def ensure_directories() -> None:
    for path in [
        PROCESSED_DIR,
        RESULTS_TABLE_DIR,
        RESULTS_SUMMARY_DIR,
        FIGURE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


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

    # 4. model prediction
    v_model_kmps = compute_model_velocity_curve(
        df_norm,
        beta=SIGMA_BETA,
        use_positive_only=USE_POSITIVE_SIGMA_ONLY,
        rs_kpc=SIGMA_RS_KPC,
    )

    if len(v_model_kmps) != len(df_norm):
        raise ValueError(
            f"Model length mismatch for {galaxy_name}: "
            f"{len(v_model_kmps)} != {len(df_norm)}"
        )

    # 5. residual table
    df_resid = build_residual_dataframe(df_norm, v_model_kmps)
    resid_path = RESULTS_TABLE_DIR / f"{galaxy_name}_residuals.csv"
    df_resid.to_csv(resid_path, index=False)

    # 6. summary metrics
    summary = summarize_fit_metrics(df_resid, n_params=n_params)

    # 7. plots
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
            print(f"[OK] {galaxy_name}")
        except Exception as exc:
            results.append(
                {
                    "galaxy": galaxy_name,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "source_file": str(file_path),
                }
            )
            print(f"[FAIL] {galaxy_name}: {exc}")

    df_summary = pd.DataFrame(results)
    df_summary.to_csv(
        RESULTS_SUMMARY_DIR / "rotation_batch_summary.csv",
        index=False,
    )

    return df_summary


if __name__ == "__main__":
    summary_df = run_all_galaxies(n_params=N_PARAMS)
    print(summary_df.head())
    print(f"\nTotal galaxies processed: {len(summary_df)}")
