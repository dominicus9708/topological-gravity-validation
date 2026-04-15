from __future__ import annotations

from pathlib import Path
import pandas as pd

from galaxy_rotation_legacy_data_loader import load_rotation_curve_file
from galaxy_rotation_legacy_normalize_sparc import (
    normalize_sparc_dataframe,
    validate_normalized_sparc,
    sort_by_radius,
)
from galaxy_rotation_legacy_sigma_model import compute_model_velocity_curve
from galaxy_rotation_legacy_residuals import (
    build_residual_dataframe,
    summarize_fit_metrics,
)
from galaxy_rotation_legacy_plot_rotation_curve import (
    plot_rotation_curve,
    plot_residuals,
)


# ----- project paths -----

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sparc_csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "sparc_normalized_beta_formula"
RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables_beta_formula"
RESULTS_SUMMARY_DIR = PROJECT_ROOT / "results" / "summaries_beta_formula"
FIGURE_DIR = PROJECT_ROOT / "figures" / "rotation_curves_beta_formula"


# ----- batch configuration -----

UNIT_MAP = {
    "r": "kpc",
    "v_obs": "km/s",
    "v_err": "km/s",
    "v_gas": "km/s",
    "v_disk": "km/s",
    "v_bul": "km/s",
}


# ----- beta inference -----

def infer_beta_from_dataframe(df_norm: pd.DataFrame) -> float:
    """
    Infer beta_g from observational acceleration scale.
    Current prescription:
        beta_g = median(a_obs)

    where a_obs is in (km/s)^2 / kpc.
    """
    if "a_obs_kmps2_per_kpc" not in df_norm.columns:
        raise ValueError("Normalized dataframe must contain 'a_obs_kmps2_per_kpc'.")

    beta_g = float(df_norm["a_obs_kmps2_per_kpc"].median())

    if beta_g <= 0:
        raise ValueError(f"Inferred beta_g must be positive, got {beta_g}")

    return beta_g


# ----- helpers -----

def ensure_directories() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def run_one_galaxy(file_path: Path) -> dict:
    galaxy_name = file_path.stem

    print(f"\n[RUN beta_formula] {galaxy_name}")

    # 1) load raw rotation data
    df_raw = load_rotation_curve_file(file_path)

    # 2) normalize units / schema
    df_norm = normalize_sparc_dataframe(
        df_raw,
        unit_map=UNIT_MAP,
        galaxy_name=galaxy_name,
    )
    validate_normalized_sparc(df_norm)
    df_norm = sort_by_radius(df_norm)

    # 3) save normalized data
    normalized_out = PROCESSED_DIR / f"{galaxy_name}_normalized_beta_formula.csv"
    df_norm.to_csv(normalized_out, index=False)

    # 4) infer beta from data
    beta_g = infer_beta_from_dataframe(df_norm)

    # 5) model prediction with inferred beta
    v_model = compute_model_velocity_curve(df_norm, beta_g=beta_g)

    # 6) residual dataframe
    df_resid = build_residual_dataframe(df_norm, v_model)

    # 7) save residual table
    residual_out = RESULTS_TABLE_DIR / f"{galaxy_name}_residuals_beta_formula.csv"
    df_resid.to_csv(residual_out, index=False)

    # 8) summarize fit metrics
    summary = summarize_fit_metrics(df_resid)
    summary["galaxy"] = galaxy_name
    summary["n_points"] = int(len(df_resid))
    summary["source_file"] = str(file_path)
    summary["beta_g"] = beta_g

    # 9) plots
    rotation_curve_out = FIGURE_DIR / f"{galaxy_name}_rotation_curve_beta_formula.png"
    residual_plot_out = FIGURE_DIR / f"{galaxy_name}_residuals_beta_formula.png"

    plot_rotation_curve(
        df_resid,
        output_path=rotation_curve_out,
        show=False,
    )
    plot_residuals(
        df_resid,
        output_path=residual_plot_out,
        show=False,
    )

    print(f"[DONE beta_formula] {galaxy_name} (beta_g={beta_g:.6g})")
    return summary


def run_batch_beta_formula() -> pd.DataFrame:
    ensure_directories()

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw input directory not found: {RAW_DIR}")

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {RAW_DIR}")

    summaries = []

    for file_path in csv_files:
        try:
            result = run_one_galaxy(file_path)
            result["status"] = "ok"
        except Exception as exc:
            print(f"[FAILED beta_formula] {file_path.name} -> {exc}")
            result = {
                "galaxy": file_path.stem,
                "source_file": str(file_path),
                "status": "failed",
                "error": str(exc),
            }

        summaries.append(result)

    df_summary = pd.DataFrame(summaries)

    summary_out = RESULTS_SUMMARY_DIR / "rotation_batch_summary_beta_formula.csv"
    df_summary.to_csv(summary_out, index=False)

    print("\nBatch complete (beta_formula).")
    print(f"Processed files: {len(df_summary)}")
    print(f"Summary written to: {summary_out}")

    return df_summary


if __name__ == "__main__":
    run_batch_beta_formula()