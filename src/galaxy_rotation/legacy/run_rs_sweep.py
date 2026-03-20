# src/run_rs_sweep.py

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


RAW_DIR = Path("data/raw/sparc_csv")
SWEEP_DIR = Path("results/rs_sweep")

SIGMA_BETA = 200.0
USE_POSITIVE_SIGMA_ONLY = True
N_PARAMS = 1

RS_VALUES = [1.0, 1.5, 2.0, 3.0, 5.0]


def ensure_directories() -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)


def run_single_galaxy_for_rs(file_path: Path, rs_kpc: float, n_params: int = N_PARAMS) -> dict:
    galaxy_name = file_path.stem

    df_raw = load_rotation_curve_file(file_path)
    df_norm = normalize_sparc_dataframe(df_raw, galaxy_name=galaxy_name)
    validate_normalized_sparc(df_norm)
    df_norm = sort_by_radius(df_norm)

    v_model_kmps = compute_model_velocity_curve(
        df_norm,
        beta=SIGMA_BETA,
        use_positive_only=USE_POSITIVE_SIGMA_ONLY,
        rs_kpc=rs_kpc,
    )

    df_resid = build_residual_dataframe(df_norm, v_model_kmps)
    summary = summarize_fit_metrics(df_resid, n_params=n_params)

    summary["status"] = "success"
    summary["source_file"] = str(file_path)
    summary["rs_kpc"] = rs_kpc
    return summary


def run_all_for_rs(rs_kpc: float, n_params: int = N_PARAMS) -> pd.DataFrame:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DIR}")

    all_files = sorted(RAW_DIR.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    results = []

    for file_path in all_files:
        galaxy_name = file_path.stem
        try:
            summary = run_single_galaxy_for_rs(file_path, rs_kpc=rs_kpc, n_params=n_params)
            results.append(summary)
            print(f"[OK][rs={rs_kpc}] {galaxy_name}")
        except Exception as exc:
            results.append(
                {
                    "galaxy": galaxy_name,
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "source_file": str(file_path),
                    "rs_kpc": rs_kpc,
                }
            )
            print(f"[FAIL][rs={rs_kpc}] {galaxy_name}: {exc}")

    df_summary = pd.DataFrame(results)
    out_path = SWEEP_DIR / f"rotation_batch_summary_rs_{str(rs_kpc).replace('.', 'p')}.csv"
    df_summary.to_csv(out_path, index=False)
    return df_summary


def summarize_rs_result(df_summary: pd.DataFrame) -> dict:
    ok = df_summary[df_summary["status"] == "success"].copy()

    if ok.empty:
        return {
            "n_success": 0,
            "mean_rmse": None,
            "median_rmse": None,
            "rmse_lt_10": 0,
            "rmse_lt_30": 0,
            "rmse_ge_60": 0,
        }

    return {
        "n_success": int(len(ok)),
        "mean_rmse": float(ok["rmse_kmps"].mean()),
        "median_rmse": float(ok["rmse_kmps"].median()),
        "rmse_lt_10": int((ok["rmse_kmps"] < 10).sum()),
        "rmse_lt_30": int((ok["rmse_kmps"] < 30).sum()),
        "rmse_ge_60": int((ok["rmse_kmps"] >= 60).sum()),
    }


def run_rs_sweep():
    ensure_directories()

    rows = []

    for rs in RS_VALUES:
        print(f"\n===== Running rs_kpc = {rs} =====")
        df_summary = run_all_for_rs(rs_kpc=rs, n_params=N_PARAMS)
        metrics = summarize_rs_result(df_summary)

        row = {"rs_kpc": rs}
        row.update(metrics)
        rows.append(row)

    df_compare = pd.DataFrame(rows)
    compare_path = SWEEP_DIR / "rs_sweep_comparison.csv"
    df_compare.to_csv(compare_path, index=False)

    print("\n===== RS SWEEP SUMMARY =====")
    print(df_compare)

    return df_compare


if __name__ == "__main__":
    run_rs_sweep()
