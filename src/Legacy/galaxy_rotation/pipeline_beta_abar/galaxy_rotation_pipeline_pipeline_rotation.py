from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json

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

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "sparc_normalized_beta_abar"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def create_run_output_dirs(mode: str = "beta_abar") -> dict:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_root = DEFAULT_OUTPUT_ROOT / f"{timestamp}_{mode}"

    summary_dir = run_root / "summaries"
    table_dir = run_root / "tables"
    figure_dir = run_root / "figures"
    log_dir = run_root / "logs"
    metadata_dir = run_root / "metadata"

    for d in [summary_dir, table_dir, figure_dir, log_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return summary_dir, table_dir, figure_dir, log_dir, metadata_dir, run_root


def run_pipeline():
    summary_dir, table_dir, figure_dir, log_dir, metadata_dir, run_root = create_run_output_dirs()

    print("[INFO] run_root =", run_root)

    results = []

    csv_files = sorted(DEFAULT_DATA_DIR.glob("*_normalized.csv"))

    for file_path in csv_files:
        try:
            df = load_processed_sparc_table(file_path)
            df = df.sort_values("r_kpc").reset_index(drop=True)

            galaxy_name = file_path.stem.replace("_normalized", "")

            r_kpc = df["r_kpc"].to_numpy(dtype=float)
            v_obs = df["v_obs_kmps"].to_numpy(dtype=float)
            v_err = df["v_err_kmps"].to_numpy(dtype=float)

            v_gas = df["v_gas_kmps"].to_numpy(dtype=float)
            v_disk = df["v_disk_kmps"].to_numpy(dtype=float)
            v_bul = df["v_bul_kmps"].to_numpy(dtype=float)

            a_newton = (v_gas**2 + v_disk**2 + v_bul**2) / r_kpc

            sigma = compute_sigma_profile(r_kpc, a_newton)

            # DEBUG
            print("[DEBUG] beta_model path =", get_structural_beta.__code__.co_filename)

            beta = get_structural_beta(df, sigma)

            a_struct = compute_structural_acceleration(r_kpc, sigma, beta)
            a_total = compute_total_acceleration(a_newton, a_struct)

            v_model = compute_rotation_velocity(a_total, r_kpc)

            rmse_val = rmse(v_obs, v_model)
            maf = mean_absolute_fractional_error(v_obs, v_model)

            results.append({
                "galaxy": galaxy_name,
                "beta": beta,
                "rmse": rmse_val,
                "maf": maf,
                "status": "ok"
            })

            # 저장
            pd.DataFrame({
                "r_kpc": r_kpc,
                "v_obs": v_obs,
                "v_model": v_model
            }).to_csv(table_dir / f"{galaxy_name}.csv", index=False)

            save_rotation_curve_plot(
                r_kpc,
                v_obs,
                v_err,
                v_model,
                galaxy_name,
                figure_dir / f"{galaxy_name}.png"
            )

            print("[OK]", galaxy_name)

        except Exception as e:
            print("[FAILED]", file_path.name, "->", e)

    # summary 저장
    df_summary = pd.DataFrame(results)
    df_summary.to_csv(summary_dir / "summary_beta_abar.csv", index=False)

    # log
    with open(log_dir / "run_log.txt", "w") as f:
        f.write(f"run_root: {run_root}\n")

    # metadata
    with open(metadata_dir / "meta.json", "w") as f:
        json.dump({"mode": "beta_abar"}, f, indent=2)

    print("DONE")


if __name__ == "__main__":
    run_pipeline()