from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _preprocessing import load_processed_sparc_table
from _sigma_model import compute_sigma_profile
from _beta_model import get_constant_beta, get_structural_beta
from _acceleration_model import (
    compute_newtonian_acceleration,
    compute_structural_acceleration,
    compute_total_acceleration,
)
from _rotation_curve_model import compute_rotation_velocity
from _metrics import rmse, mean_absolute_fractional_error
from _plotting import save_rotation_curve_plot


FILE_DIR = Path(__file__).resolve().parent
PIPELINE_NAME = "pipeline_base"


# ---------------------------------------------------------------------
# Project path resolution
# ---------------------------------------------------------------------
def _find_project_root(start_dir: Path) -> Path:
    candidates = [start_dir, *start_dir.parents]
    for candidate in candidates:
        if (candidate / "data").exists() or (candidate / "README.md").exists():
            return candidate
    return start_dir


PROJECT_ROOT = _find_project_root(FILE_DIR)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "sparc_normalized"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "galaxy_rotation" / PIPELINE_NAME


# ---------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------
def ensure_directories(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)



def _sanitize_label(value: str) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)



def _create_run_directories(beta_mode: str, output_root: Path) -> dict[str, Path]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = f"{timestamp}_{_sanitize_label(beta_mode)}"
    run_dir = output_root / run_name

    paths = {
        "run_dir": run_dir,
        "figures_dir": run_dir / "figures",
        "individual_dir": run_dir / "figures" / "individual",
        "logs_dir": run_dir / "logs",
        "metadata_dir": run_dir / "metadata",
        "summaries_dir": run_dir / "summaries",
        "tables_dir": run_dir / "tables",
    }
    ensure_directories(*paths.values())
    return paths


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------
def _resolve_galaxy_name(df: pd.DataFrame, file_path: Path) -> str:
    if "galaxy" in df.columns and not df["galaxy"].empty:
        value = str(df["galaxy"].iloc[0]).strip()
        if value:
            return value

    name = file_path.stem
    if name.endswith("_normalized"):
        name = name[: -len("_normalized")]
    return name



def _build_rotation_table(
    r_kpc: np.ndarray,
    v_obs_kmps: np.ndarray,
    v_err_kmps: np.ndarray,
    a_newton: np.ndarray,
    sigma_profile: np.ndarray,
    a_struct: np.ndarray,
    a_total: np.ndarray,
    v_model_kmps: np.ndarray,
) -> pd.DataFrame:
    residual_kmps = v_obs_kmps - v_model_kmps
    fractional_residual = residual_kmps / np.maximum(np.abs(v_obs_kmps), 1e-12)

    return pd.DataFrame(
        {
            "r_kpc": r_kpc,
            "v_obs_kmps": v_obs_kmps,
            "v_err_kmps": v_err_kmps,
            "a_newton": a_newton,
            "sigma_profile": sigma_profile,
            "a_struct": a_struct,
            "a_total": a_total,
            "v_model_kmps": v_model_kmps,
            "residual_kmps": residual_kmps,
            "fractional_residual": fractional_residual,
        }
    )


# ---------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------
def _write_text_log(lines: list[str], path: Path) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def _write_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")



def _save_distribution_plot(values: np.ndarray, title: str, xlabel: str, output_path: Path) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    bins = min(20, max(5, int(np.sqrt(values.size))))
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()



def _save_rank_tables(df_results: pd.DataFrame, tables_dir: Path, beta_mode: str, n_rows: int = 20) -> None:
    ok = df_results[df_results["status"] == "ok"].copy()
    if ok.empty:
        return

    ok["rmse_kmps"] = pd.to_numeric(ok["rmse_kmps"], errors="coerce")
    ok["mean_absolute_fractional_error"] = pd.to_numeric(
        ok["mean_absolute_fractional_error"],
        errors="coerce",
    )
    ok = ok.dropna(subset=["rmse_kmps", "mean_absolute_fractional_error"])
    if ok.empty:
        return

    suffix = _sanitize_label(beta_mode)
    ok.sort_values("rmse_kmps", ascending=True).head(n_rows).to_csv(
        tables_dir / f"best_fit_galaxies_rmse_{suffix}.csv",
        index=False,
    )
    ok.sort_values("rmse_kmps", ascending=False).head(n_rows).to_csv(
        tables_dir / f"worst_fit_galaxies_rmse_{suffix}.csv",
        index=False,
    )
    ok.sort_values("mean_absolute_fractional_error", ascending=True).head(n_rows).to_csv(
        tables_dir / f"best_fit_galaxies_mafe_{suffix}.csv",
        index=False,
    )
    ok.sort_values("mean_absolute_fractional_error", ascending=False).head(n_rows).to_csv(
        tables_dir / f"worst_fit_galaxies_mafe_{suffix}.csv",
        index=False,
    )



def _standardize_result_columns(df_results: pd.DataFrame) -> pd.DataFrame:
    ordered_columns = [
        "galaxy",
        "source_file",
        "beta_mode",
        "beta",
        "a_newton_source",
        "rmse_kmps",
        "mean_absolute_fractional_error",
        "n_points",
        "status",
        "error",
    ]
    for column in ordered_columns:
        if column not in df_results.columns:
            df_results[column] = "" if column in {"a_newton_source", "status", "error"} else np.nan
    return df_results.loc[:, ordered_columns].copy()



def _build_run_summary(
    df_results: pd.DataFrame,
    *,
    beta_mode: str,
    beta_value: float,
    data_dir: Path,
    output_root: Path,
    run_dir: Path,
) -> dict:
    total_files = int(len(df_results))
    ok = df_results[df_results["status"] == "ok"].copy()
    failed = df_results[df_results["status"] != "ok"].copy()

    summary: dict[str, object] = {
        "pipeline_name": PIPELINE_NAME,
        "generated_at": datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(data_dir),
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "beta_mode": beta_mode,
        "beta_value_input": float(beta_value),
        "total_files": total_files,
        "successful_files": int(len(ok)),
        "failed_files": int(len(failed)),
        "success_rate": float(len(ok) / total_files) if total_files else 0.0,
    }

    if not ok.empty:
        for column in ["rmse_kmps", "mean_absolute_fractional_error", "beta", "n_points"]:
            numeric = pd.to_numeric(ok[column], errors="coerce")
            numeric = numeric[np.isfinite(numeric)]
            if len(numeric) == 0:
                continue
            summary[f"{column}_mean"] = float(numeric.mean())
            summary[f"{column}_median"] = float(numeric.median())
            summary[f"{column}_min"] = float(numeric.min())
            summary[f"{column}_max"] = float(numeric.max())

    if not failed.empty and "galaxy" in failed.columns:
        summary["failed_galaxies"] = [str(value) for value in failed["galaxy"].fillna("").tolist()]

    return summary


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------
def run_rotation_pipeline(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    beta_mode: str = "constant",
    beta_value: float = 200.0,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    output_root = Path(output_root)

    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {data_dir}")

    csv_files = sorted(data_dir.glob("*_normalized.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No normalized CSV files found in: {data_dir}")

    run_paths = _create_run_directories(beta_mode=beta_mode, output_root=output_root)
    mode_label = _sanitize_label(beta_mode)

    log_lines = [
        "Base rotation pipeline run log",
        f"started_at: {datetime.now().isoformat()}",
        f"project_root: {PROJECT_ROOT}",
        f"data_dir: {data_dir}",
        f"output_root: {output_root}",
        f"run_dir: {run_paths['run_dir']}",
        f"beta_mode: {beta_mode}",
        f"beta_value: {beta_value}",
        f"n_input_files: {len(csv_files)}",
        "",
    ]

    results: list[dict] = []

    for file_path in csv_files:
        try:
            df = load_processed_sparc_table(file_path)
            df = df.sort_values("r_kpc").reset_index(drop=True)

            galaxy_name = _resolve_galaxy_name(df, file_path)
            r_kpc = df["r_kpc"].to_numpy(dtype=float)
            v_obs_kmps = df["v_obs_kmps"].to_numpy(dtype=float)
            v_err_kmps = df["v_err_kmps"].to_numpy(dtype=float)
            v_gas = df["v_gas_kmps"].to_numpy(dtype=float)
            v_disk = df["v_disk_kmps"].to_numpy(dtype=float)
            v_bulge = df["v_bul_kmps"].to_numpy(dtype=float)

            a_newton = compute_newtonian_acceleration(
                r=r_kpc,
                v_gas=v_gas,
                v_disk=v_disk,
                v_bulge=v_bulge,
            )
            sigma_profile = compute_sigma_profile(r_kpc, a_newton)

            if beta_mode == "constant":
                beta = get_constant_beta(beta_value)
            elif beta_mode == "structural":
                beta = get_structural_beta(df, sigma_profile)
            else:
                raise ValueError(f"Unknown beta_mode: {beta_mode}")

            a_struct = compute_structural_acceleration(r_kpc, sigma_profile, beta)
            a_total = compute_total_acceleration(a_newton, a_struct)
            v_model_kmps = compute_rotation_velocity(r_kpc, a_total)

            rmse_val = rmse(v_obs_kmps, v_model_kmps)
            mafe_val = mean_absolute_fractional_error(v_obs_kmps, v_model_kmps)

            rotation_table = _build_rotation_table(
                r_kpc=r_kpc,
                v_obs_kmps=v_obs_kmps,
                v_err_kmps=v_err_kmps,
                a_newton=a_newton,
                sigma_profile=sigma_profile,
                a_struct=a_struct,
                a_total=a_total,
                v_model_kmps=v_model_kmps,
            )
            rotation_table.to_csv(
                run_paths["tables_dir"] / f"{galaxy_name}_rotation_table.csv",
                index=False,
            )

            save_rotation_curve_plot(
                r_kpc=r_kpc,
                v_obs_kmps=v_obs_kmps,
                v_err_kmps=v_err_kmps,
                v_model_kmps=v_model_kmps,
                galaxy_name=galaxy_name,
                output_path=run_paths["individual_dir"] / f"{galaxy_name}_rotation_curve.png",
            )

            results.append(
                {
                    "galaxy": galaxy_name,
                    "source_file": str(file_path),
                    "beta_mode": beta_mode,
                    "beta": float(beta),
                    "a_newton_source": "computed_from_baryonic_velocity",
                    "rmse_kmps": float(rmse_val),
                    "mean_absolute_fractional_error": float(mafe_val),
                    "n_points": int(len(df)),
                    "status": "ok",
                    "error": "",
                }
            )
            log_lines.append(f"[OK] {galaxy_name}")
            print(f"[OK] {galaxy_name}")

        except Exception as exc:
            results.append(
                {
                    "galaxy": file_path.stem,
                    "source_file": str(file_path),
                    "beta_mode": beta_mode,
                    "beta": float(beta_value) if beta_mode == "constant" else np.nan,
                    "a_newton_source": "",
                    "rmse_kmps": np.nan,
                    "mean_absolute_fractional_error": np.nan,
                    "n_points": np.nan,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            log_lines.append(f"[FAILED] {file_path.name} -> {exc}")
            print(f"[FAILED] {file_path.name} -> {exc}")

    df_results = _standardize_result_columns(pd.DataFrame(results))

    summary_path = run_paths["summaries_dir"] / f"rotation_pipeline_summary_{mode_label}.csv"
    df_results.to_csv(summary_path, index=False)
    df_results.to_csv(run_paths["tables_dir"] / f"rotation_pipeline_summary_{mode_label}.csv", index=False)

    ok = df_results[df_results["status"] == "ok"].copy()
    if not ok.empty:
        _save_distribution_plot(
            pd.to_numeric(ok["rmse_kmps"], errors="coerce"),
            title=f"RMSE Distribution ({beta_mode})",
            xlabel="RMSE [km/s]",
            output_path=run_paths["figures_dir"] / f"rmse_distribution_{mode_label}.png",
        )
        _save_distribution_plot(
            pd.to_numeric(ok["mean_absolute_fractional_error"], errors="coerce"),
            title=f"MAFE Distribution ({beta_mode})",
            xlabel="Mean absolute fractional error",
            output_path=run_paths["figures_dir"] / f"mafe_distribution_{mode_label}.png",
        )
        _save_distribution_plot(
            pd.to_numeric(ok["beta"], errors="coerce"),
            title=f"Beta Distribution ({beta_mode})",
            xlabel="beta",
            output_path=run_paths["figures_dir"] / f"beta_distribution_{mode_label}.png",
        )
        _save_rank_tables(df_results, run_paths["tables_dir"], beta_mode=beta_mode, n_rows=20)

    run_summary = _build_run_summary(
        df_results,
        beta_mode=beta_mode,
        beta_value=beta_value,
        data_dir=data_dir,
        output_root=output_root,
        run_dir=run_paths["run_dir"],
    )
    _write_json(run_summary, run_paths["metadata_dir"] / f"run_summary_{mode_label}.json")

    log_lines.extend(
        [
            "",
            f"finished_at: {datetime.now().isoformat()}",
            f"successful_files: {run_summary['successful_files']}",
            f"failed_files: {run_summary['failed_files']}",
            f"summary_csv: {summary_path}",
        ]
    )
    _write_text_log(log_lines, run_paths["logs_dir"] / f"run_log_{mode_label}.txt")

    return df_results


if __name__ == "__main__":
    summary = run_rotation_pipeline(
        data_dir=DEFAULT_DATA_DIR,
        output_root=DEFAULT_OUTPUT_ROOT,
        beta_mode="constant",
        beta_value=200.0,
    )

    print(summary.head())
    print(f"\nTotal files processed: {len(summary)}")
