from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from structural_coupling_preprocessing import load_processed_sparc_table
from structural_coupling_sigma_model import compute_sigma_profile
from structural_coupling_beta_model import (
    compute_local_beta_profile,
    get_constant_beta,
    get_structural_beta,
)
from structural_coupling_acceleration_model import (
    compute_newtonian_acceleration,
    compute_structural_acceleration,
    compute_total_acceleration,
)
from structural_coupling_rotation_curve_model import compute_rotation_velocity
from structural_coupling_pipeline_metrics import rmse, mean_absolute_fractional_error
from structural_coupling_plotting import save_rotation_curve_plot

FILE_DIR = Path(__file__).resolve().parent
PIPELINE_NAME = "pipeline_structural_coupling"


def _find_project_root(start_dir: Path) -> Path:
    candidates = [start_dir, *start_dir.parents]
    for candidate in candidates:
        if (candidate / "data").exists() or (candidate / "README.md").exists():
            return candidate
    return start_dir


PROJECT_ROOT = _find_project_root(FILE_DIR)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "sparc_normalized"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "galaxy_rotation" / PIPELINE_NAME


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
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing baryonic velocity columns: {missing}")
    return compute_newtonian_acceleration(
        r=r_kpc,
        v_gas=df["v_gas_kmps"].to_numpy(dtype=float),
        v_disk=df["v_disk_kmps"].to_numpy(dtype=float),
        v_bulge=df["v_bul_kmps"].to_numpy(dtype=float),
    )


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
    for metric in ["rmse_kmps", "mean_absolute_fractional_error", "sigma_flip_severity"]:
        if metric not in ok.columns:
            continue
        ranked = ok.sort_values(metric, ascending=True)
        ranked.head(n_rows).to_csv(tables_dir / f"best_{metric}_{_sanitize_label(beta_mode)}.csv", index=False)
        ranked.tail(n_rows).to_csv(tables_dir / f"worst_{metric}_{_sanitize_label(beta_mode)}.csv", index=False)


def _build_run_summary(df_results: pd.DataFrame, *, beta_mode: str, beta_value: float, data_dir: Path, output_root: Path, run_dir: Path) -> dict:
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
        for column in [
            "beta_mean", "beta_max", "d_eff_mean", "d_eff_max", "cinfo_cap_mean", "eta_mean", "projection_mean",
            "struct_to_bar_ratio_mean", "struct_to_bar_ratio_max", "sigma_flip_severity", "rmse_kmps",
            "mean_absolute_fractional_error", "n_points",
        ]:
            if column not in ok.columns:
                continue
            numeric = pd.to_numeric(ok[column], errors="coerce")
            numeric = numeric[np.isfinite(numeric)]
            if len(numeric) == 0:
                continue
            summary[f"{column}_mean"] = float(numeric.mean())
            summary[f"{column}_median"] = float(numeric.median())
            summary[f"{column}_min"] = float(numeric.min())
            summary[f"{column}_max"] = float(numeric.max())
    if not failed.empty and "galaxy" in failed.columns:
        summary["failed_galaxies"] = [str(v) for v in failed["galaxy"].fillna("").tolist()]
    return summary


def _standardize_result_columns(df_results: pd.DataFrame) -> pd.DataFrame:
    ordered_columns = [
        "galaxy", "source_file", "beta_mode", "beta_mean", "beta_max", "d_eff_mean", "d_eff_max", "cinfo_cap_mean",
        "eta_mean", "projection_mean", "struct_to_bar_ratio_mean", "struct_to_bar_ratio_max", "sigma_flip_severity",
        "a_newton_source", "rmse_kmps", "mean_absolute_fractional_error", "n_points", "status", "error",
    ]
    for column in ordered_columns:
        if column not in df_results.columns:
            df_results[column] = "" if column in {"a_newton_source", "status", "error"} else np.nan
    return df_results.loc[:, ordered_columns].copy()


def _save_galaxy_diagnostics_table(output_path: Path, **columns) -> None:
    pd.DataFrame(columns).to_csv(output_path, index=False)


def run_rotation_pipeline(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    beta_mode: str = "local_structural_coupling_v6",
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
        "Structural coupling rotation pipeline run log",
        f"started_at: {datetime.now().isoformat()}",
        f"project_root: {PROJECT_ROOT}",
        f"data_dir: {data_dir}",
        f"output_root: {output_root}",
        f"run_dir: {run_paths['run_dir']}",
        f"beta_mode: {beta_mode}",
        f"beta_value: {beta_value}",
        f"n_input_files: {len(csv_files)}",
        "-" * 72,
    ]
    results: list[dict[str, object]] = []

    for index, file_path in enumerate(csv_files, start=1):
        try:
            df = load_processed_sparc_table(file_path)
            galaxy_name = _resolve_galaxy_name(df, file_path)

            r_kpc = df["r_kpc"].to_numpy(dtype=float)
            v_obs = df["v_obs_kmps"].to_numpy(dtype=float)
            v_err = df["v_err_kmps"].to_numpy(dtype=float)
            a_newton = _compute_baryonic_newtonian_acceleration(df, r_kpc)
            sigma_profile = compute_sigma_profile(r_kpc, a_newton)

            if beta_mode == "constant":
                beta_local = np.full_like(r_kpc, get_constant_beta(beta_value), dtype=float)
                d_eff_local = np.full_like(r_kpc, 2.20, dtype=float)
                cinfo_cap = np.full_like(r_kpc, 0.52, dtype=float)
                eta_profile = np.full_like(r_kpc, 0.46, dtype=float)
                projection_profile = np.full_like(r_kpc, 0.80, dtype=float)
            elif beta_mode in {"structural", "structural_beta"}:
                beta_scalar = get_structural_beta(df, sigma_profile)
                beta_local = np.full_like(r_kpc, beta_scalar, dtype=float)
                d_eff_local = np.full_like(r_kpc, 2.20, dtype=float)
                cinfo_cap = np.full_like(r_kpc, 0.52, dtype=float)
                eta_profile = np.full_like(r_kpc, 0.46, dtype=float)
                projection_profile = np.full_like(r_kpc, 0.80, dtype=float)
            else:
                beta_payload = compute_local_beta_profile(r_kpc, sigma_profile, a_newton, beta0=beta_value)
                beta_local = beta_payload["beta_local"]
                d_eff_local = beta_payload["d_eff_local"]
                cinfo_cap = beta_payload["cinfo_cap"]
                eta_profile = beta_payload["eta_profile"]
                projection_profile = beta_payload["projection_profile"]

            struct_diag = compute_structural_acceleration(
                r=r_kpc,
                sigma=sigma_profile,
                beta=beta_local,
                g_newton=a_newton,
                eta_profile=eta_profile,
                d_eff_local=d_eff_local,
                cinfo_cap=cinfo_cap,
                projection_profile=projection_profile,
                return_diagnostics=True,
            )
            a_struct = struct_diag["a_struct"]
            a_total = compute_total_acceleration(a_newton, a_struct)
            v_model = compute_rotation_velocity(r_kpc, a_total)
            v_baryon = compute_rotation_velocity(r_kpc, a_newton)

            rmse_val = rmse(v_obs, v_model)
            mf_error = mean_absolute_fractional_error(v_obs, v_model)

            galaxy_table_path = run_paths["tables_dir"] / f"{_sanitize_label(galaxy_name)}_rotation_table.csv"
            _save_galaxy_diagnostics_table(
                galaxy_table_path,
                r_kpc=r_kpc,
                v_obs_kmps=v_obs,
                v_err_kmps=v_err,
                v_model_kmps=v_model,
                v_baryon_kmps=v_baryon,
                a_newton=a_newton,
                sigma_profile=sigma_profile,
                sigma_smoothed=struct_diag["sigma_smoothed"],
                sigma_transferred=struct_diag["sigma_transferred"],
                beta_local=beta_local,
                d_eff_local=d_eff_local,
                cinfo_cap=cinfo_cap,
                eta_profile=eta_profile,
                projection_profile=struct_diag["projection_profile"],
                a_struct_internal=struct_diag["a_struct_internal"],
                a_struct_observed=struct_diag["a_struct_observed"],
                a_struct=a_struct,
                a_total=a_total,
                struct_to_bar_ratio=struct_diag["struct_to_bar_ratio"],
                sigma_flip_severity=np.full_like(r_kpc, struct_diag["sigma_flip_severity"], dtype=float),
            )

            plot_path = run_paths["individual_dir"] / f"{galaxy_name}_rotation_curve.png"
            save_rotation_curve_plot(
                r_kpc,
                v_obs,
                v_err,
                v_model,
                galaxy_name,
                plot_path,
                v_baryon_kmps=v_baryon,
                sigma_profile=sigma_profile,
                sigma_smoothed=struct_diag["sigma_smoothed"],
                sigma_transferred=struct_diag["sigma_transferred"],
                beta_local=beta_local,
                a_struct_internal=struct_diag["a_struct_internal"],
                a_struct=a_struct,
                d_eff_local=d_eff_local,
                cinfo_cap=cinfo_cap,
                eta_profile=eta_profile,
                projection_profile=struct_diag["projection_profile"],
            )

            ratio = np.asarray(struct_diag["struct_to_bar_ratio"], dtype=float)
            result = {
                "galaxy": galaxy_name,
                "source_file": str(file_path),
                "beta_mode": beta_mode,
                "beta_mean": float(np.nanmean(beta_local)),
                "beta_max": float(np.nanmax(beta_local)),
                "d_eff_mean": float(np.nanmean(d_eff_local)),
                "d_eff_max": float(np.nanmax(d_eff_local)),
                "cinfo_cap_mean": float(np.nanmean(cinfo_cap)),
                "eta_mean": float(np.nanmean(eta_profile)),
                "projection_mean": float(np.nanmean(struct_diag["projection_profile"])),
                "struct_to_bar_ratio_mean": float(np.nanmean(ratio)),
                "struct_to_bar_ratio_max": float(np.nanmax(ratio)),
                "sigma_flip_severity": float(struct_diag["sigma_flip_severity"]),
                "a_newton_source": "baryonic_velocity_components",
                "rmse_kmps": float(rmse_val),
                "mean_absolute_fractional_error": float(mf_error),
                "n_points": int(len(r_kpc)),
                "status": "ok",
                "error": "",
            }
            results.append(result)
            log_lines.append(
                f"[{index:03d}/{len(csv_files):03d}] [OK] {galaxy_name} | "
                f"beta_mean={result['beta_mean']:.4f} | eta_mean={result['eta_mean']:.6f} | "
                f"projection_mean={result['projection_mean']:.6f} | ratio_max={result['struct_to_bar_ratio_max']:.6f} | "
                f"sigma_flip={result['sigma_flip_severity']:.6f} | rmse={result['rmse_kmps']:.4f} | mafe={result['mean_absolute_fractional_error']:.6f}"
            )
            print(f"[OK] {galaxy_name}")
        except Exception as exc:
            failed_name = file_path.stem.replace("_normalized", "")
            error_message = str(exc)
            results.append({
                "galaxy": failed_name,
                "source_file": str(file_path),
                "beta_mode": beta_mode,
                "status": "failed",
                "error": error_message,
            })
            log_lines.append(f"[{index:03d}/{len(csv_files):03d}] [FAILED] {failed_name} -> {error_message}")
            print(f"[FAILED] {file_path.name} -> {error_message}")

    df_results = _standardize_result_columns(pd.DataFrame(results))
    summary_filename = f"rotation_pipeline_summary_{mode_label}.csv"
    summary_path = run_paths["summaries_dir"] / summary_filename
    table_copy_path = run_paths["tables_dir"] / summary_filename
    df_results.to_csv(summary_path, index=False)
    df_results.to_csv(table_copy_path, index=False)

    ok = df_results[df_results["status"] == "ok"].copy()
    if not ok.empty:
        distributions = [
            ("rmse_kmps", f"RMSE distribution ({beta_mode})", "RMSE [km/s]"),
            ("mean_absolute_fractional_error", f"MAFE distribution ({beta_mode})", "Mean absolute fractional error"),
            ("beta_mean", f"Beta mean distribution ({beta_mode})", "Beta mean"),
            ("eta_mean", f"Eta mean distribution ({beta_mode})", "Eta mean"),
            ("projection_mean", f"Projection mean distribution ({beta_mode})", "Projection mean"),
            ("struct_to_bar_ratio_max", f"Structural-to-baryonic ratio max ({beta_mode})", "max |a_struct| / |a_bar|"),
            ("sigma_flip_severity", f"Sigma flip severity ({beta_mode})", "max adjacent |Δ sigma_transfer|"),
        ]
        for col, title, xlabel in distributions:
            _save_distribution_plot(
                ok[col].to_numpy(dtype=float),
                title=title,
                xlabel=xlabel,
                output_path=run_paths["figures_dir"] / f"{col}_distribution_{mode_label}.png",
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

    log_lines.append("-" * 72)
    log_lines.append(f"finished_at: {datetime.now().isoformat()}")
    log_lines.append(f"successful_files: {run_summary['successful_files']}")
    log_lines.append(f"failed_files: {run_summary['failed_files']}")
    log_lines.append(f"summary_csv: {summary_path}")
    _write_text_log(log_lines, run_paths["logs_dir"] / f"run_log_{mode_label}.txt")
    return df_results


if __name__ == "__main__":
    summary = run_rotation_pipeline(
        data_dir=DEFAULT_DATA_DIR,
        output_root=DEFAULT_OUTPUT_ROOT,
        beta_mode="local_structural_coupling_v6",
        beta_value=200.0,
    )
    print(summary.head())
    print(f"\nTotal files processed: {len(summary)}")
