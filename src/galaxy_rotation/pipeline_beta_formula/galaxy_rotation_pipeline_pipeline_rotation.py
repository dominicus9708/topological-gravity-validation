from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd

from galaxy_rotation_pipeline_preprocessing import load_processed_sparc_table
from galaxy_rotation_pipeline_sigma_model import (
    compute_sigma_profile,
    compute_sigma_observables,
)
from galaxy_rotation_pipeline_beta_model import (
    get_constant_beta,
    get_structural_beta_details,
    get_formula_beta_details,
)
from galaxy_rotation_pipeline_acceleration_model import compute_total_acceleration
from galaxy_rotation_pipeline_rotation_curve_model import compute_rotation_velocity
from galaxy_rotation_pipeline_metrics import (
    rmse,
    mae,
    mean_absolute_fractional_error,
    median_absolute_fractional_error,
    chi_square,
    reduced_chi_square,
    valid_point_count,
)
from galaxy_rotation_pipeline_plotting import (
    save_rotation_curve_plot,
    save_beta_histogram,
    save_metric_histogram,
)

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "sparc_normalized_beta_abar"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "galaxy_rotation" / "pipeline_beta_formula"


def _resolve_galaxy_name(df: pd.DataFrame, file_path: Path) -> str:
    if "galaxy" in df.columns and not df["galaxy"].isna().all():
        name = str(df["galaxy"].iloc[0]).strip()
        if name:
            return name
    return file_path.stem.replace("_normalized", "")


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_run_output_dirs(beta_mode: str) -> dict[str, Path]:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_root = DEFAULT_OUTPUT_ROOT / f"{timestamp}_{beta_mode}"

    return {
        "run_root": _ensure_directory(run_root),
        "summary_dir": _ensure_directory(run_root / "summaries"),
        "table_dir": _ensure_directory(run_root / "tables"),
        "figure_dir": _ensure_directory(run_root / "figures"),
        "log_dir": _ensure_directory(run_root / "logs"),
        "metadata_dir": _ensure_directory(run_root / "metadata"),
    }


def _compute_component_velocities(
    r_kpc: np.ndarray,
    a_bar: np.ndarray,
    a_struct: np.ndarray,
    a_total: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r_kpc, dtype=float)
    a_bar = np.asarray(a_bar, dtype=float)
    a_struct = np.asarray(a_struct, dtype=float)
    a_total = np.asarray(a_total, dtype=float)

    v_bar = np.sqrt(np.clip(r * np.clip(a_bar, 0.0, None), 0.0, None))
    v_struct = np.sqrt(np.clip(r * np.clip(a_struct, 0.0, None), 0.0, None))
    v_total = np.sqrt(np.clip(r * np.clip(a_total, 0.0, None), 0.0, None))
    return v_bar, v_struct, v_total


def _write_run_log(log_path: Path, lines: list[str]) -> None:
    with open(log_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def load_estimated_formula_params(json_path: str | Path) -> dict:
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Estimated formula params JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required = [
        "formula_g0",
        "formula_sigma0",
        "formula_p",
        "formula_q",
        "formula_lambda_beta",
        "sigma_weight_u",
        "sigma_weight_v",
        "sigma_weight_lambda",
        "formula_g_choice",
        "formula_sigma_choice",
        "sigma_weight_g_choice",
        "sigma_weight_sigma_choice",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required keys in estimated params JSON: {missing}")

    # 차원 프록시 관련 기본값
    data.setdefault("use_dimension_proxy", None)
    data.setdefault("formula_d_ref", 3.0)
    data.setdefault("formula_dim_sensitivity", 0.85)
    data.setdefault("formula_dim_g_power", 1.0)
    data.setdefault("formula_dim_sigma_power", 1.0)
    data.setdefault("formula_d_min", 1.5)
    data.setdefault("formula_d_max", 4.5)
    data.setdefault("formula_dim_response_lambda", 0.45)
    data.setdefault("formula_dim_transition", 0.35)
    data.setdefault("formula_dim_response_min", 0.55)
    data.setdefault("formula_dim_response_max", 1.85)

    return data


def _estimate_shared_normalizations(
    csv_files: list[Path],
    D_bg: float,
    sigma_positive_only: bool,
    sigma_proxy_mode: str,
    sigma_damping_mode: str,
    upsilon_disk: float,
    upsilon_bul: float,
    g_choice: str,
    sigma_choice: str,
    sigma_weight_mode: str,
    sigma_weight_value: float,
    sigma_weight_u: float,
    sigma_weight_v: float,
    sigma_weight_lambda: float,
    sigma_weight_g_choice: str,
    sigma_weight_sigma_choice: str,
) -> dict[str, float]:
    g_values: list[float] = []
    sigma_values: list[float] = []

    for file_path in csv_files:
        try:
            df = load_processed_sparc_table(file_path)

            sigma_profile = compute_sigma_profile(
                galaxy_data=df,
                D_bg=D_bg,
                positive_only=sigma_positive_only,
                proxy_mode=sigma_proxy_mode,
                damping_mode=sigma_damping_mode,
                upsilon_disk=upsilon_disk,
                upsilon_bul=upsilon_bul,
                sigma_weight_mode=sigma_weight_mode,
                sigma_weight_value=sigma_weight_value,
                sigma_weight_g0=1.0,
                sigma_weight_sigma0=1.0,
                sigma_weight_u=sigma_weight_u,
                sigma_weight_v=sigma_weight_v,
                sigma_weight_lambda=sigma_weight_lambda,
                sigma_weight_g_choice=sigma_weight_g_choice,
                sigma_weight_sigma_choice=sigma_weight_sigma_choice,
            )

            obs = compute_sigma_observables(
                galaxy_data=df,
                sigma_raw=sigma_profile,
            )

            g_val = obs.get(g_choice, np.nan)
            s_val = obs.get(sigma_choice, np.nan)

            if np.isfinite(g_val) and g_val > 0:
                g_values.append(float(g_val))
            if np.isfinite(s_val) and s_val > 0:
                sigma_values.append(float(s_val))

        except Exception:
            continue

    g0 = float(np.median(g_values)) if g_values else 1.0
    sigma0 = float(np.median(sigma_values)) if sigma_values else 1.0
    return {"g0": g0, "sigma0": sigma0}


def run_rotation_pipeline(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    beta_mode: str = "formula",
    beta_value: float = 200.0,
    structural_mode: str = "sigma_over_r",
    sigma_proxy_mode: str = "log_ratio",
    sigma_damping_mode: str = "bar_over_obs",
    sigma_positive_only: bool = True,
    D_bg: float = 0.0,
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
    beta_min: float = 0.0,
    beta_max: float = 2000.0,
    beta_ridge: float = 1.0e-12,
    use_nonnegative_target: bool = False,

    # formula beta params
    formula_p: float = 1.0,
    formula_q: float = 1.0,
    formula_lambda_beta: float = 1.0,
    formula_g_choice: str = "gbar_char_logmean",
    formula_sigma_choice: str = "sigma_char_rms",

    # dimension proxy params
    use_dimension_proxy: bool | None = True,
    formula_d_ref: float = 3.0,
    formula_dim_sensitivity: float = 0.85,
    formula_dim_g_power: float = 1.0,
    formula_dim_sigma_power: float = 1.0,
    formula_d_min: float = 1.5,
    formula_d_max: float = 4.5,
    formula_dim_response_lambda: float = 0.45,
    formula_dim_transition: float = 0.35,
    formula_dim_response_min: float = 0.55,
    formula_dim_response_max: float = 1.85,

    # sigma weight params
    sigma_weight_mode: str = "formula",
    sigma_weight_value: float = 0.5,
    sigma_weight_u: float = 1.0,
    sigma_weight_v: float = 1.0,
    sigma_weight_lambda: float = 1.0,
    sigma_weight_g_choice: str = "gbar_char_logmean",
    sigma_weight_sigma_choice: str = "sigma_char_rms",

    # estimated params
    estimated_params_json: str | Path | None = None,
    use_estimated_params: bool = False,
) -> pd.DataFrame:
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {data_dir}")

    csv_files = sorted(data_dir.glob("*_normalized.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No normalized CSV files found in: {data_dir}")

    estimated_params = None
    if use_estimated_params:
        if estimated_params_json is None:
            raise ValueError("use_estimated_params=True requires estimated_params_json.")
        estimated_params = load_estimated_formula_params(estimated_params_json)

        formula_p = float(estimated_params["formula_p"])
        formula_q = float(estimated_params["formula_q"])
        formula_lambda_beta = float(estimated_params["formula_lambda_beta"])

        sigma_weight_u = float(estimated_params["sigma_weight_u"])
        sigma_weight_v = float(estimated_params["sigma_weight_v"])
        sigma_weight_lambda = float(estimated_params["sigma_weight_lambda"])

        formula_g_choice = str(estimated_params["formula_g_choice"])
        formula_sigma_choice = str(estimated_params["formula_sigma_choice"])
        sigma_weight_g_choice = str(estimated_params["sigma_weight_g_choice"])
        sigma_weight_sigma_choice = str(estimated_params["sigma_weight_sigma_choice"])

        json_use_dimension_proxy = estimated_params.get("use_dimension_proxy", None)
        if use_dimension_proxy is None and json_use_dimension_proxy is not None:
            use_dimension_proxy = bool(json_use_dimension_proxy)

        formula_d_ref = float(estimated_params["formula_d_ref"])
        formula_dim_sensitivity = float(estimated_params["formula_dim_sensitivity"])
        formula_dim_g_power = float(estimated_params["formula_dim_g_power"])
        formula_dim_sigma_power = float(estimated_params["formula_dim_sigma_power"])
        formula_d_min = float(estimated_params["formula_d_min"])
        formula_d_max = float(estimated_params["formula_d_max"])
        formula_dim_response_lambda = float(estimated_params["formula_dim_response_lambda"])
        formula_dim_transition = float(estimated_params["formula_dim_transition"])
        formula_dim_response_min = float(estimated_params["formula_dim_response_min"])
        formula_dim_response_max = float(estimated_params["formula_dim_response_max"])

    if use_dimension_proxy is None:
        use_dimension_proxy = True

    dirs = create_run_output_dirs(beta_mode=beta_mode)
    run_root = dirs["run_root"]
    summary_dir = dirs["summary_dir"]
    table_dir = dirs["table_dir"]
    figure_dir = dirs["figure_dir"]
    log_dir = dirs["log_dir"]
    metadata_dir = dirs["metadata_dir"]

    print("[INFO] run_root =", run_root)
    print("[INFO] beta_mode =", beta_mode)
    print("[INFO] use_dimension_proxy =", use_dimension_proxy)

    if use_estimated_params and estimated_params is not None:
        shared_norm = {
            "g0": float(estimated_params["formula_g0"]),
            "sigma0": float(estimated_params["formula_sigma0"]),
        }
    else:
        shared_norm = _estimate_shared_normalizations(
            csv_files=csv_files,
            D_bg=D_bg,
            sigma_positive_only=sigma_positive_only,
            sigma_proxy_mode=sigma_proxy_mode,
            sigma_damping_mode=sigma_damping_mode,
            upsilon_disk=upsilon_disk,
            upsilon_bul=upsilon_bul,
            g_choice=formula_g_choice,
            sigma_choice=formula_sigma_choice,
            sigma_weight_mode=sigma_weight_mode,
            sigma_weight_value=sigma_weight_value,
            sigma_weight_u=sigma_weight_u,
            sigma_weight_v=sigma_weight_v,
            sigma_weight_lambda=sigma_weight_lambda,
            sigma_weight_g_choice=sigma_weight_g_choice,
            sigma_weight_sigma_choice=sigma_weight_sigma_choice,
        )

    print("[INFO] shared g0 =", shared_norm["g0"])
    print("[INFO] shared sigma0 =", shared_norm["sigma0"])

    results: list[dict] = []

    for file_path in csv_files:
        try:
            df = load_processed_sparc_table(file_path)
            galaxy_name = _resolve_galaxy_name(df, file_path)

            r_kpc = df["r_kpc"].to_numpy(dtype=float)
            v_obs_kmps = df["v_obs_kmps"].to_numpy(dtype=float)
            v_err_kmps = df["v_err_kmps"].to_numpy(dtype=float)

            sigma_profile = compute_sigma_profile(
                galaxy_data=df,
                D_bg=D_bg,
                positive_only=sigma_positive_only,
                proxy_mode=sigma_proxy_mode,
                damping_mode=sigma_damping_mode,
                upsilon_disk=upsilon_disk,
                upsilon_bul=upsilon_bul,
                sigma_weight_mode=sigma_weight_mode,
                sigma_weight_value=sigma_weight_value,
                sigma_weight_g0=float(shared_norm["g0"]),
                sigma_weight_sigma0=float(shared_norm["sigma0"]),
                sigma_weight_u=sigma_weight_u,
                sigma_weight_v=sigma_weight_v,
                sigma_weight_lambda=sigma_weight_lambda,
                sigma_weight_g_choice=sigma_weight_g_choice,
                sigma_weight_sigma_choice=sigma_weight_sigma_choice,
            )

            if beta_mode == "constant":
                beta_const = get_constant_beta(beta_value)
                beta_details = {
                    "beta_raw": float(beta_const),
                    "beta_formula": np.nan,
                    "beta_clipped": float(beta_const),
                    "beta_saturated": float(beta_const),
                    "beta_hit_lower": False,
                    "beta_hit_upper": False,
                    "beta_saturation_ratio": np.nan,
                    "beta_z": np.nan,
                    "beta_numerator": np.nan,
                    "beta_denominator": np.nan,
                    "beta_g_choice": None,
                    "beta_sigma_choice": None,
                    "gbar_char_median": np.nan,
                    "gbar_char_logmean": np.nan,
                    "sigma_char_mean": np.nan,
                    "sigma_char_median": np.nan,
                    "sigma_char_rms": np.nan,
                    "n_sigma_positive": np.nan,
                    "use_dimension_proxy": False,
                    "d_eff_proxy": np.nan,
                    "d_eff_ratio": np.nan,
                    "d_eff_g_hat": np.nan,
                    "d_eff_sigma_hat": np.nan,
                    "d_eff_signed_delta": 0.0,
                    "d_eff_response": 1.0,
                }

            elif beta_mode == "structural":
                beta_details = get_structural_beta_details(
                    galaxy_data=df,
                    sigma_profile=sigma_profile,
                    structural_mode=structural_mode,
                    upsilon_disk=upsilon_disk,
                    upsilon_bul=upsilon_bul,
                    beta_min=beta_min,
                    beta_max=beta_max,
                    ridge=beta_ridge,
                    use_nonnegative_target=use_nonnegative_target,
                )

            elif beta_mode == "formula":
                beta_details = get_formula_beta_details(
                    galaxy_data=df,
                    sigma_profile=sigma_profile,
                    g0=float(shared_norm["g0"]),
                    sigma0=float(shared_norm["sigma0"]),
                    beta_min=beta_min,
                    beta_max=beta_max,
                    p=formula_p,
                    q=formula_q,
                    lambda_beta=formula_lambda_beta,
                    g_choice=formula_g_choice,
                    sigma_choice=formula_sigma_choice,
                    use_dimension_proxy=bool(use_dimension_proxy),
                    d_ref=formula_d_ref,
                    dim_sensitivity=formula_dim_sensitivity,
                    dim_g_power=formula_dim_g_power,
                    dim_sigma_power=formula_dim_sigma_power,
                    d_min=formula_d_min,
                    d_max=formula_d_max,
                    dim_response_lambda=formula_dim_response_lambda,
                    dim_transition=formula_dim_transition,
                    dim_response_min=formula_dim_response_min,
                    dim_response_max=formula_dim_response_max,
                )

            else:
                raise ValueError(f"Unknown beta_mode: {beta_mode}")

            beta = float(beta_details["beta_clipped"])

            a_total, a_bar, a_struct = compute_total_acceleration(
                galaxy_data=df,
                sigma_profile=sigma_profile,
                beta=beta,
                structural_mode=structural_mode,
                upsilon_disk=upsilon_disk,
                upsilon_bul=upsilon_bul,
            )

            v_model_kmps = compute_rotation_velocity(r_kpc, a_total)
            v_bar_kmps, v_struct_kmps, _ = _compute_component_velocities(
                r_kpc=r_kpc,
                a_bar=a_bar,
                a_struct=a_struct,
                a_total=a_total,
            )

            rmse_val = rmse(v_obs_kmps, v_model_kmps)
            mae_val = mae(v_obs_kmps, v_model_kmps)
            mafe_val = mean_absolute_fractional_error(v_obs_kmps, v_model_kmps)
            mdafe_val = median_absolute_fractional_error(v_obs_kmps, v_model_kmps)
            chi2_val = chi_square(v_obs_kmps, v_model_kmps, v_err_kmps)
            red_chi2_val = reduced_chi_square(
                v_obs_kmps,
                v_model_kmps,
                v_err_kmps,
                n_params=1,
            )
            n_valid = valid_point_count(v_obs_kmps, v_model_kmps, v_err_kmps)

            detail_df = pd.DataFrame(
                {
                    "galaxy": galaxy_name,
                    "r_kpc": r_kpc,
                    "v_obs_kmps": v_obs_kmps,
                    "v_err_kmps": v_err_kmps,
                    "sigma_profile": sigma_profile,
                    "a_bar_kmps2_per_kpc": a_bar,
                    "a_struct_kmps2_per_kpc": a_struct,
                    "a_total_kmps2_per_kpc": a_total,
                    "v_bar_kmps": v_bar_kmps,
                    "v_struct_kmps": v_struct_kmps,
                    "v_model_kmps": v_model_kmps,
                    "beta_raw": beta_details["beta_raw"],
                    "beta_formula": beta_details["beta_formula"],
                    "beta_clipped": beta_details["beta_clipped"],
                    "use_dimension_proxy": beta_details.get("use_dimension_proxy", False),
                    "d_eff_proxy": beta_details.get("d_eff_proxy", np.nan),
                    "d_eff_response": beta_details.get("d_eff_response", 1.0),
                    "beta_mode": beta_mode,
                }
            )
            detail_path = table_dir / f"{galaxy_name}_rotation_table.csv"
            detail_df.to_csv(detail_path, index=False)

            plot_path = save_rotation_curve_plot(
                galaxy_name=galaxy_name,
                r_kpc=r_kpc,
                v_obs_kmps=v_obs_kmps,
                v_err_kmps=v_err_kmps,
                v_model_kmps=v_model_kmps,
                output_dir=figure_dir,
                v_bar_kmps=v_bar_kmps,
                v_struct_kmps=v_struct_kmps,
                beta=float(beta),
                rmse_value=rmse_val,
                chi2_value=chi2_val,
                reduced_chi2_value=red_chi2_val,
            )

            results.append(
                {
                    "galaxy": galaxy_name,
                    "source_file": str(file_path),
                    "beta_mode": beta_mode,
                    "beta_raw": beta_details["beta_raw"],
                    "beta_formula": beta_details["beta_formula"],
                    "beta": beta_details["beta_clipped"],
                    "beta_clipped": beta_details["beta_clipped"],
                    "beta_hit_lower": beta_details["beta_hit_lower"],
                    "beta_hit_upper": beta_details["beta_hit_upper"],
                    "beta_saturation_ratio": beta_details["beta_saturation_ratio"],
                    "beta_z": beta_details["beta_z"],
                    "beta_numerator": beta_details["beta_numerator"],
                    "beta_denominator": beta_details["beta_denominator"],
                    "beta_g_choice": beta_details["beta_g_choice"],
                    "beta_sigma_choice": beta_details["beta_sigma_choice"],
                    "gbar_char_median": beta_details["gbar_char_median"],
                    "gbar_char_logmean": beta_details["gbar_char_logmean"],
                    "sigma_char_mean": beta_details["sigma_char_mean"],
                    "sigma_char_median": beta_details["sigma_char_median"],
                    "sigma_char_rms": beta_details["sigma_char_rms"],
                    "n_sigma_positive": beta_details["n_sigma_positive"],
                    "use_dimension_proxy": beta_details.get("use_dimension_proxy", False),
                    "d_eff_proxy": beta_details.get("d_eff_proxy", np.nan),
                    "d_eff_ratio": beta_details.get("d_eff_ratio", np.nan),
                    "d_eff_g_hat": beta_details.get("d_eff_g_hat", np.nan),
                    "d_eff_sigma_hat": beta_details.get("d_eff_sigma_hat", np.nan),
                    "d_eff_signed_delta": beta_details.get("d_eff_signed_delta", 0.0),
                    "d_eff_response": beta_details.get("d_eff_response", 1.0),
                    "formula_g0": float(shared_norm["g0"]) if beta_mode == "formula" else np.nan,
                    "formula_sigma0": float(shared_norm["sigma0"]) if beta_mode == "formula" else np.nan,
                    "formula_p": float(formula_p) if beta_mode == "formula" else np.nan,
                    "formula_q": float(formula_q) if beta_mode == "formula" else np.nan,
                    "formula_lambda_beta": float(formula_lambda_beta) if beta_mode == "formula" else np.nan,
                    "formula_d_ref": float(formula_d_ref) if beta_mode == "formula" else np.nan,
                    "formula_dim_sensitivity": float(formula_dim_sensitivity) if beta_mode == "formula" else np.nan,
                    "formula_dim_g_power": float(formula_dim_g_power) if beta_mode == "formula" else np.nan,
                    "formula_dim_sigma_power": float(formula_dim_sigma_power) if beta_mode == "formula" else np.nan,
                    "formula_d_min": float(formula_d_min) if beta_mode == "formula" else np.nan,
                    "formula_d_max": float(formula_d_max) if beta_mode == "formula" else np.nan,
                    "formula_dim_response_lambda": float(formula_dim_response_lambda) if beta_mode == "formula" else np.nan,
                    "formula_dim_transition": float(formula_dim_transition) if beta_mode == "formula" else np.nan,
                    "formula_dim_response_min": float(formula_dim_response_min) if beta_mode == "formula" else np.nan,
                    "formula_dim_response_max": float(formula_dim_response_max) if beta_mode == "formula" else np.nan,
                    "sigma_weight_mode": sigma_weight_mode,
                    "sigma_weight_value": float(sigma_weight_value) if sigma_weight_mode == "fixed" else np.nan,
                    "sigma_weight_u": float(sigma_weight_u) if sigma_weight_mode == "formula" else np.nan,
                    "sigma_weight_v": float(sigma_weight_v) if sigma_weight_mode == "formula" else np.nan,
                    "sigma_weight_lambda": float(sigma_weight_lambda) if sigma_weight_mode == "formula" else np.nan,
                    "sigma_weight_g_choice": sigma_weight_g_choice,
                    "sigma_weight_sigma_choice": sigma_weight_sigma_choice,
                    "structural_mode": structural_mode,
                    "sigma_proxy_mode": sigma_proxy_mode,
                    "sigma_damping_mode": sigma_damping_mode,
                    "sigma_positive_only": bool(sigma_positive_only),
                    "D_bg": float(D_bg),
                    "upsilon_disk": float(upsilon_disk),
                    "upsilon_bul": float(upsilon_bul),
                    "rmse_kmps": float(rmse_val) if np.isfinite(rmse_val) else np.nan,
                    "mae_kmps": float(mae_val) if np.isfinite(mae_val) else np.nan,
                    "mean_absolute_fractional_error": float(mafe_val) if np.isfinite(mafe_val) else np.nan,
                    "median_absolute_fractional_error": float(mdafe_val) if np.isfinite(mdafe_val) else np.nan,
                    "chi_square": float(chi2_val) if np.isfinite(chi2_val) else np.nan,
                    "reduced_chi_square": float(red_chi2_val) if np.isfinite(red_chi2_val) else np.nan,
                    "n_points_total": int(len(df)),
                    "n_points_valid": int(n_valid),
                    "detail_table": str(detail_path),
                    "plot_path": str(plot_path),
                    "status": "ok",
                    "error": None,
                }
            )

            print(f"[OK] {galaxy_name}")

        except Exception as exc:
            results.append(
                {
                    "galaxy": file_path.stem.replace("_normalized", ""),
                    "source_file": str(file_path),
                    "beta_mode": beta_mode,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(f"[FAILED] {file_path.name} -> {exc}")

    df_results = pd.DataFrame(results)

    summary_name = f"rotation_pipeline_summary_{beta_mode}.csv"
    summary_path = summary_dir / summary_name
    df_results.to_csv(summary_path, index=False)

    ok_df = df_results[df_results["status"] == "ok"].copy()

    metadata = {
        "mode": beta_mode,
        "timestamp_root": str(run_root.name),
        "data_dir": str(data_dir),
        "output_root": str(run_root),
        "n_files_found": int(len(csv_files)),
        "n_ok": int((df_results["status"] == "ok").sum()),
        "n_failed": int((df_results["status"] == "failed").sum()),
        "parameters": {
            "beta_value": float(beta_value),
            "structural_mode": structural_mode,
            "sigma_proxy_mode": sigma_proxy_mode,
            "sigma_damping_mode": sigma_damping_mode,
            "sigma_positive_only": bool(sigma_positive_only),
            "D_bg": float(D_bg),
            "upsilon_disk": float(upsilon_disk),
            "upsilon_bul": float(upsilon_bul),
            "beta_min": float(beta_min),
            "beta_max": float(beta_max),
            "beta_ridge": float(beta_ridge),
            "use_nonnegative_target": bool(use_nonnegative_target),
            "formula_p": float(formula_p),
            "formula_q": float(formula_q),
            "formula_lambda_beta": float(formula_lambda_beta),
            "formula_g_choice": formula_g_choice,
            "formula_sigma_choice": formula_sigma_choice,
            "use_dimension_proxy": bool(use_dimension_proxy),
            "formula_d_ref": float(formula_d_ref),
            "formula_dim_sensitivity": float(formula_dim_sensitivity),
            "formula_dim_g_power": float(formula_dim_g_power),
            "formula_dim_sigma_power": float(formula_dim_sigma_power),
            "formula_d_min": float(formula_d_min),
            "formula_d_max": float(formula_d_max),
            "formula_dim_response_lambda": float(formula_dim_response_lambda),
            "formula_dim_transition": float(formula_dim_transition),
            "formula_dim_response_min": float(formula_dim_response_min),
            "formula_dim_response_max": float(formula_dim_response_max),
            "shared_g0": float(shared_norm["g0"]),
            "shared_sigma0": float(shared_norm["sigma0"]),
            "sigma_weight_mode": sigma_weight_mode,
            "sigma_weight_value": float(sigma_weight_value),
            "sigma_weight_u": float(sigma_weight_u),
            "sigma_weight_v": float(sigma_weight_v),
            "sigma_weight_lambda": float(sigma_weight_lambda),
            "sigma_weight_g_choice": sigma_weight_g_choice,
            "sigma_weight_sigma_choice": sigma_weight_sigma_choice,
            "use_estimated_params": bool(use_estimated_params),
            "estimated_params_json": str(estimated_params_json) if estimated_params_json is not None else None,
        },
        "summary_csv": str(summary_path),
    }

    metadata_path = metadata_dir / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log_lines = [
        f"run_root: {run_root}",
        f"data_dir: {data_dir}",
        f"beta_mode: {beta_mode}",
        f"shared_g0: {shared_norm['g0']}",
        f"shared_sigma0: {shared_norm['sigma0']}",
        f"formula_p: {formula_p}",
        f"formula_q: {formula_q}",
        f"formula_lambda_beta: {formula_lambda_beta}",
        f"use_dimension_proxy: {use_dimension_proxy}",
        f"formula_d_ref: {formula_d_ref}",
        f"formula_dim_sensitivity: {formula_dim_sensitivity}",
        f"formula_dim_g_power: {formula_dim_g_power}",
        f"formula_dim_sigma_power: {formula_dim_sigma_power}",
        f"formula_d_min: {formula_d_min}",
        f"formula_d_max: {formula_d_max}",
        f"formula_dim_response_lambda: {formula_dim_response_lambda}",
        f"formula_dim_transition: {formula_dim_transition}",
        f"formula_dim_response_min: {formula_dim_response_min}",
        f"formula_dim_response_max: {formula_dim_response_max}",
        f"summary_csv: {summary_path}",
        f"metadata_json: {metadata_path}",
    ]
    _write_run_log(log_dir / "run_log.txt", log_lines)

    if not ok_df.empty:
        save_metric_histogram(
            metric_values=ok_df["rmse_kmps"].to_numpy(dtype=float),
            metric_name="RMSE_kmps",
            output_dir=figure_dir,
            filename="rmse_distribution.png",
            title="RMSE Distribution",
        )
        save_metric_histogram(
            metric_values=ok_df["chi_square"].to_numpy(dtype=float),
            metric_name="ChiSquare",
            output_dir=figure_dir,
            filename="chi_square_distribution.png",
            title="Chi-Square Distribution",
        )
        save_beta_histogram(
            beta_values=ok_df["beta"].to_numpy(dtype=float),
            output_dir=figure_dir,
            filename="beta_distribution.png",
            title=f"Beta Distribution ({beta_mode})",
        )

    print(f"[DONE] summary saved to: {summary_path}")
    return df_results


if __name__ == "__main__":
    estimated_json = Path(
        "outputs/galaxy_rotation/pipeline_beta_formula/estimated_formula_params.json"
    )

    summary = run_rotation_pipeline(
        data_dir=DEFAULT_DATA_DIR,
        beta_mode="formula",
        beta_value=200.0,
        structural_mode="sigma_over_r",
        sigma_proxy_mode="log_ratio",
        sigma_damping_mode="bar_over_obs",
        sigma_positive_only=True,
        D_bg=0.0,
        upsilon_disk=1.0,
        upsilon_bul=1.0,
        beta_min=0.0,
        beta_max=2000.0,
        beta_ridge=1.0e-12,
        use_nonnegative_target=False,

        formula_p=1.0,
        formula_q=1.0,
        formula_lambda_beta=1.0,
        formula_g_choice="gbar_char_logmean",
        formula_sigma_choice="sigma_char_rms",

        use_dimension_proxy=True,
        formula_d_ref=3.0,
        formula_dim_sensitivity=0.85,
        formula_dim_g_power=1.0,
        formula_dim_sigma_power=1.0,
        formula_d_min=1.5,
        formula_d_max=4.5,
        formula_dim_response_lambda=0.45,
        formula_dim_transition=0.35,
        formula_dim_response_min=0.55,
        formula_dim_response_max=1.85,

        sigma_weight_mode="formula",
        sigma_weight_value=0.5,
        sigma_weight_u=1.0,
        sigma_weight_v=1.0,
        sigma_weight_lambda=1.0,
        sigma_weight_g_choice="gbar_char_logmean",
        sigma_weight_sigma_choice="sigma_char_rms",

        estimated_params_json=estimated_json,
        use_estimated_params=True,
    )

    print(summary.head())
    print(f"\nTotal files processed: {len(summary)}")