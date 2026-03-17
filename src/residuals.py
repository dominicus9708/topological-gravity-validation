# src/residuals.py

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_float_array(x):
    return np.asarray(x, dtype=float)


def compute_velocity_residuals(v_obs_kmps, v_model_kmps):
    v_obs_kmps = _as_float_array(v_obs_kmps)
    v_model_kmps = _as_float_array(v_model_kmps)
    return v_obs_kmps - v_model_kmps


def compute_absolute_residuals(v_obs_kmps, v_model_kmps):
    return np.abs(compute_velocity_residuals(v_obs_kmps, v_model_kmps))


def compute_fractional_residuals(v_obs_kmps, v_model_kmps, eps: float = 1e-12):
    v_obs_kmps = _as_float_array(v_obs_kmps)
    residuals = compute_velocity_residuals(v_obs_kmps, v_model_kmps)
    return residuals / np.maximum(v_obs_kmps, eps)


def compute_chi2(v_obs_kmps, v_model_kmps, v_err_kmps):
    v_obs_kmps = _as_float_array(v_obs_kmps)
    v_model_kmps = _as_float_array(v_model_kmps)
    v_err_kmps = _as_float_array(v_err_kmps)

    if np.any(v_err_kmps <= 0):
        raise ValueError("All velocity uncertainties must be positive.")

    return float(np.sum(((v_obs_kmps - v_model_kmps) / v_err_kmps) ** 2))


def compute_reduced_chi2(v_obs_kmps, v_model_kmps, v_err_kmps, n_params: int = 1):
    n = len(v_obs_kmps)
    dof = n - n_params
    if dof <= 0:
        return np.nan
    return compute_chi2(v_obs_kmps, v_model_kmps, v_err_kmps) / dof


def compute_rmse(v_obs_kmps, v_model_kmps):
    v_obs_kmps = _as_float_array(v_obs_kmps)
    v_model_kmps = _as_float_array(v_model_kmps)
    return float(np.sqrt(np.mean((v_obs_kmps - v_model_kmps) ** 2)))


def compute_mae(v_obs_kmps, v_model_kmps):
    v_obs_kmps = _as_float_array(v_obs_kmps)
    v_model_kmps = _as_float_array(v_model_kmps)
    return float(np.mean(np.abs(v_obs_kmps - v_model_kmps)))


def build_residual_dataframe(df_norm: pd.DataFrame, v_model_kmps) -> pd.DataFrame:
    out = df_norm.copy()
    out["v_model_kmps"] = _as_float_array(v_model_kmps)
    out["residual_kmps"] = compute_velocity_residuals(
        out["v_obs_kmps"].values,
        out["v_model_kmps"].values,
    )
    out["abs_residual_kmps"] = np.abs(out["residual_kmps"].values)
    out["frac_residual"] = compute_fractional_residuals(
        out["v_obs_kmps"].values,
        out["v_model_kmps"].values,
    )
    return out


def summarize_fit_metrics(df_resid: pd.DataFrame, n_params: int = 1) -> dict:
    v_obs = df_resid["v_obs_kmps"].values
    v_model = df_resid["v_model_kmps"].values
    v_err = df_resid["v_err_kmps"].values

    chi2 = compute_chi2(v_obs, v_model, v_err)
    red_chi2 = compute_reduced_chi2(v_obs, v_model, v_err, n_params=n_params)
    rmse = compute_rmse(v_obs, v_model)
    mae = compute_mae(v_obs, v_model)

    galaxy_name = str(df_resid["galaxy"].iloc[0]) if "galaxy" in df_resid.columns else "unknown"

    return {
        "galaxy": galaxy_name,
        "n_points": int(len(df_resid)),
        "chi2": chi2,
        "reduced_chi2": red_chi2,
        "rmse_kmps": rmse,
        "mae_kmps": mae,
        "mean_abs_residual_kmps": float(df_resid["abs_residual_kmps"].mean()),
        "max_abs_residual_kmps": float(df_resid["abs_residual_kmps"].max()),
    }