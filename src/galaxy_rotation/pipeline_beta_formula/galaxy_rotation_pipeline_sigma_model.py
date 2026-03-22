from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_radius(r: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return np.maximum(r, eps)


def _compute_observed_acceleration(
    r_kpc: np.ndarray,
    v_obs_kmps: np.ndarray,
) -> np.ndarray:
    r = _safe_radius(r_kpc)
    v = np.asarray(v_obs_kmps, dtype=float)
    return (v ** 2) / r


def _compute_baryonic_acceleration(
    r_kpc: np.ndarray,
    v_gas_kmps: np.ndarray,
    v_disk_kmps: np.ndarray,
    v_bul_kmps: np.ndarray,
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
) -> np.ndarray:
    r = _safe_radius(r_kpc)
    v_gas = np.asarray(v_gas_kmps, dtype=float)
    v_disk = np.asarray(v_disk_kmps, dtype=float)
    v_bul = np.asarray(v_bul_kmps, dtype=float)

    v_bar2 = (
        v_gas ** 2
        + float(upsilon_disk) * (v_disk ** 2)
        + float(upsilon_bul) * (v_bul ** 2)
    )
    v_bar2 = np.clip(v_bar2, 0.0, None)

    return v_bar2 / r


def compute_sigma_observables(
    galaxy_data: pd.DataFrame,
    sigma_raw: np.ndarray,
    floor: float = 1.0e-12,
) -> dict[str, float]:
    """
    sigma damping 및 후속 beta formula용 은하 대표량 계산.
    """
    required = [
        "r_kpc",
        "v_obs_kmps",
        "v_gas_kmps",
        "v_disk_kmps",
        "v_bul_kmps",
    ]
    missing = [c for c in required if c not in galaxy_data.columns]
    if missing:
        raise ValueError(f"Missing required columns for sigma observables: {missing}")

    r = galaxy_data["r_kpc"].to_numpy(dtype=float)
    v_obs = galaxy_data["v_obs_kmps"].to_numpy(dtype=float)
    v_gas = galaxy_data["v_gas_kmps"].to_numpy(dtype=float)
    v_disk = galaxy_data["v_disk_kmps"].to_numpy(dtype=float)
    v_bul = galaxy_data["v_bul_kmps"].to_numpy(dtype=float)

    a_obs = _compute_observed_acceleration(r, v_obs)
    a_bar = _compute_baryonic_acceleration(r, v_gas, v_disk, v_bul)

    sigma_raw = np.asarray(sigma_raw, dtype=float)
    sigma_raw = np.where(np.isfinite(sigma_raw), sigma_raw, 0.0)
    sigma_pos = np.clip(sigma_raw, 0.0, None)

    gbar_char_median = float(np.median(a_bar)) if a_bar.size else np.nan
    gbar_char_logmean = (
        float(np.exp(np.mean(np.log(np.maximum(a_bar, floor)))))
        if a_bar.size else np.nan
    )

    sigma_char_mean = float(np.mean(sigma_pos)) if sigma_pos.size else np.nan
    sigma_char_median = float(np.median(sigma_pos)) if sigma_pos.size else np.nan
    sigma_char_rms = float(np.sqrt(np.mean(sigma_pos ** 2))) if sigma_pos.size else np.nan

    return {
        "a_obs": a_obs,
        "a_bar": a_bar,
        "gbar_char_median": gbar_char_median,
        "gbar_char_logmean": gbar_char_logmean,
        "sigma_char_mean": sigma_char_mean,
        "sigma_char_median": sigma_char_median,
        "sigma_char_rms": sigma_char_rms,
    }


def compute_sigma_weight(
    g_char: float,
    sigma_char: float,
    g0: float,
    sigma0: float,
    u: float = 1.0,
    v: float = 1.0,
    lambda_w: float = 1.0,
    floor: float = 1.0e-12,
) -> dict[str, float]:
    """
    고정 0.5를 대체하는 은하별 weight 계산.

    w = W / (1 + W)
    W = lambda_w * (g_char/g0)^u * (sigma_char/sigma0)^v
    """
    g_char = max(float(g_char), floor)
    sigma_char = max(float(sigma_char), floor)
    g0 = max(float(g0), floor)
    sigma0 = max(float(sigma0), floor)

    W = float(lambda_w) * (g_char / g0) ** float(u) * (sigma_char / sigma0) ** float(v)
    W = max(W, 0.0)

    w = W / (1.0 + W)

    return {
        "sigma_weight": float(w),
        "sigma_weight_z": float(W),
    }


def compute_sigma_profile(
    galaxy_data: pd.DataFrame,
    D_bg: float = 0.0,
    positive_only: bool = True,
    proxy_mode: str = "log_ratio",
    damping_mode: str = "bar_over_obs",
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
    floor: float = 1.0e-12,
    sigma_weight_mode: str = "fixed",   # fixed / formula
    sigma_weight_value: float = 0.5,
    sigma_weight_g0: float | None = None,
    sigma_weight_sigma0: float | None = None,
    sigma_weight_u: float = 1.0,
    sigma_weight_v: float = 1.0,
    sigma_weight_lambda: float = 1.0,
    sigma_weight_g_choice: str = "gbar_char_logmean",
    sigma_weight_sigma_choice: str = "sigma_char_rms",
) -> np.ndarray:
    """
    레거시 정합형 sigma profile.

    핵심 변경:
    기존의 고정 0.5 대신,
    은하별 유도 weight w_gal 을 넣을 수 있음.

    기존 fixed 모드:
        sigma *= w + (1-w) * (a_bar / a_obs)
        기본 w = 0.5

    formula 모드:
        w = function(g_char, sigma_char)
    """
    required = [
        "r_kpc",
        "v_obs_kmps",
        "v_gas_kmps",
        "v_disk_kmps",
        "v_bul_kmps",
    ]
    missing = [c for c in required if c not in galaxy_data.columns]
    if missing:
        raise ValueError(f"Missing required columns for sigma profile: {missing}")

    r = galaxy_data["r_kpc"].to_numpy(dtype=float)
    v_obs = galaxy_data["v_obs_kmps"].to_numpy(dtype=float)
    v_gas = galaxy_data["v_gas_kmps"].to_numpy(dtype=float)
    v_disk = galaxy_data["v_disk_kmps"].to_numpy(dtype=float)
    v_bul = galaxy_data["v_bul_kmps"].to_numpy(dtype=float)

    a_obs = _compute_observed_acceleration(r, v_obs)
    a_bar = _compute_baryonic_acceleration(
        r,
        v_gas,
        v_disk,
        v_bul,
        upsilon_disk=upsilon_disk,
        upsilon_bul=upsilon_bul,
    )

    if proxy_mode == "log_ratio":
        ratio = a_obs / np.maximum(a_bar, floor)
        alpha_proxy = np.log(np.maximum(ratio, floor))
    elif proxy_mode == "relative_excess":
        alpha_proxy = (a_obs - a_bar) / np.maximum(a_bar, floor)
    else:
        raise ValueError(f"Unsupported proxy_mode: {proxy_mode}")

    sigma_raw = alpha_proxy - float(D_bg)

    if damping_mode == "none":
        sigma = sigma_raw

    elif damping_mode == "bar_over_obs":
        ratio = np.clip(a_bar / np.maximum(a_obs, floor), 0.0, 1.0)

        if sigma_weight_mode == "fixed":
            w = float(sigma_weight_value)

        elif sigma_weight_mode == "formula":
            obs = compute_sigma_observables(
                galaxy_data=galaxy_data,
                sigma_raw=sigma_raw,
                floor=floor,
            )

            if sigma_weight_g0 is None or sigma_weight_sigma0 is None:
                raise ValueError(
                    "formula sigma weight mode requires sigma_weight_g0 and sigma_weight_sigma0."
                )

            if sigma_weight_g_choice not in obs:
                raise ValueError(f"Unsupported sigma_weight_g_choice: {sigma_weight_g_choice}")
            if sigma_weight_sigma_choice not in obs:
                raise ValueError(f"Unsupported sigma_weight_sigma_choice: {sigma_weight_sigma_choice}")

            g_char = float(obs[sigma_weight_g_choice])
            s_char = float(obs[sigma_weight_sigma_choice])

            weight_info = compute_sigma_weight(
                g_char=g_char,
                sigma_char=s_char,
                g0=float(sigma_weight_g0),
                sigma0=float(sigma_weight_sigma0),
                u=sigma_weight_u,
                v=sigma_weight_v,
                lambda_w=sigma_weight_lambda,
                floor=floor,
            )
            w = float(weight_info["sigma_weight"])

        else:
            raise ValueError(f"Unsupported sigma_weight_mode: {sigma_weight_mode}")

        w = float(np.clip(w, 0.0, 1.0))
        sigma = sigma_raw * (w + (1.0 - w) * ratio)

    else:
        raise ValueError(f"Unsupported damping_mode: {damping_mode}")

    if positive_only:
        sigma = np.maximum(sigma, 0.0)

    sigma = np.asarray(sigma, dtype=float)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)

    return sigma