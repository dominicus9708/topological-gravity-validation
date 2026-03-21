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
    """
    관측 회전속도로부터 관측 가속도 a_obs = v_obs^2 / r 를 계산합니다.
    단위:
        r      : kpc
        v_obs  : km/s
        a_obs  : km^2 s^-2 kpc^-1
    """
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
    """
    바리온 속도와 바리온 가속도 a_bar = v_bar^2 / r 를 계산합니다.

    v_bar^2 = v_gas^2 + upsilon_disk * v_disk^2 + upsilon_bul * v_bul^2
    """
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


def compute_sigma_profile(
    galaxy_data: pd.DataFrame,
    D_bg: float = 0.0,
    positive_only: bool = True,
    proxy_mode: str = "log_ratio",
    damping_mode: str = "bar_over_obs",
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
    floor: float = 1.0e-12,
) -> np.ndarray:
    """
    레거시 정합형 sigma profile 계산 함수.

    기본 아이디어:
        alpha_proxy ~ log(a_obs / a_bar)

    또는:
        alpha_proxy ~ (a_obs - a_bar) / a_bar

    이후:
        sigma_raw = alpha_proxy - D_bg

    그리고 필요하면 damping을 걸어 구조 신호가 과도하게 커지는 것을 완화합니다.

    파라미터:
    - D_bg:
        배경 구조 차원값. 기본 0.0
    - positive_only:
        True이면 음수 sigma를 0으로 절단
    - proxy_mode:
        "log_ratio"       -> log(a_obs / a_bar)
        "relative_excess" -> (a_obs - a_bar) / a_bar
    - damping_mode:
        "none"            -> damping 없음
        "bar_over_obs"    -> sigma *= (0.5 + 0.5 * clip(a_bar / a_obs, 0, 1))
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

    sigma = alpha_proxy - float(D_bg)

    if damping_mode == "none":
        pass
    elif damping_mode == "bar_over_obs":
        damping = np.clip(a_bar / np.maximum(a_obs, floor), 0.0, 1.0)
        sigma = sigma * (0.5 + 0.5 * damping)
    else:
        raise ValueError(f"Unsupported damping_mode: {damping_mode}")

    if positive_only:
        sigma = np.maximum(sigma, 0.0)

    sigma = np.asarray(sigma, dtype=float)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)

    return sigma