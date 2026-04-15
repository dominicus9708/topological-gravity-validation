from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_radius(r: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return np.maximum(r, eps)


def compute_baryonic_acceleration(
    galaxy_data: pd.DataFrame,
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
) -> np.ndarray:
    required = ["r_kpc", "v_gas_kmps", "v_disk_kmps", "v_bul_kmps"]
    missing = [c for c in required if c not in galaxy_data.columns]
    if missing:
        raise ValueError(f"Missing required columns for baryonic acceleration: {missing}")

    r = _safe_radius(galaxy_data["r_kpc"].to_numpy(dtype=float))
    v_gas = galaxy_data["v_gas_kmps"].to_numpy(dtype=float)
    v_disk = galaxy_data["v_disk_kmps"].to_numpy(dtype=float)
    v_bul = galaxy_data["v_bul_kmps"].to_numpy(dtype=float)

    v_bar2 = (
        v_gas ** 2
        + float(upsilon_disk) * (v_disk ** 2)
        + float(upsilon_bul) * (v_bul ** 2)
    )
    v_bar2 = np.clip(v_bar2, 0.0, None)
    a_bar = v_bar2 / r
    return np.where(np.isfinite(a_bar), a_bar, 0.0)


def compute_structural_acceleration(
    r_kpc: np.ndarray,
    sigma_profile: np.ndarray,
    beta: float,
    mode: str = "sigma_over_r",
    floor: float = 1.0e-12,
) -> np.ndarray:
    r = _safe_radius(r_kpc, eps=floor)
    sigma = np.asarray(sigma_profile, dtype=float)
    if r.shape != sigma.shape:
        raise ValueError("r_kpc and sigma_profile must have the same shape.")

    beta = float(beta)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)

    if mode == "sigma_over_r":
        g_struct = beta * sigma / r
    elif mode == "plain_sigma":
        g_struct = beta * sigma
    elif mode == "sigma_over_sqrt_r":
        g_struct = beta * sigma / np.sqrt(r)
    else:
        raise ValueError(f"Unsupported structural acceleration mode: {mode}")

    return np.where(np.isfinite(g_struct), g_struct, 0.0)


def compute_total_acceleration(
    galaxy_data: pd.DataFrame,
    sigma_profile: np.ndarray,
    beta: float,
    structural_mode: str = "sigma_over_r",
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "r_kpc" not in galaxy_data.columns:
        raise ValueError("galaxy_data must contain 'r_kpc'.")

    r = galaxy_data["r_kpc"].to_numpy(dtype=float)
    a_bar = compute_baryonic_acceleration(
        galaxy_data,
        upsilon_disk=upsilon_disk,
        upsilon_bul=upsilon_bul,
    )
    a_struct = compute_structural_acceleration(
        r_kpc=r,
        sigma_profile=sigma_profile,
        beta=beta,
        mode=structural_mode,
    )
    a_total = a_bar + a_struct
    a_total = np.where(np.isfinite(a_total), a_total, 0.0)
    a_total = np.clip(a_total, 0.0, None)
    return a_total, a_bar, a_struct
