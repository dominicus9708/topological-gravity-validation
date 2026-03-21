import numpy as np
import pandas as pd


A0 = 1000.0  # 기준 가속도 (튜닝 가능)


def get_constant_beta(value: float = 200.0) -> float:
    return float(value)


def _compute_abar_from_baryonic_velocity(df: pd.DataFrame) -> np.ndarray:
    r = df["r_kpc"].to_numpy(dtype=float)
    v_gas = df["v_gas_kmps"].to_numpy(dtype=float)
    v_disk = df["v_disk_kmps"].to_numpy(dtype=float)
    v_bul = df["v_bul_kmps"].to_numpy(dtype=float)

    mask = (r > 0) & np.isfinite(r)

    abar = np.zeros_like(r)
    abar[mask] = (v_gas[mask]**2 + v_disk[mask]**2 + v_bul[mask]**2) / r[mask]

    return abar


def get_structural_beta(df: pd.DataFrame, sigma_profile: np.ndarray) -> float:
    abar = _compute_abar_from_baryonic_velocity(df)

    abar = abar[np.isfinite(abar)]
    abar = abar[np.abs(abar) > 0]

    if abar.size == 0:
        return 1.0

    abar_med = np.median(np.abs(abar))

    # 핵심 수식
    beta = np.sqrt(abar_med / A0)

    return max(0.1, beta)