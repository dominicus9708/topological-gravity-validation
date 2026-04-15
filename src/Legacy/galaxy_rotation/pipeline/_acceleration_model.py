import numpy as np


def compute_newtonian_acceleration(
    r: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    v_gas = np.asarray(v_gas, dtype=float)
    v_disk = np.asarray(v_disk, dtype=float)
    v_bulge = np.asarray(v_bulge, dtype=float)

    v_bar_sq = v_gas**2 + v_disk**2 + v_bulge**2
    return v_bar_sq / np.maximum(r, eps)



def compute_structural_acceleration(
    r: np.ndarray,
    sigma: np.ndarray,
    beta: float,
    eps: float = 1e-12,
) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return float(beta) * sigma / np.maximum(r, eps)



def compute_total_acceleration(
    g_newton: np.ndarray,
    g_struct: np.ndarray,
) -> np.ndarray:
    g_newton = np.asarray(g_newton, dtype=float)
    g_struct = np.asarray(g_struct, dtype=float)
    return g_newton + g_struct
