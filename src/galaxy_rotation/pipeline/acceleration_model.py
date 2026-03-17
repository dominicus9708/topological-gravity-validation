import numpy as np


def compute_newtonian_acceleration(
    r: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    v_bar_sq = v_gas**2 + v_disk**2 + v_bulge**2
    return v_bar_sq / np.maximum(r, eps)


def compute_structural_acceleration(
    r: np.ndarray,
    sigma: np.ndarray,
    beta: float,
    eps: float = 1e-12,
) -> np.ndarray:
    return beta * sigma / np.maximum(r, eps)


def compute_total_acceleration(
    g_newton: np.ndarray,
    g_struct: np.ndarray,
) -> np.ndarray:
    return g_newton + g_struct
