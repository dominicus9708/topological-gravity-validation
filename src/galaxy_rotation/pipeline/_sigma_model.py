import numpy as np



def compute_sigma_profile(
    r: np.ndarray,
    g_newton: np.ndarray,
) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    g_newton = np.asarray(g_newton, dtype=float)

    dr = np.gradient(r)
    dg = np.gradient(g_newton)
    sigma = dg / np.maximum(np.abs(dr), 1e-12)
    return sigma
