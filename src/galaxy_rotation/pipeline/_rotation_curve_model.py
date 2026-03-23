import numpy as np



def compute_rotation_velocity(
    r: np.ndarray,
    g_total: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    g_total = np.asarray(g_total, dtype=float)
    return np.sqrt(np.maximum(r, eps) * np.maximum(g_total, 0.0))
