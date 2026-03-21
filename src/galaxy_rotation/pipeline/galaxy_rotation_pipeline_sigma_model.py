import numpy as np


def compute_sigma_profile(
    r: np.ndarray,
    g_newton: np.ndarray,
) -> np.ndarray:
    # 임시 예시: 반지름 방향 변화율 기반
    dr = np.gradient(r)
    dg = np.gradient(g_newton)
    sigma = dg / np.maximum(dr, 1e-12)
    return sigma
