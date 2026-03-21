from __future__ import annotations

import numpy as np


def compute_rotation_velocity(
    r_kpc: np.ndarray,
    g_total_kmps2_per_kpc: np.ndarray,
) -> np.ndarray:
    """
    총 가속도 프로파일로부터 모델 회전속도를 계산합니다.

    식:
        v_model(r) = sqrt( r * g_total(r) )

    입력:
    - r_kpc: 반지름 [kpc]
    - g_total_kmps2_per_kpc: 총 가속도 [km^2 s^-2 kpc^-1]

    출력:
    - v_model_kmps: 모델 회전속도 [km/s]

    주의:
    - 반드시 인자 순서는 (r_kpc, g_total_kmps2_per_kpc) 입니다.
    - 음수/비정상 값은 0으로 잘라 수치적으로 안정화합니다.
    """
    r = np.asarray(r_kpc, dtype=float)
    g = np.asarray(g_total_kmps2_per_kpc, dtype=float)

    if r.shape != g.shape:
        raise ValueError(
            "r_kpc and g_total_kmps2_per_kpc must have the same shape."
        )

    # 수치 안정화
    r = np.where(np.isfinite(r), r, 0.0)
    g = np.where(np.isfinite(g), g, 0.0)

    # 반지름/가속도는 물리적으로 음수가 아니어야 하므로 클리핑
    r = np.clip(r, 0.0, None)
    g = np.clip(g, 0.0, None)

    v2 = r * g
    v2 = np.clip(v2, 0.0, None)

    return np.sqrt(v2)