from __future__ import annotations

import numpy as np
import pandas as pd

from galaxy_rotation_pipeline_acceleration_model import (
    compute_baryonic_acceleration,
    compute_structural_acceleration,
)


def get_constant_beta(value: float = 200.0) -> float:
    """
    기준선 비교용 상수 beta.
    """
    return float(value)


def _compute_observed_acceleration(
    galaxy_data: pd.DataFrame,
    floor: float = 1.0e-12,
) -> np.ndarray:
    """
    관측 속도로부터 a_obs = v_obs^2 / r 계산.
    """
    required = ["r_kpc", "v_obs_kmps"]
    missing = [c for c in required if c not in galaxy_data.columns]
    if missing:
        raise ValueError(f"Missing required columns for observed acceleration: {missing}")

    r = galaxy_data["r_kpc"].to_numpy(dtype=float)
    v_obs = galaxy_data["v_obs_kmps"].to_numpy(dtype=float)

    r = np.maximum(r, floor)
    a_obs = (v_obs ** 2) / r
    a_obs = np.where(np.isfinite(a_obs), a_obs, 0.0)
    a_obs = np.clip(a_obs, 0.0, None)

    return a_obs


def _compute_required_excess_v2(
    galaxy_data: pd.DataFrame,
    a_bar: np.ndarray,
) -> np.ndarray:
    """
    관측이 요구하는 초과 velocity^2:
        Y = v_obs^2 - v_bar^2

    여기서
        v_bar^2 = r * a_bar
    """
    required = ["r_kpc", "v_obs_kmps"]
    missing = [c for c in required if c not in galaxy_data.columns]
    if missing:
        raise ValueError(f"Missing required columns for excess v^2: {missing}")

    r = galaxy_data["r_kpc"].to_numpy(dtype=float)
    v_obs = galaxy_data["v_obs_kmps"].to_numpy(dtype=float)

    v_obs2 = np.asarray(v_obs ** 2, dtype=float)
    v_bar2 = np.asarray(r * a_bar, dtype=float)

    y = v_obs2 - v_bar2
    y = np.where(np.isfinite(y), y, 0.0)

    return y


def derive_beta_from_observables(
    galaxy_data: pd.DataFrame,
    sigma_profile: np.ndarray,
    structural_mode: str = "sigma_over_r",
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
    beta_min: float = 0.0,
    beta_max: float = 2000.0,
    ridge: float = 1.0e-12,
    use_nonnegative_target: bool = False,
) -> float:
    """
    구조 신호와 관측 초과분의 최소제곱 정합으로 은하별 beta를 유도합니다.

    기본 아이디어:
        v_obs^2 - v_bar^2  ≈  beta * (r * a_struct(beta=1))

    즉
        Y_i = v_obs^2(r_i) - v_bar^2(r_i)
        X_i = r_i * a_struct(r_i; beta=1)

    최소제곱 해:
        beta = sum(X_i Y_i) / sum(X_i^2)

    파라미터:
    - structural_mode:
        acceleration_model 쪽 구조 가속도 연결 방식과 동일해야 함
    - beta_min, beta_max:
        수치 폭주 방지용 범위
    - ridge:
        0 나눗셈 방지용 작은 값
    - use_nonnegative_target:
        True이면 Y<0 구간을 0으로 잘라 보수적으로 추정
    """
    if "r_kpc" not in galaxy_data.columns:
        raise ValueError("galaxy_data must contain 'r_kpc'.")

    r = galaxy_data["r_kpc"].to_numpy(dtype=float)
    sigma = np.asarray(sigma_profile, dtype=float)

    if r.shape != sigma.shape:
        raise ValueError("r_kpc and sigma_profile must have the same shape.")

    # 1) 바리온 가속도
    a_bar = compute_baryonic_acceleration(
        galaxy_data,
        upsilon_disk=upsilon_disk,
        upsilon_bul=upsilon_bul,
    )

    # 2) 관측이 요구하는 초과 velocity^2
    y = _compute_required_excess_v2(
        galaxy_data=galaxy_data,
        a_bar=a_bar,
    )

    if use_nonnegative_target:
        y = np.clip(y, 0.0, None)

    # 3) beta=1일 때 구조항이 제공하는 velocity^2 신호
    a_struct_beta1 = compute_structural_acceleration(
        r_kpc=r,
        sigma_profile=sigma,
        beta=1.0,
        mode=structural_mode,
    )
    x = np.asarray(r * a_struct_beta1, dtype=float)

    # 4) 유효 점 필터
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        raise ValueError("No valid data points available to derive beta.")

    # 구조 신호가 거의 없으면 beta를 기본 최소값으로 둠
    denom = float(np.sum(x * x) + ridge)
    numer = float(np.sum(x * y))
    beta = numer / denom

    # 물리적/수치적 안정화
    beta = float(np.clip(beta, beta_min, beta_max))

    return beta


def get_structural_beta(
    galaxy_data: pd.DataFrame,
    sigma_profile: np.ndarray,
    structural_mode: str = "sigma_over_r",
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
    beta_min: float = 0.0,
    beta_max: float = 2000.0,
    ridge: float = 1.0e-12,
    use_nonnegative_target: bool = False,
) -> float:
    """
    파이프라인에서 직접 호출할 구조 유도 beta.

    기존의 mean(|sigma|) 휴리스틱을 대체합니다.
    """
    return derive_beta_from_observables(
        galaxy_data=galaxy_data,
        sigma_profile=sigma_profile,
        structural_mode=structural_mode,
        upsilon_disk=upsilon_disk,
        upsilon_bul=upsilon_bul,
        beta_min=beta_min,
        beta_max=beta_max,
        ridge=ridge,
        use_nonnegative_target=use_nonnegative_target,
    )