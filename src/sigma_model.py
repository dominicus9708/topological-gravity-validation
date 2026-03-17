from __future__ import annotations

import numpy as np
import pandas as pd


# =========================================================
# 0. constants.py 호환
# =========================================================
# constants.py 안에 D_bg 또는 D_BG가 있으면 사용하고,
# 없으면 0.0으로 둡니다.
try:
    from constants import D_bg as _D_BG_FROM_CONSTANTS
    DEFAULT_D_BG = float(_D_BG_FROM_CONSTANTS)
except Exception:
    try:
        from constants import D_BG as _D_BG_FROM_CONSTANTS
        DEFAULT_D_BG = float(_D_BG_FROM_CONSTANTS)
    except Exception:
        DEFAULT_D_BG = 0.0


# =========================================================
# 1. 내부 유틸
# =========================================================
def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _safe_radius(r: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    r = _as_array(r)
    return np.maximum(r, eps)


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _sorted_by_radius(df: pd.DataFrame, r_col: str = "r_kpc") -> tuple[pd.DataFrame, np.ndarray]:
    order = np.argsort(df[r_col].to_numpy(dtype=float))
    return df.iloc[order].reset_index(drop=True), order


def _restore_original_order(arr_sorted: np.ndarray, order: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr_sorted)
    out[order] = arr_sorted
    return out


def _get_required_dataframe(df_norm: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df_norm, pd.DataFrame):
        raise TypeError("df_norm must be a pandas DataFrame.")
    if "r_kpc" not in df_norm.columns:
        raise ValueError("df_norm must contain 'r_kpc'.")
    return df_norm


def _get_vobs(df_norm: pd.DataFrame) -> np.ndarray:
    if "v_obs_kmps" in df_norm.columns:
        return df_norm["v_obs_kmps"].to_numpy(dtype=float)

    col = _pick_column(df_norm, ["Vobs", "v_obs", "V_obs", "V"])
    if col is not None:
        return df_norm[col].to_numpy(dtype=float)

    raise ValueError("Could not find observed velocity column.")


def _get_aobs(df_norm: pd.DataFrame) -> np.ndarray:
    if "a_obs_kmps2_per_kpc" in df_norm.columns:
        return df_norm["a_obs_kmps2_per_kpc"].to_numpy(dtype=float)

    r = _safe_radius(df_norm["r_kpc"].to_numpy(dtype=float))
    v_obs = _get_vobs(df_norm)
    return (v_obs ** 2) / r


def _estimate_rs_kpc(r: np.ndarray) -> float:
    r = _as_array(r)
    finite = r[np.isfinite(r)]
    if finite.size == 0:
        return 1.0
    return max(float(np.nanmedian(finite)), 1.0)


# =========================================================
# 2. 가장 기본적인 sigma 정의
# =========================================================
def compute_general_sigma(alpha, D_bg: float = DEFAULT_D_BG, positive_only: bool = True) -> np.ndarray:
    """
    일반적인 sigma 정의:
        sigma = alpha - D_bg
    positive_only=True이면 음수는 0으로 자릅니다.
    """
    alpha = _as_array(alpha)
    sigma = alpha - float(D_bg)
    if positive_only:
        sigma = np.maximum(sigma, 0.0)
    return sigma


def compute_sigma(alpha, D_bg: float = DEFAULT_D_BG, positive_only: bool = True) -> np.ndarray:
    """
    과거 코드 호환용 alias.
    """
    return compute_general_sigma(alpha, D_bg=D_bg, positive_only=positive_only)


# =========================================================
# 3. 바리온 성분
# =========================================================
def compute_baryonic_velocity(
    v_gas_kmps,
    v_disk_kmps,
    v_bul_kmps,
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
) -> np.ndarray:
    """
    바리온 기준 속도:
        v_bar^2 = v_gas^2 + (upsilon_disk)*v_disk^2 + (upsilon_bul)*v_bul^2
    """
    v_gas = _as_array(v_gas_kmps)
    v_disk = _as_array(v_disk_kmps)
    v_bul = _as_array(v_bul_kmps)

    v2 = (
        v_gas ** 2
        + float(upsilon_disk) * (v_disk ** 2)
        + float(upsilon_bul) * (v_bul ** 2)
    )
    return np.sqrt(np.clip(v2, 0.0, None))


def compute_baryonic_acceleration_profile(
    df_norm: pd.DataFrame,
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
) -> np.ndarray:
    df_norm = _get_required_dataframe(df_norm)

    required = ["v_gas_kmps", "v_disk_kmps", "v_bul_kmps"]
    for c in required:
        if c not in df_norm.columns:
            raise ValueError(f"df_norm must contain '{c}'.")

    r = _safe_radius(df_norm["r_kpc"].to_numpy(dtype=float))
    v_bar = compute_baryonic_velocity(
        df_norm["v_gas_kmps"].to_numpy(dtype=float),
        df_norm["v_disk_kmps"].to_numpy(dtype=float),
        df_norm["v_bul_kmps"].to_numpy(dtype=float),
        upsilon_disk=upsilon_disk,
        upsilon_bul=upsilon_bul,
    )
    return (v_bar ** 2) / r


# =========================================================
# 4. 구조자유도/구조편차 proxy
# =========================================================
def compute_galactic_structural_degree_proxy(
    df_norm: pd.DataFrame,
    floor: float = 1.0e-12,
    proxy_mode: str = "log_ratio",
) -> np.ndarray:
    """
    은하 구조자유도 proxy를 구성합니다.

    기본값(proxy_mode='log_ratio'):
        alpha_proxy = log( a_obs / a_bar )

    대안:
        proxy_mode='relative_excess'
        alpha_proxy = (a_obs - a_bar) / max(a_bar, floor)
    """
    df_norm = _get_required_dataframe(df_norm)

    a_obs = _get_aobs(df_norm)
    a_bar = compute_baryonic_acceleration_profile(df_norm)

    if proxy_mode == "log_ratio":
        ratio = a_obs / np.maximum(a_bar, floor)
        alpha_proxy = np.log(np.maximum(ratio, floor))
    elif proxy_mode == "relative_excess":
        alpha_proxy = (a_obs - a_bar) / np.maximum(a_bar, floor)
    else:
        raise ValueError(f"Unsupported proxy_mode: {proxy_mode}")

    return alpha_proxy


def compute_galactic_sigma_proxy(
    df_norm: pd.DataFrame,
    D_bg: float = DEFAULT_D_BG,
    positive_only: bool = True,
    proxy_mode: str = "log_ratio",
    method: str = "current",
) -> np.ndarray:
    """
    은하 sigma local proxy.

    method='legacy':
        alpha_proxy - D_bg 를 그대로 사용 (필요시 양수부만 사용)

    method='current':
        alpha_proxy - D_bg 를 쓰되,
        구조 신호를 지나치게 키우지 않도록 a_bar/a_obs 비율을 곱해 완만하게 조정
    """
    alpha_proxy = compute_galactic_structural_degree_proxy(
        df_norm,
        proxy_mode=proxy_mode,
    )
    sigma_raw = compute_general_sigma(alpha_proxy, D_bg=D_bg, positive_only=False)

    if method == "legacy":
        sigma = sigma_raw
    elif method == "current":
        a_obs = _get_aobs(df_norm)
        a_bar = compute_baryonic_acceleration_profile(df_norm)
        damping = np.clip(a_bar / np.maximum(a_obs, 1.0e-12), 0.0, 1.0)
        sigma = sigma_raw * (0.5 + 0.5 * damping)
    else:
        raise ValueError(f"Unsupported method: {method}")

    if positive_only:
        sigma = np.maximum(sigma, 0.0)

    return sigma


# =========================================================
# 5. 누적 구조 보정항
# =========================================================
def compute_galactic_sigma_cumulative_effect(
    df_norm: pd.DataFrame,
    beta: float = 200.0,
    use_positive_only: bool = True,
    rs_kpc: float | None = None,
    D_bg: float = DEFAULT_D_BG,
    proxy_mode: str = "log_ratio",
    method: str = "current",
    kernel: str = "exp",
) -> np.ndarray:
    """
    Delta_v^2(r)를 계산합니다.

    과거 방식(legacy):
        - local sigma를 단순 누적
        - kernel='flat' 또는 method='legacy'로 해석 가능

    현재 방식(current):
        - local sigma를 scale radius rs_kpc를 가진 완만한 커널로 누적
        - 기본 kernel='exp'
    """
    df_norm = _get_required_dataframe(df_norm)
    df_sorted, order = _sorted_by_radius(df_norm, r_col="r_kpc")

    r = df_sorted["r_kpc"].to_numpy(dtype=float)
    sigma_local = compute_galactic_sigma_proxy(
        df_sorted,
        D_bg=D_bg,
        positive_only=use_positive_only,
        proxy_mode=proxy_mode,
        method=method,
    )

    if rs_kpc is None:
        rs_kpc = _estimate_rs_kpc(r)
    rs_kpc = max(float(rs_kpc), 1.0e-12)

    n = len(r)
    delta_v2 = np.zeros(n, dtype=float)

    # legacy는 사실상 평평한 누적 커널
    if method == "legacy" or kernel == "flat":
        dr = np.gradient(r)
        delta_v2 = float(beta) * np.cumsum(np.maximum(sigma_local, 0.0) * dr)
        return _restore_original_order(delta_v2, order)

    # current: 반경 차이에 따른 감쇠 커널
    for i in range(n):
        dr = np.diff(np.concatenate(([0.0], r[: i + 1])))

        if kernel == "exp":
            weights = np.exp(-(r[i] - r[: i + 1]) / rs_kpc)
        elif kernel == "gaussian":
            weights = np.exp(-((r[i] - r[: i + 1]) ** 2) / (2.0 * rs_kpc ** 2))
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")

        contrib = sigma_local[: i + 1] * weights * dr
        delta_v2[i] = float(beta) * np.sum(contrib)

    return _restore_original_order(delta_v2, order)


# =========================================================
# 6. 최종 모델 회전곡선
# =========================================================
def compute_model_velocity_curve(
    df_norm: pd.DataFrame,
    beta: float = 200.0,
    use_positive_only: bool = True,
    rs_kpc: float | None = None,
    D_bg: float = DEFAULT_D_BG,
    proxy_mode: str = "log_ratio",
    method: str = "current",
    kernel: str = "exp",
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
) -> np.ndarray:
    """
    최종 회전곡선:
        v_model^2(r) = v_bar^2(r) + Delta_sigma(r)
    """
    df_norm = _get_required_dataframe(df_norm)

    required = ["v_gas_kmps", "v_disk_kmps", "v_bul_kmps"]
    for c in required:
        if c not in df_norm.columns:
            raise ValueError(f"df_norm must contain '{c}'.")

    v_bar = compute_baryonic_velocity(
        df_norm["v_gas_kmps"].to_numpy(dtype=float),
        df_norm["v_disk_kmps"].to_numpy(dtype=float),
        df_norm["v_bul_kmps"].to_numpy(dtype=float),
        upsilon_disk=upsilon_disk,
        upsilon_bul=upsilon_bul,
    )

    delta_v2 = compute_galactic_sigma_cumulative_effect(
        df_norm,
        beta=beta,
        use_positive_only=use_positive_only,
        rs_kpc=rs_kpc,
        D_bg=D_bg,
        proxy_mode=proxy_mode,
        method=method,
        kernel=kernel,
    )

    v_model2 = np.clip(v_bar ** 2 + delta_v2, 0.0, None)
    return np.sqrt(v_model2)


# =========================================================
# 7. 과거 방식 / 현재 방식 래퍼
# =========================================================
def compute_galactic_sigma_proxy_legacy(
    df_norm: pd.DataFrame,
    D_bg: float = DEFAULT_D_BG,
    positive_only: bool = True,
    proxy_mode: str = "log_ratio",
) -> np.ndarray:
    return compute_galactic_sigma_proxy(
        df_norm,
        D_bg=D_bg,
        positive_only=positive_only,
        proxy_mode=proxy_mode,
        method="legacy",
    )


def compute_galactic_sigma_proxy_current(
    df_norm: pd.DataFrame,
    D_bg: float = DEFAULT_D_BG,
    positive_only: bool = True,
    proxy_mode: str = "log_ratio",
) -> np.ndarray:
    return compute_galactic_sigma_proxy(
        df_norm,
        D_bg=D_bg,
        positive_only=positive_only,
        proxy_mode=proxy_mode,
        method="current",
    )


def compute_galactic_sigma_cumulative_effect_legacy(
    df_norm: pd.DataFrame,
    beta: float = 200.0,
    use_positive_only: bool = True,
    rs_kpc: float | None = None,
    D_bg: float = DEFAULT_D_BG,
    proxy_mode: str = "log_ratio",
) -> np.ndarray:
    return compute_galactic_sigma_cumulative_effect(
        df_norm,
        beta=beta,
        use_positive_only=use_positive_only,
        rs_kpc=rs_kpc,
        D_bg=D_bg,
        proxy_mode=proxy_mode,
        method="legacy",
        kernel="flat",
    )


def compute_galactic_sigma_cumulative_effect_current(
    df_norm: pd.DataFrame,
    beta: float = 200.0,
    use_positive_only: bool = True,
    rs_kpc: float | None = None,
    D_bg: float = DEFAULT_D_BG,
    proxy_mode: str = "log_ratio",
    kernel: str = "exp",
) -> np.ndarray:
    return compute_galactic_sigma_cumulative_effect(
        df_norm,
        beta=beta,
        use_positive_only=use_positive_only,
        rs_kpc=rs_kpc,
        D_bg=D_bg,
        proxy_mode=proxy_mode,
        method="current",
        kernel=kernel,
    )


def compute_model_velocity_curve_legacy(
    df_norm: pd.DataFrame,
    beta: float = 200.0,
    use_positive_only: bool = True,
    rs_kpc: float | None = None,
    D_bg: float = DEFAULT_D_BG,
    proxy_mode: str = "log_ratio",
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
) -> np.ndarray:
    return compute_model_velocity_curve(
        df_norm,
        beta=beta,
        use_positive_only=use_positive_only,
        rs_kpc=rs_kpc,
        D_bg=D_bg,
        proxy_mode=proxy_mode,
        method="legacy",
        kernel="flat",
        upsilon_disk=upsilon_disk,
        upsilon_bul=upsilon_bul,
    )


def compute_model_velocity_curve_current(
    df_norm: pd.DataFrame,
    beta: float = 200.0,
    use_positive_only: bool = True,
    rs_kpc: float | None = None,
    D_bg: float = DEFAULT_D_BG,
    proxy_mode: str = "log_ratio",
    kernel: str = "exp",
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
) -> np.ndarray:
    return compute_model_velocity_curve(
        df_norm,
        beta=beta,
        use_positive_only=use_positive_only,
        rs_kpc=rs_kpc,
        D_bg=D_bg,
        proxy_mode=proxy_mode,
        method="current",
        kernel=kernel,
        upsilon_disk=upsilon_disk,
        upsilon_bul=upsilon_bul,
    )


# =========================================================
# 8. exports
# =========================================================
__all__ = [
    "DEFAULT_D_BG",
    "compute_general_sigma",
    "compute_sigma",
    "compute_baryonic_velocity",
    "compute_baryonic_acceleration_profile",
    "compute_galactic_structural_degree_proxy",
    "compute_galactic_sigma_proxy",
    "compute_galactic_sigma_proxy_legacy",
    "compute_galactic_sigma_proxy_current",
    "compute_galactic_sigma_cumulative_effect",
    "compute_galactic_sigma_cumulative_effect_legacy",
    "compute_galactic_sigma_cumulative_effect_current",
    "compute_model_velocity_curve",
    "compute_model_velocity_curve_legacy",
    "compute_model_velocity_curve_current",
]