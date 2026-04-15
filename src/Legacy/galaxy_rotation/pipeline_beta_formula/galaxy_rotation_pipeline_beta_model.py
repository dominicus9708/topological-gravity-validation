from __future__ import annotations

import numpy as np
import pandas as pd

from galaxy_rotation_pipeline_acceleration_model import (
    compute_baryonic_acceleration,
    compute_structural_acceleration,
)


# ---------------------------------------------------------
# 1. baseline
# ---------------------------------------------------------
def get_constant_beta(value: float = 200.0) -> float:
    return float(value)


# ---------------------------------------------------------
# 2. observables
# ---------------------------------------------------------
def _compute_required_excess_v2(
    galaxy_data: pd.DataFrame,
    a_bar: np.ndarray,
) -> np.ndarray:
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


def compute_beta_observables(
    galaxy_data: pd.DataFrame,
    sigma_profile: np.ndarray,
    floor: float = 1.0e-12,
) -> dict[str, float]:
    sigma = np.asarray(sigma_profile, dtype=float)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma_pos = np.clip(sigma, 0.0, None)

    a_bar = compute_baryonic_acceleration(galaxy_data)

    gbar_char_median = float(np.median(a_bar)) if a_bar.size else np.nan
    gbar_char_logmean = (
        float(np.exp(np.mean(np.log(np.maximum(a_bar, floor)))))
        if a_bar.size else np.nan
    )

    sigma_char_mean = float(np.mean(sigma_pos)) if sigma_pos.size else np.nan
    sigma_char_median = float(np.median(sigma_pos)) if sigma_pos.size else np.nan
    sigma_char_rms = float(np.sqrt(np.mean(sigma_pos ** 2))) if sigma_pos.size else np.nan
    n_sigma_positive = int(np.sum(sigma_pos > 0.0))

    return {
        "gbar_char_median": gbar_char_median,
        "gbar_char_logmean": gbar_char_logmean,
        "sigma_char_mean": sigma_char_mean,
        "sigma_char_median": sigma_char_median,
        "sigma_char_rms": sigma_char_rms,
        "n_sigma_positive": n_sigma_positive,
    }


# ---------------------------------------------------------
# 3. raw beta estimation (data-driven target)
# ---------------------------------------------------------
def derive_beta_raw_from_observables(
    galaxy_data: pd.DataFrame,
    sigma_profile: np.ndarray,
    structural_mode: str = "sigma_over_r",
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
    ridge: float = 1.0e-12,
    use_nonnegative_target: bool = False,
) -> dict[str, np.ndarray | float]:
    r = galaxy_data["r_kpc"].to_numpy(dtype=float)
    sigma = np.asarray(sigma_profile, dtype=float)

    if r.shape != sigma.shape:
        raise ValueError("r_kpc and sigma_profile must have the same shape.")

    a_bar = compute_baryonic_acceleration(
        galaxy_data,
        upsilon_disk=upsilon_disk,
        upsilon_bul=upsilon_bul,
    )

    y = _compute_required_excess_v2(
        galaxy_data=galaxy_data,
        a_bar=a_bar,
    )
    if use_nonnegative_target:
        y = np.clip(y, 0.0, None)

    a_struct_beta1 = compute_structural_acceleration(
        r_kpc=r,
        sigma_profile=sigma,
        beta=1.0,
        mode=structural_mode,
    )
    x = np.asarray(r * a_struct_beta1, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]

    if x_valid.size == 0:
        raise ValueError("No valid data points available to derive raw beta.")

    denominator = float(np.sum(x_valid * x_valid) + ridge)
    numerator = float(np.sum(x_valid * y_valid))
    beta_raw = numerator / denominator

    return {
        "beta_raw": float(beta_raw),
        "numerator": float(numerator),
        "denominator": float(denominator),
        "target_y": y,
        "structural_x": x,
        "a_bar": a_bar,
    }


# ---------------------------------------------------------
# 4. base formula beta
# ---------------------------------------------------------
def compute_formula_beta_from_observables(
    g_char: float,
    sigma_char: float,
    g0: float,
    sigma0: float,
    beta_min: float = 0.0,
    beta_max: float = 2000.0,
    p: float = 1.0,
    q: float = 1.0,
    lambda_beta: float = 1.0,
    floor: float = 1.0e-12,
) -> dict[str, float]:
    g_char = max(float(g_char), floor)
    sigma_char = max(float(sigma_char), floor)
    g0 = max(float(g0), floor)
    sigma0 = max(float(sigma0), floor)

    if beta_max <= beta_min:
        raise ValueError("beta_max must be greater than beta_min.")

    Z = float(lambda_beta) * (g_char / g0) ** float(p) * (sigma_char / sigma0) ** float(q)
    Z = max(float(Z), 0.0)

    beta_formula = float(beta_min + (beta_max - beta_min) * (Z / (1.0 + Z)))

    return {
        "beta_formula": beta_formula,
        "beta_clipped": beta_formula,
        "beta_z": float(Z),
        "beta_saturation_ratio": float(Z / (1.0 + Z)),
        "beta_hit_lower": False,
        "beta_hit_upper": bool(beta_formula >= beta_max * 0.999999),
    }


# ---------------------------------------------------------
# 5. dimension proxy branch
# ---------------------------------------------------------
def compute_dimension_proxy_from_observables(
    g_char: float,
    sigma_char: float,
    g0: float,
    sigma0: float,
    d_ref: float = 3.0,
    dim_sensitivity: float = 0.60,
    dim_g_power: float = 1.0,
    dim_sigma_power: float = 1.0,
    d_min: float = 1.5,
    d_max: float = 4.5,
    floor: float = 1.0e-12,
) -> dict[str, float]:
    """
    양방향 D_eff 프록시.

    핵심:
    - sigma가 baryonic scale보다 상대적으로 크면 D_eff < d_ref
    - baryonic scale이 sigma보다 상대적으로 크면 D_eff > d_ref

    정의:
        g_hat = (g_char / g0)^a
        s_hat = (sigma_char / sigma0)^b
        balance_log = log(s_hat) - log(g_hat)
        d_eff = d_ref - sensitivity * balance_log

    따라서:
    - s_hat > g_hat -> balance_log > 0 -> d_eff < d_ref
    - g_hat > s_hat -> balance_log < 0 -> d_eff > d_ref
    """
    g_char = max(float(g_char), floor)
    sigma_char = max(float(sigma_char), floor)
    g0 = max(float(g0), floor)
    sigma0 = max(float(sigma0), floor)

    g_hat = (g_char / g0) ** float(max(dim_g_power, 0.0))
    s_hat = (sigma_char / sigma0) ** float(max(dim_sigma_power, 0.0))

    g_hat = max(float(g_hat), floor)
    s_hat = max(float(s_hat), floor)

    balance_log = float(np.log(s_hat) - np.log(g_hat))
    ratio = float(s_hat / g_hat)

    d_eff_proxy = float(d_ref - float(dim_sensitivity) * balance_log)
    d_eff_proxy = float(np.clip(d_eff_proxy, d_min, d_max))

    return {
        "d_eff_proxy": d_eff_proxy,
        "d_eff_ratio": ratio,
        "d_eff_g_hat": g_hat,
        "d_eff_sigma_hat": s_hat,
        "d_eff_balance_log": balance_log,
    }


def compute_dimension_response_factor(
    d_eff_proxy: float,
    d_ref: float = 3.0,
    dim_response_lambda: float = 0.30,
    dim_transition: float = 0.50,
    response_min: float = 0.70,
    response_max: float = 1.40,
    floor: float = 1.0e-12,
) -> dict[str, float]:
    """
    양방향 응답 함수.

    D < 3  -> response > 1
    D = 3  -> response = 1
    D > 3  -> response < 1
    """
    d_eff_proxy = float(d_eff_proxy)
    dim_transition = max(float(dim_transition), floor)
    dim_response_lambda = max(float(dim_response_lambda), 0.0)
    response_min = float(min(response_min, 1.0))
    response_max = float(max(response_max, 1.0))

    signed_delta = float(d_ref - d_eff_proxy)
    response = 1.0 + dim_response_lambda * np.tanh(signed_delta / dim_transition)
    response = float(np.clip(response, response_min, response_max))

    return {
        "d_eff_signed_delta": signed_delta,
        "d_eff_response": float(response),
    }


def apply_dimension_response_to_beta(
    beta_base: float,
    beta_min: float,
    beta_max: float,
    d_eff_proxy: float,
    d_ref: float = 3.0,
    dim_response_lambda: float = 0.30,
    dim_transition: float = 0.50,
    response_min: float = 0.70,
    response_max: float = 1.40,
) -> dict[str, float]:
    response_info = compute_dimension_response_factor(
        d_eff_proxy=d_eff_proxy,
        d_ref=d_ref,
        dim_response_lambda=dim_response_lambda,
        dim_transition=dim_transition,
        response_min=response_min,
        response_max=response_max,
    )

    beta_dimensional = float(beta_base) * float(response_info["d_eff_response"])
    beta_dimensional = float(np.clip(beta_dimensional, beta_min, beta_max))

    return {
        "beta_dimensional": float(beta_dimensional),
        "d_eff_signed_delta": float(response_info["d_eff_signed_delta"]),
        "d_eff_response": float(response_info["d_eff_response"]),
    }


# ---------------------------------------------------------
# 6. public beta detail APIs
# ---------------------------------------------------------
def get_formula_beta_details(
    galaxy_data: pd.DataFrame,
    sigma_profile: np.ndarray,
    g0: float,
    sigma0: float,
    beta_min: float = 0.0,
    beta_max: float = 2000.0,
    p: float = 1.0,
    q: float = 1.0,
    lambda_beta: float = 1.0,
    g_choice: str = "gbar_char_logmean",
    sigma_choice: str = "sigma_char_rms",
    use_dimension_proxy: bool = False,
    d_ref: float = 3.0,
    dim_sensitivity: float = 0.60,
    dim_g_power: float = 1.0,
    dim_sigma_power: float = 1.0,
    d_min: float = 1.5,
    d_max: float = 4.5,
    dim_response_lambda: float = 0.30,
    dim_transition: float = 0.50,
    dim_response_min: float = 0.70,
    dim_response_max: float = 1.40,
    dim_balance_scale: float = 1.0,
    g_suppression_lambda: float = 0.0,
    g_suppression_power: float = 1.0,
) -> dict[str, float | bool]:
    obs = compute_beta_observables(
        galaxy_data=galaxy_data,
        sigma_profile=sigma_profile,
    )

    if g_choice not in obs:
        raise ValueError(f"Unsupported g_choice: {g_choice}")
    if sigma_choice not in obs:
        raise ValueError(f"Unsupported sigma_choice: {sigma_choice}")

    g_char = float(obs[g_choice])
    sigma_char = float(obs[sigma_choice])

    formula = compute_formula_beta_from_observables(
        g_char=g_char,
        sigma_char=sigma_char,
        g0=g0,
        sigma0=sigma0,
        beta_min=beta_min,
        beta_max=beta_max,
        p=p,
        q=q,
        lambda_beta=lambda_beta,
    )

    beta_base = float(formula["beta_formula"])

    d_eff_proxy = np.nan
    d_eff_ratio = np.nan
    d_eff_g_hat = np.nan
    d_eff_sigma_hat = np.nan
    d_eff_balance_log = np.nan
    d_eff_signed_delta = 0.0
    d_eff_response = 1.0
    beta_effective = beta_base

    # NOTE:
    # 현재 단계에서는 원인 분리를 위해 D_eff 응답을 "계산만" 하고,
    # beta 자체에는 다시 곱하지 않습니다.
    # 즉, D_eff는 해석/진단 지표로 유지하되 별도 차원 응답의 중복 적용은 제거합니다.
    if use_dimension_proxy:
        dim_info = compute_dimension_proxy_from_observables(
            g_char=g_char,
            sigma_char=sigma_char,
            g0=g0,
            sigma0=sigma0,
            d_ref=d_ref,
            dim_sensitivity=dim_sensitivity,
            dim_g_power=dim_g_power,
            dim_sigma_power=dim_sigma_power,
            d_min=d_min,
            d_max=d_max,
        )
        response_info = compute_dimension_response_factor(
            d_eff_proxy=float(dim_info["d_eff_proxy"]),
            d_ref=d_ref,
            dim_response_lambda=dim_response_lambda,
            dim_transition=dim_transition,
            response_min=dim_response_min,
            response_max=dim_response_max,
        )

        d_eff_proxy = float(dim_info["d_eff_proxy"])
        d_eff_ratio = float(dim_info["d_eff_ratio"])
        d_eff_g_hat = float(dim_info["d_eff_g_hat"])
        d_eff_sigma_hat = float(dim_info["d_eff_sigma_hat"])
        d_eff_balance_log = float(dim_info["d_eff_balance_log"])
        d_eff_signed_delta = float(response_info["d_eff_signed_delta"])
        d_eff_response = float(response_info["d_eff_response"])
        beta_effective = float(beta_base)

    out = {
        "beta_raw": np.nan,
        "beta_formula": float(beta_base),
        "beta_clipped": float(beta_effective),
        "beta_saturated": float(beta_effective),
        "beta_hit_lower": bool(formula["beta_hit_lower"]),
        "beta_hit_upper": bool(beta_effective >= beta_max * 0.999999),
        "beta_saturation_ratio": float(formula["beta_saturation_ratio"]),
        "beta_z": float(formula["beta_z"]),
        "beta_numerator": np.nan,
        "beta_denominator": np.nan,
        "beta_g_choice": g_choice,
        "beta_sigma_choice": sigma_choice,
        "use_dimension_proxy": bool(use_dimension_proxy),
        "dimension_response_applied": False,
        "d_eff_proxy": float(d_eff_proxy) if np.isfinite(d_eff_proxy) else np.nan,
        "d_eff_ratio": float(d_eff_ratio) if np.isfinite(d_eff_ratio) else np.nan,
        "d_eff_g_hat": float(d_eff_g_hat) if np.isfinite(d_eff_g_hat) else np.nan,
        "d_eff_sigma_hat": float(d_eff_sigma_hat) if np.isfinite(d_eff_sigma_hat) else np.nan,
        "d_eff_balance_log": float(d_eff_balance_log) if np.isfinite(d_eff_balance_log) else np.nan,
        "d_eff_signed_delta": float(d_eff_signed_delta),
        "d_eff_response": float(d_eff_response),
        "dim_balance_scale": float(dim_balance_scale),
        "g_suppression_lambda": float(g_suppression_lambda),
        "g_suppression_power": float(g_suppression_power),
    }
    out.update(obs)
    return out


# ---------------------------------------------------------
# 7. structural mode saturation
# ---------------------------------------------------------
def saturate_beta(
    beta_raw: float,
    beta_min: float = 0.0,
    beta_max: float = 2000.0,
    eps: float = 1.0e-12,
) -> dict[str, float | bool]:
    beta_raw = float(beta_raw)
    beta_min = float(beta_min)
    beta_max = float(beta_max)

    if beta_max <= beta_min:
        raise ValueError("beta_max must be greater than beta_min.")

    beta_excess = max(beta_raw - beta_min, 0.0)
    scale = (beta_max - beta_min) + float(eps)
    z = beta_excess / scale

    beta_sat = beta_min + (beta_max - beta_min) * (z / (1.0 + z))

    hit_lower = bool(beta_raw < beta_min)
    hit_upper = bool(beta_raw > beta_max)

    return {
        "beta_saturated": float(beta_sat),
        "beta_clipped": float(beta_sat),
        "beta_hit_lower": hit_lower,
        "beta_hit_upper": hit_upper,
        "beta_saturation_ratio": float(z / (1.0 + z)),
        "beta_z": float(z),
    }


def get_structural_beta_details(
    galaxy_data: pd.DataFrame,
    sigma_profile: np.ndarray,
    structural_mode: str = "sigma_over_r",
    upsilon_disk: float = 1.0,
    upsilon_bul: float = 1.0,
    beta_min: float = 0.0,
    beta_max: float = 2000.0,
    ridge: float = 1.0e-12,
    use_nonnegative_target: bool = False,
) -> dict[str, float | bool]:
    raw_info = derive_beta_raw_from_observables(
        galaxy_data=galaxy_data,
        sigma_profile=sigma_profile,
        structural_mode=structural_mode,
        upsilon_disk=upsilon_disk,
        upsilon_bul=upsilon_bul,
        ridge=ridge,
        use_nonnegative_target=use_nonnegative_target,
    )

    sat_info = saturate_beta(
        beta_raw=float(raw_info["beta_raw"]),
        beta_min=beta_min,
        beta_max=beta_max,
    )

    obs_info = compute_beta_observables(
        galaxy_data=galaxy_data,
        sigma_profile=sigma_profile,
    )

    out = {
        "beta_raw": float(raw_info["beta_raw"]),
        "beta_formula": np.nan,
        "beta_clipped": float(sat_info["beta_clipped"]),
        "beta_saturated": float(sat_info["beta_saturated"]),
        "beta_hit_lower": bool(sat_info["beta_hit_lower"]),
        "beta_hit_upper": bool(sat_info["beta_hit_upper"]),
        "beta_saturation_ratio": float(sat_info["beta_saturation_ratio"]),
        "beta_z": float(sat_info["beta_z"]),
        "beta_numerator": float(raw_info["numerator"]),
        "beta_denominator": float(raw_info["denominator"]),
        "beta_g_choice": None,
        "beta_sigma_choice": None,
        "use_dimension_proxy": False,
        "dimension_response_applied": False,
        "d_eff_proxy": np.nan,
        "d_eff_ratio": np.nan,
        "d_eff_g_hat": np.nan,
        "d_eff_sigma_hat": np.nan,
        "d_eff_balance_log": np.nan,
        "d_eff_signed_delta": 0.0,
        "d_eff_response": 1.0,
    }
    out.update(obs_info)
    return out