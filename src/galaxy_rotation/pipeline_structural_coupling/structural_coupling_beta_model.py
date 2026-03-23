import numpy as np
import pandas as pd


def get_constant_beta(value: float = 200.0) -> float:
    return float(value)


def get_structural_beta(galaxy_data: pd.DataFrame, sigma_profile: np.ndarray) -> float:
    sigma_mean = float(np.mean(np.abs(np.asarray(sigma_profile, dtype=float))))
    return max(1.0, sigma_mean)


def _smooth_profile(arr: np.ndarray, window: int = 3) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if window <= 1 or arr.size < window:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="same")


def _safe_normalize(arr: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    scale = np.nanpercentile(np.abs(arr), 75)
    scale = max(float(scale), floor)
    return arr / scale


def compute_local_effective_dimension(
    sigma: np.ndarray,
    a_bar: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    a_bar = np.asarray(a_bar, dtype=float)
    r = np.asarray(r, dtype=float)

    base = np.abs(sigma) / (np.abs(a_bar) / (r + 1e-6) + 1e-8)
    base = _smooth_profile(base, window=5)
    base = np.tanh(base)

    d_eff = 2.0 + 0.5 * base
    return np.clip(d_eff, 2.0, 2.5)


def compute_local_cinfo_cap(d_eff: np.ndarray) -> np.ndarray:
    d_eff = np.asarray(d_eff, dtype=float)
    cap = 0.38 + 0.48 * ((d_eff - 2.0) / 0.5)
    return np.clip(cap, 0.30, 0.92)


def compute_eta_profile(d_eff: np.ndarray, cinfo_cap: np.ndarray) -> np.ndarray:
    d_eff = np.asarray(d_eff, dtype=float)
    cinfo_cap = np.asarray(cinfo_cap, dtype=float)

    eta_base = 0.24 + 0.28 * ((d_eff - 2.0) / 0.5)
    eta = eta_base * (0.78 + 0.42 * cinfo_cap)
    return np.clip(eta, 0.18, 0.62)


def compute_thickness_factor(
    r: np.ndarray,
    d_eff: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
) -> np.ndarray:
    """
    Empirical observer-thickness attenuation.
    Outer disk is treated as observationally thicker / more diffuse, so
    structural coupling is projected less directly there.

    This is not a measured scale-height input. It is a SPARC-only proxy
    intended for v8 until explicit thickness data are available.
    """
    r = np.asarray(r, dtype=float)
    eps = 1e-8
    r_max = max(float(np.max(r)), eps)
    r_norm = np.clip(r / r_max, 0.0, 1.0)

    # Base attenuation: stronger decay in outer disk
    outer_decay = 1.0 / (1.0 + (r_norm / 0.62) ** 2.2)

    # Keep center from being artificially over-amplified
    center_moderation = 0.90 + 0.10 * (1.0 - np.exp(-4.0 * r_norm))

    thickness = outer_decay * center_moderation

    if d_eff is not None:
        d_eff = np.asarray(d_eff, dtype=float)
        d_term = np.clip((d_eff - 2.0) / 0.5, 0.0, 1.0)
        # higher structural accessibility mildly resists outer attenuation
        thickness *= (0.88 + 0.18 * d_term)

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        sigma_sm = _smooth_profile(sigma, window=5)
        sigma_term = np.tanh(np.abs(_safe_normalize(sigma_sm)))
        # strong local structure remains somewhat more visible
        thickness *= (0.90 + 0.12 * sigma_term)

    return np.clip(thickness, 0.18, 1.00)


def compute_projection_profile(
    r: np.ndarray,
    d_eff: np.ndarray,
    cinfo_cap: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    v8:
    observer projection with thickness attenuation included.
    """
    r = np.asarray(r, dtype=float)
    d_eff = np.asarray(d_eff, dtype=float)
    cinfo_cap = np.asarray(cinfo_cap, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    eps = 1e-8
    r_max = max(float(np.max(r)), eps)
    r_norm = np.clip(r / r_max, 0.0, 1.0)

    sigma_sm = _smooth_profile(sigma, window=5)

    # sigma magnitude
    sigma_norm = np.abs(_safe_normalize(sigma_sm))
    sigma_mag_term = 0.68 + 0.34 * np.tanh(sigma_norm)

    # sigma gradient damping
    if len(r) > 1:
        ds_dr = np.gradient(sigma_sm, r + eps)
    else:
        ds_dr = np.zeros_like(sigma_sm)
    grad_norm = np.abs(_safe_normalize(ds_dr))
    grad_damping = 1.0 - 0.26 * np.tanh(grad_norm)
    grad_damping = np.clip(grad_damping, 0.60, 1.00)

    # d_eff term
    d_eff_frac = np.clip((d_eff - 2.0) / 0.5, 0.0, 1.0)
    d_eff_term = 0.52 + 0.58 * (d_eff_frac ** 1.20)

    # cinfo term
    cinfo_frac = np.clip((cinfo_cap - 0.30) / 0.62, 0.0, 1.0)
    cinfo_term = 0.62 + 0.42 * (cinfo_frac ** 1.10)

    # radial visibility
    radial_outer = 1.0 - 0.42 * (r_norm ** 0.85)
    radial_outer = np.clip(radial_outer, 0.45, 1.0)
    radial_inner = 0.86 + 0.14 * (1.0 - np.exp(-4.0 * r_norm))
    radial_term = np.clip(radial_outer * radial_inner, 0.45, 1.0)

    # new thickness factor
    thickness_factor = compute_thickness_factor(r, d_eff=d_eff, sigma=sigma_sm)

    projection = (
        radial_term
        * d_eff_term
        * cinfo_term
        * sigma_mag_term
        * grad_damping
        * thickness_factor
    )

    return np.clip(projection, 0.10, 1.00)


def compute_local_beta_profile(
    r: np.ndarray,
    sigma: np.ndarray,
    a_bar: np.ndarray,
    beta0: float = 200.0,
) -> dict:
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    a_bar = np.asarray(a_bar, dtype=float)

    d_eff_local = compute_local_effective_dimension(sigma, a_bar, r)
    cinfo_cap = compute_local_cinfo_cap(d_eff_local)

    sigma_sm = _smooth_profile(sigma, window=5)
    sigma_norm = np.abs(_safe_normalize(sigma_sm))

    beta_raw = beta0 * (0.42 + 0.58 * np.tanh(sigma_norm))
    beta_limit = beta0 * cinfo_cap
    beta_local = beta_limit * np.tanh(beta_raw / (beta_limit + 1e-8))

    eta_profile = compute_eta_profile(d_eff_local, cinfo_cap)
    projection_profile = compute_projection_profile(r, d_eff_local, cinfo_cap, sigma_sm)
    thickness_factor = compute_thickness_factor(r, d_eff=d_eff_local, sigma=sigma_sm)

    return {
        "beta_local": beta_local,
        "d_eff_local": d_eff_local,
        "cinfo_cap": cinfo_cap,
        "eta_profile": eta_profile,
        "projection_profile": projection_profile,
        "thickness_factor": thickness_factor,
    }
