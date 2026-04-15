import numpy as np


def _smooth_profile(arr: np.ndarray, window: int = 3) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if window <= 1 or arr.size < window:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="same")


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


def _sigma_transfer(sigma: np.ndarray, scale: float | None = None) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float)
    if scale is None:
        scale = max(float(np.nanpercentile(np.abs(sigma), 75)), 1e-6)
    return np.tanh(sigma / scale)


def compute_structural_acceleration(
    r: np.ndarray,
    sigma: np.ndarray,
    beta,
    g_newton=None,
    a_bar=None,
    eta_profile: np.ndarray | None = None,
    projection_profile: np.ndarray | None = None,
    d_eff_local=None,
    cinfo_cap=None,
    return_diagnostics: bool = False,
    eps: float = 1e-12,
):
    """
    Compatibility layer for pipeline_structural_coupling.

    Important v8 fix:
    sigma_flip_severity is returned as a scalar, not an (N-1)-length array,
    so per-radius tables do not fail with shape mismatch.
    """
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if a_bar is None and g_newton is not None:
        a_bar = np.asarray(g_newton, dtype=float)
    elif a_bar is not None:
        a_bar = np.asarray(a_bar, dtype=float)

    sigma_smoothed = _smooth_profile(sigma, window=5)
    sigma_transferred = _sigma_transfer(sigma_smoothed)

    # constant beta / scalar mode
    if np.isscalar(beta):
        beta_local = np.full_like(sigma_transferred, float(beta), dtype=float)
    else:
        beta_local = np.asarray(beta, dtype=float)

    a_struct_internal = beta_local * sigma_transferred / np.maximum(r, eps)

    if projection_profile is None:
        projection_profile = np.ones_like(a_struct_internal, dtype=float)
    else:
        projection_profile = np.asarray(projection_profile, dtype=float)

    a_struct_observed = a_struct_internal * projection_profile

    if a_bar is None or eta_profile is None:
        a_struct = a_struct_observed.copy()
        struct_to_bar_ratio = np.full_like(a_struct, np.nan, dtype=float)
    else:
        eta_profile = np.asarray(eta_profile, dtype=float)
        limit = eta_profile * np.abs(a_bar)
        a_struct = a_struct_observed * np.tanh(
            limit / (np.abs(a_struct_observed) + 1e-8)
        )
        struct_to_bar_ratio = np.abs(a_struct) / (np.abs(a_bar) + 1e-8)

    # v8 fix: scalar summary value instead of (N-1)-length array
    sigma_flip_severity = (
        float(np.max(np.abs(np.diff(sigma_transferred))))
        if len(sigma_transferred) > 1
        else 0.0
    )

    if return_diagnostics:
        return {
            "sigma_smoothed": sigma_smoothed,
            "sigma_transferred": sigma_transferred,
            "projection_profile": projection_profile,
            "a_struct_internal": a_struct_internal,
            "a_struct_observed": a_struct_observed,
            "a_struct_raw": a_struct_observed,
            "a_struct": a_struct,
            "struct_to_bar_ratio": struct_to_bar_ratio,
            "sigma_flip_severity": sigma_flip_severity,
        }

    return a_struct


def compute_total_acceleration(
    g_newton: np.ndarray,
    g_struct,
) -> np.ndarray:
    g_newton = np.asarray(g_newton, dtype=float)
    if isinstance(g_struct, dict):
        g_struct = np.asarray(g_struct["a_struct"], dtype=float)
    else:
        g_struct = np.asarray(g_struct, dtype=float)
    return g_newton + g_struct
