# src/units.py

from __future__ import annotations

import numpy as np

# ----- fundamental conversion constants -----

KPC_IN_M = 3.085677581491367e19
PC_IN_M = 3.085677581491367e16
MPC_IN_M = 3.085677581491367e22

MSUN_IN_KG = 1.98847e30


# ----- helpers -----

def _as_float_array(x):
    return np.asarray(x, dtype=float)


# ----- distance conversion -----

def to_kpc(x, unit: str):
    """
    Convert distance-like quantity to kpc.

    Supported units:
    - 'kpc'
    - 'pc'
    - 'Mpc'
    - 'm'
    """
    x = _as_float_array(x)

    if unit == "kpc":
        return x
    if unit == "pc":
        return x * 1e-3
    if unit == "Mpc":
        return x * 1e3
    if unit == "m":
        return x / KPC_IN_M

    raise ValueError(f"Unsupported distance unit: {unit}")


# ----- velocity conversion -----

def to_kmps(v, unit: str):
    """
    Convert velocity-like quantity to km/s.

    Supported units:
    - 'km/s'
    - 'm/s'
    """
    v = _as_float_array(v)

    if unit == "km/s":
        return v
    if unit == "m/s":
        return v * 1e-3

    raise ValueError(f"Unsupported velocity unit: {unit}")


# ----- mass conversion -----

def to_msun(m, unit: str):
    """
    Convert mass-like quantity to solar mass (Msun).

    Supported units:
    - 'Msun'
    - 'kg'
    """
    m = _as_float_array(m)

    if unit == "Msun":
        return m
    if unit == "kg":
        return m / MSUN_IN_KG

    raise ValueError(f"Unsupported mass unit: {unit}")


# ----- derived quantities -----

def velocity_squared_over_radius(v_kmps, r_kpc):
    """
    Compute v^2 / r in units of (km/s)^2 / kpc.
    """
    v_kmps = _as_float_array(v_kmps)
    r_kpc = _as_float_array(r_kpc)

    if np.any(r_kpc <= 0):
        raise ValueError("All radius values must be positive.")

    return (v_kmps ** 2) / r_kpc


def safe_quadrature_sum(*terms):
    """
    sqrt(t1^2 + t2^2 + ...)
    """
    arrays = [_as_float_array(t) for t in terms]
    total = np.zeros_like(arrays[0], dtype=float)

    for arr in arrays:
        total += arr ** 2

    return np.sqrt(total)