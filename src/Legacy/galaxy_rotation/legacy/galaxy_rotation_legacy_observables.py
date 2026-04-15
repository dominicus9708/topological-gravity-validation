# src/observables.py

from __future__ import annotations

import numpy as np

from galaxy_rotation_legacy_constants import G_ASTRO


def compute_newtonian_acceleration(m_r_msun, r_kpc):
    """
    Newtonian acceleration in astrophysical units:

        g_N = G_ASTRO * M(r) / r^2

    Units
    -----
    G_ASTRO : (km/s)^2 * kpc / Msun
    M       : Msun
    r       : kpc

    Therefore:
        g_N : (km/s)^2 / kpc
    """
    m_r_msun = np.asarray(m_r_msun, dtype=float)
    r_kpc = np.asarray(r_kpc, dtype=float)

    if np.any(r_kpc <= 0):
        raise ValueError("All radius values must be positive.")

    return G_ASTRO * m_r_msun / (r_kpc**2)


def compute_rotation_velocity_from_acceleration(g_kmps2_per_kpc, r_kpc):
    """
    From centripetal relation:
        v^2 / r = g
        v = sqrt(g * r)

    Units:
        g : (km/s)^2 / kpc
        r : kpc
        v : km/s
    """
    g_kmps2_per_kpc = np.asarray(g_kmps2_per_kpc, dtype=float)
    r_kpc = np.asarray(r_kpc, dtype=float)

    if np.any(r_kpc <= 0):
        raise ValueError("All radius values must be positive.")

    v_sq = g_kmps2_per_kpc * r_kpc
    v_sq = np.maximum(v_sq, 0.0)
    return np.sqrt(v_sq)


def compute_total_velocity_quadrature(*velocity_terms):
    """
    Combine multiple velocity contributions in quadrature:
        v_tot = sqrt(v1^2 + v2^2 + ...)

    All inputs must be in km/s.
    """
    if not velocity_terms:
        raise ValueError("At least one velocity term is required.")

    arrays = [np.asarray(v, dtype=float) for v in velocity_terms]
    total_sq = np.zeros_like(arrays[0], dtype=float)

    for arr in arrays:
        total_sq += arr**2

    total_sq = np.maximum(total_sq, 0.0)
    return np.sqrt(total_sq)
