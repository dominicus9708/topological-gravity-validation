from __future__ import annotations

import numpy as np

from galaxy_rotation_legacy_constants import C_INFO_NORM


# ----- compatibility aliases -----

c_info = C_INFO_NORM
xi = 1.0


# ----- core topological-gravity relations -----

def compute_information_velocity_scale(r_kpc):
    r_kpc = np.asarray(r_kpc, dtype=float)

    if np.any(r_kpc <= 0):
        raise ValueError("All radius values must be positive.")

    return c_info / (1.0 + xi * r_kpc)


def compute_topological_potential(r_kpc):
    r_kpc = np.asarray(r_kpc, dtype=float)

    if np.any(r_kpc <= 0):
        raise ValueError("All radius values must be positive.")

    return c_info * np.log(1.0 + xi * r_kpc)


def compute_topological_acceleration(r_kpc):
    r_kpc = np.asarray(r_kpc, dtype=float)

    if np.any(r_kpc <= 0):
        raise ValueError("All radius values must be positive.")

    return c_info * xi / (1.0 + xi * r_kpc)