"""
Structure-independent topological gravity prototype.

This module implements a validation-oriented version of the topological
gravity term where the structural acceleration is generated from the
structural information itself, rather than as a direct multiplicative
correction to the baryonic field.

Core equations
--------------
sigma(r) = D_eff(r) - D_bg

g_topo(r) = g_scale * [ sigma(r) + lambda_reorg * L0_kpc * d sigma / dr ]

g_eff(r) = g_bar(r) + g_topo(r)

v_model(r) = sqrt( r * g_eff(r) )

Units
-----
- r_kpc          : kpc
- g_bar          : (km/s)^2 / kpc
- g_scale        : (km/s)^2 / kpc
- sigma          : dimensionless
- d sigma / dr   : 1 / kpc
- v_model        : km/s
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


ArrayLike = np.ndarray


@dataclass(frozen=True)
class StructuralTopogravityResult:
    sigma: ArrayLike
    dsigma_dr: ArrayLike
    g_topo: ArrayLike
    g_eff: ArrayLike
    v_model: ArrayLike


def _as_1d_float_array(values: ArrayLike, name: str) -> ArrayLike:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _validate_same_length(*named_arrays: tuple[str, ArrayLike]) -> None:
    lengths = {name: arr.size for name, arr in named_arrays}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Array length mismatch: {lengths}")


def compute_sigma(d_eff: ArrayLike, d_bg: float) -> ArrayLike:
    """Compute structural deviation sigma = D_eff - D_bg."""
    d_eff_arr = _as_1d_float_array(d_eff, "d_eff")
    if not np.isfinite(float(d_bg)):
        raise ValueError("d_bg must be finite")
    return d_eff_arr - float(d_bg)


def compute_dsigma_dr(r_kpc: ArrayLike, sigma: ArrayLike) -> ArrayLike:
    """Compute the radial derivative of sigma using numpy.gradient."""
    r_arr = _as_1d_float_array(r_kpc, "r_kpc")
    sigma_arr = _as_1d_float_array(sigma, "sigma")
    _validate_same_length(("r_kpc", r_arr), ("sigma", sigma_arr))

    if np.any(np.diff(r_arr) <= 0):
        raise ValueError("r_kpc must be strictly increasing")

    return np.gradient(sigma_arr, r_arr)


def compute_topological_acceleration(
    r_kpc: ArrayLike,
    d_eff: ArrayLike,
    d_bg: float,
    g_scale: float,
    lambda_reorg: float = 0.0,
    l0_kpc: float = 1.0,
    clip_g_eff_to_zero: bool = True,
    g_bar: Optional[ArrayLike] = None,
) -> StructuralTopogravityResult:
    """
    Compute structure-independent topological acceleration and rotation.

    Parameters
    ----------
    r_kpc:
        Radial coordinate in kpc.
    d_eff:
        Effective structural dimension profile.
    d_bg:
        Background structural dimension.
    g_scale:
        Structural acceleration scale in (km/s)^2 / kpc.
    lambda_reorg:
        Weight for the reorganization term d sigma / dr.
    l0_kpc:
        Length scale used to make the derivative term dimensionless.
    clip_g_eff_to_zero:
        If True, clip negative total effective acceleration to zero before
        taking the square root.
    g_bar:
        Optional baryonic acceleration profile. If omitted, the model returns
        the pure topological contribution as the effective field.
    """
    r_arr = _as_1d_float_array(r_kpc, "r_kpc")
    d_eff_arr = _as_1d_float_array(d_eff, "d_eff")
    _validate_same_length(("r_kpc", r_arr), ("d_eff", d_eff_arr))

    if not np.isfinite(float(g_scale)):
        raise ValueError("g_scale must be finite")
    if not np.isfinite(float(lambda_reorg)):
        raise ValueError("lambda_reorg must be finite")
    if not np.isfinite(float(l0_kpc)) or float(l0_kpc) <= 0.0:
        raise ValueError("l0_kpc must be a positive finite scalar")

    sigma = compute_sigma(d_eff_arr, d_bg)
    dsigma_dr = compute_dsigma_dr(r_arr, sigma)
    g_topo = float(g_scale) * (sigma + float(lambda_reorg) * float(l0_kpc) * dsigma_dr)

    if g_bar is None:
        g_eff = g_topo.copy()
    else:
        g_bar_arr = _as_1d_float_array(g_bar, "g_bar")
        _validate_same_length(("r_kpc", r_arr), ("g_bar", g_bar_arr))
        g_eff = g_bar_arr + g_topo

    if clip_g_eff_to_zero:
        g_eff = np.clip(g_eff, 0.0, None)

    v_model = np.sqrt(r_arr * g_eff)

    return StructuralTopogravityResult(
        sigma=sigma,
        dsigma_dr=dsigma_dr,
        g_topo=g_topo,
        g_eff=g_eff,
        v_model=v_model,
    )


def result_to_dict(result: StructuralTopogravityResult) -> Dict[str, ArrayLike]:
    """Convert the dataclass result to a plain dictionary."""
    return {
        "sigma": result.sigma,
        "dsigma_dr": result.dsigma_dr,
        "g_topo": result.g_topo,
        "g_eff": result.g_eff,
        "v_model": result.v_model,
    }


def example_usage() -> Dict[str, ArrayLike]:
    """Small self-contained example for quick manual validation."""
    r_kpc = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    d_eff = np.array([2.55, 2.58, 2.63, 2.67, 2.70])
    g_bar = np.array([8000.0, 4200.0, 2800.0, 2100.0, 1700.0])

    result = compute_topological_acceleration(
        r_kpc=r_kpc,
        d_eff=d_eff,
        d_bg=2.50,
        g_scale=300.0,
        lambda_reorg=0.25,
        l0_kpc=1.0,
        g_bar=g_bar,
        clip_g_eff_to_zero=True,
    )
    return result_to_dict(result)


if __name__ == "__main__":
    output = example_usage()
    for key, value in output.items():
        print(f"{key}: {value}")
