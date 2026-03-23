from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np



def _to_array(values, *, fill_value=np.nan) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.array([float(arr)], dtype=float)
    arr = np.where(np.isfinite(arr), arr, fill_value)
    return arr



def _ensure_same_length(reference: np.ndarray, values, *, default=np.nan) -> np.ndarray:
    arr = _to_array(values)
    if arr.size == 0:
        return np.full_like(reference, default, dtype=float)
    if arr.size == 1:
        return np.full_like(reference, float(arr[0]), dtype=float)
    if arr.size != reference.size:
        raise ValueError("Diagnostic arrays must have the same length as r_kpc.")
    return arr



def save_rotation_curve_plot(
    r_kpc,
    v_obs_kmps,
    v_err_kmps,
    v_model_kmps,
    galaxy_name: str,
    output_path: str | Path,
    *,
    v_baryon_kmps=None,
    sigma_profile=None,
    sigma_smoothed=None,
    sigma_transferred=None,
    beta_local=None,
    a_struct_internal=None,
    a_struct=None,
    d_eff_local=None,
    cinfo_cap=None,
    eta_profile=None,
    projection_profile=None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    r_kpc = np.asarray(r_kpc, dtype=float)
    v_obs_kmps = np.asarray(v_obs_kmps, dtype=float)
    v_err_kmps = np.asarray(v_err_kmps, dtype=float)
    v_model_kmps = np.asarray(v_model_kmps, dtype=float)

    v_baryon_kmps = _ensure_same_length(r_kpc, v_baryon_kmps)
    sigma_profile = _ensure_same_length(r_kpc, sigma_profile)
    sigma_smoothed = _ensure_same_length(r_kpc, sigma_smoothed)
    sigma_transferred = _ensure_same_length(r_kpc, sigma_transferred)
    beta_local = _ensure_same_length(r_kpc, beta_local)
    a_struct_internal = _ensure_same_length(r_kpc, a_struct_internal)
    a_struct = _ensure_same_length(r_kpc, a_struct)
    d_eff_local = _ensure_same_length(r_kpc, d_eff_local)
    cinfo_cap = _ensure_same_length(r_kpc, cinfo_cap)
    eta_profile = _ensure_same_length(r_kpc, eta_profile)
    projection_profile = _ensure_same_length(r_kpc, projection_profile)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(9, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.6, 1.4]},
    )

    ax = axes[0]
    ax.errorbar(r_kpc, v_obs_kmps, yerr=v_err_kmps, fmt="o", markersize=4, capsize=3, label="Observed")
    ax.plot(r_kpc, v_model_kmps, linewidth=2, label="Model")
    if np.isfinite(v_baryon_kmps).any():
        ax.plot(r_kpc, v_baryon_kmps, linestyle="--", linewidth=1.8, label="Baryonic")
    ax.set_ylabel("Rotation velocity [km/s]")
    ax.set_title(f"Structural Coupling Diagnostics: {galaxy_name}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1]
    if np.isfinite(sigma_profile).any():
        ax.plot(r_kpc, sigma_profile, linewidth=1.6, label="Sigma raw")
    if np.isfinite(sigma_smoothed).any():
        ax.plot(r_kpc, sigma_smoothed, linewidth=1.5, linestyle="--", label="Sigma smoothed")
    if np.isfinite(sigma_transferred).any():
        ax.plot(r_kpc, sigma_transferred, linewidth=1.5, linestyle=":", label="Sigma transferred")
    if np.isfinite(a_struct_internal).any():
        ax.plot(r_kpc, a_struct_internal, linewidth=1.6, linestyle="-.", label="Structural acc. internal")
    if np.isfinite(a_struct).any():
        ax.plot(r_kpc, a_struct, linewidth=1.8, label="Structural acc. observed")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("Structural terms")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax = axes[2]
    if np.isfinite(beta_local).any():
        ax.plot(r_kpc, beta_local, linewidth=1.8, label="Beta local")
    if np.isfinite(d_eff_local).any():
        ax.plot(r_kpc, d_eff_local, linewidth=1.5, linestyle="--", label="D_eff local")
    if np.isfinite(cinfo_cap).any():
        ax.plot(r_kpc, cinfo_cap, linewidth=1.5, linestyle=":", label="Cinfo cap")
    if np.isfinite(eta_profile).any():
        ax.plot(r_kpc, eta_profile, linewidth=1.4, linestyle="-.", label="Eta cap")
    if np.isfinite(projection_profile).any():
        ax.plot(r_kpc, projection_profile, linewidth=1.4, linestyle=(0, (3, 1, 1, 1)), label="Observer projection")
    ax.set_xlabel("Radius [kpc]")
    ax.set_ylabel("Coupling controls")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
