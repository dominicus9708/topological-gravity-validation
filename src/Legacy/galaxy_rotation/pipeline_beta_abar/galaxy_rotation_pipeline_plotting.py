from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_rotation_curve_plot(
    r_kpc,
    v_obs_kmps,
    v_err_kmps,
    v_model_kmps,
    galaxy_name: str,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    r_kpc = np.asarray(r_kpc, dtype=float)
    v_obs_kmps = np.asarray(v_obs_kmps, dtype=float)
    v_err_kmps = np.asarray(v_err_kmps, dtype=float)
    v_model_kmps = np.asarray(v_model_kmps, dtype=float)

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        r_kpc,
        v_obs_kmps,
        yerr=v_err_kmps,
        fmt="o",
        markersize=4,
        capsize=3,
        label="Observed",
    )
    plt.plot(
        r_kpc,
        v_model_kmps,
        linewidth=2,
        label="Model",
    )

    plt.xlabel("Radius [kpc]")
    plt.ylabel("Rotation velocity [km/s]")
    plt.title(f"Rotation Curve: {galaxy_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()