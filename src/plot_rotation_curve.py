# src/plot_rotation_curve.py

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_rotation_curve(
    df_resid: pd.DataFrame,
    output_path: str | None = None,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(
        df_resid["r_kpc"],
        df_resid["v_obs_kmps"],
        yerr=df_resid["v_err_kmps"],
        fmt="o",
        capsize=3,
        label="Observed",
    )

    ax.plot(
        df_resid["r_kpc"],
        df_resid["v_model_kmps"],
        label="Model",
    )

    ax.set_xlabel("Radius [kpc]")
    ax.set_ylabel("Velocity [km/s]")

    galaxy_name = str(df_resid["galaxy"].iloc[0]) if "galaxy" in df_resid.columns else "unknown"
    ax.set_title(f"Rotation Curve: {galaxy_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_residuals(
    df_resid: pd.DataFrame,
    output_path: str | None = None,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.axhline(0.0)
    ax.plot(
        df_resid["r_kpc"],
        df_resid["residual_kmps"],
        marker="o",
    )

    ax.set_xlabel("Radius [kpc]")
    ax.set_ylabel("Residual [km/s]")

    galaxy_name = str(df_resid["galaxy"].iloc[0]) if "galaxy" in df_resid.columns else "unknown"
    ax.set_title(f"Residuals: {galaxy_name}")
    ax.grid(True, alpha=0.3)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)