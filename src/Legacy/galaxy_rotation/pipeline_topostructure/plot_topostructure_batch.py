from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Batch plotting for pointwise topostructure outputs
#
# Expected input:
#   *_pointwise_topostructure.csv produced by galaxy_rotation_topostructure_pipeline_v2.py
#
# Output:
#   one PNG per galaxy with three panels
#   1) observed / baryonic / topostructure model velocities
#   2) residual comparison (bar-only vs topo)
#   3) structural quantities (sigma, beta, thickness)
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot batch topostructure galaxy outputs.")
    parser.add_argument("--input-dir", required=True, help="Directory containing *_pointwise_topostructure.csv files.")
    parser.add_argument("--output-dir", required=True, help="Directory where plot PNG files will be written.")
    parser.add_argument("--glob", default="*_pointwise_topostructure.csv", help="Input file glob pattern.")
    parser.add_argument("--dpi", type=int, default=160, help="PNG DPI. Default: 160")
    return parser.parse_args()



def infer_galaxy_name(df: pd.DataFrame, file_path: Path) -> str:
    if "galaxy" in df.columns and df["galaxy"].notna().any():
        return str(df["galaxy"].dropna().iloc[0]).strip()
    stem = file_path.stem
    suffix = "_pointwise_topostructure"
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem



def validate_columns(df: pd.DataFrame, file_path: Path) -> None:
    required = [
        "r_kpc",
        "v_obs_kms",
        "v_bar_kms",
        "v_eff_kms",
        "rotation_residual_bar_only_kms",
        "rotation_residual_kms",
        "sigma_i",
        "beta_i",
        "h_i_kpc",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path.name} missing required columns: {missing}")



def plot_one_file(file_path: Path, output_dir: Path, dpi: int) -> None:
    df = pd.read_csv(file_path)
    validate_columns(df, file_path)
    df = df.sort_values("r_kpc").reset_index(drop=True)
    galaxy = infer_galaxy_name(df, file_path)

    x = df["r_kpc"].to_numpy(dtype=float)
    v_obs = df["v_obs_kms"].to_numpy(dtype=float)
    v_bar = df["v_bar_kms"].to_numpy(dtype=float)
    v_eff = df["v_eff_kms"].to_numpy(dtype=float)
    resid_bar = df["rotation_residual_bar_only_kms"].to_numpy(dtype=float)
    resid_topo = df["rotation_residual_kms"].to_numpy(dtype=float)
    sigma = df["sigma_i"].to_numpy(dtype=float)
    beta = df["beta_i"].to_numpy(dtype=float)
    thickness = df["h_i_kpc"].to_numpy(dtype=float)

    has_err = "v_err_kms" in df.columns and np.isfinite(df["v_err_kms"]).any()
    err = df["v_err_kms"].to_numpy(dtype=float) if has_err else None

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"{galaxy} Topostructure Diagnostics", fontsize=14)

    # Panel 1: velocities
    ax = axes[0]
    if has_err:
        ax.errorbar(x, v_obs, yerr=err, fmt="o", markersize=4, linewidth=1, label="Observed")
    else:
        ax.plot(x, v_obs, marker="o", linewidth=1.2, label="Observed")
    ax.plot(x, v_bar, linewidth=1.6, label="Baryonic / Newtonian proxy")
    ax.plot(x, v_eff, linewidth=1.6, label="Topostructure model")
    ax.set_ylabel("Velocity (km/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Panel 2: residuals
    ax = axes[1]
    ax.axhline(0.0, linewidth=1.0)
    ax.plot(x, resid_bar, marker="o", linewidth=1.2, label="Residual: obs - baryon")
    ax.plot(x, resid_topo, marker="o", linewidth=1.2, label="Residual: obs - topo")
    ax.set_ylabel("Residual (km/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Panel 3: structural terms
    ax = axes[2]
    ax.plot(x, sigma, marker="o", linewidth=1.2, label="sigma")
    ax.plot(x, beta, marker="o", linewidth=1.2, label="beta")
    ax.set_ylabel("sigma / beta")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(x, thickness, linestyle="--", linewidth=1.2, label="thickness h")
    ax2.set_ylabel("Thickness (kpc)")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax.set_xlabel("Radius (kpc)")

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    output_path = output_dir / f"{galaxy.replace('/', '_').replace(' ', '_')}_topostructure_diagnostics.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)



def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files: List[Path] = sorted(input_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched {args.glob!r} in {input_dir}")

    ok = 0
    failed = 0
    for file_path in files:
        try:
            plot_one_file(file_path, output_dir, args.dpi)
            ok += 1
            print(f"[OK] {file_path.name}")
        except Exception as exc:
            failed += 1
            print(f"[FAILED] {file_path.name} -> {exc}")

    print("\nBatch plotting complete.")
    print(f"Output directory: {output_dir}")
    print(f"Success: {ok}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
